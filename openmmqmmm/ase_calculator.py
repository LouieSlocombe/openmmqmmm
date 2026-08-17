from typing import ClassVar

import numpy as np
from ase import Atoms, units
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator,
    CalculatorSetupError,
    PropertyNotImplementedError,
    all_changes,
)
from ase.symbols import symbols2numbers
from ase.utils import workdir

from openmmqmmm.exceptions import InputError
from openmmqmmm.qmmm import QMMMTheory


class OpenMMQMMMCalculator(Calculator):
    """Expose an openmmqmmm QM/MM theory as an ASE energy/forces calculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]
    _implemented_properties: ClassVar[frozenset[str]] = frozenset(implemented_properties)

    def __init__(
        self,
        qmmm_theory: QMMMTheory,
        *,
        charge: int | None = None,
        mult: int | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(qmmm_theory, QMMMTheory):
            raise CalculatorSetupError("qmmm_theory must be an openmmqmmm.QMMMTheory object")

        self.qmmm_theory = qmmm_theory
        try:
            self.charge, self.mult = qmmm_theory.resolve_qm_charge_mult(charge=charge, mult=mult)
        except InputError as err:
            raise CalculatorSetupError(str(err)) from err
        self._expected_symbols = tuple(qmmm_theory.elems)
        self._expected_numbers = np.asarray(symbols2numbers(self._expected_symbols))
        self._reference_cell = None
        self._reference_pbc = None
        super().__init__(**kwargs)

    def _validate_atoms(self, atoms: Atoms, system_changes: list[str]) -> np.ndarray:
        first_calculation = self._reference_cell is None
        if (first_calculation or "numbers" in system_changes) and not np.array_equal(
            atoms.numbers, self._expected_numbers
        ):
            raise CalculatorSetupError(
                "ASE atoms must retain the atom count, elements, and ordering of the Fragment used to create QMMMTheory"
            )

        # Calculator.calculate() owns a copy of the Atoms, so its position array can
        # be passed through without allocating another full-system copy here.
        positions = atoms.positions
        if not np.all(np.isfinite(positions)):
            raise CalculatorSetupError("ASE atom positions must all be finite")

        if first_calculation:
            self._reference_cell = atoms.cell.array.copy()
            self._reference_pbc = atoms.pbc.copy()
        elif (
            "cell" in system_changes and not np.allclose(atoms.cell.array, self._reference_cell, rtol=0.0, atol=1e-12)
        ) or ("pbc" in system_changes and not np.array_equal(atoms.pbc, self._reference_pbc)):
            raise CalculatorSetupError(
                "Changing the ASE cell or periodic boundary conditions is unsupported; QMMMTheory uses a fixed "
                "OpenMM system and does not provide stress"
            )

        return positions

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        """Calculate QM/MM energy and, when requested, atomic forces for ASE."""
        if properties is None:
            properties = self.implemented_properties
        unsupported = set(properties).difference(self._implemented_properties)
        if unsupported:
            raise PropertyNotImplementedError(f"Unsupported ASE properties: {sorted(unsupported)}")

        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise CalculatorSetupError("An ASE Atoms object is required")

        positions = self._validate_atoms(self.atoms, system_changes)
        needs_forces = "forces" in properties

        with workdir(self._directory):
            output = self.qmmm_theory.run(
                current_coords=positions,
                elems=self._expected_symbols,
                grad=needs_forces,
                label=self.prefix,
                charge=self.charge,
                mult=self.mult,
            )

        if needs_forces:
            energy, gradient = output
            gradient = np.asarray(gradient, dtype=float)
            if gradient.shape != positions.shape:
                raise CalculationFailed(
                    f"QMMMTheory returned gradient shape {gradient.shape}; expected {positions.shape}"
                )
            if not np.all(np.isfinite(gradient)):
                raise CalculationFailed("QMMMTheory returned a non-finite gradient")
            forces = -gradient * units.Hartree / units.Bohr
        else:
            energy = output

        energy = float(energy)
        if not np.isfinite(energy):
            raise CalculationFailed("QMMMTheory returned a non-finite energy")

        self.results = {"energy": energy * units.Hartree}
        if needs_forces:
            self.results["forces"] = forces

        # Keep the original Fragment useful to callers after an ASE optimization.
        coordinates = positions.copy()
        self.qmmm_theory.fragment.coords = coordinates
        self.qmmm_theory.coords = coordinates
