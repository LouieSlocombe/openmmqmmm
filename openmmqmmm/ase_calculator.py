from pathlib import Path
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
from ase.utils import workdir

from openmmqmmm.qmmm import QMMMTheory


class OpenMMQMMMCalculator(Calculator):
    """Expose an openmmqmmm QM/MM theory as an ASE energy/forces calculator."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

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
        self.charge = self._resolve_electronic_state("charge", charge)
        self.mult = self._resolve_electronic_state("mult", mult)
        self._expected_symbols = tuple(qmmm_theory.elems)
        self._reference_cell = None
        self._reference_pbc = None
        super().__init__(**kwargs)

    def _resolve_electronic_state(self, name: str, supplied_value: int | None) -> int:
        theory_value = getattr(self.qmmm_theory, f"qm_{name}", None)
        if theory_value is not None:
            if supplied_value is not None and supplied_value != theory_value:
                raise CalculatorSetupError(
                    f"{name}={supplied_value} conflicts with QMMMTheory.qm_{name}={theory_value}"
                )
            return theory_value

        if supplied_value is not None:
            return supplied_value

        fragment_value = getattr(self.qmmm_theory.fragment, name, None)
        if fragment_value is not None:
            return fragment_value

        raise CalculatorSetupError(
            f"QM-region {name} is undefined; set QMMMTheory.qm_{name} or pass {name}= to the calculator"
        )

    def _validate_atoms(self, atoms: Atoms) -> np.ndarray:
        symbols = tuple(atoms.get_chemical_symbols())
        if symbols != self._expected_symbols:
            raise CalculatorSetupError(
                "ASE atoms must retain the atom count, elements, and ordering of the Fragment used to create QMMMTheory"
            )

        positions = np.asarray(atoms.get_positions())
        if not np.all(np.isfinite(positions)):
            raise CalculatorSetupError("ASE atom positions must all be finite")

        cell = np.asarray(atoms.cell)
        pbc = np.asarray(atoms.pbc)
        if self._reference_cell is None:
            self._reference_cell = cell.copy()
            self._reference_pbc = pbc.copy()
        elif not np.allclose(cell, self._reference_cell, rtol=0.0, atol=1e-12) or not np.array_equal(
            pbc, self._reference_pbc
        ):
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
        unsupported = set(properties) - set(self.implemented_properties)
        if unsupported:
            raise PropertyNotImplementedError(f"Unsupported ASE properties: {sorted(unsupported)}")

        super().calculate(atoms, properties, system_changes)
        if self.atoms is None:
            raise CalculatorSetupError("An ASE Atoms object is required")

        positions = self._validate_atoms(self.atoms)
        symbols = self.atoms.get_chemical_symbols()
        needs_forces = "forces" in properties
        calculation_directory = Path(self._directory).resolve()

        with workdir(calculation_directory):
            output = self.qmmm_theory.run(
                current_coords=positions,
                elems=symbols,
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
        self.qmmm_theory.fragment.coords = positions.copy()
        self.qmmm_theory.coords = positions.copy()
