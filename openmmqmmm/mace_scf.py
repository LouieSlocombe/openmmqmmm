"""Finite MACE-SCF calculations through the QM theory protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike, fspath
from typing import Any

import numpy as np
from ase import Atoms

from openmmqmmm.constants import BOHR_TO_ANG, HARTREE_TO_EV
from openmmqmmm.exceptions import InputError, InternalError, require
from openmmqmmm.utils import require_int_in_range

_CALCULATORS = {
    "fixedpoint": ("mace_scf.calculators.fixedpoint_scf", "MACEFixedPointSCF"),
    "qeq": ("mace_scf.calculators.qeq", "MACEQEqCalculator"),
}
_GRADIENT_FACTOR = -BOHR_TO_ANG / HARTREE_TO_EV


class MACESCFTheory:
    """Use a finite MACE-SCF model as the QM backend of :class:`QMMMTheory`.

    Supply either ``model_path`` (a FixedPoint or QEq checkpoint selected by
    ``model_type``) or an existing ``calculator`` with MACE-SCF's ASE embedding
    interface. Checkpoint loading imports the optional ``mace_scf`` package only
    when needed. ``calculator_kwargs`` forwards model-specific constructor
    options, including ``scf_options`` for FixedPoint models.

    Coordinates are in Angstrom, charges in elementary-charge units, and
    returned energies and gradients in Hartree and Hartree/Bohr. The calculator
    receives nonperiodic QM atoms, even inside a periodic OpenMM system.
    ``mult=1`` is required: these models have no spin/multiplicity input.
    ``numcores`` is protocol metadata; configure PyTorch threads separately.
    An injected calculator is owned by this theory and must not be shared with
    another running calculation.
    """

    def __init__(
        self,
        *,
        model_path: str | PathLike[str] | None = None,
        model_type: str = "fixedpoint",
        calculator: Any | None = None,
        device: str = "cpu",
        default_dtype: str = "float64",
        calculator_kwargs: Mapping[str, Any] | None = None,
        numcores: int = 1,
        label: str = "MACE-SCF",
    ) -> None:
        if (model_path is None) == (calculator is None):
            raise InputError("Supply exactly one of model_path or calculator to MACESCFTheory")
        if model_type not in _CALCULATORS:
            raise InputError("model_type must be 'fixedpoint' or 'qeq'")
        if calculator is not None and calculator_kwargs:
            raise InputError("calculator_kwargs cannot be used with an existing calculator")
        self.set_numcores(numcores)
        if calculator is None:
            options = dict(calculator_kwargs or {})
            reserved = {"model_path", "device", "default_dtype"}.intersection(options)
            if reserved:
                raise InputError(f"Pass {sorted(reserved)} directly, not in calculator_kwargs")
            options.setdefault("pbc_handling", "realspace")
            if model_type == "fixedpoint":
                options.setdefault("require_converged", True)
            module_name, class_name = _CALCULATORS[model_type]
            module = require(
                module_name,
                hint="python -m pip install -e /path/to/mace-scf (requires its supported Python version)",
                feature="MACESCFTheory checkpoint loading",
            )
            calculator = getattr(module, class_name)(
                model_path=fspath(model_path), device=device, default_dtype=default_dtype, **options
            )
        if not callable(getattr(calculator, "calculate", None)) or not callable(getattr(calculator, "embed", None)):
            raise InputError("calculator must provide MACE-SCF's calculate() and embed() interface")
        self.calculator = calculator
        self._validate_calculator()
        self.theorytype = "QM"
        self.theorynamelabel = "MACESCFTheory"
        self.analytic_hessian = False
        self.label = label
        self.qmatoms: list[int] = []
        self.runcalls = 0
        self._environment: Any | None = None
        self._environment_size: int | None = None
        self._atomic_charges: np.ndarray | None = None
        self.energy: float | None = None
        self.gradient: np.ndarray | None = None
        self.pcgradient: np.ndarray | None = None

    def _validate_calculator(self) -> None:
        calc = self.calculator
        keyspec = getattr(calc, "keyspec", None)
        if keyspec is None:
            keyspec = getattr(calc, "keyspec_singlepoint", None)
        if keyspec is not None and "total_charge" not in keyspec.info_keys:
            raise InputError("MACESCFTheory requires a total_charge input; use a FixedPoint or QEq calculator")
        if getattr(calc, "use_compile", False) or getattr(calc, "compiled_evaluator", None) is not None:
            raise InputError("MACESCFTheory requires eager evaluation (use_compile=False)")
        scf_options = getattr(calc, "scf_options", None)
        constant_charge = (
            scf_options.get("constant_charge", True)
            if isinstance(scf_options, Mapping)
            else getattr(scf_options, "constant_charge", True)
        )
        if not constant_charge:
            raise InputError("MACESCFTheory requires constant_charge=True to honor the QM-region charge")

    def _charge_key(self) -> str:
        calc = self.calculator
        keyspec = getattr(calc, "keyspec", None)
        if keyspec is None:
            keyspec = getattr(calc, "keyspec_singlepoint", None)
        return getattr(keyspec, "info_keys", {}).get("total_charge", getattr(calc, "total_charge_key", "total_charge"))

    @staticmethod
    def _coordinates(values: Any, count: int, name: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise InputError(f"{name} must be finite coordinates with shape ({count}, 3)") from error
        if count == 0 and array.size == 0:
            array = array.reshape(0, 3)
        if array.shape != (count, 3) or not np.isfinite(array).all():
            raise InputError(f"{name} must be finite coordinates with shape ({count}, 3)")
        return array

    @staticmethod
    def _result_array(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise InternalError(f"MACE-SCF returned invalid {name}") from error
        if array.shape != shape or not np.isfinite(array).all():
            raise InternalError(f"MACE-SCF returned invalid {name}; expected finite shape {shape}")
        return array

    def _set_environment(self, charges: np.ndarray, positions: np.ndarray) -> None:
        # Replacing the count is necessary for truncated fields and boundary
        # charge sites. The setters invalidate SCF history when the field moves.
        if self._environment_size != len(charges) or getattr(self.calculator, "pcpot", None) is not self._environment:
            self._environment = self.calculator.embed(charges)
            self._environment_size = len(charges)
        else:
            self._environment.set_charges(charges)
        self._environment.set_positions(positions)

    def run(
        self,
        *,
        current_coords: Sequence[Sequence[float]] | np.ndarray | None = None,
        elems: Sequence[str] | None = None,
        qm_elems: Sequence[str] | None = None,
        charge: float | None = None,
        mult: int | None = None,
        grad: bool = False,
        pc: bool = False,
        current_mm_coords: Sequence[Sequence[float]] | np.ndarray | None = None,
        mm_charges: Sequence[float] | np.ndarray | None = None,
        numcores: int | None = None,
        label: str | None = None,
        hessian: bool = False,
    ) -> float | tuple[float, np.ndarray] | tuple[float, np.ndarray, np.ndarray]:
        """Return energy, optionally QM gradients and gradients on MM charge sites.

        ``pc=True, grad=True`` returns ``(energy, qm_gradient, pc_gradient)``;
        without embedding it returns ``(energy, gradient)``. Energy-only calls
        return a scalar. The model's total energy already includes QM-MM
        electrostatics; MM internal energy is the driver's responsibility.
        """
        if hessian:
            raise InputError("MACESCFTheory does not provide analytic Hessians; use numerical_frequencies")
        if isinstance(mult, (bool, np.bool_)) or mult != 1:
            raise InputError("MACESCFTheory requires mult=1; MACE-SCF has no spin/multiplicity input")
        if isinstance(charge, (bool, np.bool_)) or not isinstance(charge, (int, float, np.integer, np.floating)):
            raise InputError("charge must be a finite number")
        if not np.isfinite(charge):
            raise InputError("charge must be a finite number")
        if numcores is not None:
            require_int_in_range(numcores, "numcores must be a positive integer")
        symbols = qm_elems if qm_elems is not None else elems
        if symbols is None or len(symbols) == 0:
            raise InputError("Provide nonempty elems or qm_elems")
        coords = self._coordinates(current_coords, len(symbols), "current_coords")
        try:
            atoms = Atoms(symbols=symbols, positions=coords, pbc=False)
        except (KeyError, TypeError, ValueError) as error:
            raise InputError("elems or qm_elems must contain valid chemical symbols") from error
        atoms.info[self._charge_key()] = float(charge)
        charges = np.empty(0)
        positions = np.empty((0, 3))
        if pc:
            if mm_charges is None or current_mm_coords is None:
                raise InputError("pc=True requires mm_charges and current_mm_coords (empty arrays are allowed)")
            try:
                charges = np.asarray(mm_charges, dtype=float)
            except (TypeError, ValueError) as error:
                raise InputError("mm_charges must be a finite one-dimensional array") from error
            if charges.ndim != 1 or not np.isfinite(charges).all():
                raise InputError("mm_charges must be a finite one-dimensional array")
            positions = self._coordinates(current_mm_coords, len(charges), "current_mm_coords")
        self._validate_calculator()
        self.energy = self.gradient = self.pcgradient = self._atomic_charges = None
        self._set_environment(charges, positions)
        properties = ["energy"]
        if grad:
            properties.append("forces")
            if pc and len(charges):
                properties.append("point_charge_forces")
        # Request all needed quantities in one SCF evaluation. Calling directly
        # also prevents ASE geometry caches from hiding electronic-state changes.
        self.calculator.results = {}
        self.calculator.calculate(atoms, properties=properties)
        self.runcalls += 1
        results = self.calculator.results
        for key in ("density_converged", "charge_converged"):
            if key in results and not np.all(results[key]):
                raise InternalError(f"MACE-SCF did not converge: {key}=False")
        energy = float(self._result_array(results.get("energy"), (), "energy")) / HARTREE_TO_EV
        gradient = pcgradient = None
        if grad:
            gradient = self._result_array(results.get("forces"), coords.shape, "QM forces") * _GRADIENT_FACTOR
            if pc:
                forces = self._environment.get_forces(self.calculator) if len(charges) else np.empty((0, 3))
                pcgradient = self._result_array(forces, positions.shape, "point-charge forces") * _GRADIENT_FACTOR
        populations = results.get("partial_charges")
        if populations is None and "density_coefficients" in results:
            density = np.asarray(results["density_coefficients"])
            if density.ndim == 2 and density.shape[1] > 0:
                populations = density[:, 0]
        if populations is not None:
            self._atomic_charges = self._result_array(populations, (len(atoms),), "atomic charges").copy()
        self.energy, self.gradient, self.pcgradient = energy, gradient, pcgradient
        if grad:
            return (energy, gradient, pcgradient) if pc else (energy, gradient)
        return energy

    def get_atomic_charges(self) -> np.ndarray:
        """Return QM populations from the last successful calculation, in input order."""
        if self._atomic_charges is None:
            raise InputError("No atomic charges are available; run a model that returns QM populations first")
        return self._atomic_charges.copy()

    def set_numcores(self, numcores: int) -> None:
        """Store the driver's core-count metadata without changing global PyTorch threads."""
        self.numcores = require_int_in_range(numcores, "numcores must be a positive integer")

    def cleanup(self) -> None:
        """Clear cached results and embedding state while retaining the loaded model."""
        self.calculator.embed([])
        self.calculator.reset()
        self._environment = self._environment_size = None
        self.energy = self.gradient = self.pcgradient = self._atomic_charges = None
