from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import openmm
import openmm.unit

import openmmqmmm.constants
from openmmqmmm.exceptions import InputError, InternalError, MissingDependencyError

logger = logging.getLogger(__name__)

RPMD_PYTHON_FORCE_NAME = "openmmqmmm bead-specific external force"


class _RPMDPythonForceProvider:
    """Evaluate an external theory for the coordinates OpenMM is currently processing."""

    def __init__(
        self,
        theory: Any,
        elems: Sequence[str],
        charge: int,
        mult: int,
        *,
        periodic: bool = False,
        cache_size: int = 64,
    ) -> None:
        self.theory = theory
        self.elems = tuple(elems)
        self.charge = charge
        self.mult = mult
        self.periodic = bool(periodic)
        self.cache_size = max(1, int(cache_size))
        self.evaluation_count = 0
        self.cache_hits = 0
        self.last_energy_hartree = None
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        # Cached force arrays are disposable and can make serialized Systems very large.
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._cache = OrderedDict(self._cache)
        self._lock = threading.RLock()

    def _evaluate(self, coords_angstrom: npt.NDArray[np.float64]) -> tuple[float, npt.ArrayLike]:
        raise NotImplementedError

    def _cache_key(self, state: openmm.State, positions_nm: npt.NDArray[np.float64]) -> tuple[bytes, ...]:
        key = [np.ascontiguousarray(positions_nm, dtype=np.float64).tobytes()]
        if self.periodic:
            box_nm = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(openmm.unit.nanometer)
            key.append(np.ascontiguousarray(box_nm, dtype=np.float64).tobytes())
        return tuple(key)

    def __call__(self, state: openmm.State) -> tuple[float, npt.NDArray[np.float64]]:
        positions_nm = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer), dtype=np.float64
        )
        expected_shape = (len(self.elems), 3)
        if positions_nm.shape != expected_shape:
            raise InternalError(
                f"RPMD external force received positions with shape {positions_nm.shape}; expected {expected_shape}."
            )

        key = self._cache_key(state, positions_nm)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                self._cache.move_to_end(key)
                energy_kj_mol, forces_kj_mol_nm = cached
                return energy_kj_mol, forces_kj_mol_nm.copy()

            coords_angstrom = positions_nm * 10.0
            try:
                energy_hartree, gradient = self._evaluate(coords_angstrom)
            except Exception as error:
                raise RuntimeError(
                    f"RPMD external-force evaluation failed for {len(self.elems)} atoms: {error}"
                ) from error

            energy_hartree = float(energy_hartree)
            gradient = np.asarray(gradient, dtype=np.float64)
            if gradient.shape != expected_shape:
                raise InternalError(
                    f"RPMD external theory returned a gradient with shape {gradient.shape}; expected {expected_shape}."
                )
            if not np.isfinite(energy_hartree) or not np.all(np.isfinite(gradient)):
                raise InternalError("RPMD external theory returned a non-finite energy or gradient.")

            energy_kj_mol = energy_hartree * openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
            forces_kj_mol_nm = -gradient * openmmqmmm.constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
            self.evaluation_count += 1
            self.last_energy_hartree = energy_hartree
            self._cache[key] = (energy_kj_mol, forces_kj_mol_nm.copy())
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return energy_kj_mol, forces_kj_mol_nm

    def clear_cache(self) -> None:
        """Discard cached coordinate evaluations without resetting lifetime statistics."""
        with self._lock:
            self._cache.clear()


class RPMDQMMMForceProvider(_RPMDPythonForceProvider):
    """Provide bead-specific QM/MM energies and gradients to ``openmm.PythonForce``."""

    def _evaluate(self, coords_angstrom: npt.NDArray[np.float64]) -> tuple[float, npt.ArrayLike]:
        return self.theory.run_openmm_python_force(
            current_coords=coords_angstrom,
            elems=self.elems,
            charge=self.charge,
            mult=self.mult,
        )


class RPMDExternalQMForceProvider(_RPMDPythonForceProvider):
    """Provide bead-specific full-QM energies and gradients to ``openmm.PythonForce``."""

    def _evaluate(self, coords_angstrom: npt.NDArray[np.float64]) -> tuple[float, npt.ArrayLike]:
        result = self.theory.run(
            current_coords=coords_angstrom,
            elems=self.elems,
            grad=True,
            charge=self.charge,
            mult=self.mult,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise InternalError("External QM theory must return an (energy, gradient) pair when grad=True.")
        return result


def add_rpmd_python_force(
    system: openmm.System,
    provider: _RPMDPythonForceProvider,
    *,
    periodic: bool = False,
) -> tuple[openmm.PythonForce, int]:
    """Add an isolated-force-group ``PythonForce`` and return it with its group index."""
    if not hasattr(openmm, "PythonForce"):
        raise MissingDependencyError("QM/MM RPMD requires OpenMM 8.5 or newer with openmm.PythonForce support.")

    if any(force.getName() == RPMD_PYTHON_FORCE_NAME for force in system.getForces()):
        raise InputError(
            "This OpenMM System already carries the openmmqmmm bead-specific PythonForce from a previous "
            "MolecularDynamicsEngine or export_rpmd_potential call. A second one would silently double the "
            "QM force; reuse the existing engine or export, or build a fresh theory object."
        )

    used_groups = {force.getForceGroup() for force in system.getForces()}
    force_group = next((group for group in range(31, -1, -1) if group not in used_groups), None)
    if force_group is None:
        raise InputError("QM/MM RPMD requires a dedicated OpenMM force group, but all 32 groups are already used.")

    force = openmm.PythonForce(provider)
    force.setName(RPMD_PYTHON_FORCE_NAME)
    force.setForceGroup(force_group)
    force.setUsesPeriodicBoundaryConditions(bool(periodic))
    system.addForce(force)
    logger.info("Added bead-specific PythonForce in force group %s", force_group)
    return force, force_group
