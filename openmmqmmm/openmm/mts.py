"""Experimental NVE multiple-time-step export with an embedded QM reference."""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import openmm
import openmm.app
import openmm.unit

from openmmqmmm.coords import check_charge_mult
from openmmqmmm.exceptions import InputError, MissingDependencyError
from openmmqmmm.openmm.nqe_export import modeller_from_topology
from openmmqmmm.openmm.rpmd_force import RPMD_PYTHON_FORCE_NAME, RPMDQMMMForceProvider
from openmmqmmm.openmm.theory import OpenMMTheory
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.utils import require_int_in_range, require_positive_finite

_CORRECTION_FORCE_NAME = "openmmqmmm MTS high-minus-reference correction"


@dataclasses.dataclass(frozen=True)
class MTSPotentialExport:
    """Experimental NVE System, Modeller, MTS integrator and independent QM providers."""

    system: openmm.System
    modeller: openmm.app.Modeller
    integrator: openmm.MTSIntegrator
    high_provider: RPMDQMMMForceProvider
    reference_provider: RPMDQMMMForceProvider
    reference_force: openmm.PythonForce
    correction_force: openmm.PythonForce
    fast_group: int
    slow_group: int
    inner_timestep: float
    inner_steps: int


class _QMCorrection:
    def __init__(self, high: RPMDQMMMForceProvider, reference: RPMDQMMMForceProvider) -> None:
        self.high = high
        self.reference = reference

    def __call__(self, state: openmm.State) -> tuple[float, npt.NDArray[np.float64]]:
        high_energy, high_forces = self.high(state)
        reference_energy, reference_forces = self.reference(state)
        return high_energy - reference_energy, high_forces - reference_forces


def _backend_chain(theory: Any) -> list[Any]:
    chain = []
    seen = set()
    while theory is not None and id(theory) not in seen:
        seen.add(id(theory))
        chain.append(theory)
        theory = getattr(theory, "theory", None)
    return chain


def _validate_reference(high: Any, reference: Any) -> None:
    from openmmqmmm.orca import ORCATheory

    if not callable(getattr(reference, "run", None)):
        raise InputError("reference_qm_theory must provide run() with consistent energies and gradients")
    high_chain, reference_chain = _backend_chain(high), _backend_chain(reference)
    if {id(backend) for backend in high_chain}.intersection(id(backend) for backend in reference_chain):
        raise InputError("The high and reference QM theories must be independent objects")
    high_files = {
        Path(backend.filename).expanduser().resolve() for backend in high_chain if isinstance(backend, ORCATheory)
    }
    reference_files = {
        Path(backend.filename).expanduser().resolve() for backend in reference_chain if isinstance(backend, ORCATheory)
    }
    if high_files.intersection(reference_files):
        raise InputError("High and reference ORCA theories must use distinct filenames to isolate outputs and orbitals")
    require_int_in_range(getattr(reference, "numcores", 1), "The reference QM numcores must be a positive integer")


def _native_groups(system: openmm.System) -> set[int]:
    groups = {force.getForceGroup() for force in system.getForces()}
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            reciprocal = force.getReciprocalSpaceForceGroup()
            if reciprocal >= 0:
                groups.add(reciprocal)
    return groups


def export_mts_potential(
    *,
    theory: QMMMTheory,
    reference_qm_theory: Any,
    inner_timestep: float = 0.00025,
    inner_steps: int = 2,
    charge: int | None = None,
    mult: int | None = None,
) -> MTSPotentialExport:
    """Export experimental NVE QM/MM with fast MM/reference and slow high-minus-reference forces.

    Timesteps are in ps; one integrator step advances ``inner_timestep * inner_steps``.
    The returned System is the theory's MM System. Successful export switches the theory
    to external-force mode, so use a fresh theory for subsequent standalone calculations.
    Both QM backends must define conservative potentials with the same electronic state
    and support the selected embedding. This prototype supplies no thermostat or barostat.
    """
    if not isinstance(theory, QMMMTheory) or not isinstance(theory.mm_theory, OpenMMTheory):
        raise InputError("export_mts_potential requires a QMMMTheory with an OpenMMTheory MM system")
    inner_timestep = require_positive_finite(inner_timestep, "inner_timestep must be finite and positive (ps)")
    inner_steps = require_int_in_range(inner_steps, "inner_steps must be a positive integer")
    if theory.truncated_pc or theory.update_qm_region_charges:
        raise InputError(
            "MTS does not support truncated_pc or update_qm_region_charges: each term needs a fixed potential"
        )
    if theory.linkatoms and theory.linkatom_forceproj_method == "none":
        raise InputError("MTS requires link-atom force projection for a conservative potential")
    if theory.openmm_externalforce or theory.exit_after_customexternalforce_update:
        raise InputError("MTS requires a fresh standalone QM/MM theory, not an already exported or MD-bound theory")
    if not hasattr(openmm, "PythonForce"):
        raise MissingDependencyError("MTS QM/MM requires OpenMM with PythonForce support")
    system = theory.mm_theory.system
    excluded = (
        openmm.DrudeForce,
        openmm.AndersenThermostat,
        openmm.MonteCarloBarostat,
        openmm.MonteCarloAnisotropicBarostat,
        openmm.MonteCarloMembraneBarostat,
        openmm.MonteCarloFlexibleBarostat,
    )
    for force in system.getForces():
        if isinstance(force, excluded):
            raise InputError(f"Experimental NVE MTS does not support {type(force).__name__}")
        if isinstance(force, openmm.PythonForce):
            raise InputError("MTS cannot attach to a System that already carries a PythonForce")
    _validate_reference(theory.qm_theory, reference_qm_theory)
    charge, mult = check_charge_mult(
        charge, mult, theory.theorytype, theory.fragment, "export_mts_potential", theory=theory
    )
    groups = _native_groups(system)
    available = [group for group in range(31, -1, -1) if group not in groups]
    if len(available) < 2:
        raise InputError("MTS requires two unused force groups, including any reciprocal-space groups")
    fast_group, slow_group = available[:2]

    # Preserve prepared boundary/charge bookkeeping without applying QM/MM force
    # surgery twice to the shared native MM system or copying a live MM Context.
    reference = copy.deepcopy(
        theory, {id(theory.mm_theory): theory.mm_theory, id(theory.qm_theory): reference_qm_theory}
    )
    reference.qm_theory_name = type(reference_qm_theory).__name__
    reference.numcores = int(getattr(reference_qm_theory, "numcores", 1))
    coords = reference._image_periodic_coords(theory.fragment.coords)
    modeller = modeller_from_topology(topology=theory.mm_theory.topology, coords_angstrom=coords)
    periodic = bool(theory.mm_theory.periodic)
    high_provider = RPMDQMMMForceProvider(theory, theory.elems, charge, mult, periodic=periodic)
    reference_provider = RPMDQMMMForceProvider(reference, theory.elems, charge, mult, periodic=periodic)
    reference_force = openmm.PythonForce(reference_provider)
    correction_force = openmm.PythonForce(_QMCorrection(high_provider, reference_provider))
    for force, group, name in (
        (reference_force, fast_group, RPMD_PYTHON_FORCE_NAME),
        (correction_force, slow_group, _CORRECTION_FORCE_NAME),
    ):
        force.setName(name)
        force.setForceGroup(group)
        force.setUsesPeriodicBoundaryConditions(periodic)
    schedule = [(slow_group, 1), *((group, inner_steps) for group in sorted(groups | {fast_group}))]
    integrator = openmm.MTSIntegrator(inner_timestep * inner_steps * openmm.unit.picosecond, schedule)
    original_force_count = system.getNumForces()
    missing = object()
    original_qmatoms = getattr(reference_qm_theory, "qmatoms", missing)
    assigned_qmatoms = False
    try:
        reference_qm_theory.qmatoms = reference.qmatoms
        assigned_qmatoms = True
        system.addForce(reference_force)
        system.addForce(correction_force)
    except Exception:
        while system.getNumForces() > original_force_count:
            system.removeForce(system.getNumForces() - 1)
        if assigned_qmatoms:
            if original_qmatoms is missing:
                del reference_qm_theory.qmatoms
            else:
                reference_qm_theory.qmatoms = original_qmatoms
        raise
    theory.openmm_externalforce = reference.openmm_externalforce = True
    theory.exit_after_customexternalforce_update = reference.exit_after_customexternalforce_update = True
    return MTSPotentialExport(
        system=system,
        modeller=modeller,
        integrator=integrator,
        high_provider=high_provider,
        reference_provider=reference_provider,
        reference_force=reference_force,
        correction_force=correction_force,
        fast_group=fast_group,
        slow_group=slow_group,
        inner_timestep=inner_timestep,
        inner_steps=inner_steps,
    )
