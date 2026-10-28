from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import openmm

from openmmqmmm._mace_polar import _create_polar_potential, _polar_name, _polar_options
from openmmqmmm.exceptions import InputError, MissingDependencyError, require
from openmmqmmm.openmm.theory import OpenMMTheory

logger = logging.getLogger(__name__)

_SYSTEM_ARGUMENTS = frozenset(
    {"topology", "system", "atoms", "removeConstraints", "embedding", "interpolate", "returnInfo"}
)
_TRANSIENT_ATTRIBUTES = frozenset(
    {
        "system",
        "nonbonded_force",
        "forcegroups",
        "simulation",
        "integrator",
        "restraint",
        "energy",
        "gradient",
        "cell_gradient",
        "_cell_gradient_snapshot",
        "_exception_charge_scales",
    }
)
_SUPPORTED_FORCES = (
    openmm.HarmonicBondForce,
    openmm.HarmonicAngleForce,
    openmm.PeriodicTorsionForce,
    openmm.RBTorsionForce,
    openmm.CMAPTorsionForce,
    openmm.CustomBondForce,
    openmm.CustomAngleForce,
    openmm.CustomTorsionForce,
    openmm.NonbondedForce,
    openmm.CustomNonbondedForce,
    openmm.CustomExternalForce,
    openmm.CMMotionRemover,
    openmm.AndersenThermostat,
    openmm.MonteCarloBarostat,
    openmm.MonteCarloAnisotropicBarostat,
    openmm.MonteCarloMembraneBarostat,
    openmm.MonteCarloFlexibleBarostat,
)


class MLMMTheory(OpenMMTheory):
    """Mechanically embedded OpenMM-ML potential for whole molecules in an independent MM system."""

    def __init__(
        self,
        *,
        mm_theory: OpenMMTheory,
        mlatoms: Sequence[int],
        potential: Any,
        potential_kwargs: Mapping[str, Any] | None = None,
        system_kwargs: Mapping[str, Any] | None = None,
        remove_constraints: bool = True,
        label: str = "ML/MM",
    ) -> None:
        if not isinstance(mm_theory, OpenMMTheory) or isinstance(mm_theory, MLMMTheory):
            raise InputError("mm_theory must be a classical OpenMMTheory, not an MLMMTheory")
        if getattr(mm_theory, "_qmmm_prepared", False):
            raise InputError("mm_theory has already been modified for QM/MM; build a fresh classical OpenMMTheory")
        if not isinstance(remove_constraints, bool):
            raise InputError("remove_constraints must be a bool")

        selected = _validate_region(mm_theory, mlatoms)
        _validate_forces(mm_theory.system, set(selected))
        potential_options = _options(potential_kwargs, "potential_kwargs")
        system_options = _options(system_kwargs, "system_kwargs")
        reserved = _SYSTEM_ARGUMENTS.intersection(system_options)
        if reserved:
            raise InputError(f"system_kwargs cannot override MLMMTheory arguments: {sorted(reserved)}")
        if not isinstance(potential, str) and potential_options:
            raise InputError("potential_kwargs can only be used with a potential name")

        polar_name = _polar_name(potential)
        polar_settings = None
        if polar_name is not None:
            if not isinstance(potential, str):
                model_path = getattr(potential._impl, "modelPath", None)
                if model_path is not None:
                    potential_options["modelPath"] = model_path
            polar_model, polar_settings = _polar_options(
                polar_name, potential_options, system_options, mm_theory.system, mm_theory.periodic
            )
        if isinstance(potential, str) or polar_name is not None:
            if isinstance(potential, str) and not potential.strip():
                raise InputError("potential must be a nonempty OpenMM-ML potential name")
            openmmml = require(
                "openmmml",
                feature="MLMMTheory",
                hint="conda env update -n openmmqmmm -f build_tools/environment_ml.yml",
            )
            try:
                if polar_name is not None:
                    ml_potential, system_options = _create_polar_potential(openmmml, polar_model, polar_settings)
                else:
                    ml_potential = openmmml.MLPotential(potential, **potential_options)
            except KeyError as exc:
                raise InputError(f"Unknown OpenMM-ML potential or model option: {potential!r}: {exc}") from exc
            except ImportError as exc:
                if isinstance(exc, MissingDependencyError):
                    raise
                raise MissingDependencyError(
                    f"The OpenMM-ML backend for {potential!r} is unavailable; install its model dependencies."
                ) from exc
            potential_name = polar_name or potential
        else:
            ml_potential = potential
            potential_name = type(potential).__name__
        if not callable(getattr(ml_potential, "createMixedSystem", None)):
            raise InputError("potential must be an OpenMM-ML name or an object providing createMixedSystem")

        # Keep topology/loader aliases together, but never copy a Context or its
        # bound Integrator, cached results, or force wrappers from the old System.
        state = {key: value for key, value in vars(mm_theory).items() if key not in _TRANSIENT_ATTRIBUTES}
        self.__dict__.update(copy.deepcopy(state))
        if polar_settings is not None:
            self.ml_charge = polar_settings["charge"]
            self.ml_mult = polar_settings["multiplicity"]
            self.ml_precision = polar_settings["precision"]
            self.ml_model = polar_model
            logger.info("MACE-POLAR model/checkpoint: %s", polar_model)
            # The source theory's boundary condition is authoritative, including
            # nonperiodic systems loaded from PDB files containing a CRYST1 cell.
            self.topology.setPeriodicBoxVectors(
                mm_theory.system.getDefaultPeriodicBoxVectors() if self.periodic else None
            )
        classical_system = copy.deepcopy(mm_theory.system)
        restraints = []
        for index in reversed(range(classical_system.getNumForces())):
            force = classical_system.getForce(index)
            if force.getName() == "OpenMMQMMM restraint":
                # Upstream removes internal bonded terms, including user restraints.
                restraints.append(copy.deepcopy(force))
                classical_system.removeForce(index)
        try:
            mixed_system = ml_potential.createMixedSystem(
                self.topology,
                classical_system,
                selected,
                removeConstraints=remove_constraints,
                embedding="mechanical",
                interpolate=False,
                returnInfo=False,
                **system_options,
            )
        except ImportError as exc:
            raise MissingDependencyError(
                f"The OpenMM-ML backend for {potential_name!r} is unavailable; install its model dependencies."
            ) from exc
        if not isinstance(mixed_system, openmm.System):
            raise InputError("OpenMM-ML createMixedSystem must return an OpenMM System")
        if mixed_system.getNumParticles() != mm_theory.system.getNumParticles():
            raise InputError("MLMMTheory requires a mixed System with unchanged particle count")
        for force in reversed(restraints):
            mixed_system.addForce(force)

        self.system = mixed_system
        self.mlatoms = selected
        self.potential_name = potential_name
        self.embedding = "mechanical"
        self.remove_constraints = remove_constraints
        self.label = label
        self.theorytype = "MM"
        self.theorynamelabel = "MLMMTheory"
        self.numatoms = self.system.getNumParticles()
        self.allatoms = list(range(self.numatoms))
        self.charges = []
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                self.nonbonded_force = force
        self.getatomcharges()
        self.compute_dof()
        if self.do_energy_decomposition:
            self.forcegroupify()
        logger.info("ML/MM potential %s: %s ML atoms, mechanical embedding", potential_name, len(selected))


def _options(options: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping) or any(not isinstance(key, str) for key in options):
        raise InputError(f"{name} must be a mapping with string keys")
    return dict(options)


def _validate_region(mm_theory: OpenMMTheory, mlatoms: Sequence[int]) -> list[int]:
    system = mm_theory.system
    topology = mm_theory.topology
    if topology.getNumAtoms() != system.getNumParticles():
        raise InputError("ML/MM requires matching topology atom and System particle counts")
    try:
        indices = list(mlatoms)
    except TypeError as exc:
        raise InputError("mlatoms must be a nonempty sequence of integer atom indices") from exc
    if not indices:
        raise InputError("mlatoms must be a nonempty sequence of integer atom indices")
    if any(isinstance(atom, (bool, np.bool_)) or not isinstance(atom, (int, np.integer)) for atom in indices):
        raise InputError("mlatoms must contain integer atom indices, not bools or floats")
    indices = [int(atom) for atom in indices]
    if len(set(indices)) != len(indices):
        raise InputError("mlatoms contains duplicate atom indices")
    if any(atom < 0 or atom >= system.getNumParticles() for atom in indices):
        raise InputError(f"mlatoms indices must be between 0 and {system.getNumParticles() - 1}")
    selected = set(indices)
    for atom in selected:
        if system.isVirtualSite(atom):
            raise InputError("Native OpenMM virtual sites cannot be included in mlatoms")
    for atom in range(system.getNumParticles()):
        if system.isVirtualSite(atom):
            site = system.getVirtualSite(atom)
            if any(site.getParticle(parent) in selected for parent in range(site.getNumParticles())):
                raise InputError("Virtual sites and their parent atoms must remain entirely in the MM region")
    if any(atom.element is None for atom in topology.atoms()):
        # OpenMM-ML 1.8 reads atomic_number for every atom, even without link atoms.
        raise InputError(
            "OpenMM-ML requires elements for all topology atoms; element-less extra particles are unsupported"
        )
    if any((first.index in selected) != (second.index in selected) for first, second in topology.bonds()):
        raise InputError("ML/MM requires whole molecules: a covalent bond crosses the ML/MM boundary")
    return sorted(indices)


def _validate_forces(system: openmm.System, selected: set[int]) -> None:
    for force in system.getForces():
        if force.getName() == "OpenMMQMMM restraint":
            continue
        if not isinstance(force, _SUPPORTED_FORCES):
            raise InputError(
                f"ML/MM mechanical embedding does not support {type(force).__name__}; "
                "its internal ML-region interactions cannot safely be removed"
            )
        if isinstance(force, openmm.NonbondedForce):
            for offset in range(force.getNumParticleParameterOffsets()):
                _name, atom, *_scales = force.getParticleParameterOffset(offset)
                if atom in selected:
                    raise InputError("Nonbonded particle parameter offsets on ML atoms are unsupported")
            for offset in range(force.getNumExceptionParameterOffsets()):
                _name, exception, *_scales = force.getExceptionParameterOffset(offset)
                first, second, *_parameters = force.getExceptionParameters(exception)
                if first in selected and second in selected:
                    raise InputError("Nonbonded exception parameter offsets within the ML region are unsupported")
