"""Internal AMOEBA force partitioning for mechanical QM/MM embedding."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import openmm
import openmm.app

from openmmqmmm.exceptions import InputError

logger = logging.getLogger(__name__)

_BONDED_TYPES = (
    openmm.HarmonicBondForce,
    openmm.HarmonicAngleForce,
    openmm.PeriodicTorsionForce,
    openmm.RBTorsionForce,
    openmm.CustomBondForce,
    openmm.CustomAngleForce,
    openmm.CustomTorsionForce,
    openmm.CustomCompoundBondForce,
    openmm.AmoebaTorsionTorsionForce,
)


def _bonded_terms(force: openmm.Force) -> list[tuple[tuple[int, ...], tuple]]:
    """Return particle dependencies and parameters of supported bonded terms."""
    if isinstance(force, openmm.AmoebaTorsionTorsionForce):
        terms = [force.getTorsionTorsionParameters(i) for i in range(force.getNumTorsionTorsions())]
        return [(tuple(p for p in term[:6] if p >= 0), tuple(term)) for term in terms]
    if isinstance(force, openmm.CustomCompoundBondForce):
        terms = [force.getBondParameters(i) for i in range(force.getNumBonds())]
        return [(tuple(term[0]), tuple(term)) for term in terms]
    if isinstance(force, (openmm.HarmonicBondForce, openmm.CustomBondForce)):
        kind, count = "Bond", 2
    elif isinstance(force, (openmm.HarmonicAngleForce, openmm.CustomAngleForce)):
        kind, count = "Angle", 3
    else:
        kind, count = "Torsion", 4
    terms = [getattr(force, f"get{kind}Parameters")(i) for i in range(getattr(force, f"getNum{kind}s")())]
    return [(tuple(term[:count]), tuple(term)) for term in terms]


def _validate_amoeba_qm_region(system: openmm.System, topology: openmm.app.Topology, qmatoms: Sequence[int]) -> None:
    """Require whole molecules and independent local frames before force mutation."""
    selected = set(qmatoms)
    multipoles = [force for force in system.getForces() if isinstance(force, openmm.AmoebaMultipoleForce)]
    if len(multipoles) != 1:
        raise InputError("AMOEBA QM/MM requires exactly one AmoebaMultipoleForce")
    for first, second in topology.bonds():
        if (first.index in selected) != (second.index in selected):
            raise InputError("AMOEBA mechanical embedding requires whole molecules in the QM region")
    for force in system.getForces():
        dependencies = []
        if isinstance(force, openmm.AmoebaMultipoleForce):
            for particle in range(force.getNumMultipoles()):
                parameters = force.getMultipoleParameters(particle)
                dependencies.extend((particle, index) for index in parameters[4:7] if index >= 0)
                for covalent_map in force.getCovalentMaps(particle):
                    dependencies.extend((particle, index) for index in covalent_map)
        elif isinstance(force, openmm.AmoebaVdwForce):
            dependencies.extend(
                (particle, force.getParticleParameters(particle)[0]) for particle in range(force.getNumParticles())
            )
        elif isinstance(force, _BONDED_TYPES):
            if force.getName() == "OpenMMQMMM restraint":
                continue
            dependencies.extend(atoms for atoms, _parameters in _bonded_terms(force))
        elif isinstance(force, (openmm.CMMotionRemover, openmm.CustomExternalForce, openmm.MonteCarloBarostat)):
            continue
        else:
            raise InputError(f"AMOEBA QM/MM does not support force type {type(force).__name__}")
        if any(
            any(atom in selected for atom in atoms) and not all(atom in selected for atom in atoms)
            for atoms in dependencies
        ):
            raise InputError(
                "AMOEBA mechanical embedding requires whole molecules, local multipole axes, "
                "polarization groups and reduced van der Waals sites within the same region"
            )
    for particle in range(system.getNumParticles()):
        if system.isVirtualSite(particle):
            site = system.getVirtualSite(particle)
            if any((particle in selected) != (site.getParticle(i) in selected) for i in range(site.getNumParticles())):
                raise InputError("AMOEBA QM/MM virtual sites and their parents must belong to the same region")


def _copy_bonded_without_qm(force: openmm.Force, selected: set[int]) -> openmm.Force | None:
    """Rebuild bonded terms through the API, preserving custom functions and settings."""
    terms = _bonded_terms(force)
    retained = [parameters for particles, parameters in terms if not all(i in selected for i in particles)]
    if len(retained) == len(terms):
        return None
    if isinstance(force, openmm.AmoebaTorsionTorsionForce):
        replacement = openmm.AmoebaTorsionTorsionForce()
        for index in range(force.getNumTorsionTorsionGrids()):
            replacement.setTorsionTorsionGrid(index, force.getTorsionTorsionGrid(index))
        for term in retained:
            replacement.addTorsionTorsion(*term)
    else:
        if isinstance(force, (openmm.HarmonicBondForce, openmm.CustomBondForce, openmm.CustomCompoundBondForce)):
            kind = "Bond"
        elif isinstance(force, (openmm.HarmonicAngleForce, openmm.CustomAngleForce)):
            kind = "Angle"
        else:
            kind = "Torsion"
        if isinstance(force, openmm.CustomCompoundBondForce):
            replacement = openmm.CustomCompoundBondForce(force.getNumParticlesPerBond(), force.getEnergyFunction())
        elif hasattr(force, "getEnergyFunction"):
            replacement = type(force)(force.getEnergyFunction())
        else:
            replacement = type(force)()
        if hasattr(force, "getEnergyFunction"):
            for i in range(getattr(force, f"getNumPer{kind}Parameters")()):
                getattr(replacement, f"addPer{kind}Parameter")(getattr(force, f"getPer{kind}ParameterName")(i))
            for i in range(force.getNumGlobalParameters()):
                replacement.addGlobalParameter(force.getGlobalParameterName(i), force.getGlobalParameterDefaultValue(i))
            for i in range(force.getNumEnergyParameterDerivatives()):
                replacement.addEnergyParameterDerivative(force.getEnergyParameterDerivativeName(i))
            if isinstance(force, openmm.CustomCompoundBondForce):
                for i in range(force.getNumTabulatedFunctions()):
                    replacement.addTabulatedFunction(
                        force.getTabulatedFunctionName(i), openmm.XmlSerializer.clone(force.getTabulatedFunction(i))
                    )
        for term in retained:
            getattr(replacement, f"add{kind}")(*term)
    replacement.setName(force.getName())
    replacement.setForceGroup(force.getForceGroup())
    replacement.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    return replacement


def _prepare_amoeba_mechanical(system: openmm.System, qmatoms: Sequence[int]) -> None:
    """Subtract the QM AMOEBA reference while retaining the full polarization response."""
    selected = set(qmatoms)
    replacements = []
    subtraction = None
    for index, force in enumerate(system.getForces()):
        if isinstance(force, openmm.AmoebaMultipoleForce):
            reference = openmm.XmlSerializer.clone(force)
            for particle in range(reference.getNumMultipoles()):
                if particle not in selected:
                    parameters = list(reference.getMultipoleParameters(particle))
                    parameters[0:3] = [0.0, [0.0] * 3, [0.0] * 9]
                    parameters[-1] = 0.0
                    reference.setMultipoleParameters(particle, *parameters)
            # Polarization is many-body: pair exclusions cannot remove an
            # isolated QM reference while preserving its response to MM sites.
            subtraction = openmm.CustomCVForce("-qm_reference")
            subtraction.setName("OpenMMQMMM AMOEBA QM multipole subtraction")
            subtraction.setForceGroup(force.getForceGroup())
            subtraction.addCollectiveVariable("qm_reference", reference)
        elif isinstance(force, openmm.AmoebaVdwForce):
            for particle in selected:
                force.setParticleExclusions(particle, sorted(set(force.getParticleExclusions(particle)) | selected))
        elif isinstance(force, _BONDED_TYPES) and force.getName() != "OpenMMQMMM restraint":
            replacement = _copy_bonded_without_qm(force, selected)
            if replacement is not None:
                replacements.append((index, replacement))
    for index, _force in reversed(replacements):
        system.removeForce(index)
    for _index, force in replacements:
        system.addForce(force)
    if subtraction is not None:
        system.addForce(subtraction)


def _nonbonded_forces(system: openmm.System) -> list[openmm.Force]:
    """Include nested AMOEBA reference forces when updating periodic cutoffs."""
    forces = []
    for force in system.getForces():
        if isinstance(
            force,
            (openmm.NonbondedForce, openmm.CustomNonbondedForce, openmm.AmoebaMultipoleForce, openmm.AmoebaVdwForce),
        ):
            forces.append(force)
        elif isinstance(force, openmm.CustomCVForce):
            for i in range(force.getNumCollectiveVariables()):
                nested = force.getCollectiveVariable(i)
                if isinstance(nested, (openmm.AmoebaMultipoleForce, openmm.AmoebaVdwForce)):
                    forces.append(nested)
    return forces
