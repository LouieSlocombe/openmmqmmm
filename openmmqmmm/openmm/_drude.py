from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import openmm
import openmm.app

from openmmqmmm.exceptions import InputError


def _drude_dependencies(system: openmm.System) -> list[tuple[int, ...]]:
    """Return each oscillator's shell, parent and optional anisotropy references."""
    return [
        tuple(int(atom) for atom in force.getParticleParameters(index)[:5] if atom >= 0)
        for force in system.getForces()
        if isinstance(force, openmm.DrudeForce)
        for index in range(force.getNumParticles())
    ]


def _validate_drude_qm_region(
    system: openmm.System, topology: openmm.app.Topology, qmatoms: Sequence[int]
) -> list[tuple[int, int]]:
    """Keep entire oscillators in MM and return their periodic imaging links."""
    selected = set(qmatoms)
    dependencies = _drude_dependencies(system)
    if any(selected.intersection(atoms) for atoms in dependencies):
        raise InputError(
            "Drude particles, their parent atoms and anisotropy references must remain entirely in the MM region"
        )
    if any((first.index in selected) != (second.index in selected) for first, second in topology.bonds()):
        raise InputError("Drude QM/MM requires whole molecules: a covalent bond crosses the QM/MM boundary")
    _validate_drude_charge_updates(system, (), ())
    return [(atoms[0], reference) for atoms in dependencies for reference in atoms[1:]]


def _validate_drude_charge_updates(
    system: openmm.System, atomlist: Sequence[int], atomcharges: Sequence[float]
) -> None:
    """Prevent a charge assignment from changing the Drude spring or screening model."""
    drude_forces = [force for force in system.getForces() if isinstance(force, openmm.DrudeForce)]
    if not drude_forces:
        return
    nonbonded = [force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)]
    if len(nonbonded) != 1:
        raise InputError("Drude charge handling requires exactly one NonbondedForce")
    force = nonbonded[0]
    updates = dict(zip(atomlist, atomcharges, strict=True))
    shells = set()
    for drude in drude_forces:
        for index in range(drude.getNumParticles()):
            shell, _parent, _axis2, _axis3, _axis4, charge, *_parameters = drude.getParticleParameters(index)
            shells.add(shell)
            expected = charge.value_in_unit(openmm.unit.elementary_charge)
            actual = force.getParticleParameters(shell)[0].value_in_unit(openmm.unit.elementary_charge)
            if not np.isclose(actual, expected, atol=1e-12, rtol=0):
                raise InputError(f"Drude particle {shell} has inconsistent NonbondedForce and DrudeForce charges")
            if shell in updates and not np.isclose(updates[shell], expected, atol=1e-12, rtol=0):
                raise InputError(
                    f"Cannot change Drude particle {shell}'s charge independently of its polarizability and screening"
                )
    for index in range(force.getNumParticleParameterOffsets()):
        _name, particle, charge, _sigma, _epsilon = force.getParticleParameterOffset(index)
        if particle in shells and charge != 0:
            raise InputError("Charge parameter offsets on Drude particles are unsupported")
