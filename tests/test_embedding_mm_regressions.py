"""Physical regressions for MM terms retained or removed by QM/MM embedding."""

import numpy as np
import openmm
import pytest
from openmm import unit

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR, HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError


def _system(charges):
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    for charge in charges:
        system.addParticle(12)
        nonbonded.addParticle(charge, 0.3, 0.2)
    system.addForce(nonbonded)
    mm = OpenMMTheory.__new__(OpenMMTheory)
    mm.system = system
    mm.nonbonded_force = nonbonded
    mm.charges = list(charges)
    mm.numatoms = len(charges)
    mm.delete_qm1_mm1_bonded = False
    return mm, nonbonded


def _evaluate(system, positions, derivatives=False):
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True, getParameterDerivatives=derivatives)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    result = energy, forces, dict(state.getEnergyParameterDerivatives()) if derivatives else {}
    del context, integrator
    return result


class _ZeroQM:
    numcores = 1
    theorytype = "QM"


@pytest.mark.parametrize("embedding", ["mech", "elstat"])
def test_all_qm_exceptions_remove_base_parameters_and_every_offset(embedding):
    mm, nb = _system([0.5, 0.5])
    exception = nb.addException(0, 1, 0.125, 0.3, 0.1)
    nb.addGlobalParameter("lambda", 1)
    nb.addExceptionParameterOffset("lambda", exception, 0.15, 0.3, 1)
    fragment = Fragment(elems=["C", "C"], coords=[[0, 0, 0], [4, 0, 0]], conncalc=False)
    QMMMTheory(
        fragment=fragment,
        qm_theory=_ZeroQM(),
        mm_theory=mm,
        qmatoms=[0, 1],
        embedding=embedding,
        qm_charge=0,
        qm_mult=1,
    )
    energy, forces, _ = _evaluate(mm.system, fragment.coords * 0.1)
    assert energy == pytest.approx(0, abs=1e-12)
    assert forces == pytest.approx(np.zeros((2, 3)), abs=1e-12)
    assert nb.getExceptionParameterOffset(0)[2:] == [0, 0, 0]


def test_charge_update_preserves_exception_scaling_across_zero_and_keeps_exclusions():
    mm, nb = _system([0.2, -0.5, 0.3])
    scaled = nb.addException(0, 1, -0.05, 0.27, 0.6)
    excluded = nb.addException(0, 2, 0, 0.28, 0)
    mm.update_charges([0], [0.0])
    mm.update_charges([0], [0.8])
    parameters = nb.getExceptionParameters(scaled)
    assert parameters[2].value_in_unit(unit.elementary_charge**2) == pytest.approx(-0.2)
    assert parameters[3].value_in_unit(unit.nanometer) == pytest.approx(0.27)
    assert parameters[4].value_in_unit(unit.kilojoules_per_mole) == pytest.approx(0.6)
    assert nb.getExceptionParameters(excluded)[2].value_in_unit(unit.elementary_charge**2) == 0
    # The excluded pair must stay excluded even after the scaling was cached.
    mm.addexceptions([0, 1])
    mm.update_charges([0], [0.4])
    assert nb.getExceptionParameters(scaled)[2].value_in_unit(unit.elementary_charge**2) == 0


def test_charge_update_changes_scaled_pair_energy_and_force():
    mm, nb = _system([0.2, -0.5])
    nb.addException(0, 1, -0.05, 0.3, 0)
    mm.update_charges([0], [0.8])
    energy, forces, _ = _evaluate(mm.system, [[0, 0, 0], [0.4, 0, 0]])
    expected_energy = 138.93545764438198 * -0.2 / 0.4
    assert energy == pytest.approx(expected_energy, abs=1e-10)
    assert forces[0, 0] == pytest.approx(-expected_energy / 0.4, abs=1e-9)
    assert forces.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


@pytest.mark.parametrize("kind", ["zero_endpoint", "charge_offset"])
def test_ambiguous_nonzero_exception_update_fails_without_mutation(kind):
    mm, nb = _system([0.0 if kind == "zero_endpoint" else 0.2, -0.5])
    nb.addException(0, 1, -0.05, 0.3, 0.2)
    if kind == "charge_offset":
        nb.addGlobalParameter("lambda", 1)
        nb.addExceptionParameterOffset("lambda", 0, 0.1, 0, 0)
    before = openmm.XmlSerializer.serialize(mm.system)
    old_charges = mm.charges.copy()
    with pytest.raises(InputError, match="Cannot infer fixed-charge Coulomb scaling"):
        mm.update_charges([0], [0.8])
    assert openmm.XmlSerializer.serialize(mm.system) == before
    assert mm.charges == old_charges


def test_ambiguous_exception_can_be_suppressed_but_not_later_resurrected():
    mm, nb = _system([0, -0.5])
    nb.addException(0, 1, -0.05, 0.3, 0.2)
    mm.update_charges([0], [0])
    assert nb.getExceptionParameters(0)[2].value_in_unit(unit.elementary_charge**2) == 0
    with pytest.raises(InputError, match="Cannot infer fixed-charge Coulomb scaling"):
        mm.update_charges([0], [0.8])


@pytest.mark.parametrize("parameter_count", [0, 1, 2, 3])
def test_custom_torsion_removal_preserves_arbitrary_expression_and_force_metadata(parameter_count):
    mm, nb = _system([0] * 8)
    for atom in range(8):
        nb.setParticleParameters(atom, 0, 0.3, 0)
    expression = "lambda*(1+cos(theta))" + "".join(f"+p{i}" for i in range(parameter_count))
    force = openmm.CustomTorsionForce(expression)
    force.addGlobalParameter("lambda", 1.7)
    force.addEnergyParameterDerivative("lambda")
    for i in range(parameter_count):
        force.addPerTorsionParameter(f"p{i}")
    force.addTorsion(0, 1, 2, 3, list(range(1, parameter_count + 1)))
    force.addTorsion(4, 5, 6, 7, list(range(1, parameter_count + 1)))
    force.setName("arbitrary torsion")
    force.setForceGroup(7)
    force.setUsesPeriodicBoundaryConditions(True)
    mm.system.addForce(force)
    mm.forcegroups = {force: 7}
    positions = np.array([[0, 0, 0], [0.15, 0, 0], [0.2, 0.1, 0], [0.3, 0.1, 0.1]])
    positions = np.vstack([positions, positions + np.array([0.5, 0, 0])])
    before_energy, before_forces, before_derivatives = _evaluate(mm.system, positions, derivatives=True)
    mm.modify_bonded_forces([0, 1, 2])
    replacement = next(f for f in mm.system.getForces() if isinstance(f, openmm.CustomTorsionForce))
    assert replacement.getEnergyFunction() == expression
    assert replacement.getNumTorsions() == 1
    assert replacement.getNumPerTorsionParameters() == parameter_count
    assert replacement.getGlobalParameterDefaultValue(0) == 1.7
    assert replacement.getName() == "arbitrary torsion"
    assert replacement.getForceGroup() == 7
    assert replacement.usesPeriodicBoundaryConditions()
    assert {(force.getName(), group) for force, group in mm.forcegroups.items()} == {
        ("NonbondedForce", 0),
        ("arbitrary torsion", 7),
    }
    energy, forces, derivatives = _evaluate(mm.system, positions, derivatives=True)
    assert energy == pytest.approx(before_energy / 2, abs=1e-12)
    assert derivatives["lambda"] == pytest.approx(before_derivatives["lambda"] / 2, abs=1e-12)
    assert forces[:4] == pytest.approx(np.zeros((4, 3)), abs=1e-12)
    assert forces[4:] == pytest.approx(before_forces[4:], abs=1e-12)


def test_library_global_parameter_torsion_can_be_removed():
    mm, nb = _system([0] * 4)
    for atom in range(4):
        nb.setParticleParameters(atom, 0, 0.3, 0)
    mm.add_custom_torsion_force(0, 1, 2, 3, 0.0, 1.0)
    mm.modify_bonded_forces([0, 1, 2, 3])
    energy, forces, _ = _evaluate(mm.system, [[0, 0, 0], [0.15, 0, 0], [0.2, 0.1, 0], [0.3, 0.1, 0.1]])
    assert energy == pytest.approx(0, abs=1e-12)
    assert forces == pytest.approx(np.zeros((4, 3)), abs=1e-12)


def test_periodic_dispersion_tail_is_retained_after_direct_qm_pair_exclusion():
    mm, nb = _system([0, 0])
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(1)
    nb.setUseDispersionCorrection(True)
    mm.addexceptions([0, 1])
    energies = []
    for length in (3, 4):
        mm.system.setDefaultPeriodicBoxVectors(*np.eye(3) * length)
        energy, forces, _ = _evaluate(mm.system, [[0, 0, 0], [0.4, 0, 0]])
        energies.append(energy)
        assert forces == pytest.approx(np.zeros((2, 3)), abs=1e-12)
        # Homogeneous LJ tail: 8*pi*N^2*epsilon*sigma^6/(3V*rc^3)
        # times ((sigma/rc)^6/3 - 1), including the retained MM density.
        expected = 8 * np.pi * 4 * 0.2 * 0.3**6 / (3 * length**3) * (0.3**6 / 3 - 1)
        assert energy == pytest.approx(expected, abs=1e-12)
    assert energies[0] < 0
    assert energies[0] / energies[1] == pytest.approx(4**3 / 3**3)
    nb.setUseDispersionCorrection(False)
    assert _evaluate(mm.system, [[0, 0, 0], [0.4, 0, 0]])[0] == pytest.approx(0, abs=1e-12)


def test_mechanical_periodic_coulomb_images_remain_after_direct_qm_pair_exclusion():
    mm, nb = _system([1, -1])
    for index, charge in enumerate([1, -1]):
        nb.setParticleParameters(index, charge, 0.3, 0)
    nb.setNonbondedMethod(openmm.NonbondedForce.Ewald)
    nb.setCutoffDistance(1)
    nb.setEwaldErrorTolerance(1e-7)
    mm.addexceptions([0, 1])
    energies = []
    for length in (3, 4):
        mm.system.setDefaultPeriodicBoxVectors(*np.eye(3) * length)
        energy, _forces, _ = _evaluate(mm.system, [[0, 0, 0], [length / 5, 0, 0]])
        energies.append(energy)
    assert abs(energies[0]) > 0.1
    # At fixed fractional positions an electrostatic lattice energy scales as 1/L.
    assert energies[0] / energies[1] == pytest.approx(4 / 3, rel=1e-5)


def test_ljpme_dispersion_images_remain_after_direct_qm_pair_exclusion():
    mm, nb = _system([0, 0])
    nb.setNonbondedMethod(openmm.NonbondedForce.LJPME)
    nb.setCutoffDistance(1)
    nb.setEwaldErrorTolerance(1e-6)
    mm.system.setDefaultPeriodicBoxVectors(*np.eye(3) * 3)
    mm.addexceptions([0, 1])
    positions = [[0, 0, 0], [0.6, 0, 0]]
    energy, forces, _ = _evaluate(mm.system, positions)
    assert energy < -1e-6
    assert np.max(np.abs(forces)) > 1e-6
    # This residual belongs to the LJ lattice, not the excluded direct pair.
    for index in range(2):
        nb.setParticleParameters(index, 0, 0.3, 0)
    energy, forces, _ = _evaluate(mm.system, positions)
    assert energy == pytest.approx(0, abs=1e-12)
    assert forces == pytest.approx(np.zeros((2, 3)), abs=1e-12)


def test_standalone_mm_recomputes_native_sites_and_returns_only_independent_gradient_rows():
    fragment = Fragment(
        elems=["H", "H", "He", "He"],
        coords=[[0.0, 0, 0], [3.0, 0, 0], [8.0, 0, 0], [99.0, 0, 0]],
        conncalc=False,
    )
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    mm.system.setParticleMass(3, 0)
    mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(0, 1, 0.25, 0.75))
    mm.update_charges([0, 1, 2, 3], [0, 0, -1, 0.5])
    energy, gradient = mm.run(current_coords=fragment.coords, grad=True)
    assert gradient[3] == pytest.approx(np.zeros(3), abs=1e-12)
    expected = 138.93545764438198 * -0.5 / ((8 - 2.25) * 0.1)
    assert energy * HARTREE_TO_KJ_PER_MOL == pytest.approx(expected, abs=1e-10)
    for atom in range(4):
        plus, minus = fragment.coords.copy(), fragment.coords.copy()
        plus[atom, 0] += 1e-5
        minus[atom, 0] -= 1e-5
        difference = (mm.run(current_coords=plus) - mm.run(current_coords=minus)) / (2e-5 * ANG_TO_BOHR)
        assert gradient[atom, 0] == pytest.approx(difference, abs=1e-10)
