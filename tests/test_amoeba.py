"""AMOEBA references use independently constructed native OpenMM systems."""

import numpy as np
import openmm
import openmm.app
import pytest
from conftest import _AnalyticQM, central_difference, evaluate

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory, ZeroTheory, constants, export_rpmd_potential
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.polarizable import _nonbonded_forces

WATER_COORDS = np.array(
    [
        [0, 0, 0],
        [0.96, 0, 0],
        [-0.3, 0.91, 0],
        [3, 0.1, 0.1],
        [3.95, 0.2, 0.1],
        [2.8, 1, 0.2],
        [0.2, 3, 0.5],
        [0.4, 3.9, 0.7],
        [1.1, 2.7, 0.4],
    ]
)


def _water_topology(n_waters):
    topology = openmm.app.Topology()
    chain = topology.addChain()
    for _ in range(n_waters):
        residue = topology.addResidue("HOH", chain)
        oxygen = topology.addAtom("O", openmm.app.element.oxygen, residue)
        first = topology.addAtom("H1", openmm.app.element.hydrogen, residue)
        second = topology.addAtom("H2", openmm.app.element.hydrogen, residue)
        topology.addBond(oxygen, first)
        topology.addBond(oxygen, second)
    return topology


def _tight_polarization(system):
    for force in system.getForces():
        if isinstance(force, openmm.AmoebaMultipoleForce):
            force.setMutualInducedTargetEpsilon(1e-8)


def _native_system(n_waters, xml="amoeba2018.xml"):
    system = openmm.app.ForceField(xml).createSystem(
        _water_topology(n_waters), nonbondedMethod=openmm.app.NoCutoff, constraints=None, rigidWater=False
    )
    _tight_polarization(system)
    return system


def _amoeba_mm(n_waters=3, xml="amoeba2018.xml", **kwargs):
    mm = OpenMMTheory(
        topoforce=True,
        topology=_water_topology(n_waters),
        forcefield=openmm.app.ForceField(xml),
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
        **kwargs,
    )
    _tight_polarization(mm.system)
    fragment = Fragment(elems=["O", "H", "H"] * n_waters, coords=WATER_COORDS[: 3 * n_waters], charge=0, mult=1)
    return mm, fragment


def _qmmm(mm, fragment, qmatoms, qm=None, **kwargs):
    options = {"embedding": "mech"}
    options.update(kwargs)
    return QMMMTheory(
        fragment=fragment,
        mm_theory=mm,
        qm_theory=ZeroTheory() if qm is None else qm,
        qmatoms=qmatoms,
        qm_charge=0,
        qm_mult=1,
        **options,
    )


@pytest.mark.parametrize("xml", ["amoeba2013.xml", "amoeba2018.xml"])
def test_amoeba_mm_matches_native_energy_force_and_monopoles(xml):
    mm, fragment = _amoeba_mm(xml=xml)
    energy, forces = evaluate(_native_system(3, xml), fragment.coords / 10)
    actual_energy, gradient = mm.run(current_coords=fragment.coords, grad=True)
    assert actual_energy == pytest.approx(energy / constants.HARTREE_TO_KJ_PER_MOL, abs=1e-12)
    assert gradient == pytest.approx(-forces / constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, abs=1e-11)
    assert mm.charges == pytest.approx([-0.51966, 0.25983, 0.25983] * 3)
    assert mm.amoeba_multipole_force is not None
    assert mm.amoeba_vdw_force is not None


@pytest.mark.parametrize("qm_waters", [1, 2, 3])
def test_mechanical_amoeba_matches_native_full_minus_isolated_qm(qm_waters):
    mm, fragment = _amoeba_mm()
    qmatoms = list(range(3 * qm_waters))
    full_energy, full_forces = evaluate(_native_system(3), fragment.coords / 10)
    qm_energy, qm_forces = evaluate(_native_system(qm_waters), fragment.coords[qmatoms] / 10)
    expected_forces = full_forces.copy()
    expected_forces[qmatoms] -= qm_forces
    theory = _qmmm(mm, fragment, qmatoms)
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    assert energy == pytest.approx((full_energy - qm_energy) / constants.HARTREE_TO_KJ_PER_MOL, abs=1e-11)
    assert gradient == pytest.approx(-expected_forces / constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, abs=1e-10)


def test_mechanical_amoeba_gradient_is_conservative():
    mm, fragment = _amoeba_mm()
    theory = _qmmm(mm, fragment, list(range(6)))
    _energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    for atom, axis in [(0, 0), (1, 1), (6, 2), (8, 0)]:
        numerical = central_difference(lambda coords: theory.run(current_coords=coords), fragment.coords, atom, axis)
        assert gradient[atom, axis] == pytest.approx(numerical, abs=2e-8)


def test_amoeba_python_force_matches_native_subtraction_and_quantum_force():
    mm, fragment = _amoeba_mm()
    qm = _AnalyticQM(force_constant=0.001)
    theory = _qmmm(mm, fragment, [0, 1, 2], qm=qm)
    export = export_rpmd_potential(theory=theory, num_beads=1)
    integrator = openmm.VerletIntegrator(0.0001)
    context = openmm.Context(export.system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(fragment.coords / 10)
    context.setVelocities(np.zeros((9, 3)))
    integrator.step(2)
    state = context.getState(getEnergy=True, getForces=True, getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
    full_energy, full_forces = evaluate(_native_system(3), positions)
    isolated_energy, isolated_forces = evaluate(_native_system(1), positions[:3])
    qm_energy, qm_gradient = qm.run(current_coords=positions[:3] * 10, grad=True)
    expected_energy = full_energy - isolated_energy + qm_energy * constants.HARTREE_TO_KJ_PER_MOL
    full_forces[:3] -= isolated_forces + qm_gradient * constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    assert state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) == pytest.approx(
        expected_energy, abs=1e-8
    )
    assert state.getForces(asNumpy=True).value_in_unit(
        openmm.unit.kilojoules_per_mole / openmm.unit.nanometer
    ) == pytest.approx(full_forces, abs=1e-7)
    del context, integrator


@pytest.mark.parametrize("option", ["split_molecule", "electrostatic", "override_charges", "unsupported_force"])
def test_amoeba_rejects_unsupported_partition_before_mutation(option):
    mm, fragment = _amoeba_mm()
    qmatoms = [0, 1, 2]
    kwargs = {}
    if option == "split_molecule":
        qmatoms = [0, 1]
    elif option == "electrostatic":
        kwargs["embedding"] = "elstat"
    elif option == "override_charges":
        kwargs["charges"] = [0.0] * 9
    else:
        mm.system.addForce(openmm.AmoebaWcaDispersionForce())
    before = openmm.XmlSerializer.serialize(mm.system)
    with pytest.raises(InputError, match="AMOEBA"):
        _qmmm(mm, fragment, qmatoms, **kwargs)
    assert openmm.XmlSerializer.serialize(mm.system) == before


@pytest.mark.parametrize("dependency", ["multipole_axis", "vdw_parent", "polarization_group"])
def test_amoeba_rejects_cross_region_dependencies(dependency):
    mm, fragment = _amoeba_mm()
    if dependency == "multipole_axis":
        force = mm.amoeba_multipole_force
        parameters = list(force.getMultipoleParameters(0))
        parameters[4] = 3
        force.setMultipoleParameters(0, *parameters)
    elif dependency == "vdw_parent":
        force = mm.amoeba_vdw_force
        parameters = list(force.getParticleParameters(1))
        parameters[0] = 3
        force.setParticleParameters(1, *parameters)
    else:
        force = mm.amoeba_multipole_force
        force.setCovalentMap(0, openmm.AmoebaMultipoleForce.PolarizationCovalent11, [0, 1, 2, 3])
    with pytest.raises(InputError, match="same region"):
        _qmmm(mm, fragment, [0, 1, 2])


def test_periodic_amoeba_applies_settings_and_updates_every_cutoff():
    mm, fragment = _amoeba_mm(
        periodic=True,
        periodic_cell_dimensions=[30, 30, 30, 90, 90, 90],
        periodic_nonbonded_cutoff=10,
        dispersion_correction=False,
        ewalderrortolerance=2e-5,
        pme_parameters=(3.0, 32, 32, 32),
    )
    assert not mm.amoeba_vdw_force.getUseDispersionCorrection()
    assert mm.amoeba_multipole_force.getEwaldErrorTolerance() == pytest.approx(2e-5)
    assert list(mm.amoeba_multipole_force.getPMEParameters()[1:]) == [32, 32, 32]
    _qmmm(mm, fragment, [0, 1, 2])
    mm.update_cell(periodic_cell_dimensions=[15, 15, 15, 90, 90, 90])
    assert len(_nonbonded_forces(mm.system)) == 3
    for force in _nonbonded_forces(mm.system):
        assert force.getCutoffDistance().value_in_unit(openmm.unit.nanometer) == pytest.approx(0.7485)
    mm.update_cell(periodic_cell_dimensions=[30, 30, 30, 90, 90, 90])
    for force in _nonbonded_forces(mm.system):
        assert force.getCutoffDistance().value_in_unit(openmm.unit.nanometer) == pytest.approx(1.0)
    assert np.isfinite(mm.run(current_coords=fragment.coords))


def test_amoeba_custom_compound_bond_removal_preserves_expression_and_tabulated_function():
    from openmmqmmm.openmm.polarizable import _copy_bonded_without_qm

    force = openmm.CustomCompoundBondForce(3, "k*(distance(p1,p2)-r0)^2+f(distance(p2,p3))")
    force.addPerBondParameter("r0")
    force.addGlobalParameter("k", 2.0)
    force.addEnergyParameterDerivative("k")
    force.addTabulatedFunction("f", openmm.Continuous1DFunction([1.0, 2.0], 0.0, 1.0))
    force.addBond([0, 1, 2], [0.1])
    force.addBond([3, 4, 5], [0.2])
    force.setName("AmoebaStretchBendForce")
    force.setForceGroup(7)
    replacement = _copy_bonded_without_qm(force, {0, 1, 2})
    system = openmm.System()
    for _ in range(6):
        system.addParticle(1.0)
    system.addForce(replacement)
    positions = WATER_COORDS[:6] / 10
    distance = np.linalg.norm(positions[3] - positions[4])
    expected = 2.0 * (distance - 0.2) ** 2 + 1.0 + np.linalg.norm(positions[4] - positions[5])
    energy, forces, derivatives = evaluate(system, positions, derivatives=True)
    assert energy == pytest.approx(expected)
    assert forces[:3] == pytest.approx(np.zeros((3, 3)))
    assert derivatives["k"] == pytest.approx((distance - 0.2) ** 2)
    assert replacement.getForceGroup() == 7
    assert replacement.getName() == force.getName()


def test_amoeba_torsion_torsion_removal_preserves_grid_and_chirality():
    from openmmqmmm.openmm.polarizable import _copy_bonded_without_qm

    force = openmm.AmoebaTorsionTorsionForce()
    grid = [[[float(x), float(y), 2.0, 0.0, 0.0, 0.0] for y in range(-180, 181, 15)] for x in range(-180, 181, 15)]
    force.setTorsionTorsionGrid(0, grid)
    force.addTorsionTorsion(0, 1, 2, 3, 4, 5, 0)
    force.addTorsionTorsion(6, 7, 8, 9, 10, 11, 0)
    replacement = _copy_bonded_without_qm(force, set(range(6)))
    system = openmm.System()
    for _ in range(12):
        system.addParticle(1.0)
    system.addForce(replacement)
    positions = np.vstack([WATER_COORDS[:6] / 10, WATER_COORDS[:6] / 10 + 1.0])
    energy, forces = evaluate(system, positions)
    assert energy == pytest.approx(2.0)
    assert forces == pytest.approx(np.zeros((12, 3)), abs=1e-12)
    assert replacement.getTorsionTorsionParameters(0) == [6, 7, 8, 9, 10, 11, 0]
