"""Physical energy/force and barostat regressions for external-QM dynamics."""

import csv
from pathlib import Path

import numpy as np
import openmm
import pytest
from conftest import _AnalyticQM, _make_analytic_qmmm

from openmmqmmm import (
    Fragment,
    MolecularDynamicsEngine,
    OpenMMTheory,
    QMMMTheory,
    constants,
    export_rpmd_potential,
    single_point,
)
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider, add_rpmd_python_force
from openmmqmmm.openmm.theory import ForceReporter


@pytest.mark.parametrize("kind", ["mech", "elstat", "external"])
@pytest.mark.parametrize("integrator", ["VerletIntegrator", "LangevinMiddleIntegrator", "NoseHooverIntegrator"])
def test_classical_reports_physical_energy_and_forces_at_saved_positions(kind, integrator):
    qmmm, fragment, qm = _make_analytic_qmmm("elstat" if kind == "elstat" else "mech")
    theory = qm if kind == "external" else qmmm
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        platform="Reference",
        integrator=integrator,
        timestep=0.0001,
        temperature=1,
        traj_frequency=1,
        datafilename="state.csv",
        energy_file_option="energy.txt",
        force_file_option="enabled",
        trajectory_file_option="XYZ",
    )

    # A native MM contribution ensures reporters include the full potential.
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.07, 30.0)
    engine.openmmobject.system.addForce(bond)

    def zero_velocities(current_engine):
        current_engine.simulation.context.setVelocities(np.zeros((2, 3)))

    engine.run(simulation_steps=2, pre_dynamics_hook=zero_velocities)
    state = engine.simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
    qm_energy, gradient = qm.run(current_coords=positions_nm * 10, grad=True)
    displacement = positions_nm[1] - positions_nm[0]
    distance = np.linalg.norm(displacement)
    expected_energy = qm_energy * constants.HARTREE_TO_KJ_PER_MOL + 0.5 * 30.0 * (distance - 0.07) ** 2
    expected_force = -gradient * constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    expected_force[0] += 30.0 * (distance - 0.07) * displacement / distance
    expected_force[1] -= 30.0 * (distance - 0.07) * displacement / distance
    force_unit = openmm.unit.kilojoules_per_mole / openmm.unit.nanometer
    assert state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) == pytest.approx(expected_energy)
    assert state.getForces(asNumpy=True).value_in_unit(force_unit) == pytest.approx(expected_force)
    engine.close()

    rows = list(csv.reader(Path("state.csv").read_text().splitlines()))
    potential_column = next(index for index, label in enumerate(rows[0]) if "Potential Energy" in label)
    assert float(rows[-1][potential_column]) == pytest.approx(expected_energy)
    assert float(Path("energy.txt").read_text().splitlines()[-1]) == pytest.approx(
        expected_energy / constants.HARTREE_TO_KJ_PER_MOL
    )
    force_lines = Path(engine.trajfilename + "_force.txt").read_text().splitlines()
    assert float(force_lines[-3]) == pytest.approx(expected_energy, rel=1e-5)
    saved_forces = np.array([[float(value) for value in line.split()] for line in force_lines[-2:]])
    assert saved_forces == pytest.approx(expected_force, rel=1e-5, abs=1e-7)
    xyz_lines = Path("OpenMMMD_traj.xyz").read_text().splitlines()
    saved_coords = np.array([[float(value) for value in line.split()[1:]] for line in xyz_lines[-2:]])
    assert saved_coords == pytest.approx(positions_nm * 10, abs=1e-6)


def test_second_classical_engine_cannot_accumulate_a_frozen_qm_force():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator")
    engine.close()
    with pytest.raises(InputError, match="second one would silently double"):
        MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator")
    assert sum(isinstance(force, openmm.PythonForce) for force in qmmm.mm_theory.system.getForces()) == 1
    assert not any(isinstance(force, openmm.CustomExternalForce) for force in qmmm.mm_theory.system.getForces())


@pytest.mark.parametrize("option", ["truncated_pc", "update_qm_region_charges"])
def test_classical_callback_rejects_history_dependent_potential(option):
    qmmm, fragment, _qm = _make_analytic_qmmm(**{option: True})
    with pytest.raises(InputError, match=f"does not support {option}"):
        MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator")
    assert not any(isinstance(force, openmm.PythonForce) for force in qmmm.mm_theory.system.getForces())


@pytest.mark.parametrize("invalid_option", ["special_wrapping", "truncated_pc", "update_qm_region_charges"])
def test_rejected_engine_leaves_standalone_qmmm_energy_intact(invalid_option):
    theory_kwargs = {} if invalid_option == "special_wrapping" else {invalid_option: True}
    engine_kwargs = {"special_wrapping": True} if invalid_option == "special_wrapping" else {}
    qmmm, fragment, qm = _make_analytic_qmmm(**theory_kwargs)
    qm.charges = [0.0, 0.0]
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.07, 30.0)
    qmmm.mm_theory.system.addForce(bond)
    with pytest.raises(InputError, match=f"does not support {invalid_option}"):
        MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator", **engine_kwargs)
    assert not qmmm.openmm_externalforce
    assert not qmmm.exit_after_customexternalforce_update
    result = single_point(fragment=fragment, theory=qmmm, grad=False)
    expected_qm_energy = qm.run(current_coords=fragment.coords)
    expected_mm_energy = 0.5 * 30.0 * (0.1 - 0.07) ** 2 / constants.HARTREE_TO_KJ_PER_MOL
    assert result.energy == pytest.approx(expected_qm_energy + expected_mm_energy, abs=1e-12)


def test_periodic_callback_uses_instantaneous_box_and_invalidates_cache():
    class BoxDependentPotential:
        def __init__(self):
            self.boxes = []

        def run_openmm_python_force(self, *, current_coords, periodic_box_vectors, **kwargs):
            self.boxes.append(np.array(periodic_box_vectors))
            return np.sum(periodic_box_vectors**2) * 1e-6, np.zeros_like(current_coords)

    potential = BoxDependentPotential()
    provider = RPMDQMMMForceProvider(potential, ["H"], 0, 1, periodic=True)
    system = openmm.System()
    system.addParticle(1)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.eye(3) * 2])
    add_rpmd_python_force(system, provider, periodic=True)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.1, 0.2, 0.3]])
    for edge_nm in [2.0, 3.0, 2.0]:
        context.setPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.eye(3) * edge_nm])
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
        assert energy == pytest.approx(3 * (10 * edge_nm) ** 2 * 1e-6 * constants.HARTREE_TO_KJ_PER_MOL)
    assert len(potential.boxes) == 2
    assert potential.boxes[0] == pytest.approx(np.eye(3) * 20)
    assert potential.boxes[1] == pytest.approx(np.eye(3) * 30)


def test_force_reporter_can_request_its_own_energy():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator")
    simulation = engine.openmmobject.create_simulation()
    engine.openmmobject.set_positions(fragment.coords, simulation)
    simulation.context.setVelocities(np.zeros((2, 3)))
    reporter = ForceReporter("only_force.txt", 1)
    simulation.reporters.append(reporter)
    simulation.step(1)
    assert float(Path("only_force.txt").read_text().splitlines()[1]) > 0
    engine.close()


def test_barostat_rejects_trial_with_large_actual_qm_energy():
    class StiffPairQM(_AnalyticQM):
        def __init__(self, coords):
            super().__init__()
            self.equilibrium = np.linalg.norm(np.asarray(coords)[1] - np.asarray(coords)[0]) * constants.ANG_TO_BOHR
            self.energies = []

        def run(self, *, current_coords, grad=False, **kwargs):
            separation = (current_coords[1] - current_coords[0]) * constants.ANG_TO_BOHR
            distance = np.linalg.norm(separation)
            displacement = distance - self.equilibrium
            energy = 0.5e6 * displacement**2
            gradient = 1e6 * displacement * separation / distance
            self.energies.append(energy)
            return (energy, np.array([-gradient, gradient])) if grad else energy

    fragment = Fragment(elems=["H", "H"], coords=[[2.5, 0, 0], [5, 0, 0]], charge=0, mult=1)
    qm = StiffPairQM(fragment.coords)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        periodic=True,
        periodic_cell_dimensions=[20, 20, 20, 90, 90, 90],
        periodic_nonbonded_cutoff=5,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0, 1],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
    )
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        barostat="MonteCarloBarostat",
        barostat_frequency=1,
        temperature=1,
        timestep=1e-12,
        enforce_periodic_box=False,
        traj_frequency=100,
    )
    for force in mm.system.getForces():
        if isinstance(force, openmm.MonteCarloBarostat):
            force.setRandomNumberSeed(42)
    create_integrator = mm.create_integrator

    def seeded_integrator():
        create_integrator()
        mm.integrator.setRandomNumberSeed(43)

    mm.create_integrator = seeded_integrator
    engine.run(simulation_steps=1, pre_dynamics_hook=lambda e: e.simulation.context.setVelocities(np.zeros((2, 3))))
    state = engine.simulation.context.getState(getPositions=True, getEnergy=True)
    actual_coords = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    assert len(qm.energies) >= 2, "Barostat trial coordinates must invoke the QM potential"
    assert max(qm.energies) * constants.HARTREE_TO_KJ_PER_MOL > 1000
    assert state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) < 1e-10
    assert actual_coords == pytest.approx(fragment.coords, abs=1e-10)
    assert state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(openmm.unit.nanometer) == pytest.approx(
        np.eye(3) * 2
    )
    engine.close()


def _make_qmmm_with_wrapped_mm_bond():
    fragment = Fragment(elems=["He"] * 3, coords=[[5, 0, 0], [29.3, 0, 0], [0.7, 0, 0]], conncalc=False)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        periodic=True,
        periodic_cell_dimensions=[30, 30, 30, 90, 90, 90],
        periodic_nonbonded_cutoff=5,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    atoms = list(mm.topology.atoms())
    mm.topology.addBond(atoms[1], atoms[2])
    bond = openmm.HarmonicBondForce()
    bond.addBond(1, 2, 0.14, 100.0)
    mm.system.addForce(bond)
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=_AnalyticQM(force_constant=0),
        mm_theory=mm,
        qmatoms=[0],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
    )
    return qmmm, fragment


@pytest.mark.parametrize("mode", ["classical", "rpmd", "export"])
def test_periodic_initial_native_mm_bonds_use_contiguous_molecules(mode):
    qmmm, fragment = _make_qmmm_with_wrapped_mm_bond()
    original_coords = fragment.coords.copy()
    if mode == "export":
        exported = export_rpmd_potential(theory=qmmm, num_beads=2)
        integrator = openmm.RPMDIntegrator(2, 300, 1, 0.001)
        _context = openmm.Context(exported.system, integrator, openmm.Platform.getPlatformByName("Reference"))
        for copy in range(2):
            integrator.setPositions(copy, exported.modeller.positions)
        states = [integrator.getState(copy, getPositions=True, getEnergy=True) for copy in range(2)]
    else:
        engine = MolecularDynamicsEngine(
            fragment=fragment,
            theory=qmmm,
            integrator="RPMDIntegrator" if mode == "rpmd" else "VerletIntegrator",
            rpmd_num_copies=2,
            enforce_periodic_box=False,
        )
        engine.run(simulation_steps=0)
        if mode == "rpmd":
            states = [
                engine.simulation.integrator.getState(copy, getPositions=True, getEnergy=True) for copy in range(2)
            ]
        else:
            states = [engine.simulation.context.getState(getPositions=True, getEnergy=True)]
        engine.close()
    for state in states:
        coords = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        assert np.linalg.norm(coords[1] - coords[2]) == pytest.approx(1.4, abs=1e-12)
        assert state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) == pytest.approx(0, abs=1e-12)
    assert fragment.coords == pytest.approx(original_coords), "Initialization must not mutate the user's fragment"


@pytest.mark.parametrize("restart_mode", ["in_memory", "state_file"])
def test_periodic_restart_preserves_saved_coordinate_images(restart_mode):
    qmmm, fragment = _make_qmmm_with_wrapped_mm_bond()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        integrator="VerletIntegrator",
        enforce_periodic_box=False,
    )
    engine.run(simulation_steps=0)
    state = engine.simulation.context.getState(getPositions=True)
    saved_coords = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    saved_coords[1:] += [60, 0, 0]
    engine.simulation.context.setPositions(saved_coords * openmm.unit.angstrom)
    if restart_mode == "state_file":
        engine.simulation.saveState("wrapped_restart.xml")
        engine.run(simulation_steps=0, statefile="wrapped_restart.xml")
    else:
        engine.run(simulation_steps=0, restart=True)
    restored = engine.simulation.context.getState(getPositions=True)
    assert restored.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom) == pytest.approx(saved_coords)
    engine.close()
