"""Drude dynamics preserves the cold bath and includes external forces in SCF."""

import numpy as np
import openmm
import pytest
from conftest import _make_analytic_qmmm, dummy_mm, use_pme
from openmm import unit

from openmmqmmm import Fragment, MolecularDynamicsEngine, OpenMMTheory, export_rpmd_potential
from openmmqmmm.exceptions import InputError


def _drude_mm(*, periodic=False):
    fragment = Fragment(elems=["C", "H"], coords=[[0, 0, 0], [0.02, 0, 0]], conncalc=False)
    mm = dummy_mm(fragment)
    mm.system.setParticleMass(0, 11.6)
    mm.system.setParticleMass(1, 0.4)
    mm.nonbonded_force.setParticleParameters(0, 0.5, 0.3, 0)
    mm.nonbonded_force.setParticleParameters(1, -0.5, 0.3, 0)
    mm.nonbonded_force.addException(0, 1, 0, 0.3, 0)
    drude = openmm.DrudeForce()
    drude.addParticle(1, 0, -1, -1, -1, -0.5, 0.001, 1, 1)
    mm.system.addForce(drude)
    if periodic:
        use_pme(mm)
    return fragment, mm


@pytest.mark.parametrize(
    "integrator", ["VerletIntegrator", "LangevinMiddleIntegrator", "RPMDIntegrator", "QTBIntegrator"]
)
def test_drude_dynamics_rejects_ordinary_integrators_before_mutation(integrator):
    fragment, mm = _drude_mm()
    before = openmm.XmlSerializer.serialize(mm.system)
    with pytest.raises(InputError, match="Drude dynamics requires"):
        MolecularDynamicsEngine(fragment=fragment, theory=mm, integrator=integrator)
    assert openmm.XmlSerializer.serialize(mm.system) == before


@pytest.mark.parametrize("integrator", ["DrudeLangevinIntegrator", "DrudeSCFIntegrator"])
def test_drude_integrators_require_a_drude_force(integrator):
    fragment = Fragment(elems=["C"], coords=[[0, 0, 0]], conncalc=False)
    mm = dummy_mm(fragment)
    with pytest.raises(InputError, match="requires a system containing a DrudeForce"):
        MolecularDynamicsEngine(fragment=fragment, theory=mm, integrator=integrator)


@pytest.mark.parametrize(
    "integrator,barostat,andersen,match",
    [
        ("DrudeLangevinIntegrator", None, True, "heat the Drude displacements"),
        ("DrudeSCFIntegrator", "MonteCarloBarostat", False, "requires anderson_thermostat"),
    ],
)
def test_drude_dynamics_rejects_incompatible_thermostats(integrator, barostat, andersen, match):
    fragment, mm = _drude_mm(periodic=barostat is not None)
    before = openmm.XmlSerializer.serialize(mm.system)
    with pytest.raises(InputError, match=match):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=mm,
            integrator=integrator,
            barostat=barostat,
            anderson_thermostat=andersen,
        )
    assert openmm.XmlSerializer.serialize(mm.system) == before


@pytest.mark.parametrize("integrator", ["DrudeLangevinIntegrator", "DrudeSCFIntegrator"])
def test_barostat_preserves_drude_integrator_and_thermostat(integrator):
    fragment, mm = _drude_mm(periodic=True)
    # The new Langevin stage must discard an old stage's hot shell thermostat.
    mm.system.addForce(openmm.AndersenThermostat(300, 1))
    andersen = integrator == "DrudeSCFIntegrator"
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=mm,
        integrator=integrator,
        barostat="MonteCarloBarostat",
        anderson_thermostat=andersen,
        timestep=0.0001,
    )
    try:
        assert engine.integrator == integrator
        assert mm.integrator_name == integrator
        thermostats = [f for f in mm.system.getForces() if isinstance(f, openmm.AndersenThermostat)]
        assert len(thermostats) == int(andersen)
        engine.run(simulation_steps=1)
        state = engine.simulation.context.getState(getEnergy=True, getPositions=True)
        assert np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        assert np.linalg.norm(positions[1] - positions[0]) < 0.02
    finally:
        engine.close()


def test_drude_scf_minimization_includes_python_force():
    # An imposed field shifts the spring minimum by F/k. Assign it a separate
    # force group, just as the QM/MM callback has, to catch a Drude-only SCF loop.
    _, mm = _drude_mm()
    mm.system.setParticleMass(0, 0)
    mm.system.setParticleMass(1, 0)
    field_force = 10.0
    sampled_shell_positions = []

    def external_field(state):
        coords = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        sampled_shell_positions.append(float(coords[1, 0]))
        forces = np.zeros((2, 3))
        forces[1, 0] = field_force
        return -field_force * coords[1, 0], forces

    force = openmm.PythonForce(external_field)
    force.setForceGroup(3)
    mm.system.addForce(force)
    mm.set_simulation_parameters(integrator="DrudeSCFIntegrator", timestep=0.0001)
    simulation = mm.create_simulation()
    simulation.integrator.setMinimizationErrorTolerance(1e-6)
    simulation.context.setPositions(np.zeros((2, 3)))
    simulation.step(1)
    state = simulation.context.getState(getPositions=True, getForces=True)
    coords = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    spring_constant = 138.93545764438198 * 0.5**2 / 0.001
    assert coords[1, 0] == pytest.approx(field_force / spring_constant, abs=1e-10)
    assert len(set(sampled_shell_positions)) > 1
    assert (
        np.linalg.norm(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)[1]) < 1e-5
    )


def test_singlepoint_drude_context_can_use_verlet_without_integrating():
    fragment, mm = _drude_mm()
    assert mm.integrator_name == "VerletIntegrator"
    energy, gradient = mm.run(current_coords=fragment.coords, grad=True)
    assert energy > 0
    assert np.isfinite(gradient).all()


def test_direct_rpmd_context_rejects_drude_particles():
    _, mm = _drude_mm()
    mm.set_simulation_parameters(integrator="RPMDIntegrator")
    with pytest.raises(InputError, match="must not be treated as quantum nuclei"):
        mm.create_simulation()


@pytest.mark.parametrize("num_beads", [1, 4])
def test_nqe_export_rejects_drude_particles_before_mutation(num_beads):
    qmmm, _, _ = _make_analytic_qmmm()
    # Add an oscillator after partitioning solely to isolate export validation.
    force = openmm.DrudeForce()
    force.addParticle(1, 0, -1, -1, -1, -0.5, 0.001, 1, 1)
    qmmm.mm_theory.system.addForce(force)
    before = openmm.XmlSerializer.serialize(qmmm.mm_theory.system)
    with pytest.raises(InputError, match="export_rpmd_potential does not support Drude particles"):
        export_rpmd_potential(theory=qmmm, num_beads=num_beads)
    assert openmm.XmlSerializer.serialize(qmmm.mm_theory.system) == before
    assert not qmmm.openmm_externalforce
    assert not qmmm.exit_after_customexternalforce_update


def test_drude_scf_factory_uses_requested_step_size():
    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.set_simulation_parameters(integrator="DrudeSCFIntegrator", timestep=0.0002)
    theory.create_integrator()
    assert isinstance(theory.integrator, openmm.DrudeSCFIntegrator)
    assert theory.integrator.getStepSize().value_in_unit(unit.picosecond) == pytest.approx(0.0002)
