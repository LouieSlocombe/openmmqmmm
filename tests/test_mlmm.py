"""ML/MM assembly, independent mechanical-embedding references, and native dynamics."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import openmm
import pytest
from conftest import central_difference_gradient, dummy_mm

from openmmqmmm import Fragment, MLMMTheory, MolecularDynamicsEngine, OpenMMTheory, QMMMTheory, ZeroTheory, single_point
from openmmqmmm.constants import HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError, MissingDependencyError

CHARGES = np.array([0.2, -0.2, 0.3, -0.3])
SIGMAS = np.array([0.20, 0.22, 0.24, 0.26])
EPSILONS = np.array([0.05, 0.10, 0.15, 0.20])
COULOMB = 138.93545764438198


def _source(*, periodic=False, **options):
    coords = np.array([[0.2, 0.2, 0.3], [1.4, 0.4, 0.6], [5.0, 1.0, 0.3], [6.6, 1.5, 0.1]])
    if periodic:
        coords[2:] += [30.0, 0.0, 0.0]
    fragment = Fragment(elems=["H", "H", "He", "He"], coords=coords, conncalc=False)
    mm = dummy_mm(fragment, bonds=[(0, 1), (2, 3)], **options)
    for index, (charge, sigma, epsilon) in enumerate(zip(CHARGES, SIGMAS, EPSILONS, strict=True)):
        mm.nonbonded_force.setParticleParameters(index, charge, sigma, epsilon)
    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.10, 1000.0)
    bond_force.addBond(2, 3, 0.15, 700.0)
    bond_force.setUsesPeriodicBoundaryConditions(periodic)
    mm.system.addForce(bond_force)
    mm.getatomcharges()
    if periodic:
        mm.periodic = True
        mm.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        mm.nonbonded_force.setCutoffDistance(1.2)
        mm.nonbonded_force.setReactionFieldDielectric(1.0)
        mm.nonbonded_force.setUseDispersionCorrection(False)
        mm.update_cell(periodic_cell_vectors=np.eye(3) * 30.0)
    return fragment, mm


class _RecordingPotential:
    def __init__(self):
        self.calls = []

    def createMixedSystem(self, topology, system, atoms, **kwargs):  # noqa: N802
        self.calls.append((topology, system, list(atoms), kwargs))
        return system


def test_import_does_not_load_optional_ml_backend():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, openmmqmmm; assert 'openmmml' not in sys.modules"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_object_potential_works_without_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "openmmml", None)
    fragment, source = _source()
    potential = _RecordingPotential()
    theory = MLMMTheory(mm_theory=source, mlatoms=[np.int64(0), 1], potential=potential, label="test ML/MM")

    assert isinstance(theory, OpenMMTheory)
    assert theory.theorytype == "MM"
    assert theory.theorynamelabel == "MLMMTheory"
    assert theory.label == "test ML/MM"
    assert theory.mlatoms == [0, 1]
    assert theory.system is not source.system
    assert theory.topology is not source.topology
    assert theory.platform_choice == "Reference"
    assert theory.run(current_coords=fragment.coords) == pytest.approx(source.run(current_coords=fragment.coords))


def test_missing_optional_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "openmmml", None)
    _fragment, source = _source()
    with pytest.raises(MissingDependencyError, match=r"openmmml.*|openmm-ml") as error:
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="ani2x")
    assert "install" in str(error.value).lower()
    assert "environment_ml.yml" in str(error.value)


def test_named_potential_and_system_options_reach_the_correct_backend_calls(monkeypatch):
    _fragment, source = _source()
    potential = _RecordingPotential()
    constructor_calls = []

    def construct(name, **kwargs):
        constructor_calls.append((name, kwargs))
        return potential

    monkeypatch.setitem(sys.modules, "openmmml", SimpleNamespace(MLPotential=construct))
    MLMMTheory(
        mm_theory=source,
        mlatoms=[0, 1],
        potential="local-model",
        potential_kwargs={"modelPath": "local.pt"},
        system_kwargs={"precision": "double", "forceGroup": 7},
        remove_constraints=False,
    )

    assert constructor_calls == [("local-model", {"modelPath": "local.pt"})]
    topology, system, atoms, options = potential.calls[0]
    assert topology is not source.topology
    assert system is not source.system
    assert atoms == [0, 1]
    assert options["removeConstraints"] is False
    assert options["precision"] == "double"
    assert options["forceGroup"] == 7
    assert options.get("embedding", "mechanical") == "mechanical"
    assert options.get("interpolate", False) is False
    assert options.get("returnInfo", False) is False


@pytest.mark.parametrize("atoms", [[], [0, 0], [-1], [4], [True], [np.bool_(False)], [0.0], ["0"], None])
def test_invalid_ml_atoms_are_rejected_before_backend_call(atoms):
    _fragment, source = _source()
    potential = _RecordingPotential()
    with pytest.raises(InputError):
        MLMMTheory(mm_theory=source, mlatoms=atoms, potential=potential)
    assert potential.calls == []


def test_covalent_boundary_is_rejected_without_mutating_source():
    _fragment, source = _source()
    xml_before = openmm.XmlSerializer.serialize(source.system)
    potential = _RecordingPotential()
    with pytest.raises(InputError, match=r"bond|boundar|molecule"):
        MLMMTheory(mm_theory=source, mlatoms=[0], potential=potential)
    assert potential.calls == []
    assert openmm.XmlSerializer.serialize(source.system) == xml_before


@pytest.mark.parametrize(
    "key", ["topology", "system", "atoms", "removeConstraints", "embedding", "interpolate", "returnInfo"]
)
def test_reserved_system_options_cannot_override_mechanical_embedding(key):
    _fragment, source = _source()
    potential = _RecordingPotential()
    with pytest.raises(InputError, match=key):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=potential, system_kwargs={key: True})
    assert potential.calls == []


def test_mismatched_topology_and_system_are_rejected():
    _fragment, source = _source()
    source.system.addParticle(1.0)
    potential = _RecordingPotential()
    with pytest.raises(InputError, match=r"topology|particles|atoms"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=potential)
    assert potential.calls == []


@pytest.mark.parametrize("selected", [[4], [0, 1]])
def test_virtual_sites_cannot_be_selected_or_depend_on_ml_atoms(selected):
    fragment = Fragment(elems=["H"] * 5, coords=np.arange(15).reshape(5, 3), conncalc=False)
    source = dummy_mm(fragment, bonds=[(0, 1)])
    source.system.setParticleMass(4, 0.0)
    source.system.setVirtualSite(4, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    potential = _RecordingPotential()
    with pytest.raises(InputError, match=r"virtual|Virtual"):
        MLMMTheory(mm_theory=source, mlatoms=selected, potential=potential)
    assert potential.calls == []


def test_backend_failure_does_not_mutate_source_system_or_topology():
    _fragment, source = _source()
    xml_before = openmm.XmlSerializer.serialize(source.system)
    atom_name = next(source.topology.atoms()).name

    class FailingPotential:
        def createMixedSystem(self, topology, system, atoms, **kwargs):  # noqa: N802
            system.setParticleMass(0, 123.0)
            next(topology.atoms()).name = "changed"
            raise RuntimeError("model construction failed")

    with pytest.raises(RuntimeError, match="model construction failed"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=FailingPotential())
    assert openmm.XmlSerializer.serialize(source.system) == xml_before
    assert next(source.topology.atoms()).name == atom_name


def test_mixed_force_references_and_result_caches_are_refreshed():
    fragment, source = _source(periodic=True, do_energy_decomposition=True)
    source.forcegroupify()
    source.run(current_coords=fragment.coords, grad=True)
    source.create_simulation(internal=True)
    source._exception_charge_scales = {(0, 1): 0.5}
    xml_before = openmm.XmlSerializer.serialize(source.system)

    class ReplacingPotential:
        def createMixedSystem(self, topology, system, atoms, **kwargs):  # noqa: N802
            for index in reversed(range(system.getNumForces())):
                if isinstance(system.getForce(index), openmm.NonbondedForce):
                    system.removeForce(index)
            replacement = openmm.NonbondedForce()
            replacement.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            for index in range(system.getNumParticles()):
                replacement.addParticle(index * 0.01, 0.2, 0.0)
            system.addForce(replacement)
            return system

    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=ReplacingPotential())
    assert theory.nonbonded_force.getParticleParameters(3)[0].value_in_unit(openmm.unit.elementary_charge) == 0.03
    assert theory.charges == pytest.approx([0.0, 0.01, 0.02, 0.03])
    assert theory.getatomcharges() == pytest.approx(theory.charges)
    assert not hasattr(theory, "simulation")
    assert not hasattr(theory, "_cell_gradient_snapshot")
    assert not hasattr(theory, "_exception_charge_scales")
    assert not hasattr(theory, "energy")
    assert not hasattr(theory, "gradient")
    assert len(theory.forcegroups) == theory.system.getNumForces()
    assert {int(force.this) for force in theory.forcegroups} == {int(force.this) for force in theory.system.getForces()}
    assert openmm.XmlSerializer.serialize(source.system) == xml_before
    theory.update_charges([2], [0.7])
    assert source.getatomcharges() == pytest.approx(CHARGES)


@pytest.mark.parametrize("subclass", [False, True])
def test_qmmm_prepared_source_is_rejected(subclass):
    fragment, source = _source()
    if subclass:

        class ClassicalTheory(OpenMMTheory):
            pass

        source.__class__ = ClassicalTheory
    QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory(),
        mm_theory=source,
        charges=list(source.charges),
        qmatoms=[0, 1],
        qm_charge=0,
        qm_mult=1,
    )
    with pytest.raises(InputError, match=r"QM/MM|QMMM"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=_RecordingPotential())


def test_unknown_source_force_is_rejected():
    _fragment, source = _source()
    force = openmm.CustomCompoundBondForce(2, "distance(p1,p2)^2")
    force.addBond([0, 1], [])
    source.system.addForce(force)
    with pytest.raises(InputError, match="CustomCompoundBondForce"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=_RecordingPotential())


def test_ml_nonbonded_parameter_offsets_are_rejected():
    _fragment, source = _source()
    source.nonbonded_force.addGlobalParameter("lambda_charge", 1.0)
    source.nonbonded_force.addParticleParameterOffset("lambda_charge", 0, 0.1, 0.0, 0.0)
    with pytest.raises(InputError, match=r"offset|Offset"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=_RecordingPotential())


@pytest.mark.parametrize("potential", [None, object(), "", "   "])
def test_invalid_potential_is_rejected(potential):
    _fragment, source = _source()
    with pytest.raises(InputError, match="potential"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=potential)


@pytest.mark.parametrize(
    "options",
    [
        {"potential_kwargs": []},
        {"system_kwargs": []},
        {"system_kwargs": {1: True}},
        {"potential_kwargs": {"modelPath": "unused.pt"}},
        {"remove_constraints": 1},
    ],
)
def test_invalid_backend_options_are_rejected(options):
    _fragment, source = _source()
    potential = _RecordingPotential()
    with pytest.raises(InputError):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=potential, **options)
    assert potential.calls == []


@pytest.mark.parametrize("incorrect_particles", [False, True])
def test_backend_must_return_a_system_with_unchanged_particle_count(incorrect_particles):
    _fragment, source = _source()

    class InvalidPotential:
        def createMixedSystem(self, topology, system, atoms, **kwargs):  # noqa: N802
            if incorrect_particles:
                system.addParticle(1.0)
                return system
            return {"system": system}

    with pytest.raises(InputError, match=r"System|particle"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=InvalidPotential())
    assert source.system.getNumParticles() == 4


@pytest.fixture
def native_ml_name():
    pytest.importorskip("openmmml")
    from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

    class HarmonicImpl(MLPotentialImpl):
        def getMLLongRange(self):  # noqa: N802
            return False

        def addForces(self, topology, system, atoms, forceGroup, **kwargs):  # noqa: N802, N803
            selected = list(range(system.getNumParticles())) if atoms is None else list(atoms)
            assert len(selected) == 2
            force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
            force.addGlobalParameter("k", 1200.0)
            force.addGlobalParameter("r0", 0.13)
            force.addBond(*selected, [])
            force.setUsesPeriodicBoundaryConditions(topology.getPeriodicBoxVectors() is not None)
            force.setForceGroup(forceGroup)
            system.addForce(force)

    class HarmonicFactory(MLPotentialImplFactory):
        def createImpl(self, name, **kwargs):  # noqa: N802
            return HarmonicImpl()

    name = "openmmqmmm-test-harmonic"
    MLPotential.registerImplFactory(name, HarmonicFactory())
    return name


def _reference_energy(coords, *, periodic=False):
    positions = np.asarray(coords) * 0.1

    def distance(first, second):
        delta = positions[first] - positions[second]
        if periodic:
            delta -= 3.0 * np.round(delta / 3.0)
        return np.linalg.norm(delta)

    energy = 0.5 * 1200.0 * (distance(0, 1) - 0.13) ** 2
    energy += 0.5 * 700.0 * (distance(2, 3) - 0.15) ** 2
    for first in range(4):
        for second in range(first + 1, 4):
            if (first, second) == (0, 1):
                continue
            radius = distance(first, second)
            sigma = 0.5 * (SIGMAS[first] + SIGMAS[second])
            epsilon = np.sqrt(EPSILONS[first] * EPSILONS[second])
            energy += 4 * epsilon * ((sigma / radius) ** 12 - (sigma / radius) ** 6)
            coulomb = 1.0 / radius - (1.0 / 1.2 if periodic else 0.0)
            energy += COULOMB * CHARGES[first] * CHARGES[second] * coulomb
    return energy / HARTREE_TO_KJ_PER_MOL


@pytest.mark.parametrize("periodic", [False, True])
def test_real_openmm_ml_mechanical_energy_and_gradient_match_independent_reference(native_ml_name, periodic):
    fragment, source = _source(periodic=periodic)
    xml_before = openmm.XmlSerializer.serialize(source.system)
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=native_ml_name)

    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    expected_energy = _reference_energy(fragment.coords, periodic=periodic)
    expected_gradient = central_difference_gradient(
        lambda coords: _reference_energy(coords, periodic=periodic), fragment.coords
    )

    assert energy == pytest.approx(expected_energy, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-8, atol=1e-10)
    assert openmm.XmlSerializer.serialize(source.system) == xml_before
    result = single_point(fragment=fragment, theory=theory, grad=True)
    assert result.energy == pytest.approx(expected_energy)
    np.testing.assert_allclose(result.gradient, expected_gradient, rtol=2e-8, atol=1e-10)
    if periodic:
        shifted = fragment.coords.copy()
        shifted[:2] += [30, 0, 0]
        assert theory.run(current_coords=shifted) == pytest.approx(energy, abs=1e-12)


def test_real_openmm_ml_preserves_pme_cross_and_image_interactions(native_ml_name):
    fragment, source = _source(periodic=True)
    source.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    source.nonbonded_force.setEwaldErrorTolerance(1e-6)
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=native_ml_name)
    reference_simulation = source.create_simulation()

    def reference_energy(coords):
        reference_simulation.context.setPositions(np.asarray(coords) * openmm.unit.angstrom)
        energy = reference_simulation.context.getState(getEnergy=True).getPotentialEnergy()
        radius = np.linalg.norm(np.asarray(coords)[0] - np.asarray(coords)[1]) * 0.1
        sigma = 0.5 * (SIGMAS[0] + SIGMAS[1])
        epsilon = np.sqrt(EPSILONS[0] * EPSILONS[1])
        # Local ML replaces the direct intramolecular MM terms; periodic images remain under PME.
        internal_mm = 0.5 * 1000.0 * (radius - 0.10) ** 2
        internal_mm += COULOMB * CHARGES[0] * CHARGES[1] / radius
        internal_mm += 4 * epsilon * ((sigma / radius) ** 12 - (sigma / radius) ** 6)
        ml_energy = 0.5 * 1200.0 * (radius - 0.13) ** 2
        return (energy.value_in_unit(openmm.unit.kilojoules_per_mole) - internal_mm + ml_energy) / HARTREE_TO_KJ_PER_MOL

    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    assert energy == pytest.approx(reference_energy(fragment.coords), rel=1e-9, abs=1e-12)
    expected_gradient = central_difference_gradient(reference_energy, fragment.coords)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-10)


@pytest.mark.parametrize("remove_constraints", [False, True])
def test_real_openmm_ml_only_removes_constraints_inside_ml_region(native_ml_name, remove_constraints):
    _fragment, source = _source(bondconstraints=[[0, 1, 1.0], [2, 3, 1.5]])
    theory = MLMMTheory(
        mm_theory=source,
        mlatoms=[0, 1],
        potential=native_ml_name,
        remove_constraints=remove_constraints,
    )

    assert source.system.getNumConstraints() == 2
    assert theory.system.getNumConstraints() == (1 if remove_constraints else 2)
    assert theory.dof == source.dof + int(remove_constraints)
    assert theory.user_constraints == source.user_constraints
    assert theory.user_constraints is not source.user_constraints
    with pytest.raises(InputError, match="constraints"):
        theory.run(current_coords=source.fragment.coords)


def test_real_openmm_ml_preserves_user_restraints_within_ml_region(native_ml_name):
    fragment, source = _source(restraints=[[0, 1, 1.0, 2.0]])
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=native_ml_name)
    theory.force_run = True
    radius_angstrom = np.linalg.norm(fragment.coords[0] - fragment.coords[1])
    restraint_energy = 0.5 * 2.0 * 4.184 * (radius_angstrom - 1.0) ** 2 / HARTREE_TO_KJ_PER_MOL

    assert theory.run(current_coords=fragment.coords) == pytest.approx(
        _reference_energy(fragment.coords) + restraint_energy, rel=1e-10
    )
    assert theory.user_restraints == source.user_restraints


@pytest.mark.parametrize("integrator", ["VerletIntegrator", "RPMDIntegrator"])
def test_real_openmm_ml_native_dynamics_and_rpmd(native_ml_name, integrator):
    fragment, source = _source()
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=native_ml_name)
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        integrator=integrator,
        rpmd_num_copies=2,
        timestep=0.000001,
        traj_frequency=2,
    )
    try:
        engine.run(simulation_steps=2)
        assert engine.theory_runtype == "MM"
        assert engine.rpmd_force_provider is None
        assert engine.simulation.currentStep == 2
        if integrator == "RPMDIntegrator":
            state = engine.simulation.integrator.getState(0, getPositions=True, getEnergy=True)
        else:
            state = engine.simulation.context.getState(getPositions=True, getEnergy=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
        assert np.isfinite(positions).all()
        assert np.isfinite(state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole))
    finally:
        engine.close()
