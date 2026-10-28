"""MACE-POLAR option handling and long-range mechanical embedding without model downloads."""

import sys
from types import SimpleNamespace

import numpy as np
import openmm
import pytest
from conftest import central_difference_gradient, dummy_mm

from openmmqmmm import Fragment, MLMMTheory, MolecularDynamicsEngine
from openmmqmmm.constants import HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError, MissingDependencyError

POLAR_NAMES = ["mace-polar-1-small", "mace-polar-1-medium", "mace-polar-1-large"]
CHARGES = [0.2, 0.1, -0.15, -0.15]
ML_SIGMAS = [0.12, 0.14]
ML_EPSILONS = [0.02, 0.03]


def _source(*, periodic=False, method=openmm.NonbondedForce.PME):
    fragment = Fragment(
        elems=["H", "H", "He", "He"],
        coords=[[0.0, 0.0, 0.0], [1.3, 0.3, 0.2], [5.0, 1.0, 0.3], [6.7, 1.5, 0.4]],
        conncalc=False,
    )
    theory = dummy_mm(fragment, bonds=[(0, 1), (2, 3)])
    for index, charge in enumerate(CHARGES):
        theory.nonbonded_force.setParticleParameters(index, charge, 0.12 + 0.02 * index, 0.02 + 0.01 * index)
    bonds = openmm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.10, 1000.0)
    bonds.addBond(2, 3, 0.15, 700.0)
    theory.system.addForce(bonds)
    theory.getatomcharges()
    if periodic:
        theory.periodic = True
        theory.nonbonded_force.setNonbondedMethod(method)
        theory.nonbonded_force.setCutoffDistance(1.0)
        theory.nonbonded_force.setUseDispersionCorrection(False)
        theory.nonbonded_force.setEwaldErrorTolerance(1e-6)
        theory.nonbonded_force.setPMEParameters(3.5, 32, 32, 32)
        theory.update_cell(periodic_cell_vectors=np.eye(3) * 30.0)
        bonds.setUsesPeriodicBoundaryConditions(True)
    return fragment, theory


class _RecordingPotential:
    def __init__(self, name):
        self._impl = SimpleNamespace(name=name)
        self.calls = []

    def createMixedSystem(self, topology, system, atoms, **kwargs):  # noqa: N802
        self.calls.append((topology, system, list(atoms), kwargs))
        return system


@pytest.fixture
def polar_calculator_factory(monkeypatch):
    calls = []
    calculator = object()

    def construct(**kwargs):
        calls.append(kwargs)
        return calculator

    module = SimpleNamespace(mace_polar=construct)
    monkeypatch.setitem(sys.modules, "mace.calculators", module)
    monkeypatch.setitem(sys.modules, "graph_longrange", SimpleNamespace())
    return SimpleNamespace(calls=calls, calculator=calculator, module=module)


@pytest.fixture
def recording_factory(monkeypatch, polar_calculator_factory):
    calls = []

    def construct(name, **kwargs):
        potential = _RecordingPotential(name)
        calls.append((name, kwargs, potential))
        return potential

    monkeypatch.setitem(sys.modules, "openmmml", SimpleNamespace(MLPotential=construct))
    return calls


@pytest.mark.parametrize("name", [*POLAR_NAMES, "mace-polar"])
def test_polar_presets_select_full_energy_and_default_electronic_state(
    recording_factory, polar_calculator_factory, name
):
    _fragment, source = _source()
    xml_before = openmm.XmlSerializer.serialize(source.system)
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=name)

    actual_name, constructor_options, potential = recording_factory[0]
    assert actual_name == "ase"
    assert constructor_options == {}
    options = potential.calls[0][3]
    assert options["calculator"] is polar_calculator_factory.calculator
    assert options["info"] == {"charge": 0, "spin": 1}
    assert options["mlLongRange"] is True
    assert "returnEnergyType" not in options
    model = (
        "polar-1-m"
        if name == "mace-polar"
        else {"small": "polar-1-s", "medium": "polar-1-m", "large": "polar-1-l"}[name.rsplit("-", 1)[1]]
    )
    assert polar_calculator_factory.calls[0]["model"] == model
    assert polar_calculator_factory.calls[0]["default_dtype"] == "float64"
    assert theory.potential_name == ("mace-polar-1-medium" if name == "mace-polar" else name)
    assert openmm.XmlSerializer.serialize(source.system) == xml_before


def test_explicit_polar_state_and_model_options_are_forwarded_without_mutating_options(
    recording_factory, polar_calculator_factory
):
    _fragment, source = _source()
    options = {"charge": np.int64(-1), "multiplicity": np.int64(2), "device": "cpu", "precision": "double"}
    original = options.copy()
    MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar-1-small", system_kwargs=options)

    passed = recording_factory[0][2].calls[0][3]
    assert passed["info"] == {"charge": -1, "spin": 2}
    assert polar_calculator_factory.calls[0]["device"] == "cpu"
    assert polar_calculator_factory.calls[0]["default_dtype"] == "float64"
    assert options == original


@pytest.mark.parametrize("field", ["charge", "multiplicity"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.0, "1", None])
def test_polar_electronic_state_requires_integers(recording_factory, field, value):
    _fragment, source = _source()
    with pytest.raises(InputError, match=field):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar", system_kwargs={field: value})
    assert all(not potential.calls for _name, _options, potential in recording_factory)


@pytest.mark.parametrize("multiplicity", [0, -1])
def test_polar_multiplicity_must_be_positive(recording_factory, multiplicity):
    _fragment, source = _source()
    with pytest.raises(InputError, match="multiplicity"):
        MLMMTheory(
            mm_theory=source,
            mlatoms=[0, 1],
            potential="mace-polar",
            system_kwargs={"multiplicity": multiplicity},
        )


@pytest.mark.parametrize("energy_type", ["interaction_energy", "forces", None])
def test_polar_cannot_omit_coordinate_dependent_electrostatic_energy(recording_factory, energy_type):
    _fragment, source = _source()
    with pytest.raises(InputError, match="returnEnergyType"):
        MLMMTheory(
            mm_theory=source,
            mlatoms=[0, 1],
            potential="mace-polar",
            system_kwargs={"returnEnergyType": energy_type},
        )


@pytest.mark.parametrize("option", ["spin", "total_charge", "total_spin", "external_field"])
def test_ignored_backend_electronic_options_raise_instead_of_silently_changing_physics(recording_factory, option):
    _fragment, source = _source()
    with pytest.raises(InputError, match=option):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar", system_kwargs={option: 2})


@pytest.mark.parametrize("name", ["mace-polar", "mace-polar-1-large"])
def test_polar_local_checkpoint_uses_ase_factory_and_long_range_flag(
    recording_factory, polar_calculator_factory, name, tmp_path
):
    _fragment, source = _source()
    checkpoint = tmp_path / "trained-polar.model"
    checkpoint.write_text("Fixture only: model loading is mocked.")
    theory = MLMMTheory(
        mm_theory=source,
        mlatoms=[0, 1],
        potential=name,
        potential_kwargs={"modelPath": "trained-polar.model"},
    )

    actual_name, constructor_options, potential = recording_factory[0]
    assert actual_name == "ase"
    assert constructor_options == {}
    assert polar_calculator_factory.calls[0]["model"] == str(checkpoint)
    assert potential.calls[0][3]["mlLongRange"] is True
    assert potential.calls[0][3]["calculator"] is polar_calculator_factory.calculator
    assert "polar" in theory.potential_name


def test_missing_polar_checkpoint_is_rejected_before_loading_a_model(recording_factory, polar_calculator_factory):
    _fragment, source = _source()
    with pytest.raises(InputError, match="modelPath"):
        MLMMTheory(
            mm_theory=source,
            mlatoms=[0, 1],
            potential="mace-polar",
            potential_kwargs={"modelPath": "absent.model"},
        )
    assert polar_calculator_factory.calls == []


@pytest.mark.parametrize("local", [False, True])
def test_polar_cannot_be_declared_short_range(recording_factory, local):
    _fragment, source = _source(periodic=True)
    with pytest.raises(InputError, match="mlLongRange"):
        MLMMTheory(
            mm_theory=source,
            mlatoms=[0, 1],
            potential="mace-polar",
            potential_kwargs={"modelPath": "trained-polar.model"} if local else None,
            system_kwargs={"mlLongRange": False},
        )


@pytest.mark.parametrize(
    "options",
    [
        {"potential_kwargs": {"charge": 1}},
        {"system_kwargs": {"precision": "mixed"}},
        {"system_kwargs": {"calculator": object()}},
        {"system_kwargs": {"info": {"charge": 1}}},
    ],
)
def test_polar_rejects_unsupported_or_misplaced_backend_options(recording_factory, options):
    _fragment, source = _source()
    with pytest.raises(InputError):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar", **options)


@pytest.mark.parametrize(
    "method",
    [openmm.NonbondedForce.NoCutoff, openmm.NonbondedForce.CutoffPeriodic, openmm.NonbondedForce.LJPME],
)
def test_periodic_polar_requires_long_range_mm_electrostatics(recording_factory, method):
    _fragment, source = _source(periodic=True, method=method)
    xml_before = openmm.XmlSerializer.serialize(source.system)
    with pytest.raises(InputError, match=r"PME|Ewald|long.range"):
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar")
    assert openmm.XmlSerializer.serialize(source.system) == xml_before


def test_prebuilt_polar_potential_receives_the_same_safe_defaults(recording_factory):
    _fragment, source = _source()
    potential = _RecordingPotential("mace-polar-1-medium")
    MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=potential)
    assert potential.calls == []
    assert recording_factory[0][0] == "ase"
    assert recording_factory[0][2].calls[0][3]["info"] == {"charge": 0, "spin": 1}


def test_nonpolar_models_keep_existing_energy_and_electronic_option_behavior(recording_factory):
    _fragment, source = _source()
    MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-off23-small")
    passed = recording_factory[0][2].calls[0][3]
    assert "returnEnergyType" not in passed
    assert "charge" not in passed
    assert "multiplicity" not in passed


def test_missing_polar_dependencies_produce_an_actionable_error(monkeypatch, recording_factory):
    _fragment, source = _source()
    monkeypatch.setitem(sys.modules, "graph_longrange", None)
    with pytest.raises(MissingDependencyError, match=r"mace|MACE") as error:
        MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar")
    assert "graph" in str(error.value).lower()
    assert "install" in str(error.value).lower()


@pytest.fixture
def native_polar_calculator(monkeypatch, polar_calculator_factory):
    pytest.importorskip("openmmml")
    pytest.importorskip("ase")
    from ase import units
    from ase.calculators.calculator import Calculator, all_changes

    calls = []

    class HarmonicCalculator(Calculator):
        implemented_properties = ("energy", "forces")

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            calls.append(atoms.copy())
            displacement = atoms.positions[0] - atoms.positions[1]
            radius_angstrom = np.linalg.norm(displacement)
            radius_nm = radius_angstrom * 0.1
            energy = 0.5 * 1200 * (radius_nm - 0.13) ** 2
            force = -1200 * (radius_nm - 0.13) * 0.1 * displacement / radius_angstrom
            conversion = units.kJ / units.mol
            self.results = {"energy": energy * conversion, "forces": np.array([force, -force]) * conversion}

    def construct(**kwargs):
        polar_calculator_factory.calls.append(kwargs)
        return HarmonicCalculator()

    monkeypatch.setattr(polar_calculator_factory.module, "mace_polar", construct)
    return SimpleNamespace(factory_calls=polar_calculator_factory.calls, evaluations=calls)


@pytest.mark.parametrize("name", [*POLAR_NAMES, "mace-polar"])
def test_real_openmm_ml_ase_bridge_receives_polar_state(native_polar_calculator, name):
    fragment, source = _source()
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential=name)
    assert np.isfinite(theory.run(current_coords=fragment.coords))
    assert native_polar_calculator.evaluations[0].info == {"charge": 0, "spin": 1}
    assert native_polar_calculator.evaluations[0].get_chemical_symbols() == ["H", "H"]
    assert any(isinstance(force, openmm.PythonForce) for force in theory.system.getForces())


@pytest.mark.parametrize("integrator", ["VerletIntegrator", "RPMDIntegrator"])
def test_real_polar_ase_force_runs_in_classical_and_ring_polymer_dynamics(native_polar_calculator, integrator):
    fragment, source = _source()
    theory = MLMMTheory(mm_theory=source, mlatoms=[0, 1], potential="mace-polar")
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
        assert engine.simulation.currentStep == 2
        assert engine.theory_runtype == "MM"
        assert native_polar_calculator.evaluations
        if integrator == "RPMDIntegrator":
            state = engine.simulation.integrator.getState(0, getEnergy=True)
        else:
            state = engine.simulation.context.getState(getEnergy=True)
        assert np.isfinite(state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole))
    finally:
        engine.close()


@pytest.mark.parametrize("local", [False, True])
def test_real_long_range_polar_embedding_removes_entire_ml_pme_energy(native_polar_calculator, local, tmp_path):
    fragment, source = _source(periodic=True)
    checkpoint = tmp_path / "test-polar.model"
    if local:
        checkpoint.write_text("Fixture only: model loading is mocked.")
    xml_before = openmm.XmlSerializer.serialize(source.system)
    theory = MLMMTheory(
        mm_theory=source,
        mlatoms=[0, 1],
        potential="mace-polar-1-small",
        potential_kwargs={"modelPath": "test-polar.model"} if local else None,
    )
    classical = source.create_simulation()
    ml_only_system = openmm.System()
    ml_only_force = openmm.NonbondedForce()
    ml_only_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    ml_only_force.setCutoffDistance(1.0)
    ml_only_force.setEwaldErrorTolerance(1e-6)
    ml_only_force.setPMEParameters(3.5, 32, 32, 32)
    for index in range(4):
        ml_only_system.addParticle(1.0)
        ml_only_force.addParticle(CHARGES[index] if index < 2 else 0.0, 1.0, 0.0)
    ml_only_system.addForce(ml_only_force)
    ml_only_system.setDefaultPeriodicBoxVectors(*np.eye(3) * 3.0)
    integrator = openmm.VerletIntegrator(0.001)
    ml_only_context = openmm.Context(ml_only_system, integrator, openmm.Platform.getPlatformByName("Reference"))

    def reference_energy(coords):
        coords = np.asarray(coords)
        for context in (classical.context, ml_only_context):
            context.setPositions(coords * openmm.unit.angstrom)
        full_energy = classical.context.getState(getEnergy=True).getPotentialEnergy()
        ml_pme_energy = ml_only_context.getState(getEnergy=True).getPotentialEnergy()
        radius = np.linalg.norm(coords[0] - coords[1]) * 0.1
        sigma = 0.5 * sum(ML_SIGMAS)
        epsilon = np.sqrt(np.prod(ML_EPSILONS))
        internal_lj = 4 * epsilon * ((sigma / radius) ** 12 - (sigma / radius) ** 6)
        internal_bond = 0.5 * 1000 * (radius - 0.10) ** 2
        polar_energy = 0.5 * 1200 * (radius - 0.13) ** 2
        reference = (full_energy - ml_pme_energy).value_in_unit(openmm.unit.kilojoules_per_mole)
        return (reference - internal_lj - internal_bond + polar_energy) / HARTREE_TO_KJ_PER_MOL

    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    assert energy == pytest.approx(reference_energy(fragment.coords), rel=1e-9, abs=1e-11)
    np.testing.assert_allclose(
        gradient,
        central_difference_gradient(reference_energy, fragment.coords),
        rtol=3e-6,
        atol=3e-10,
    )
    assert openmm.XmlSerializer.serialize(source.system) == xml_before
    assert native_polar_calculator.factory_calls[0]["model"] == (str(checkpoint) if local else "polar-1-s")
    assert native_polar_calculator.evaluations[0].pbc.tolist() == [True, True, True]
    np.testing.assert_allclose(native_polar_calculator.evaluations[0].cell, np.eye(3) * 30.0)


@pytest.mark.parametrize("periodic", [False, True])
def test_polar_boundary_conditions_follow_the_source_system_instead_of_stale_topology(
    native_polar_calculator, periodic
):
    fragment, source = _source(periodic=periodic)
    stale_cell = None if periodic else np.eye(3) * 2.0
    source.topology.setPeriodicBoxVectors(stale_cell)
    theory = MLMMTheory(
        mm_theory=source,
        mlatoms=[0, 1],
        potential="mace-polar",
        system_kwargs={"charge": 1, "multiplicity": 2, "precision": "single"},
    )
    theory.run(current_coords=fragment.coords)
    evaluated = native_polar_calculator.evaluations[0]
    assert evaluated.pbc.tolist() == [periodic] * 3
    assert evaluated.info == {"charge": 1, "spin": 2}
    assert (theory.ml_charge, theory.ml_mult, theory.ml_precision) == (1, 2, "single")
    assert native_polar_calculator.factory_calls[0]["default_dtype"] == "float32"
    if periodic:
        np.testing.assert_allclose(evaluated.cell, np.eye(3) * 30.0)
        assert source.topology.getPeriodicBoxVectors() is None
    else:
        np.testing.assert_allclose(
            source.topology.getPeriodicBoxVectors().value_in_unit(openmm.unit.nanometer), stale_cell
        )
