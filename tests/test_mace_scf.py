"""MACE-SCF protocol regressions using an analytic ASE embedding calculator."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import openmm
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from conftest import central_difference_gradient, dummy_mm

from openmmqmmm import Fragment, MACESCFTheory, OpenMMQMMMCalculator, QMMMTheory, export_rpmd_potential, single_point
from openmmqmmm.constants import (
    BOHR_TO_ANG,
    HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM,
    HARTREE_TO_EV,
    HARTREE_TO_KJ_PER_MOL,
)
from openmmqmmm.exceptions import InputError, InternalError, MissingDependencyError

QM_COORDS = np.array([[0.2, -0.4, 0.3], [1.1, 0.3, -0.2]])
MM_COORDS = np.array([[4.1, 0.8, -0.7], [-2.3, 1.4, 0.5]])
MM_CHARGES = np.array([0.3, -0.2])


def _reference(coords, charge=0, mm_coords=(), mm_charges=()):
    """Return eV and eV/Angstrom gradients for a polynomial with QM--MM coupling."""
    coords = np.asarray(coords, dtype=float)
    mm_coords = np.asarray(mm_coords, dtype=float).reshape(-1, 3)
    mm_charges = np.asarray(mm_charges, dtype=float)
    weights = np.array([0.7, 1.1, 1.3])
    energy = 0.5 * np.sum(weights * coords**2) + charge * (1.7 + 0.2 * np.sum(coords))
    gradient = weights * coords + 0.2 * charge
    displacement = coords[:, None, :] - mm_coords[None, :, :]
    coupling_gradient = mm_charges[None, :, None] * weights * displacement
    energy += 0.5 * np.sum(displacement * coupling_gradient)
    gradient += np.sum(coupling_gradient, axis=1)
    pc_gradient = -np.sum(coupling_gradient, axis=0)
    return float(energy), gradient, pc_gradient


class _PointCharges:
    def __init__(self, calculator, charges):
        self.calculator = calculator
        self.charges = np.array(charges, dtype=float, copy=True)
        self.positions = np.empty((0, 3)) if len(self.charges) == 0 else None

    def set_positions(self, positions):
        self.positions = np.array(positions, dtype=float, copy=True).reshape(-1, 3)
        assert len(self.positions) == len(self.charges)
        self.calculator.results = {}

    def set_charges(self, charges):
        charges = np.array(charges, dtype=float, copy=True)
        assert charges.shape == self.charges.shape
        self.charges = charges
        self.calculator.results = {}

    def get_forces(self, calculator):
        assert calculator is self.calculator
        assert calculator.pcpot is self
        if len(self.charges) == 0:
            return np.empty((0, 3))
        return calculator.get_property("point_charge_forces", calculator.atoms)


class _EmbeddedCalculator(Calculator):
    implemented_properties = ("energy", "forces", "point_charge_forces")

    def __init__(self, charge_key="total_charge", charge_result="partial_charges"):
        super().__init__()
        self.total_charge_key = charge_key
        self.expected_charge_key = charge_key
        self.charge_result = charge_result
        self.pcpot = None
        self.calls = []
        self.environments = []

    def embed(self, charges, positions=None):
        self.pcpot = _PointCharges(self, charges)
        self.environments.append(self.pcpot)
        self.results = {}
        if positions is not None:
            self.pcpot.set_positions(positions)
        return self.pcpot

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls.append((self.atoms.copy(), tuple(properties)))
        charge = self.atoms.info[self.expected_charge_key]
        positions, charges = ((), ()) if self.pcpot is None else (self.pcpot.positions, self.pcpot.charges)
        energy, gradient, pc_gradient = _reference(self.atoms.positions, charge, positions, charges)
        partial_charges = np.full(len(atoms), charge / len(atoms))
        self.results = {"energy": energy}
        self.results[self.charge_result] = (
            partial_charges
            if self.charge_result == "partial_charges"
            else np.column_stack([partial_charges, np.zeros((len(atoms), 3))])
        )
        if "forces" in properties:
            self.results["forces"] = -gradient
        if "point_charge_forces" in properties:
            self.results["point_charge_forces"] = -pc_gradient


def _run(theory, **overrides):
    options = {"current_coords": QM_COORDS, "elems": ["H", "H"], "charge": 0, "mult": 1}
    options.update(overrides)
    return theory.run(**options)


def _pc_run(theory, **overrides):
    options = {"pc": True, "current_mm_coords": MM_COORDS, "mm_charges": MM_CHARGES}
    options.update(overrides)
    return _run(theory, **options)


@pytest.mark.parametrize(
    "model_type,module_name,class_name",
    [
        ("fixedpoint", "mace_scf.calculators.fixedpoint_scf", "MACEFixedPointSCF"),
        ("qeq", "mace_scf.calculators.qeq", "MACEQEqCalculator"),
    ],
)
def test_checkpoint_factory_loads_the_selected_optional_calculator(
    monkeypatch, tmp_path, model_type, module_name, class_name
):
    module = ModuleType(module_name)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return _EmbeddedCalculator()

    setattr(module, class_name, factory)
    monkeypatch.setitem(sys.modules, module_name, module)
    model_path = tmp_path / "trained.model"
    model_path.touch()

    theory = MACESCFTheory(
        model_path=model_path,
        model_type=model_type,
        device="cuda",
        default_dtype="float32",
        calculator_kwargs={"pbc_handling": "auto"},
    )

    assert calls == [
        {
            "model_path": str(model_path),
            "device": "cuda",
            "default_dtype": "float32",
            "pbc_handling": "auto",
            **({"require_converged": True} if model_type == "fixedpoint" else {}),
        }
    ]
    assert _run(theory) == pytest.approx(_reference(QM_COORDS)[0] / HARTREE_TO_EV)


def test_optional_dependency_is_only_needed_when_loading_a_checkpoint(monkeypatch):
    original_import = importlib.import_module

    def import_module(name, package=None):
        if name.startswith("mace_scf"):
            raise ModuleNotFoundError("mace_scf is intentionally unavailable")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)

    theory = MACESCFTheory(calculator=_EmbeddedCalculator())
    assert np.isfinite(_run(theory))
    with pytest.raises(MissingDependencyError, match="mace-scf"):
        MACESCFTheory(model_path="trained.model")


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"model_path": "trained.model", "calculator": _EmbeddedCalculator()},
        {"calculator": _EmbeddedCalculator(), "calculator_kwargs": {"device": "cpu"}},
        {"calculator": _EmbeddedCalculator(), "model_type": "unknown"},
        {"calculator": object()},
        {"calculator": _EmbeddedCalculator(), "numcores": 0},
        {"model_path": "trained.model", "calculator_kwargs": {"device": "cuda"}},
    ],
)
def test_invalid_constructor_options_are_rejected(options):
    with pytest.raises(InputError):
        MACESCFTheory(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"use_compile": True},
        {"compiled_evaluator": object()},
        {"scf_options": {"constant_charge": False}},
        {"scf_options": SimpleNamespace(constant_charge=False)},
    ],
)
def test_incompatible_calculator_settings_are_rejected(options):
    calculator = _EmbeddedCalculator()
    calculator.__dict__.update(options)

    with pytest.raises(InputError):
        MACESCFTheory(calculator=calculator)


@pytest.mark.parametrize("pc", [False, True])
def test_ase_energy_and_forces_convert_to_theory_units(pc):
    calculator = _EmbeddedCalculator()
    theory = MACESCFTheory(calculator=calculator)
    run = _pc_run if pc else _run

    result = run(theory, grad=True, charge=-0.5)

    expected = _reference(QM_COORDS, -0.5, MM_COORDS if pc else (), MM_CHARGES if pc else ())
    assert result[0] == pytest.approx(expected[0] / HARTREE_TO_EV, abs=1e-14)
    np.testing.assert_allclose(result[1], expected[1] * BOHR_TO_ANG / HARTREE_TO_EV, atol=1e-14)
    if pc:
        np.testing.assert_allclose(result[2], expected[2] * BOHR_TO_ANG / HARTREE_TO_EV, atol=1e-14)
    else:
        assert len(result) == 2
    assert len(calculator.calls) == 1
    assert not calculator.calls[-1][0].pbc.any()
    np.testing.assert_array_equal(calculator.calls[-1][0].positions, QM_COORDS)


def test_energy_then_gradient_requests_current_point_charge_forces_once():
    calculator = _EmbeddedCalculator()
    theory = MACESCFTheory(calculator=calculator)

    energy = _pc_run(theory)
    assert calculator.calls[-1][1] == ("energy",)
    assert "forces" not in calculator.results
    assert "point_charge_forces" not in calculator.results
    gradient_energy, gradient, pc_gradient = _pc_run(theory, grad=True)

    assert len(calculator.calls) == 2
    assert set(calculator.calls[-1][1]) == {"energy", "forces", "point_charge_forces"}
    assert gradient_energy == pytest.approx(energy)
    assert np.linalg.norm(gradient) > 0
    assert np.linalg.norm(pc_gradient) > 0


def test_changed_point_charge_positions_charges_and_count_are_respected():
    calculator = _EmbeddedCalculator()
    theory = MACESCFTheory(calculator=calculator)
    environments = [
        (MM_COORDS, MM_CHARGES),
        (MM_COORDS + np.array([0.4, -0.2, 0.1]), MM_CHARGES),
        (MM_COORDS, [-0.1, 0.6]),
        (MM_COORDS[:1], [0.7]),
        (np.empty((0, 3)), []),
    ]
    for coords, charges in environments:
        energy, gradient, pc_gradient = _pc_run(theory, current_mm_coords=coords, mm_charges=charges, grad=True)
        expected = _reference(QM_COORDS, 0, coords, charges)
        assert energy == pytest.approx(expected[0] / HARTREE_TO_EV, abs=1e-14)
        np.testing.assert_allclose(gradient, expected[1] * BOHR_TO_ANG / HARTREE_TO_EV, atol=1e-14)
        np.testing.assert_allclose(pc_gradient, expected[2] * BOHR_TO_ANG / HARTREE_TO_EV, atol=1e-14)
    assert len(calculator.calls) == len(environments)


def test_nonembedded_run_clears_both_injected_and_previous_environments():
    calculator = _EmbeddedCalculator()
    calculator.embed(MM_CHARGES, MM_COORDS)
    theory = MACESCFTheory(calculator=calculator)
    vacuum = _reference(QM_COORDS)[0] / HARTREE_TO_EV

    assert _run(theory) == pytest.approx(vacuum)
    assert _pc_run(theory) != pytest.approx(vacuum)
    assert _run(theory) == pytest.approx(vacuum)
    assert len(calculator.pcpot.charges) == 0


@pytest.mark.parametrize("key_source", ["attribute", "keyspec", "keyspec_singlepoint"])
def test_charge_metadata_changes_are_forwarded_even_with_unchanged_coordinates(key_source):
    calculator = _EmbeddedCalculator(charge_key="qm_charge")
    if key_source != "attribute":
        setattr(calculator, key_source, SimpleNamespace(info_keys={"total_charge": "qm_charge"}))
        calculator.total_charge_key = "wrong_fallback"
    theory = MACESCFTheory(calculator=calculator)

    for charge in (0, -1, 0.5):
        energy = _run(theory, charge=charge)
        assert energy == pytest.approx(_reference(QM_COORDS, charge)[0] / HARTREE_TO_EV)
        assert calculator.calls[-1][0].info["qm_charge"] == charge
    assert len(calculator.calls) == 3


def test_qmmm_element_argument_takes_precedence_over_whole_system_elements():
    calculator = _EmbeddedCalculator()
    theory = MACESCFTheory(calculator=calculator)

    _run(theory, elems=["C", "C", "C"], qm_elems=["O", "H"])

    assert calculator.calls[-1][0].get_chemical_symbols() == ["O", "H"]


def test_local_source_calculator_cannot_silently_ignore_qm_charge():
    calculator = _EmbeddedCalculator()
    calculator.keyspec = SimpleNamespace(info_keys={"external_field": "external_field"})
    with pytest.raises(InputError, match="total_charge"):
        MACESCFTheory(calculator=calculator)


@pytest.mark.parametrize(
    "overrides",
    [
        {"current_coords": None},
        {"current_coords": [[0, 0]]},
        {"current_coords": np.full((2, 3), np.nan)},
        {"elems": None},
        {"elems": ["H"]},
        {"charge": None},
        {"charge": np.inf},
        {"mult": None},
        {"mult": 2},
        {"mult": 0},
        {"hessian": True},
        {"pc": True},
        {"pc": True, "current_mm_coords": MM_COORDS, "mm_charges": [1]},
        {"pc": True, "current_mm_coords": MM_COORDS, "mm_charges": [0.1, np.nan]},
        {"pc": True, "current_mm_coords": [[0, 0]], "mm_charges": [0.1]},
    ],
)
def test_invalid_requests_fail_before_calculation(overrides):
    calculator = _EmbeddedCalculator()
    theory = MACESCFTheory(calculator=calculator)

    with pytest.raises(InputError):
        _run(theory, **overrides)

    assert calculator.calls == []


@pytest.mark.parametrize(
    "property_name,value",
    [
        ("energy", np.nan),
        ("energy", [1, 2]),
        ("forces", np.zeros((1, 3))),
        ("forces", np.full((2, 3), np.nan)),
        ("point_charge_forces", np.zeros((1, 3))),
        ("point_charge_forces", np.full((2, 3), np.inf)),
    ],
)
def test_invalid_backend_results_are_rejected(property_name, value):
    calculator = _EmbeddedCalculator()
    original_calculate = calculator.calculate

    def calculate(*args, **kwargs):
        original_calculate(*args, **kwargs)
        calculator.results[property_name] = value

    calculator.calculate = calculate
    theory = MACESCFTheory(calculator=calculator)

    with pytest.raises(InternalError):
        _pc_run(theory, grad=True)


@pytest.mark.parametrize("property_name", ["density_converged", "charge_converged"])
def test_unconverged_backend_results_are_not_returned(property_name):
    calculator = _EmbeddedCalculator()
    original_calculate = calculator.calculate

    def calculate(*args, **kwargs):
        original_calculate(*args, **kwargs)
        calculator.results[property_name] = False

    calculator.calculate = calculate
    theory = MACESCFTheory(calculator=calculator)

    with pytest.raises(InternalError, match="converge"):
        _pc_run(theory, grad=True)
    assert theory.energy is None


@pytest.mark.parametrize("charge_result", ["partial_charges", "density_coefficients"])
def test_atomic_charge_accessor_and_single_point_workflow(charge_result):
    calculator = _EmbeddedCalculator(charge_result=charge_result)
    theory = MACESCFTheory(calculator=calculator)
    fragment = Fragment(elems=["H", "H"], coords=QM_COORDS, charge=-1, mult=1, conncalc=False)

    result = single_point(fragment=fragment, theory=theory, grad=True, result_write_to_disk=False)

    expected = _reference(QM_COORDS, -1)
    assert result.energy == pytest.approx(expected[0] / HARTREE_TO_EV)
    np.testing.assert_allclose(result.gradient, expected[1] * BOHR_TO_ANG / HARTREE_TO_EV)
    np.testing.assert_allclose(theory.get_atomic_charges(), [-0.5, -0.5])


def test_cleanup_discards_embedding_and_population_state_but_retains_the_model():
    calculator = _EmbeddedCalculator()
    theory = MACESCFTheory(calculator=calculator)
    _pc_run(theory, charge=-1, grad=True)

    theory.cleanup()

    assert theory.calculator is calculator
    assert calculator.results == {}
    assert len(calculator.pcpot.charges) == 0
    with pytest.raises(InputError, match="atomic charges"):
        theory.get_atomic_charges()
    assert _run(theory) == pytest.approx(_reference(QM_COORDS)[0] / HARTREE_TO_EV)


def _embedded_qmmm():
    calculator = _EmbeddedCalculator()
    qm_theory = MACESCFTheory(calculator=calculator)
    coords = np.vstack([QM_COORDS[:1], MM_COORDS])
    fragment = Fragment(elems=["H", "H", "H"], coords=coords, charge=0, mult=1, conncalc=False)
    mm_theory = dummy_mm(fragment)
    mm_theory.update_charges([0, 1, 2], [0.0, *MM_CHARGES])
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm_theory,
        mm_theory=mm_theory,
        qmatoms=[0],
        embedding="elstat",
        qm_charge=0,
        qm_mult=1,
        dipole_correction=False,
    )
    return theory, fragment, calculator


def test_qmmm_combined_gradient_matches_energy_differences_on_both_regions():
    theory, fragment, calculator = _embedded_qmmm()
    coords = fragment.coords

    result = single_point(fragment=fragment, theory=theory, grad=True, result_write_to_disk=False)
    gradient = result.gradient.copy()
    numerical = central_difference_gradient(
        lambda positions: theory.run(current_coords=positions, elems=fragment.elems), coords
    )

    np.testing.assert_allclose(gradient, numerical, atol=1e-9, rtol=1e-7)
    assert np.linalg.norm(gradient[1:]) > 0
    assert len(calculator.calls[-1][0]) == 1
    atoms = Atoms(fragment.elems, positions=coords)
    atoms.calc = OpenMMQMMMCalculator(theory)
    np.testing.assert_allclose(atoms.get_forces(), -gradient * units.Hartree / units.Bohr, atol=1e-8)
    assert atoms.get_potential_energy() == pytest.approx(result.energy * units.Hartree)


def test_openmm_python_force_runs_dynamics_and_matches_direct_qmmm_at_moved_geometry():
    theory, fragment, calculator = _embedded_qmmm()
    reference, _reference_fragment, _reference_calculator = _embedded_qmmm()
    export = export_rpmd_potential(theory=theory, num_beads=1)
    integrator = openmm.VerletIntegrator(0.0001)
    context = openmm.Context(export.system, integrator, openmm.Platform.getPlatformByName("Reference"))
    moved_coords = fragment.coords + np.array([[0.1, -0.2, 0.3], [-0.3, 0.2, -0.1], [0.2, 0.1, -0.2]])
    context.setPositions(moved_coords * 0.1)
    context.setVelocities(np.zeros_like(moved_coords))

    integrator.step(3)
    state = context.getState(getEnergy=True, getForces=True, getPositions=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    energy, gradient = reference.run(current_coords=positions, grad=True)

    assert len(calculator.calls) > 1
    assert np.isfinite(positions).all()
    assert np.max(np.abs(positions - moved_coords)) > 1e-8
    assert np.linalg.norm(gradient[0]) > 0
    assert np.linalg.norm(gradient[1:]) > 0
    assert state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) == pytest.approx(
        energy * HARTREE_TO_KJ_PER_MOL, abs=1e-9
    )
    np.testing.assert_allclose(
        state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoules_per_mole / openmm.unit.nanometer),
        -gradient * HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM,
        atol=1e-8,
        rtol=1e-12,
    )
    del context, integrator
