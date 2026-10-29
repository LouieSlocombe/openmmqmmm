"""Reject cached truncated-field forces while retaining explicit energy caching."""

import numpy as np
import pytest
from conftest import central_difference_gradient

from openmmqmmm import Fragment, NumGrad, QMMMTheory, numerical_frequencies, optimize_geometry
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError


class _HarmonicFieldQM:
    numcores = 1
    theorytype = "QM"

    def __init__(self):
        self.calls = []

    def run(self, *, current_coords, current_mm_coords, mm_charges, grad=False, **_kwargs):
        qm_coords = np.asarray(current_coords) * ANG_TO_BOHR
        mm_coords = np.asarray(current_mm_coords).reshape(-1, 3) * ANG_TO_BOHR
        difference = qm_coords[:, None, :] - mm_coords[None, :, :]
        weighted = np.asarray(mm_charges)[None, :, None] * difference
        energy = 0.5 * np.sum(difference * weighted)
        self.calls.append((grad, len(mm_coords)))
        if grad:
            return energy, weighted.sum(axis=1), -weighted.sum(axis=0)
        return energy


def _theory(**kwargs):
    fragment = Fragment(elems=["H", "He"], coords=[[0.0, 0, 0], [5.0, 0, 0]], charge=0, mult=1, conncalc=False)
    qm = _HarmonicFieldQM()
    options = {"truncated_pc": True, "truncated_pc_radius": 1.0, "truncated_pc_recalc_iter": 50}
    options.update(kwargs)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        qmatoms=[0],
        charges=[0, 0.25],
        qm_charge=0,
        qm_mult=1,
        **options,
    )
    return theory, fragment, qm


@pytest.mark.parametrize("radius", [0, -1, np.inf, -np.inf, np.nan, True, np.bool_(False), None, "bad"])
def test_invalid_truncation_radius_is_rejected_at_construction(radius):
    with pytest.raises(InputError, match="truncated_pc_radius"):
        _theory(truncated_pc_radius=radius)


@pytest.mark.parametrize("interval", [0, -1, 1.5, 2.0, True, np.bool_(True), None, "1"])
def test_invalid_truncation_interval_is_rejected_at_construction(interval):
    with pytest.raises(InputError, match="truncated_pc_recalc_iter"):
        _theory(truncated_pc_recalc_iter=interval)


def test_cached_gradient_is_rejected_before_runprep_or_qm_work():
    theory, fragment, qm = _theory()
    with pytest.raises(InputError, match="Disable truncated_pc or set truncated_pc_recalc_iter=1"):
        theory.run(current_coords=fragment.coords, grad=True)

    assert qm.calls == []
    assert theory.runcalls == 0
    assert theory.truncated_pc_calls == 0
    assert not hasattr(theory, "pointcharges_original")


def test_energy_cache_remains_available_and_gradient_rejection_does_not_advance_it():
    theory, fragment, qm = _theory()
    reference = theory.run(current_coords=fragment.coords)
    changed = fragment.coords.copy()
    changed[1, 0] += 1
    cached = theory.run(current_coords=changed)
    calls = qm.calls.copy()
    with pytest.raises(InputError, match="Cached truncated-PC corrections"):
        theory.run(current_coords=changed, grad=True)

    fresh_theory, _fragment, _qm = _theory()
    refreshed = fresh_theory.run(current_coords=changed)
    assert cached == pytest.approx(reference)
    assert cached != pytest.approx(refreshed)
    assert theory.truncated_pc_recalc_flag is False
    assert theory.runcalls == 2
    assert theory.truncated_pc_calls == 2
    assert qm.calls == calls


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("truncated_pc_radius", np.nan), ("truncated_pc_recalc_iter", 0), ("truncated_pc_recalc_iter", False)],
)
def test_mutated_invalid_truncation_controls_fail_before_qm_work(attribute, value):
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=1)
    setattr(theory, attribute, value)
    with pytest.raises(InputError, match=attribute):
        theory.run(current_coords=fragment.coords)
    assert qm.calls == []
    assert theory.runcalls == 0


def test_direct_truncation_helper_refuses_cached_gradients_before_changing_state():
    theory, fragment, _qm = _theory()
    with pytest.raises(InputError, match="truncated_pc_recalc_iter=1"):
        theory.truncated_pc_function(fragment.coords[:1], require_gradients=True)
    assert theory.truncated_pc_calls == 0


def test_refresh_every_call_gradients_match_full_field_and_energy_derivatives():
    theory, fragment, _qm = _theory(truncated_pc_recalc_iter=np.int64(1))
    reference, _fragment, _qm = _theory(truncated_pc=False)
    reference_energy, reference_gradient = reference.run(current_coords=fragment.coords, grad=True)
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()

    assert energy == pytest.approx(reference_energy, abs=1e-14)
    assert gradient == pytest.approx(reference_gradient, abs=1e-14)
    numeric = central_difference_gradient(lambda coords: theory.run(current_coords=coords), fragment.coords)
    assert gradient == pytest.approx(numeric, abs=1e-9)


@pytest.mark.parametrize("runmode", ["serial", "parallel"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_numerical_frequencies_reject_cached_gradients_before_workspace_setup(runmode, wrapped, tmp_path):
    theory, fragment, qm = _theory()
    with pytest.raises(InputError, match="truncated_pc_recalc_iter=1"):
        numerical_frequencies(
            theory=NumGrad(theory) if wrapped else theory,
            fragment=fragment,
            hessatoms=[0],
            runmode=runmode,
            IR=False,
            qrrho=False,
        )
    assert qm.calls == []
    assert not (tmp_path / "Numfreq_dir").exists()


@pytest.mark.parametrize("runmode", ["serial", "parallel"])
@pytest.mark.parametrize("depth", [1, 2])
def test_numgrad_rejects_cached_corrections_before_any_displacement(runmode, depth):
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    wrapped = theory
    for _ in range(depth):
        wrapped = NumGrad(wrapped, runmode=runmode)
    with pytest.raises(InputError, match="truncated_pc_recalc_iter=1"):
        wrapped.run(current_coords=fragment.coords, elems=fragment.elems, grad=True)
    assert qm.calls == []
    assert theory.truncated_pc_calls == 0


def test_numgrad_allows_energy_only_caching_and_refresh_every_call_derivatives():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    wrapped = NumGrad(theory)
    energy = wrapped.run(current_coords=fragment.coords, elems=fragment.elems)
    assert np.isfinite(energy)
    assert qm.calls
    theory.truncated_pc_recalc_iter = 1
    kwargs = {"current_coords": fragment.coords, "elems": fragment.elems, "grad": True}
    _, expected = theory.run(**kwargs)
    for _ in range(2):
        _, gradient = wrapped.run(**kwargs)
        assert gradient == pytest.approx(expected, abs=1e-9)
    theory.truncated_pc_recalc_iter = 3
    calls = len(qm.calls)
    with pytest.raises(InputError, match="truncated_pc_recalc_iter=1"):
        wrapped.run(**kwargs)
    assert len(qm.calls) == calls


@pytest.mark.parametrize("num_grad", [False, True])
def test_optimizer_rejects_cached_corrections_before_setup(num_grad, tmp_path):
    theory, fragment, qm = _theory()
    before = set(tmp_path.iterdir())
    with pytest.raises(InputError, match="truncated_pc_recalc_iter=1"):
        optimize_geometry(theory=theory, fragment=fragment, num_grad=num_grad)
    assert qm.calls == []
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("enabled", [np.bool_(True), np.bool_(False)])
def test_numpy_boolean_truncation_runs_consistently(enabled):
    theory, fragment, _qm = _theory(truncated_pc=enabled, truncated_pc_recalc_iter=1)
    reference, _fragment, _qm = _theory(truncated_pc=False)
    assert theory.truncated_pc is bool(enabled)
    assert theory.run(current_coords=fragment.coords) == reference.run(current_coords=fragment.coords)


@pytest.mark.parametrize("enabled", [0, 1, "true", None, [], np.array([True, False])])
def test_invalid_truncation_flag_is_rejected(enabled):
    with pytest.raises(InputError, match="truncated_pc must be a Boolean"):
        _theory(truncated_pc=enabled)
    theory, fragment, qm = _theory()
    theory.truncated_pc = enabled
    with pytest.raises(InputError, match="truncated_pc must be a Boolean"):
        theory.run(current_coords=fragment.coords)
    assert qm.calls == []
    assert theory.runcalls == 0


def test_disable_and_reenable_truncation_restores_full_field_and_refreshes():
    theory, fragment, qm = _theory()
    reference, _fragment, _qm = _theory(truncated_pc=False)
    theory.run(current_coords=fragment.coords)
    theory.truncated_pc = False
    changed = fragment.coords.copy()
    changed[1, 0] += 1
    energy, gradient = theory.run(current_coords=changed, grad=True)
    expected_energy, expected_gradient = reference.run(current_coords=changed, grad=True)
    assert energy == expected_energy
    assert gradient == pytest.approx(expected_gradient)

    theory.truncated_pc = True
    assert theory.run(current_coords=changed) == expected_energy
    assert qm.calls[-2:] == [(False, 0), (False, 1)]


@pytest.mark.parametrize("state", [{"charge": 1, "mult": 1}, {"charge": 0, "mult": 3}])
def test_changed_electronic_state_refreshes_correction(state, monkeypatch):
    theory, fragment, qm = _theory()
    theory.qm_charge = theory.qm_mult = None
    original_run = qm.run

    def state_dependent_run(*, charge, mult, **kwargs):
        return (charge + mult) * original_run(**kwargs)

    monkeypatch.setattr(qm, "run", state_dependent_run)
    initial = theory.run(current_coords=fragment.coords, charge=0, mult=1)
    energy = theory.run(current_coords=fragment.coords, **state)
    assert energy == pytest.approx((state["charge"] + state["mult"]) * initial)
    assert len(qm.calls) == 4


@pytest.mark.parametrize("control", ["radius", "interval", "backend"])
def test_changed_truncation_configuration_refreshes_correction(control):
    theory, fragment, qm = _theory()
    theory.run(current_coords=fragment.coords)
    changed = fragment.coords.copy()
    changed[1, 0] += 1
    if control == "radius":
        theory.truncated_pc_radius = 10.0
    elif control == "interval":
        theory.truncated_pc_recalc_iter = 10
    else:
        qm = theory.qm_theory = _HarmonicFieldQM()
    reference, _fragment, _qm = _theory(truncated_pc=False)
    before = len(qm.calls)
    assert theory.run(current_coords=changed) == reference.run(current_coords=changed)
    assert theory.truncated_pc_recalc_flag is True
    assert qm.calls[before:] == ([(False, 1)] if control == "radius" else [(False, 0), (False, 1)])


@pytest.mark.parametrize("failed_call", [1, 2])
def test_failed_refresh_is_recalculated_before_reusing_correction(failed_call, monkeypatch):
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    theory.run(current_coords=fragment.coords)
    theory.run(current_coords=fragment.coords)
    changed = fragment.coords.copy()
    changed[1, 0] = 0.5  # The scheduled refresh now selects the previously omitted site.
    original_run = qm.run
    attempts = 0

    def fail_during_refresh(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == failed_call:
            raise RuntimeError("simulated SCF failure")
        return original_run(**kwargs)

    monkeypatch.setattr(qm, "run", fail_during_refresh)
    with pytest.raises(RuntimeError, match="simulated SCF failure"):
        theory.run(current_coords=changed)
    monkeypatch.setattr(qm, "run", original_run)

    reference, _fragment, _qm = _theory(truncated_pc=False)
    before = len(qm.calls)
    assert theory.run(current_coords=changed) == reference.run(current_coords=changed)
    assert theory.truncated_pc_recalc_flag is True
    assert qm.calls[before:] == [(False, 1)]


def test_failed_cached_evaluation_forces_full_refresh_on_retry(monkeypatch):
    theory, fragment, qm = _theory()
    theory.run(current_coords=fragment.coords)
    changed = fragment.coords.copy()
    changed[1, 0] += 1
    original_run = qm.run

    def fail_cached_call(**_kwargs):
        raise RuntimeError("simulated SCF failure")

    monkeypatch.setattr(qm, "run", fail_cached_call)
    with pytest.raises(RuntimeError, match="simulated SCF failure"):
        theory.run(current_coords=changed)
    monkeypatch.setattr(qm, "run", original_run)

    reference, _fragment, _qm = _theory(truncated_pc=False)
    assert theory.run(current_coords=changed) == reference.run(current_coords=changed)
    assert qm.calls[-2:] == [(False, 0), (False, 1)]


@pytest.mark.parametrize("grad", [False, True])
def test_refresh_returns_full_backend_result_without_cancellation(grad, monkeypatch):
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=1)

    def disparate_fields(*, current_coords, current_mm_coords, grad, **_kwargs):
        value = 1.0 if len(current_mm_coords) else 1e20
        if grad:
            return value, np.full_like(current_coords, value), np.full_like(current_mm_coords, value)
        return value

    monkeypatch.setattr(qm, "run", disparate_fields)
    result = theory.run(current_coords=fragment.coords, grad=grad)
    if grad:
        energy, gradient = result
        assert energy == 1.0
        assert gradient == pytest.approx(np.ones_like(fragment.coords))
    else:
        assert result == 1.0
