"""Measure cached-field errors and require explicit opt-in to approximate energies."""

import numpy as np
import pytest

from openmmqmmm import Fragment, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError


class _ChargeWeightedQM:
    """A deterministic field whose exact energy can be evaluated independently."""

    numcores = 1
    theorytype = "QM"

    def __init__(self):
        self.calls = []

    def run(self, *, current_coords, current_mm_coords, mm_charges, grad=False, **_kwargs):
        qm_coords = np.asarray(current_coords)
        mm_coords = np.asarray(current_mm_coords).reshape(-1, 3)
        charges = np.asarray(mm_charges)
        self.calls.append((grad, charges.copy(), mm_coords.copy()))
        displacement = mm_coords[:, 0] - qm_coords[0, 0]
        energy = float(np.sum(charges * displacement**2))
        if not grad:
            return energy
        pc_gradient = np.zeros_like(mm_coords)
        pc_gradient[:, 0] = 2 * charges * displacement / ANG_TO_BOHR
        qm_gradient = np.zeros_like(qm_coords)
        qm_gradient[0] = -pc_gradient.sum(axis=0)
        return energy, qm_gradient, pc_gradient


def _theory(**kwargs):
    fragment = Fragment(
        elems=["H", "He", "Ne"],
        coords=[[0, 0, 0], [1, 0, 0], [5, 0, 0]],
        charge=0,
        mult=1,
        conncalc=False,
    )
    qm = _ChargeWeightedQM()
    options = {"truncated_pc": True, "truncated_pc_radius": 2.0}
    options.update(kwargs)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        qmatoms=[0],
        charges=[0, 1, 0.5],
        qm_charge=0,
        qm_mult=1,
        **options,
    )
    return theory, fragment, qm


def _change_selection(fragment):
    coords = fragment.coords.copy()
    coords[1, 0], coords[2, 0] = 3, 1
    return coords


def _reach_changed_selection_refresh(theory, fragment):
    assert theory.run(current_coords=fragment.coords) == pytest.approx(13.5)
    assert theory.run(current_coords=fragment.coords) == pytest.approx(13.5)
    return theory.run(current_coords=_change_selection(fragment))


@pytest.mark.parametrize("grad", [False, True])
def test_default_truncation_uses_one_full_calculation_per_evaluation(grad):
    theory, fragment, qm = _theory()
    assert theory.truncated_pc_recalc_iter == 1
    result = theory.run(current_coords=fragment.coords, grad=grad)
    energy = result[0] if grad else result
    assert energy == pytest.approx(13.5)
    assert len(qm.calls) == 1
    assert qm.calls[0][0] is grad
    assert qm.calls[0][1] == pytest.approx([1, 0.5])
    assert theory.truncated_pc_diagnostics["last_refresh_error"] is None
    assert theory.truncated_pc_diagnostics["fallback"] is False


def test_first_refresh_has_no_measured_cached_error_and_reports_charge_counts():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    assert theory.run(current_coords=fragment.coords) == pytest.approx(13.5)
    diagnostics = theory.truncated_pc_diagnostics
    assert diagnostics["last_refresh_error"] is None
    assert diagnostics["selected_count"] == 1
    assert diagnostics["full_count"] == 2
    assert diagnostics["selected_charge"] == pytest.approx(1)
    assert diagnostics["full_charge"] == pytest.approx(1.5)
    assert diagnostics["fallback"] is False
    assert qm.calls[-1][1] == pytest.approx([1, 0.5])


def test_complete_selection_uses_one_full_call_even_in_cached_mode():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3, truncated_pc_radius=10.0)
    for shift in (0, 0.5, 1):
        coords = fragment.coords.astype(float)
        coords[2, 0] += shift
        before = len(qm.calls)
        assert theory.run(current_coords=coords) == pytest.approx(1 + 0.5 * (5 + shift) ** 2)
        assert len(qm.calls) - before == 1
        assert theory.truncated_pc_diagnostics["mode"] == "full"
    assert theory.truncated_pc_diagnostics["last_refresh_error"] == 0.0


def test_refresh_measures_old_cached_potential_even_when_selection_changes():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    assert _reach_changed_selection_refresh(theory, fragment) == pytest.approx(9.5)

    # At the refresh geometry the old approximation is 9 + 12.5 = 21.5 Eh;
    # the full result is 9.5 Eh. Comparing corrections alone gives 3.5, not 12.
    diagnostics = theory.truncated_pc_diagnostics
    assert diagnostics["last_refresh_error"] == pytest.approx(12)
    assert diagnostics["selected_count"] == 1
    assert diagnostics["selected_charge"] == pytest.approx(0.5)
    assert diagnostics["fallback"] is False
    old_subset, new_subset, full_field = qm.calls[-3:]
    assert old_subset[1] == pytest.approx([1])
    assert new_subset[1] == pytest.approx([0.5])
    assert full_field[1] == pytest.approx([1, 0.5])


def test_same_selection_refresh_reuses_subset_evaluation_for_error_measurement():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    theory.run(current_coords=fragment.coords)
    theory.run(current_coords=fragment.coords)
    changed = fragment.coords.copy()
    changed[2, 0] = 6
    before = len(qm.calls)
    assert theory.run(current_coords=changed) == pytest.approx(19)
    assert len(qm.calls) - before == 2
    assert theory.truncated_pc_diagnostics["last_refresh_error"] == pytest.approx(-5.5)
    assert qm.calls[-1][1] == pytest.approx([1, 0.5])


def test_exceeded_error_tolerance_latches_full_field_until_explicit_reset():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3, truncated_pc_error_tolerance=1.0)
    assert _reach_changed_selection_refresh(theory, fragment) == pytest.approx(9.5)
    assert theory.truncated_pc_diagnostics["fallback"] is True
    assert theory.truncated_pc_diagnostics["last_refresh_error"] == pytest.approx(12)
    assert theory.truncated_pc is True
    assert theory.truncated_pc_recalc_iter == 3

    changed = fragment.coords.copy()
    changed[1, 0], changed[2, 0] = 4, 2
    before = len(qm.calls)
    assert theory.run(current_coords=changed) == pytest.approx(18)
    assert len(qm.calls) - before == 1
    assert qm.calls[-1][1] == pytest.approx([1, 0.5])
    assert theory.truncated_pc_diagnostics["fallback"] is True

    theory.reset_truncated_pc_cache()
    before = len(qm.calls)
    assert theory.run(current_coords=fragment.coords) == pytest.approx(13.5)
    assert len(qm.calls) - before == 2
    assert theory.truncated_pc_diagnostics["fallback"] is False
    assert theory.truncated_pc_diagnostics["last_refresh_error"] is None


def test_configuration_change_clears_full_field_fallback():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3, truncated_pc_error_tolerance=1.0)
    _reach_changed_selection_refresh(theory, fragment)
    assert theory.truncated_pc_diagnostics["fallback"] is True
    theory.truncated_pc_error_tolerance = 100.0
    before = len(qm.calls)
    assert theory.run(current_coords=fragment.coords) == pytest.approx(13.5)
    assert len(qm.calls) - before == 2
    assert theory.truncated_pc_diagnostics["fallback"] is False
    assert theory.truncated_pc_diagnostics["last_refresh_error"] is None


@pytest.mark.parametrize("tolerance", [0, -1, np.nan, np.inf, -np.inf, True, np.bool_(False), "bad"])
def test_invalid_error_tolerance_is_rejected_before_backend_work(tolerance):
    with pytest.raises(InputError, match="truncated_pc_error_tolerance"):
        _theory(truncated_pc_error_tolerance=tolerance)
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    theory.truncated_pc_error_tolerance = tolerance
    with pytest.raises(InputError, match="truncated_pc_error_tolerance"):
        theory.run(current_coords=fragment.coords)
    assert qm.calls == []


def test_missing_tolerance_retains_explicitly_requested_cached_approximation():
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3)
    _reach_changed_selection_refresh(theory, fragment)
    changed = _change_selection(fragment)
    changed[1, 0] = 4
    before = len(qm.calls)
    assert theory.run(current_coords=changed) == pytest.approx(9.5)
    assert len(qm.calls) - before == 1
    assert qm.calls[-1][1] == pytest.approx([0.5])
    assert theory.truncated_pc_diagnostics["fallback"] is False
    assert theory.truncated_pc_diagnostics["last_refresh_error"] == pytest.approx(12)


@pytest.mark.parametrize("failed_call", [1, 2, 3])
def test_failed_changed_selection_refresh_does_not_publish_partial_diagnostics(failed_call, monkeypatch):
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3, truncated_pc_error_tolerance=1.0)
    theory.run(current_coords=fragment.coords)
    theory.run(current_coords=fragment.coords)
    previous_diagnostics = theory.truncated_pc_diagnostics.copy()
    original_run = qm.run
    attempts = 0

    def fail_refresh(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == failed_call:
            raise RuntimeError("simulated SCF failure")
        return original_run(**kwargs)

    monkeypatch.setattr(qm, "run", fail_refresh)
    with pytest.raises(RuntimeError, match="simulated SCF failure"):
        theory.run(current_coords=_change_selection(fragment))
    assert theory.truncated_pc_diagnostics == previous_diagnostics
    assert theory.truncated_pc_diagnostics["last_refresh_error"] is None
    assert theory.truncated_pc_diagnostics["fallback"] is False

    monkeypatch.setattr(qm, "run", original_run)
    before = len(qm.calls)
    assert theory.run(current_coords=_change_selection(fragment)) == pytest.approx(9.5)
    assert len(qm.calls) - before == 2
    assert qm.calls[-1][1] == pytest.approx([1, 0.5])
    assert theory.truncated_pc_diagnostics["last_refresh_error"] is None
    assert theory.truncated_pc_diagnostics["fallback"] is False


def test_failed_full_only_evaluation_preserves_error_triggered_fallback(monkeypatch):
    theory, fragment, qm = _theory(truncated_pc_recalc_iter=3, truncated_pc_error_tolerance=1.0)
    _reach_changed_selection_refresh(theory, fragment)
    previous_diagnostics = theory.truncated_pc_diagnostics.copy()
    assert previous_diagnostics["fallback"] is True
    original_run = qm.run

    def fail_full_call(**_kwargs):
        raise RuntimeError("simulated SCF failure")

    monkeypatch.setattr(qm, "run", fail_full_call)
    with pytest.raises(RuntimeError, match="simulated SCF failure"):
        theory.run(current_coords=fragment.coords)
    assert theory.truncated_pc_diagnostics == previous_diagnostics

    monkeypatch.setattr(qm, "run", original_run)
    changed = fragment.coords.copy()
    changed[1, 0], changed[2, 0] = 4, 2
    before = len(qm.calls)
    assert theory.run(current_coords=changed) == pytest.approx(18)
    assert len(qm.calls) - before == 1
    assert qm.calls[-1][1] == pytest.approx([1, 0.5])
    assert theory.truncated_pc_diagnostics["fallback"] is True
    assert theory.truncated_pc_diagnostics["last_refresh_error"] == pytest.approx(12)
