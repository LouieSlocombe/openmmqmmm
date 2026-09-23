"""Reject cached truncated-field forces while retaining explicit energy caching."""

import numpy as np
import pytest

from openmmqmmm import Fragment, QMMMTheory, numerical_frequencies
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
    options = {"truncated_pc": True, "truncated_pc_radius": 1.0}
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
    step = 1e-5
    for atom in range(fragment.numatoms):
        for axis in range(3):
            plus, minus = fragment.coords.copy(), fragment.coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numeric = (theory.run(current_coords=plus) - theory.run(current_coords=minus)) / (2 * step * ANG_TO_BOHR)
            assert gradient[atom, axis] == pytest.approx(numeric, abs=1e-9)


@pytest.mark.parametrize("runmode", ["serial", "parallel"])
def test_numerical_frequencies_reject_cached_gradients_before_workspace_setup(runmode, tmp_path):
    theory, fragment, qm = _theory()
    with pytest.raises(InputError, match="truncated_pc_recalc_iter=1"):
        numerical_frequencies(
            theory=theory,
            fragment=fragment,
            hessatoms=[0],
            runmode=runmode,
            IR=False,
            qrrho=False,
        )
    assert qm.calls == []
    assert not (tmp_path / "Numfreq_dir").exists()
