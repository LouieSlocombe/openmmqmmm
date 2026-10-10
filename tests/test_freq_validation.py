"""Frequency-analysis guards: option validation, optional-property bookkeeping, and thermochemistry edge cases."""

import logging

import numpy as np
import pytest
from conftest import make_subregion_qmmm
from test_freq import WATER_COORDS, WATER_FREQUENCIES, SoftModeHessianTheory, _complete_displacement_gradients

import openmmqmmm.freq
from openmmqmmm import Fragment, ZeroTheory, analytic_frequencies, numerical_frequencies
from openmmqmmm.exceptions import InputError, InternalError
from openmmqmmm.freq import (
    _assemble_hessian,
    _get_center,
    _property_mapping_is_complete,
    approximate_full_hessian_from_smaller,
    calc_thermochemistry,
    detect_linear,
)

FREQ_LOGGER = "openmmqmmm.freq"


@pytest.fixture
def water():
    return Fragment(coordsstring=WATER_COORDS, charge=0, mult=1)


def test_both_entry_points_require_a_fragment_and_a_theory(water):
    with pytest.raises(InputError, match="requires a fragment and a theory"):
        analytic_frequencies(fragment=water)
    with pytest.raises(InputError, match="requires a fragment and a theory"):
        numerical_frequencies(theory=ZeroTheory())


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rotmode_threshold": "tight"}, "rotmode_threshold must be a non-negative finite number"),
        ({"masses": 5}, "masses must be a sequence"),
    ],
)
def test_analytic_option_shapes_are_validated(water, options, message):
    with pytest.raises(InputError, match=message):
        analytic_frequencies(fragment=water, theory=SoftModeHessianTheory(np.eye(9)), **options)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"force_projection": "yes"}, "force_projection must be True, False, or None"),
        ({"hessatoms": 5}, "hessatoms must be a sequence"),
    ],
)
def test_numerical_option_shapes_are_validated(water, options, message):
    with pytest.raises(InputError, match=message):
        numerical_frequencies(fragment=water, theory=ZeroTheory(), charge=0, mult=1, **options)


def test_qmmm_numerical_frequencies_need_an_explicit_hessian_region():
    qmmm, fragment = make_subregion_qmmm(qm_charge=0, qm_mult=1)
    with pytest.raises(InputError, match="required for QM/MM"):
        numerical_frequencies(fragment=fragment, theory=qmmm)


def test_only_one_numerical_frequency_job_may_run_per_process(water):
    lock = openmmqmmm.freq._NUMFREQ_PROCESS_LOCK
    assert lock.acquire(blocking=False)
    try:
        with pytest.raises(InputError, match="already active in this process"):
            numerical_frequencies(fragment=water, theory=ZeroTheory(), charge=0, mult=1)
    finally:
        lock.release()


def test_property_mapping_separates_missing_from_malformed_entries():
    labels = ["a", "b", "c", "d"]
    values = {
        "a": np.zeros(3),
        "b": [[1.0, 2.0], [3.0]],
        "c": np.zeros(0),
        "d": np.array(["x", "y", "z"], dtype=object),
    }
    with pytest.raises(InternalError, match=r"missing displacement labels: c; malformed labels: b, d"):
        _property_mapping_is_complete(values, labels, "dipole", (3,))

    with pytest.raises(InternalError, match=r"malformed displacement labels: b, d"):
        _property_mapping_is_complete(values, ["a", "b", "d"], "dipole", (3,))
    assert _property_mapping_is_complete({"a": np.zeros(3)}, ["a"], "dipole", (3,)) is True


def test_missing_displacement_gradients_are_named():
    labels, gradients = _complete_displacement_gradients()
    del gradients[labels[-1]]
    with pytest.raises(InternalError, match=r"missing displacement labels: 0_2_-"):
        _assemble_hessian(
            npoint=2,
            hessatoms=[0],
            displacement_bohr=0.01,
            grads=gradients,
            dipoles={},
            polarizabilities={},
            IR=False,
            Raman=False,
        )


class _IrTheory(SoftModeHessianTheory):
    def __init__(self, ir_intensities):
        super().__init__(np.eye(9))
        self.ir_intensities = ir_intensities


@pytest.mark.parametrize(
    ("analytic_ir", "expected"),
    [([], None), ([1.0, 2.0, 3.0], [0.0] * 6 + [1.0, 2.0, 3.0])],
)
def test_analytic_ir_intensities_are_dropped_when_empty_and_padded_when_short(water, analytic_ir, expected):
    result = analytic_frequencies(fragment=water, theory=_IrTheory(analytic_ir), charge=0, mult=1)
    if expected is None:
        assert result.ir_intensities is None
    else:
        assert list(result.ir_intensities) == expected


class _ZeroPolarizabilityTheory(ZeroTheory):
    def get_polarizability_tensor(self):
        return np.zeros((3, 3))


@pytest.mark.parametrize("theory", [ZeroTheory(), _ZeroPolarizabilityTheory()])
def test_raman_without_polarizabilities_is_skipped_with_a_warning(water, theory, caplog):
    with caplog.at_level(logging.WARNING, logger=FREQ_LOGGER):
        result = numerical_frequencies(fragment=water, theory=theory, charge=0, mult=1, Raman=True)

    if isinstance(theory, _ZeroPolarizabilityTheory):
        assert "No polarizability information found" in caplog.text
        assert not np.any(result.raman_activities)
    else:
        assert "Problem getting polarizability tensor" in caplog.text
        assert result.raman_activities is None
        assert result.depolarization_ratios is None


def test_thermochemistry_can_use_the_hessian_geometry_and_skips_imaginary_modes(water, caplog):
    partial = calc_thermochemistry(
        vfreq=WATER_FREQUENCIES,
        atoms=[1, 2],
        fragment=water,
        multiplicity=None,
        use_full_geo_in_rotational_analysis=False,
    )
    full = calc_thermochemistry(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=2)
    with caplog.at_level(logging.INFO, logger=FREQ_LOGGER):
        saddle = calc_thermochemistry(
            vfreq=[*WATER_FREQUENCIES[:-1], 500j], atoms=[0, 1, 2], fragment=water, multiplicity=1
        )

    assert partial["TS_el"] == 0.0
    assert full["TS_el"] > 0.0
    assert partial["ZPVE"] < full["ZPVE"]
    assert "is imaginary. Skipping in thermochemistry" in caplog.text
    assert saddle["ZPVE"] < full["ZPVE"]


def test_thermochemistry_rejects_an_unknown_qrrho_method(water):
    with pytest.raises(InputError, match="Unknown QRRHO_method"):
        calc_thermochemistry(
            vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, qrrho_method="Ad hoc"
        )


def test_centre_of_mass_needs_masses_or_elements():
    with pytest.raises(InputError, match="masses or elems"):
        _get_center(np.zeros((2, 3)))
    centre = _get_center(np.array([[0.0, 0, 0], [1.0, 0, 0]]), elems=["H", "H"])
    assert centre == pytest.approx((0.5, 0.0, 0.0))


def test_embedding_a_small_hessian_validates_region_and_rest_choice(water):
    with pytest.raises(InputError, match="not all present in large_atomindices"):
        approximate_full_hessian_from_smaller(water, np.eye(3), [0], large_atomindices=[1, 2])
    with pytest.raises(InputError, match="not available in this ORCA\\+OpenMM build"):
        approximate_full_hessian_from_smaller(water, np.eye(3), [0], rest_hessian="xtb")
    unknown = Fragment(coordsstring=WATER_COORDS)
    with pytest.raises(InputError, match="carries neither"):
        approximate_full_hessian_from_smaller(unknown, np.eye(3), [0], rest_hessian="Lindh")


def test_linearity_from_raw_coordinates_treats_one_and_two_atoms_as_linear():
    assert detect_linear(coords=np.zeros((1, 3)), elems=["He"]) is True
    assert detect_linear(coords=np.array([[0.0, 0, 0], [1.0, 0, 0]]), elems=["H", "H"]) is True
    assert detect_linear(coords=np.array([[0.0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]), elems=["C", "O", "O"]) is True
    bent = np.array([[0.0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    assert detect_linear(coords=bent, elems=["O", "H", "H"]) is False
