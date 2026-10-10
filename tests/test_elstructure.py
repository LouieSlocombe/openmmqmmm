"""CM5 charge corrections, the postg (XDM) driver, and electronic-smearing entropies."""

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from openmmqmmm.elements import cm5_dz, cm5_radii
from openmmqmmm.elstructure import _alpha, calc_cm5, get_ec_entropy, xdm_run
from openmmqmmm.exceptions import ExternalProgramError, InputError


def test_cm5_pair_outside_the_tabulated_set_uses_the_element_dz_difference():
    hydrogen, chlorine = 1, 17
    distance = 1.3
    hirshfeld = np.array([0.2, -0.2])

    charges = calc_cm5([hydrogen, chlorine], [[0, 0, 0], [distance, 0, 0]], hirshfeld)

    bond_term = np.exp(-_alpha * (distance - cm5_radii[hydrogen - 1] - cm5_radii[chlorine - 1]))
    transfer = (cm5_dz[hydrogen - 1] - cm5_dz[chlorine - 1]) * bond_term
    assert charges == pytest.approx(hirshfeld + np.array([transfer, -transfer]))
    assert charges.sum() == pytest.approx(hirshfeld.sum())


POSTG_OUTPUT = """\
postg version 1.0
dispersion energy            -0.0123456789
dispersion forces (Hartree/bohr)
# atom         Fx             Fy             Fz
    1     0.0010000000   -0.0020000000    0.0030000000
    2    -0.0010000000    0.0020000000   -0.0030000000
dispersion force constant matrix
    1 1  0.1 0.2 0.3
"""


@pytest.fixture
def fake_postg(tmp_path):
    bindir = tmp_path / "postg_bin"
    bindir.mkdir()
    postg = bindir / "postg"
    postg.write_text(f'#!/bin/sh\necho "$@" > "{tmp_path}/postg_args"\ncat <<"EOF"\n{POSTG_OUTPUT}EOF\n')
    postg.chmod(postg.stat().st_mode | stat.S_IXUSR)
    return bindir


def test_xdm_run_looks_up_functional_parameters_and_negates_forces_into_a_gradient(fake_postg, tmp_path):
    energy, gradient = xdm_run(wfxfile="mol.wfx", postgdir=str(fake_postg), functional="B3LYP")

    assert energy == pytest.approx(-0.0123456789)
    np.testing.assert_allclose(gradient, [[-0.001, 0.002, -0.003], [0.001, -0.002, 0.003]])
    assert (tmp_path / "postg_args").read_text().split() == ["0.6356", "1.5119", "mol.wfx", "B3LYP"]
    assert "dispersion force constant matrix" in Path("xdm-postg.out").read_text()


def test_xdm_run_finds_postg_on_path_and_prefers_explicit_parameters(fake_postg, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", f"{fake_postg}{os.pathsep}{os.environ['PATH']}")

    xdm_run(wfxfile="mol.wfx", a1=0.5, a2=2.5, functional="pbe")

    assert (tmp_path / "postg_args").read_text().split() == ["0.5", "2.5", "mol.wfx", "pbe"]


def test_xdm_run_without_postg_anywhere_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(ExternalProgramError, match="postg"):
        xdm_run(wfxfile="mol.wfx", functional="pbe")


@pytest.mark.parametrize(
    ("method", "half_filled_term"),
    [
        ("fermi", -np.log(2)),
        ("gaussian", -1 / (2 * np.sqrt(np.pi))),
        ("linear", -0.5 + np.sqrt(2) * 0.5**1.5 * 2 / 3),
    ],
)
def test_smearing_entropy_at_half_filling_matches_the_closed_form(method, half_filled_term):
    sigma = 0.02
    occupations = np.array([2.0, 1.0, 1.0, 0.0])

    entropy = get_ec_entropy(occupations, sigma, method=method)

    assert entropy == pytest.approx(2 * sigma * 2 * half_filled_term)


def test_smearing_entropy_is_symmetric_about_half_filling_and_zero_for_integer_occupations():
    assert get_ec_entropy(np.array([2.0, 0.0]), 0.1) == 0
    assert get_ec_entropy(np.array([0.4]), 0.1) == pytest.approx(get_ec_entropy(np.array([1.6]), 0.1))
    assert get_ec_entropy(np.array([0.4]), 0.1) < 0


def test_unknown_smearing_method_is_rejected():
    with pytest.raises(InputError, match="plateau"):
        get_ec_entropy(np.array([1.0]), 0.1, method="plateau")
