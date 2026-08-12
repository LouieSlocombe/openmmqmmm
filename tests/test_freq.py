"""Tests for the frequency and thermochemistry engine.

freq.py is the largest untested module in the package (1099 statements, 4% covered),
even though it produces the numbers that end up in papers. The checks here are against
independent physics rather than recorded output: the Sackur-Tetrode equation for the
translational entropy, the harmonic ZPVE sum, and G = H - TS.
"""

import math

import numpy as np
import pytest

from openmmqmmm import Fragment, ZeroTheory, numerical_frequencies
from openmmqmmm.freq import (
    approximate_full_hessian_from_smaller,
    calc_rotational_constants,
    detect_linear,
    read_hessian,
    thermochemcalc,
    write_hessian,
)

HARTREE_TO_KJ_PER_MOL = 2625.4996394799
# 0.5 * h * c in Hartree per cm**-1, the harmonic zero-point energy per unit wavenumber
HALF_HC = 4.5563352812122295e-06

WATER_COORDS = "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n"
# ORCA HF/def2-SVP harmonic frequencies for the geometry above. thermochemcalc takes
# the full 3N list, translations and rotations included.
WATER_FREQUENCIES = [0.0] * 6 + [1790.72, 4113.36, 4212.31]


@pytest.fixture
def water():
    return Fragment(coordsstring=WATER_COORDS, charge=0, mult=1)


def test_translational_entropy_matches_sackur_tetrode():
    """A lone argon atom has only translational entropy, known in closed form.

    This pins the whole translational branch — masses, units and the Eh conversion —
    against physics rather than against a previously recorded number.
    """
    argon = Fragment(coordsstring="Ar 0.0 0.0 0.0\n", charge=0, mult=1)
    result = thermochemcalc(vfreq=[], atoms=[0], fragment=argon, multiplicity=1)

    boltzmann, planck, avogadro = 1.380649e-23, 6.62607015e-34, 6.02214076e23
    gas_constant = boltzmann * avogadro
    mass, temperature, pressure = 39.948e-3 / avogadro, 298.15, 101325.0
    entropy = gas_constant * (
        math.log((2 * math.pi * mass * boltzmann * temperature / planck**2) ** 1.5 * boltzmann * temperature / pressure)
        + 2.5
    )
    expected_ts = entropy * temperature / 1000 / HARTREE_TO_KJ_PER_MOL

    assert result["TS_trans"] == pytest.approx(expected_ts, rel=1e-3)
    assert result["TS_rot"] == 0.0, "A single atom cannot rotate"
    assert result["TS_vib"] == 0.0, "A single atom has no vibrations"
    assert result["ZPVE"] == 0.0


def test_zpve_is_the_harmonic_sum(water):
    """ZPVE must be 0.5*h*c*sum(nu) over the real modes."""
    result = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1)
    assert result["ZPVE"] == pytest.approx(0.5 * sum(WATER_FREQUENCIES) * HALF_HC, rel=1e-6)


def test_thermal_corrections_match_orca(water):
    """Cross-check the thermochemistry against ORCA for the same molecule and modes.

    Reference values are the ones ORCA 6.1.1 printed for this HF/def2-SVP frequency
    run (see orca_outputs/h2o_freq.out). The thermal vibrational correction caught a
    real error: the Bose-Einstein factor was written 1/exp(x-1) instead of
    1/(exp(x)-1), which made this term 2.7x too large.
    """
    # Water is C2v, so sigma = 2. openmmqmmm does not detect point groups and defaults
    # to 1; ORCA used 2, so it has to be supplied here for the comparison to be like for like.
    result = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, symmetry_number=2)

    assert result["vibenergycorr"] == pytest.approx(0.00000144, abs=2e-8)
    assert result["TS_trans"] == pytest.approx(0.01644380, rel=1e-3)
    assert result["TS_rot"] == pytest.approx(0.00496523, rel=1e-3)
    assert result["TS_vib"] == pytest.approx(0.00000161, abs=2e-8)


def test_symmetry_number_lowers_rotational_entropy(water):
    """sigma defaults to 1 and must be supplied for symmetric molecules.

    Getting this wrong is a silent RT*ln(sigma) error in every reaction free energy,
    so the default is pinned here deliberately rather than assumed.
    """
    gas_constant_hartree_per_kelvin = 3.166811563e-6
    default = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1)
    c2v = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, symmetry_number=2)

    assert default["TS_rot"] - c2v["TS_rot"] == pytest.approx(
        gas_constant_hartree_per_kelvin * 298.15 * math.log(2), rel=1e-3
    )


def test_thermal_vibrational_energy_reaches_the_classical_limit(water):
    """As h*nu/kT -> 0 each mode must approach the classical RT of energy.

    The previous formula went to zero instead, so low-frequency modes — the ones
    that dominate the entropy of floppy biomolecular systems — were badly wrong.
    """
    gas_constant_hartree_per_kelvin = 3.166811563e-6
    temperature = 298.15
    # 1 cm-1 corresponds to a vibrational temperature of ~1.44 K, far below 298 K
    soft_modes = [0.0] * 6 + [1.0, 1.0, 1.0]

    result = thermochemcalc(vfreq=soft_modes, atoms=[0, 1, 2], fragment=water, multiplicity=1, temp=temperature)

    classical_limit = 3 * gas_constant_hartree_per_kelvin * temperature
    assert result["E_vib"] == pytest.approx(classical_limit, rel=1e-2)


def test_gibbs_correction_is_enthalpy_minus_entropy_term(water):
    """G = H - TS must hold across the returned corrections."""
    result = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1)
    assert result["Gcorr"] == pytest.approx(result["Hcorr"] - result["TS_tot"], abs=1e-12)
    assert result["TS_tot"] == pytest.approx(
        result["TS_trans"] + result["TS_rot"] + result["TS_vib"] + result["TS_el"], abs=1e-12
    )


def test_entropy_increases_with_temperature(water):
    """Entropy is monotonic in temperature; the ZPVE is not temperature dependent."""
    cold = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, temp=200.0)
    hot = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, temp=400.0)

    assert hot["TS_tot"] > cold["TS_tot"]
    assert hot["ZPVE"] == pytest.approx(cold["ZPVE"])


def test_electronic_entropy_follows_multiplicity(water):
    """A degenerate ground state contributes R*ln(multiplicity)."""
    singlet = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1)
    triplet = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=3)

    assert singlet["TS_el"] == 0.0, "A non-degenerate state has no electronic entropy"
    assert triplet["TS_el"] > 0.0
    assert triplet["TS_el"] / singlet["Hcorr"] < 1.0  # sanity: a small correction, not a dominant term


def test_rotational_constants_of_water(water):
    """Water is an asymmetric top: three distinct constants, near the known values."""
    constants = sorted(calc_rotational_constants(water), reverse=True)

    assert len(constants) == 3
    # Experimental A/B/C are 27.88 / 14.51 / 9.29 cm-1; this geometry is close but not
    # the exact experimental equilibrium one, so allow a few percent.
    assert constants[0] == pytest.approx(27.88, rel=0.05)
    assert constants[1] == pytest.approx(14.51, rel=0.05)
    assert constants[2] == pytest.approx(9.29, rel=0.05)


def test_detect_linear():
    linear = Fragment(coordsstring="C 0.0 0.0 0.0\nO 0.0 0.0 1.13\nO 0.0 0.0 -1.13\n", charge=0, mult=1)
    bent = Fragment(coordsstring=WATER_COORDS, charge=0, mult=1)

    assert detect_linear(fragment=linear) is True
    assert detect_linear(fragment=bent) is False


def test_hessian_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    hessian = rng.random((9, 9))
    hessian = hessian + hessian.T  # Hessians are symmetric

    hessfile = str(tmp_path / "Hessian")
    write_hessian(hessian, hessfile=hessfile)

    assert np.allclose(read_hessian(hessfile), hessian)


def test_approximate_full_hessian_embeds_the_small_one():
    """The computed block must survive into the full Hessian unchanged."""
    fragment = Fragment(coordsstring=WATER_COORDS + "H 0.0 0.0 3.0\n", charge=0, mult=1)
    hessatoms = [0, 1, 2]
    rng = np.random.default_rng(1)
    small = rng.random((9, 9))
    small = small + small.T

    full = approximate_full_hessian_from_smaller(fragment, small, hessatoms)

    assert full.shape == (12, 12), "3N x 3N for the whole fragment"
    assert np.allclose(full[:9, :9], small), "The computed block is kept exactly"
    assert np.allclose(full, full.T), "The result must stay symmetric"


def test_numerical_frequencies_on_a_flat_surface():
    """A zero potential gives a zero Hessian and therefore zero frequencies.

    This exercises the whole displacement/assembly path — geometry displacement,
    gradient collection and mass weighting — without needing a QM program.
    """
    fragment = Fragment(coordsstring=WATER_COORDS, charge=0, mult=1)

    result = numerical_frequencies(fragment=fragment, theory=ZeroTheory())

    assert len(result.frequencies) == 9, "3N frequencies for three atoms"
    assert np.allclose(result.frequencies, 0.0, atol=1e-6), "A flat surface has no curvature"
    assert result.hessian.shape == (9, 9)
    assert np.allclose(result.hessian, 0.0, atol=1e-10)
