import inspect
import math

import numpy as np
import pytest

import openmmqmmm.freq
from openmmqmmm import Fragment, ZeroTheory, analytic_frequencies, constants, numerical_frequencies
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError
from openmmqmmm.freq import (
    approximate_full_hessian_from_smaller,
    calc_rotational_constants,
    detect_linear,
    read_hessian,
    s_vib_qrrho_grimme,
    s_vib_qrrho_truhlar,
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
    """A lone argon atom has only translational entropy, known in closed form."""
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
    # Water is C2v, so sigma = 2. openmmqmmm does not detect point groups and defaults
    # to 1; ORCA used 2, so it has to be supplied here for the comparison to be like for like.
    result = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, symmetry_number=2)

    assert result["vibenergycorr"] == pytest.approx(0.00000144, abs=2e-8)
    assert result["TS_trans"] == pytest.approx(0.01644380, rel=1e-3)
    assert result["TS_rot"] == pytest.approx(0.00496523, rel=1e-3)
    assert result["TS_vib"] == pytest.approx(0.00000161, abs=2e-8)


def test_symmetry_number_lowers_rotational_entropy(water):
    """Sigma defaults to 1 and must be supplied for symmetric molecules."""
    gas_constant_hartree_per_kelvin = 3.166811563e-6
    default = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1)
    c2v = thermochemcalc(vfreq=WATER_FREQUENCIES, atoms=[0, 1, 2], fragment=water, multiplicity=1, symmetry_number=2)

    assert default["TS_rot"] - c2v["TS_rot"] == pytest.approx(
        gas_constant_hartree_per_kelvin * 298.15 * math.log(2), rel=1e-3
    )


def test_thermal_vibrational_energy_reaches_the_classical_limit(water):
    """As h*nu/kT -> 0 each mode must approach the classical RT of energy."""
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
    fragment = Fragment(coordsstring=WATER_COORDS + "H 0.0 0.0 3.0\n", charge=0, mult=1)
    hessatoms = [0, 1, 2]
    rng = np.random.default_rng(1)
    small = rng.random((9, 9))
    small = small + small.T

    full = approximate_full_hessian_from_smaller(fragment, small, hessatoms)

    assert full.shape == (12, 12), "3N x 3N for the whole fragment"
    assert np.allclose(full[:9, :9], small), "The computed block is kept exactly"
    assert np.allclose(full, full.T), "The result must stay symmetric"


def test_sub_region_hessian_fragment_does_not_inherit_the_whole_system_charge(monkeypatch):
    """A model Hessian over part of a fragment must not be run with the whole fragment's charge."""
    fragment = Fragment(coordsstring=WATER_COORDS + "H 0.0 0.0 3.0\n", charge=-1, mult=1)
    seen = {}

    def fake_model_hessian(subfragment, model="Almloef", *, charge=None, mult=None):
        seen["subfragment_charge"] = subfragment.charge
        seen["charge"] = charge
        return np.zeros((9, 9))

    monkeypatch.setattr(openmmqmmm.freq, "calc_model_hessian_orca", fake_model_hessian)
    rng = np.random.default_rng(1)
    small = rng.random((9, 9))
    small = small + small.T

    approximate_full_hessian_from_smaller(
        fragment, small, [0, 1, 2], large_atomindices=[0, 1, 2], rest_hessian="Almloef", charge=0, mult=1
    )

    assert seen["subfragment_charge"] is None, "The sub-region fragment carries no charge of its own"
    assert seen["charge"] == 0, "The explicitly supplied sub-region charge is what gets used"


def test_sub_region_model_hessian_without_a_charge_is_rejected():
    fragment = Fragment(coordsstring=WATER_COORDS + "H 0.0 0.0 3.0\n", charge=-1, mult=1)
    rng = np.random.default_rng(1)
    small = rng.random((9, 9))
    small = small + small.T

    with pytest.raises(InputError, match="describe the whole system"):
        approximate_full_hessian_from_smaller(
            fragment, small, [0, 1, 2], large_atomindices=[0, 1, 2], rest_hessian="Almloef"
        )


def test_whole_fragment_model_hessian_may_use_the_fragment_charge(monkeypatch):
    """With no sub-region there is no region confusion, so the fragment's charge is correct."""
    fragment = Fragment(coordsstring=WATER_COORDS, charge=-1, mult=2)
    seen = {}

    def fake_model_hessian(subfragment, model="Almloef", *, charge=None, mult=None):
        seen["charge"], seen["mult"] = charge, mult
        return np.zeros((9, 9))

    monkeypatch.setattr(openmmqmmm.freq, "calc_model_hessian_orca", fake_model_hessian)
    rng = np.random.default_rng(1)
    small = rng.random((9, 9))
    small = small + small.T

    approximate_full_hessian_from_smaller(fragment, small, [0, 1, 2], rest_hessian="Almloef")

    assert (seen["charge"], seen["mult"]) == (-1, 2)


def test_model_hessian_does_not_mutate_the_callers_fragment(monkeypatch):
    fragment = Fragment(coordsstring=WATER_COORDS, charge=-1, mult=2)
    monkeypatch.setattr(openmmqmmm.freq, "calc_model_hessian_orca", lambda *_a, **_kw: np.zeros((9, 9)))
    rng = np.random.default_rng(1)
    small = rng.random((9, 9))
    small = small + small.T

    approximate_full_hessian_from_smaller(fragment, small, [0, 1, 2], rest_hessian="Almloef", charge=0, mult=1)

    assert (fragment.charge, fragment.mult) == (-1, 2), "The supplied Hessian charge must not overwrite the fragment"


def test_analytic_frequencies_rejects_a_theory_without_an_analytic_hessian():
    """QMMMTheory and the wrapper theories never define analytic_hessian at all."""
    fragment = Fragment(coordsstring=WATER_COORDS, charge=0, mult=1)

    with pytest.raises(InputError, match="numerical_frequencies"):
        analytic_frequencies(fragment=fragment, theory=ZeroTheory(), charge=0, mult=1)


def test_numerical_frequencies_on_a_flat_surface():
    """A zero potential gives a zero Hessian and therefore zero frequencies."""
    fragment = Fragment(coordsstring=WATER_COORDS, charge=0, mult=1)

    result = numerical_frequencies(fragment=fragment, theory=ZeroTheory())

    assert len(result.frequencies) == 9, "3N frequencies for three atoms"
    assert np.allclose(result.frequencies, 0.0, atol=1e-6), "A flat surface has no curvature"
    assert result.hessian.shape == (9, 9)
    assert np.allclose(result.hessian, 0.0, atol=1e-10)


# Quasi-RRHO vibrational entropy. Both methods take a cut-off frequency, threaded from
# the numerical_frequencies(qrrho_omega_0=...) argument. The Truhlar implementation
# accepted that cut-off and then compared against, and raised to, a hardcoded 100 cm-1:
# the log reported the requested value while the arithmetic used 100.

LOW_MODE_FREQUENCIES = [30.0, 75.0, 150.0, 1600.0, 3700.0]
ROOM_TEMPERATURE = 298.15


def _harmonic_ts_vib(freqs, temperature):
    """T*S_vib for a set of harmonic oscillators, straight from the standard expression."""
    total = 0.0
    R = constants.GAS_CONSTANT_HARTREE_PER_K
    for freq in freqs:
        vibtemp = (freq * constants.LIGHT_SPEED_CM_PER_S * constants.PLANCK_HARTREE_S) / R
        total += temperature * (
            R * (vibtemp / temperature) / (math.exp(vibtemp / temperature) - 1)
            - R * math.log(1 - math.exp(-vibtemp / temperature))
        )
    return total


def test_truhlar_cutoff_below_every_mode_is_plain_harmonic():
    """With nothing to raise, the quasi-harmonic result must be the harmonic one."""
    result = s_vib_qrrho_truhlar(LOW_MODE_FREQUENCIES, ROOM_TEMPERATURE, lowfreq_thresh=1.0)
    assert result == pytest.approx(_harmonic_ts_vib(LOW_MODE_FREQUENCIES, ROOM_TEMPERATURE), rel=1e-12)


@pytest.mark.parametrize("cutoff", [50.0, 100.0, 200.0])
def test_truhlar_raises_low_modes_to_the_requested_cutoff(cutoff):
    """The cut-off argument must be the one used, not a hardcoded 100 cm-1."""
    result = s_vib_qrrho_truhlar(LOW_MODE_FREQUENCIES, ROOM_TEMPERATURE, lowfreq_thresh=cutoff)
    raised = [max(freq, cutoff) for freq in LOW_MODE_FREQUENCIES]

    assert result == pytest.approx(_harmonic_ts_vib(raised, ROOM_TEMPERATURE), rel=1e-12)


def test_truhlar_entropy_decreases_as_the_cutoff_rises():
    """A stiffer mode carries less entropy, and only modes under the cut-off are touched."""
    entropies = [s_vib_qrrho_truhlar(LOW_MODE_FREQUENCIES, ROOM_TEMPERATURE, lowfreq_thresh=c) for c in (50, 100, 200)]

    assert entropies[0] > entropies[1] > entropies[2]


def test_truhlar_default_cutoff_is_the_published_one():
    """Riberio et al. use 100 cm-1; the default must not drift away from the citation."""
    assert inspect.signature(s_vib_qrrho_truhlar).parameters["lowfreq_thresh"].default == 100


def test_grimme_cutoff_changes_the_interpolation():
    """Grimme's omega_0 is the midpoint of the vibration/rotation weighting, not a floor."""
    # Moment of inertia for the free-rotor limit, in SI; any positive value serves here.
    inertia = 1e-44
    entropies = [
        s_vib_qrrho_grimme(LOW_MODE_FREQUENCIES, ROOM_TEMPERATURE, omega_0=c, i_av=inertia) for c in (50, 100, 200)
    ]

    assert len(set(entropies)) == 3, "Each cut-off must give a different interpolation"
    # Per mode the result is a linear blend of a vibrational and a free-rotor entropy with
    # weight w = 1/(1+(omega_0/f)^4), monotonic in omega_0, so the total is monotonic too.
    # Which way it runs is set by which of the two terms is larger, and that depends on the
    # moment of inertia rather than on the cut-off, so only monotonicity is asserted.
    assert entropies == sorted(entropies) or entropies == sorted(entropies, reverse=True)


# The two entry points must agree on which quasi-RRHO scheme runs. numerical_frequencies
# forwarded qrrho_method to thermochemcalc and analytic_frequencies did not, so
# analytic_frequencies(qrrho_method="Truhlar") silently produced Grimme numbers.


class SoftModeHessianTheory:
    """A theory that hands back a fixed Hessian, so analytic_frequencies runs offline."""

    def __init__(self, hessian):
        self.theorytype = "QM"
        self.theorynamelabel = "SoftModeHessian"
        self.analytic_hessian = True
        self.numcores = 1
        self.hessian = hessian

    def set_numcores(self, numcores):
        self.numcores = numcores

    def run(self, current_coords=None, elems=None, charge=None, mult=None, hessian=False, **kwargs):
        return 0.0


@pytest.fixture
def soft_mode_theory(water):
    """A water Hessian softened until its lowest real mode falls under 100 cm-1."""
    numfreq = numerical_frequencies(theory=ZeroTheory(), fragment=water, charge=0, mult=1)
    # ZeroTheory gives a zero Hessian; add a small diagonal so the modes are real but soft.
    return SoftModeHessianTheory(np.eye(numfreq.hessian.shape[0]) * 1e-5)


def test_analytic_frequencies_forwards_the_qrrho_method(water, soft_mode_theory):
    """qrrho_method must reach thermochemcalc, not be dropped on the way."""
    grimme = analytic_frequencies(
        fragment=water, theory=soft_mode_theory, charge=0, mult=1, qrrho_method="Grimme"
    ).thermochemistry
    truhlar = analytic_frequencies(
        fragment=water, theory=soft_mode_theory, charge=0, mult=1, qrrho_method="Truhlar"
    ).thermochemistry

    assert grimme["TS_vib"] != pytest.approx(truhlar["TS_vib"]), (
        "Truhlar and Grimme must give different vibrational entropies for a sub-cut-off mode; "
        "equal values mean qrrho_method never reached thermochemcalc"
    )


class _HarmonicTheory:
    """E = 1/2 sum_i k_i (r_i - r0_i)^2 in bohr, so the exact Hessian is diag(FORCE_CONSTANTS)."""

    theorytype = "QM"
    numcores = 1

    def __init__(self, reference_coords):
        self.reference = np.ravel(np.asarray(reference_coords, float)) * ANG_TO_BOHR

    def run(self, current_coords=None, elems=None, grad=False, charge=None, mult=None, **kwargs):
        """Return the harmonic energy and its analytic gradient in Eh and Eh/bohr."""
        displacement = np.ravel(np.asarray(current_coords, float)) * ANG_TO_BOHR - self.reference
        energy = 0.5 * float(np.sum(FORCE_CONSTANTS * displacement * displacement))
        return energy, (FORCE_CONSTANTS * displacement).reshape(-1, 3)


FORCE_CONSTANTS = np.array([0.4, 0.7, 1.1, 0.3, 0.9, 0.5, 0.6, 0.2, 0.8])
HARMONIC_COORDS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.98], [0.93, 0.0, -0.26]]


def _numerical_hessian(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    fragment = Fragment(elems=["O", "H", "H"], coords=HARMONIC_COORDS, charge=0, mult=1)
    numerical_frequencies(
        fragment=fragment,
        theory=_HarmonicTheory(HARMONIC_COORDS),
        displacement=0.005,
        charge=0,
        mult=1,
        **kwargs,
    )
    return fragment.hessian


@pytest.mark.parametrize("npoint", [1, 2])
def test_numerical_hessian_reproduces_an_analytic_one(tmp_path, monkeypatch, npoint):
    """A linear gradient makes both difference formulas exact, so this pins the whole pipeline."""
    hessian = _numerical_hessian(tmp_path, monkeypatch, npoint=npoint)
    assert hessian == pytest.approx(np.diag(FORCE_CONSTANTS), abs=1e-10)


def test_partial_numerical_hessian_covers_only_the_requested_atoms(tmp_path, monkeypatch):
    hessian = _numerical_hessian(tmp_path, monkeypatch, npoint=2, hessatoms=[1, 2])
    assert hessian.shape == (6, 6)
    assert hessian == pytest.approx(np.diag(FORCE_CONSTANTS[3:]), abs=1e-10)
