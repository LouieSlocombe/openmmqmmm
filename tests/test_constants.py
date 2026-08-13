"""Guards against the stale/inconsistent-constant class of bug.

Every value in openmmqmmm.constants is derived from scipy.constants, so these check the
derivations rather than the arithmetic: internal identities that must hold exactly, and
spot-checks against published CODATA at a tolerance loose enough to survive a CODATA
revision but tight enough to catch a typo.
"""

import math

import pytest

from openmmqmmm import constants


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (constants.ANG_TO_BOHR * constants.BOHR_TO_ANG, 1.0),
        (constants.BOHR_TO_NM * 10, constants.BOHR_TO_ANG),
        (constants.BOHR_TO_M * 1e10, constants.BOHR_TO_ANG),
        (constants.ANG_TO_M * 1e10, 1.0),
        (constants.HARTREE_TO_KCAL_PER_MOL * constants.KCAL_TO_KJ, constants.HARTREE_TO_KJ_PER_MOL),
        (
            constants.GAS_CONSTANT_HARTREE_PER_K * constants.HARTREE_TO_KCAL_PER_MOL,
            constants.GAS_CONSTANT_KCAL_PER_MOL_K,
        ),
        (constants.GAS_CONSTANT_HARTREE_PER_K * constants.HARTREE_TO_J, constants.BOLTZMANN_J_PER_K),
        (constants.PLANCK_HARTREE_S * constants.HARTREE_TO_J, constants.PLANCK_J_S),
        (constants.GHZ_TO_WAVENUMBER * constants.LIGHT_SPEED_CM_PER_S, 1e9),
        # The zero-point factor is half a wavenumber expressed in Hartree.
        (2 * constants.HALF_HC_HARTREE_PER_WAVENUMBER * constants.HARTREE_TO_WAVENUMBER, 1.0),
        (constants.HC_J_CM * constants.HARTREE_TO_WAVENUMBER, constants.HARTREE_TO_J),
    ],
)
def test_conversions_are_mutually_consistent(lhs, rhs):
    assert lhs == pytest.approx(rhs, rel=1e-12)


@pytest.mark.parametrize(
    ("name", "codata"),
    [
        ("PLANCK_J_S", 6.62607015e-34),
        ("BOLTZMANN_J_PER_K", 1.380649e-23),
        ("LIGHT_SPEED_CM_PER_S", 2.99792458e10),
        ("HARTREE_TO_J", 4.3597447222060e-18),
        ("HARTREE_TO_KJ_PER_MOL", 2625.4996394798),
        ("HARTREE_TO_KCAL_PER_MOL", 627.5094740631),
        ("HARTREE_TO_EV", 27.211386245981),
        ("HARTREE_TO_WAVENUMBER", 219474.6313632),
        ("BOHR_TO_ANG", 0.529177210544),
        ("AMU_TO_KG", 1.66053906892e-27),
        ("GAS_CONSTANT_HARTREE_PER_K", 3.166811563e-06),
        ("GAS_CONSTANT_KCAL_PER_MOL_K", 1.9872042586e-03),
        # The three factors that used to be undocumented literals inside freq.py.
        ("ROT_CONSTANT_GHZ_AMU_ANG2", 505.3790084),
        ("TRANS_PARTITION_PREFACTOR", 0.02560748669),
        ("IR_INTENSITY_AU_TO_KM_PER_MOL", 974.8801098),
    ],
)
def test_values_match_published_references(name, codata):
    assert getattr(constants, name) == pytest.approx(codata, rel=1e-8)


def test_rotational_constant_factor_reproduces_the_closed_form():
    """Cross-check against the closed form rather than against the stored number."""
    inertia_amu_ang2 = 3.5
    inertia_si = inertia_amu_ang2 * constants.AMU_TO_KG * constants.ANG_TO_M**2
    expected_ghz = constants.PLANCK_J_S / (8 * math.pi**2 * inertia_si) / 1e9
    assert constants.ROT_CONSTANT_GHZ_AMU_ANG2 / inertia_amu_ang2 == pytest.approx(expected_ghz, rel=1e-12)


def test_energy_unit_table_covers_every_documented_unit():
    """reaction_energy(unit=...) and single_point_fragments(unit=...) accept these names."""
    table = constants.ENERGY_UNIT_FROM_HARTREE
    assert set(table) == {"Eh", "mEh", "eV", "meV", "kcal/mol", "kcalpermol", "kJ/mol", "kJpermol", "cm-1"}
    assert table["Eh"] == 1.0
    assert table["mEh"] == 1000.0
    assert table["kcal/mol"] == table["kcalpermol"]
    assert table["kJ/mol"] == table["kJpermol"]
    assert table["meV"] == pytest.approx(table["eV"] * 1000)
