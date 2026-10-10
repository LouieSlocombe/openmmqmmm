"""Constraint, region, gradient-check and nuclear-repulsion helpers in openmmqmmm.coords."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from openmmqmmm import Fragment, check_gradient_for_bad_atoms
from openmmqmmm.constants import BOHR_TO_ANG
from openmmqmmm.coords import (
    _actindex_to_fullindex,
    _get_molecule_members_np,
    define_xh_constraints,
    expand_qm_pc_region,
    expand_qm_region,
    fullindex_to_actindex,
    get_connected_atoms,
    get_water_constraints,
    list_of_masses,
    nuc_nuc_repulsion,
    simple_get_water_constraints,
    threshold_conn,
    total_nuclear_charge,
)
from openmmqmmm.elements import eldict_covrad
from openmmqmmm.exceptions import InputError, InternalError

WATER_ELEMS = ["O", "H", "H"]
WATER_COORDS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
# Tetrahedral methanol: C-H 1.09 A, C-O 1.44 A, O-H 0.97 A; every non-bonded pair is above 1.7 A
METHANOL_ELEMS = ["C", "H", "H", "H", "O", "H"]
METHANOL_COORDS = [
    [0.0, 0.0, 0.0],
    [0.63, 0.63, 0.63],
    [-0.63, -0.63, 0.63],
    [0.63, -0.63, -0.63],
    [-0.83, 0.83, -0.83],
    [-1.39, 1.39, -1.39],
]


def _water_methanol():
    methanol = [[x + 5.0, y, z] for x, y, z in METHANOL_COORDS]
    return Fragment(coords=WATER_COORDS + methanol, elems=WATER_ELEMS + METHANOL_ELEMS)


def _methanol_two_waters():
    waters = [[x + 5.0, y, z] for x, y, z in WATER_COORDS] + [[x, y + 5.0, z] for x, y, z in WATER_COORDS]
    return Fragment(coords=METHANOL_COORDS + waters, elems=METHANOL_ELEMS + WATER_ELEMS * 2)


def test_define_xh_constraints_pairs_each_hydrogen_with_its_heavy_atom():
    assert define_xh_constraints(Fragment(coords=METHANOL_COORDS, elems=METHANOL_ELEMS)) == [
        [0, 1],
        [0, 2],
        [0, 3],
        [4, 5],
    ]


def test_define_xh_constraints_reports_full_system_indices_for_an_active_region():
    fragment = _water_methanol()

    assert define_xh_constraints(fragment) == [[0, 1], [0, 2], [3, 4], [3, 5], [3, 6], [7, 8]]
    assert define_xh_constraints(fragment, actatoms=[3, 4, 5, 6, 7, 8]) == [[3, 4], [3, 5], [3, 6], [7, 8]]
    assert define_xh_constraints(fragment, actatoms=[7, 8, 3, 4]) == [[7, 8], [3, 4]]


def test_define_xh_constraints_drops_excluded_hydrogens():
    fragment = _water_methanol()

    assert define_xh_constraints(fragment, excludeatoms=[1, 8]) == [[0, 2], [3, 4], [3, 5], [3, 6]]
    assert define_xh_constraints(fragment, actatoms=[3, 4, 5, 6, 7, 8], excludeatoms=[4, 5, 6]) == [[7, 8]]


def test_define_xh_constraints_rejects_a_hydrogen_without_a_partner():
    fragment = Fragment(coords=[*WATER_COORDS, [10.0, 10.0, 10.0]], elems=[*WATER_ELEMS, "H"])

    with pytest.raises(InternalError, match=r"XHpair is strange: \[3\]"):
        define_xh_constraints(fragment)


def _mm_object(resnames, elements):
    return SimpleNamespace(resnames=resnames, mm_elements=elements)


METHANOL_TWO_WATERS_RESNAMES = ["MOH"] * 6 + ["HOH"] * 3 + ["WAT"] * 3
METHANOL_TWO_WATERS_ELEMS = METHANOL_ELEMS + WATER_ELEMS * 2


@pytest.mark.parametrize("watermodel", ["tip3p", "spc"])
def test_get_water_constraints_builds_three_constraints_per_selected_water(watermodel):
    mm = _mm_object(METHANOL_TWO_WATERS_RESNAMES, METHANOL_TWO_WATERS_ELEMS)

    constraints = get_water_constraints(openmmtheoryobject=mm, atomlist=list(range(12)), watermodel=watermodel)

    assert constraints == [[6, 7], [6, 8], [7, 8], [9, 10], [9, 11], [10, 11]]


def test_get_water_constraints_keys_on_the_oxygen_being_in_atomlist():
    mm = _mm_object(METHANOL_TWO_WATERS_RESNAMES, METHANOL_TWO_WATERS_ELEMS)

    assert get_water_constraints(openmmtheoryobject=mm, atomlist=[9]) == [[9, 10], [9, 11], [10, 11]]
    assert get_water_constraints(openmmtheoryobject=mm, atomlist=[7, 8, 10, 11]) == []
    assert get_water_constraints(openmmtheoryobject=mm, atomlist=[0, 4]) == []


def test_get_water_constraints_accepts_the_tip_residue_name():
    mm = _mm_object(["TIP"] * 3, WATER_ELEMS)

    assert get_water_constraints(openmmtheoryobject=mm, atomlist=[0, 1, 2]) == [[0, 1], [0, 2], [1, 2]]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"atomlist": [0]}, "requires openmmtheoryobject and atomlist"),
        ({"openmmtheoryobject": _mm_object(["HOH"] * 3, WATER_ELEMS)}, "requires openmmtheoryobject and atomlist"),
        (
            {"openmmtheoryobject": _mm_object(["HOH"] * 3, WATER_ELEMS), "atomlist": [0], "watermodel": "tip4p"},
            "unknown",
        ),
        ({"openmmtheoryobject": _mm_object([], WATER_ELEMS), "atomlist": [0]}, "No resnames found"),
        ({"openmmtheoryobject": _mm_object(["HOH"] * 3, []), "atomlist": [0]}, "No mm_elements found"),
    ],
)
def test_get_water_constraints_rejects_incomplete_input(kwargs, message):
    with pytest.raises(InputError, match=message):
        get_water_constraints(**kwargs)


def test_simple_get_water_constraints_walks_oxygens_from_the_starting_index():
    fragment = _methanol_two_waters()

    assert simple_get_water_constraints(fragment, starting_index=6) == [
        [6, 7],
        [6, 8],
        [7, 8],
        [9, 10],
        [9, 11],
        [10, 11],
    ]
    assert simple_get_water_constraints(fragment, starting_index=6, onlyHH=True) == [[7, 8], [10, 11]]
    assert simple_get_water_constraints(fragment, starting_index=9) == [[9, 10], [9, 11], [10, 11]]


def test_simple_get_water_constraints_rejects_a_missing_or_non_oxygen_start():
    fragment = _methanol_two_waters()

    with pytest.raises(InputError, match="must provide a starting_index"):
        simple_get_water_constraints(fragment)
    with pytest.raises(InputError, match=r"(?s)not oxygen.*starting index \(7\)"):
        simple_get_water_constraints(fragment, starting_index=7)


def test_index_conversion_between_active_region_and_full_system():
    actatoms = [5, 9, 13]

    assert fullindex_to_actindex(9, actatoms) == 1
    assert _actindex_to_fullindex(2, actatoms) == 13
    assert [_actindex_to_fullindex(fullindex_to_actindex(i, actatoms), actatoms) for i in actatoms] == actatoms


def test_check_gradient_for_bad_atoms_lists_atoms_over_the_threshold(caplog):
    fragment = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS)
    gradient = np.array([[0.0, 0.0, 0.0], [50000.0, 0.0, 0.0], [0.0, -46000.0, 0.0]])

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        bad = check_gradient_for_bad_atoms(fragment=fragment, gradient=gradient, threshold=45000)

    assert bad == [1, 2]
    assert "abnormally high values" in caplog.text
    rows = [message for message in caplog.messages if message.strip().startswith(("1 ", "2 "))]
    assert len(rows) == 2
    assert rows[0].split()[:5] == ["1", "H", "0.960000", "0.000000", "0.000000"]
    assert rows[1].split()[1] == "H"
    assert float(rows[1].split()[6]) == pytest.approx(-46000.0)


def test_check_gradient_for_bad_atoms_reports_a_clean_gradient(caplog):
    fragment = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        bad = check_gradient_for_bad_atoms(fragment=fragment, gradient=np.full((3, 3), 10.0), threshold=100)

    assert bad == []
    assert "No atoms with gradients larger than threshold: 100" in caplog.text


def test_nuc_nuc_repulsion_of_two_protons_one_bohr_apart_is_one_hartree():
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, BOHR_TO_ANG]])

    assert nuc_nuc_repulsion(coords, [1, 1]) == pytest.approx(1.0)
    assert nuc_nuc_repulsion(coords * 2.0, [2, 2]) == pytest.approx(2.0)


def test_nuc_nuc_repulsion_sums_every_pair_once():
    coords = BOHR_TO_ANG * np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    charges = [1.0, 2.0, 3.0]

    pairwise = 1.0 * 2.0 / 1.0 + 1.0 * 3.0 / 2.0 + 2.0 * 3.0 / np.sqrt(5.0)
    assert nuc_nuc_repulsion(coords, charges) == pytest.approx(pairwise)
    assert nuc_nuc_repulsion(coords, np.array(charges)) == pytest.approx(pairwise)


def test_total_nuclear_charge_treats_unknown_elements_as_dummies_with_one_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.coords"):
        charge = total_nuclear_charge(["O", "Xx", "H", "Xx"])

    assert charge == 9
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Unknown element 'Xx'" in warnings[0].getMessage()


def test_list_of_masses_treats_unknown_elements_as_massless_with_one_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.coords"):
        masses = list_of_masses(["H", "Xx", "Yy"])

    assert masses[0] == pytest.approx(1.008, abs=0.001)
    assert masses[1:] == [0.0, 0.0]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_list_of_masses_rejects_elements_beyond_the_mass_table():
    with pytest.raises(InputError, match=r"No atomic mass available for element 'Rf' \(Z=104\)"):
        list_of_masses(["H", "Rf"])


def test_threshold_conn_scales_the_summed_covalent_radii_before_adding_the_tolerance():
    expected = 2.0 * (eldict_covrad["O"] + eldict_covrad["H"]) + 0.1

    assert threshold_conn("O", "H", 2.0, 0.1) == pytest.approx(expected)
    assert threshold_conn("H", "O", 2.0, 0.1) == pytest.approx(expected)


def test_get_connected_atoms_excludes_the_atom_itself():
    fragment = _water_methanol()

    assert get_connected_atoms(fragment.coords, fragment.elems, 1.0, 0.1, 0) == [1, 2]
    assert get_connected_atoms(fragment.coords, fragment.elems, 1.0, 0.1, 3) == [4, 5, 6, 7]
    assert get_connected_atoms(fragment.coords, fragment.elems, 1.0, 0.1, 8) == [7]


def test_expand_qm_region_computes_connectivity_when_the_fragment_has_none():
    fragment = Fragment(coords=[*WATER_COORDS, [10.0, 0.0, 0.0], [10.7, 0.0, 0.0]], elems=[*WATER_ELEMS, "H", "H"])

    assert fragment.connectivity == []
    assert expand_qm_region(fragment=fragment, initial_atoms=[1], radius=1.0) == [0, 1, 2]
    assert expand_qm_region(fragment=fragment, initial_atoms=[1], radius=9.5) == [0, 1, 2, 3, 4]


def test_expand_qm_region_requires_fragment_atoms_and_radius():
    fragment = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS)

    with pytest.raises(InputError, match="Provide fragment, initial_atoms and radius"):
        expand_qm_region(fragment=fragment, initial_atoms=[0])


def test_expand_qm_pc_region_rejects_missing_or_non_qmmm_theories():
    fragment = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS)

    with pytest.raises(InputError, match="requires fragment and theory"):
        expand_qm_pc_region(fragment=fragment)
    with pytest.raises(InputError, match="not a QMMMTheory"):
        expand_qm_pc_region(theory=object(), fragment=fragment)


@pytest.mark.parametrize("membs", [1, [1], [1, 3]])
def test_get_molecule_members_np_accepts_an_atom_index_or_seed_list(membs):
    fragment = Fragment(coords=[*WATER_COORDS, [10.0, 0.0, 0.0], [10.7, 0.0, 0.0]], elems=[*WATER_ELEMS, "H", "H"])

    members = _get_molecule_members_np(fragment.coords, fragment.elems, 99, 1.0, 0.1, membs=membs)

    assert members == ([0, 1, 2, 3, 4] if membs == [1, 3] else [0, 1, 2])
