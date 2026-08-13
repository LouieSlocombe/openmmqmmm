import numpy as np
import pytest

from openmmqmmm import (
    Fragment,
    angle_between_atoms,
    calculate_rmsd,
    dihedral_between_atoms,
    distance_between_atoms,
    read_xyzfile,
    write_xyzfile,
)
from openmmqmmm.coords import (
    _build_connectivity,
    angle,
    dihedral,
    distance,
    eldict_covrad,
    elemlisttoformula,
    get_centroid,
    nucchargelist,
    threshold_conn,
    totmasslist,
)

# A unit square walked corner to corner, so every quantity below is exact.
UNIT_SQUARE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


def test_distance():
    assert distance([0.0, 0.0, 0.0], [3.0, 4.0, 0.0]) == pytest.approx(5.0)
    assert distance([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_angle_of_a_right_corner():
    assert angle(UNIT_SQUARE[0], UNIT_SQUARE[1], UNIT_SQUARE[2]) == pytest.approx(90.0)


def test_angle_of_a_straight_line():
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert angle(*collinear) == pytest.approx(180.0)


def test_dihedral_of_a_planar_arrangement():
    assert abs(dihedral(*UNIT_SQUARE)) == pytest.approx(0.0, abs=1e-9)


def test_dihedral_of_a_right_angle_twist():
    twisted = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    assert abs(dihedral(*twisted)) == pytest.approx(90.0)


def test_geometry_helpers_on_a_fragment():
    fragment = Fragment(coords=UNIT_SQUARE, elems=["C", "C", "C", "C"], charge=0, mult=1)

    assert distance_between_atoms(fragment=fragment, atoms=[0, 1]) == pytest.approx(1.0)
    assert angle_between_atoms(fragment=fragment, atoms=[0, 1, 2]) == pytest.approx(90.0)
    assert abs(dihedral_between_atoms(fragment=fragment, atoms=[0, 1, 2, 3])) == pytest.approx(0.0, abs=1e-9)


def test_get_centroid():
    assert get_centroid(UNIT_SQUARE) == pytest.approx([0.5, 0.5, 0.0])


def test_elemlisttoformula_is_deterministic_hill_notation():
    assert elemlisttoformula(["H", "O", "H"]) == "H2O1"
    assert elemlisttoformula(["O", "H", "H"]) == "H2O1", "Input order must not matter"
    # Hill notation: carbon first, then hydrogen, then the rest alphabetically.
    assert elemlisttoformula(["O", "H", "C", "H", "N", "H"]) == "C1H3N1O1"


def test_nuclear_charges_and_mass():
    assert nucchargelist(["H", "O", "H"]) == pytest.approx(10.0)
    # Water is about 18 amu.
    assert totmasslist(["H", "O", "H"]) == pytest.approx(18.0, abs=0.1)


def test_xyzfile_roundtrip(tmp_path):
    elems = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])

    name = str(tmp_path / "water")
    write_xyzfile(elems, coords, name)

    read_elems, read_coords = read_xyzfile(f"{name}.xyz")
    assert read_elems == elems
    assert np.allclose(read_coords, coords)


def test_rmsd_of_identical_structures_is_zero():
    fragment = Fragment(coords=UNIT_SQUARE, elems=["C", "C", "C", "C"], charge=0, mult=1)
    assert calculate_rmsd(fragment, fragment) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_is_invariant_under_translation():
    """RMSD aligns the structures first, so a rigid shift must not change it."""
    original = Fragment(coords=UNIT_SQUARE, elems=["C", "C", "C", "C"], charge=0, mult=1)
    shifted = Fragment(coords=UNIT_SQUARE + 10.0, elems=["C", "C", "C", "C"], charge=0, mult=1)

    assert calculate_rmsd(original, shifted) == pytest.approx(0.0, abs=1e-9)


def test_rmsd_detects_a_real_difference():
    original = Fragment(coords=UNIT_SQUARE, elems=["C", "C", "C", "C"], charge=0, mult=1)
    distorted_coords = UNIT_SQUARE.copy()
    distorted_coords[0, 2] += 1.0
    distorted = Fragment(coords=distorted_coords, elems=["C", "C", "C", "C"], charge=0, mult=1)

    assert calculate_rmsd(original, distorted) > 0.1


# The package has two connectivity implementations: calc_conn_py / get_connected_atoms_np
# (used by Fragment.calc_connectivity) and _build_connectivity (used by the internal
# coordinate table). They read their covalent radii from the same table now, but used to
# read from two copies of it, and only one copy carried the overrides that stop ions and
# TIP4P dummy sites bonding to everything nearby.


def _neighbours_via_calc_conn(coords, elems):
    neighbours = [set() for _ in elems]
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            if distance(coords[i], coords[j]) < threshold_conn(elems[i], elems[j], scale=1.0, tol=0.4):
                neighbours[i].add(j)
                neighbours[j].add(i)
    return neighbours


def test_ions_do_not_bond_in_either_connectivity_path():
    # Na+ at the origin with three oxygens at 2.4 A -- a typical first solvation shell,
    # well inside the sum of the unmodified Alvarez radii for Na (1.66) and O (0.66).
    coords = np.array([[0.0, 0.0, 0.0], [2.4, 0.0, 0.0], [0.0, 2.4, 0.0], [0.0, 0.0, 2.4]])
    elems = ["Na", "O", "O", "O"]

    assert _build_connectivity(coords, elems)[0] == set()
    assert _neighbours_via_calc_conn(coords, elems)[0] == set()


def test_both_paths_use_the_same_covalent_radii():
    # The overrides that keep solvated ions and dummy sites from bonding
    assert eldict_covrad["Na"] < 0.01
    assert eldict_covrad["K"] < 0.01
    assert eldict_covrad["M"] == 0.0

    # A geometry that discriminates: two sodiums 2.5 A apart. Under the raw Alvarez radius
    # (1.66) the threshold is 3.72 A and they bond; under the override it is 0.4 A and they
    # do not. Both implementations must land on the same side.
    two_sodiums = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    assert _build_connectivity(two_sodiums, ["Na", "Na"]) == [set(), set()]
    assert _neighbours_via_calc_conn(two_sodiums, ["Na", "Na"]) == [set(), set()]

    # And the lanthanides, which the second table omitted entirely, are present
    for symbol in ("Gd", "Eu", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "U", "Po", "At", "Rn"):
        assert symbol in eldict_covrad


def test_both_connectivity_paths_agree_on_a_normal_molecule():
    # Water, at its equilibrium geometry.
    coords = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    elems = ["O", "H", "H"]

    assert _build_connectivity(coords, elems) == _neighbours_via_calc_conn(coords, elems)
    assert _build_connectivity(coords, elems) == [{1, 2}, {0}, {0}]
