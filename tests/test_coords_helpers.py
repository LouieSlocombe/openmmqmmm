"""Tests for the geometry and file-I/O helpers in coords.py.

coords.py is the second-largest module and only its Fragment-reading paths were
covered. The geometry primitives here (distance, angle, dihedral, RMSD, centroid)
underpin every constraint, active-region definition and analysis in the package, and
they are checkable against exact values from elementary geometry.
"""

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
from openmmqmmm.coords import angle, dihedral, distance, elemlisttoformula, get_centroid, nucchargelist, totmasslist

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
    """Four coplanar atoms in a cis arrangement have a zero dihedral."""
    assert abs(dihedral(*UNIT_SQUARE)) == pytest.approx(0.0, abs=1e-9)


def test_dihedral_of_a_right_angle_twist():
    """Rotating the last atom out of the plane by 90 degrees gives a 90 degree dihedral."""
    twisted = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    assert abs(dihedral(*twisted)) == pytest.approx(90.0)


def test_geometry_helpers_on_a_fragment():
    """The fragment-level wrappers must agree with the raw coordinate helpers."""
    fragment = Fragment(coords=UNIT_SQUARE, elems=["C", "C", "C", "C"], charge=0, mult=1)

    assert distance_between_atoms(fragment=fragment, atoms=[0, 1]) == pytest.approx(1.0)
    assert angle_between_atoms(fragment=fragment, atoms=[0, 1, 2]) == pytest.approx(90.0)
    assert abs(dihedral_between_atoms(fragment=fragment, atoms=[0, 1, 2, 3])) == pytest.approx(0.0, abs=1e-9)


def test_get_centroid():
    assert get_centroid(UNIT_SQUARE) == pytest.approx([0.5, 0.5, 0.0])


def test_elemlisttoformula_is_deterministic_hill_notation():
    """The formula must be stable and in Hill notation.

    It used to be built by iterating a set, so the same molecule produced a different
    string in every process — and the formula is embedded in the calculation labels
    that single_point_fragments builds.
    """
    assert elemlisttoformula(["H", "O", "H"]) == "H2O1"
    assert elemlisttoformula(["O", "H", "H"]) == "H2O1", "Input order must not matter"
    # Hill notation: carbon first, then hydrogen, then the rest alphabetically.
    assert elemlisttoformula(["O", "H", "C", "H", "N", "H"]) == "C1H3N1O1"


def test_nuclear_charges_and_mass():
    assert nucchargelist(["H", "O", "H"]) == pytest.approx(10.0)
    # Water is about 18 amu.
    assert totmasslist(["H", "O", "H"]) == pytest.approx(18.0, abs=0.1)


def test_xyzfile_roundtrip(tmp_path):
    """Coordinates written to XYZ must read back unchanged."""
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
