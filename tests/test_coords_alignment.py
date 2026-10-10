"""Kabsch alignment helpers in openmmqmmm.coords: flexible_align, reordering, subsets and the file wrappers."""

import numpy as np
import openmm.app
import openmm.unit
import pytest
from rmsd import reorder_hungarian

from openmmqmmm import Fragment, calculate_rmsd, flexible_align
from openmmqmmm.coords import (
    _reorder,
    _resolve_alignment_subsets,
    flexible_align_pdb,
    flexible_align_xyz,
    read_xyzfile,
)
from openmmqmmm.exceptions import InputError

# An asymmetric five-atom arrangement, so the optimal superposition is unique
ELEMS = ["C", "N", "O", "H", "S"]
COORDS = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 0.9], [1.0, 1.0, 1.0]])
TRANSLATION = np.array([2.0, -3.0, 4.0])
PERMUTATION = [4, 2, 0, 3, 1]


def _rotation(degrees):
    """Right-handed rotation about an axis that is not aligned with x, y or z."""
    axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    theta = np.radians(degrees)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.cos(theta) * np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * np.outer(axis, axis)


def _moved(rotate=True, translate=True):
    coords = COORDS @ _rotation(37.0).T if rotate else COORDS.copy()
    return coords + TRANSLATION if translate else coords


def _pairwise_distances(coords):
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)


def test_flexible_align_superimposes_a_rotated_and_translated_copy():
    target = _moved()
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=target.copy(), elems=ELEMS)

    aligned = flexible_align(fragment_a, fragment_b)

    assert aligned.elems == ELEMS
    assert aligned.coords == pytest.approx(target, abs=1e-9)
    assert fragment_a.coords == pytest.approx(COORDS)
    assert calculate_rmsd(aligned, fragment_b) == pytest.approx(0.0, abs=1e-9)


def test_flexible_align_rotate_only_leaves_a_pure_translation_to_the_target():
    target = _moved()
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=target.copy(), elems=ELEMS)

    rotated = flexible_align(fragment_a, fragment_b, rotate_only=True).coords

    residual = rotated - target
    assert residual == pytest.approx(np.tile(residual[0], (5, 1)), abs=1e-9)
    assert np.linalg.norm(residual[0]) > 1.0
    assert _pairwise_distances(rotated) == pytest.approx(_pairwise_distances(COORDS), abs=1e-9)


def test_flexible_align_rotate_only_recovers_a_pure_rotation_exactly():
    target = _moved(translate=False)
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=target.copy(), elems=ELEMS)

    assert flexible_align(fragment_a, fragment_b, rotate_only=True).coords == pytest.approx(target, abs=1e-9)


@pytest.mark.parametrize("reorder_method", ["brute", "hungarian", "inertia_hungarian", "distance"])
def test_flexible_align_reordering_matches_a_permuted_target(reorder_method):
    target = _moved()
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=target[PERMUTATION].copy(), elems=[ELEMS[i] for i in PERMUTATION])

    aligned = flexible_align(fragment_a, fragment_b, reordering=True, reorder_method=reorder_method)

    assert aligned.elems == ELEMS
    assert aligned.coords == pytest.approx(target, abs=1e-9)


def test_reorder_returns_the_permutation_that_restores_the_reference_order():
    p_coord = COORDS.copy()
    q_coord = _moved(rotate=False)[PERMUTATION]
    q_elems = np.array([ELEMS[i] for i in PERMUTATION])

    order = _reorder(reorder_hungarian, p_coord, q_coord, np.array(ELEMS), q_elems)

    assert order == list(np.argsort(PERMUTATION))
    assert list(q_elems[order]) == ELEMS
    assert p_coord == pytest.approx(COORDS - COORDS.mean(axis=0))


def test_flexible_align_with_paired_subsets_fits_on_matching_atoms_only():
    target = _moved()
    reversed_order = list(range(4, -1, -1))
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=target[reversed_order].copy(), elems=[ELEMS[i] for i in reversed_order])

    aligned = flexible_align(fragment_a, fragment_b, subset=[[0, 1, 2, 3], [4, 3, 2, 1]])

    assert aligned.coords == pytest.approx(target, abs=1e-9)


def test_resolve_alignment_subsets_handles_shared_and_paired_index_lists():
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=COORDS[::-1].copy(), elems=ELEMS[::-1])

    coords_a, elems_a, coords_b, elems_b = _resolve_alignment_subsets(fragment_a, fragment_b, [0, 2])
    assert elems_a == ["C", "O"]
    assert elems_b == ["S", "O"]
    assert coords_a == pytest.approx(COORDS[[0, 2]])
    assert coords_b == pytest.approx(COORDS[[4, 2]])

    coords_a, elems_a, coords_b, elems_b = _resolve_alignment_subsets(fragment_a, fragment_b, [[1, 3], [3, 1]])
    assert elems_a == elems_b == ["N", "H"]
    assert coords_a == pytest.approx(coords_b)

    coords_a, elems_a, coords_b, elems_b = _resolve_alignment_subsets(fragment_a, fragment_b, None, heavyatomsonly=True)
    assert elems_a == ["C", "N", "O", "S"]
    assert elems_b == ["S", "O", "N", "C"]
    assert coords_a == pytest.approx(COORDS[[0, 1, 2, 4]])


def test_resolve_alignment_subsets_rejects_unequal_pairs():
    fragment = Fragment(coords=COORDS.copy(), elems=ELEMS)

    with pytest.raises(InputError, match="Length of subsets not equal"):
        _resolve_alignment_subsets(fragment, fragment, [[0, 1, 2], [0, 1]])


def test_calculate_rmsd_heavy_atoms_only_ignores_a_displaced_hydrogen():
    displaced = COORDS.copy()
    displaced[3] += [0.0, 0.0, 1.0]
    fragment_a = Fragment(coords=COORDS.copy(), elems=ELEMS)
    fragment_b = Fragment(coords=displaced, elems=ELEMS)

    assert calculate_rmsd(fragment_a, fragment_b, heavyatomsonly=True) == pytest.approx(0.0, abs=1e-9)
    assert calculate_rmsd(fragment_a, fragment_b) > 0.1


def test_flexible_align_xyz_writes_the_aligned_structure_next_to_the_first_file(tmp_path):
    target = _moved()
    Fragment(coords=COORDS.copy(), elems=ELEMS).write_xyzfile(str(tmp_path / "mobile.xyz"))
    Fragment(coords=target.copy(), elems=ELEMS).write_xyzfile(str(tmp_path / "reference.xyz"))

    flexible_align_xyz(str(tmp_path / "mobile.xyz"), str(tmp_path / "reference.xyz"))

    elems, coords = read_xyzfile(str(tmp_path / "mobile_aligned.xyz"))
    assert elems == ELEMS
    assert np.asarray(coords) == pytest.approx(target, abs=1e-6)


def test_flexible_align_pdb_writes_the_aligned_structure_through_openmm(tmp_path):
    target = _moved()
    Fragment(coords=COORDS.copy(), elems=ELEMS).write_pdbfile_openmm(filename=str(tmp_path / "mobile"))
    Fragment(coords=target.copy(), elems=ELEMS).write_pdbfile_openmm(filename=str(tmp_path / "reference"))

    flexible_align_pdb(str(tmp_path / "mobile.pdb"), str(tmp_path / "reference.pdb"))

    written = openmm.app.PDBFile(str(tmp_path / "mobile_aligned.pdb"))
    assert [atom.element.symbol for atom in written.topology.atoms()] == ELEMS
    positions = np.asarray(written.positions.value_in_unit(openmm.unit.angstrom))
    assert positions == pytest.approx(target, abs=2e-3)
