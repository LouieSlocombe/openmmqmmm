"""Preserve complete charge groups while measuring the requested QM distance."""

import numpy as np
import pytest

from openmmqmmm.embedding_selection import build_charge_groups, select_charge_groups
from openmmqmmm.exceptions import InputError


def test_groups_keep_molecule_connected_through_qm_atoms_and_preserve_pc_order():
    # Atom 0 is QM and bridges two MM arms. The PC list is deliberately unsorted.
    groups = build_charge_groups(6, [4, 2, 5, 1, 3], atom_links=[(1, 0), (0, 2), (3, 4)])
    assert [group.tolist() for group in groups] == [[0, 4], [1, 3], [2]]
    coords = np.array([[8, 0, 0], [0.5, 0, 0], [10, 0, 0], [9, 0, 0], [12, 0, 0]])
    assert select_charge_groups(coords, [[0, 0, 0]], groups, 1).tolist() == [1, 3]


def test_added_dipoles_and_native_virtual_sites_stay_with_host_molecule():
    # Original site 3 is a native virtual site; added sites 4/5 form a dipole.
    groups = build_charge_groups(4, [1, 2, 3], atom_links=[(1, 2), (1, 3)], virtual_site_parents=[(1, 2), (1, 2)])
    assert [group.tolist() for group in groups] == [[0, 1, 2, 3, 4]]
    coords = np.array([[4, 0, 0], [5, 0, 0], [4.5, 0, 0], [3.9, 0, 0], [4.1, 0, 0]])
    selected = select_charge_groups(coords, [[0, 0, 0]], groups, 4.0)
    assert selected.tolist() == [0, 1, 2, 3, 4]
    charges = np.array([0.0, 0.3, 0.0, -3.333333, 3.333333])
    assert charges[selected].sum() == pytest.approx(0.3)


def test_overlapping_atom_groups_and_boundary_parents_merge_transitively():
    groups = build_charge_groups(
        6,
        [1, 2, 3, 4, 5],
        atom_groups=[(0, 1), (1, 2), (3, 4)],
        virtual_site_parents=[(2, 3)],
    )
    assert [group.tolist() for group in groups] == [[0, 1, 2, 3, 5], [4]]


def test_nearest_selection_includes_charge_at_edge_of_extended_qm_region():
    qm_coords = np.array([[-10, 0, 0], [10, 0, 0]])
    coords = np.array([[11, 0, 0], [0.5, 0, 0]])
    groups = [[0], [1]]
    assert select_charge_groups(coords, qm_coords, groups, 2, selection="nearest").tolist() == [0]
    assert select_charge_groups(coords, qm_coords, groups, 2, selection="centroid").tolist() == [1]


@pytest.mark.parametrize("selection", ["nearest", "centroid"])
def test_strict_radius_completes_neutral_molecule_and_keeps_original_site_order(selection):
    coords = np.array([[1, 0, 0], [0.9, 0, 0], [2, 0, 0], [0.5, 0, 0]])
    # The atom just outside the radius belongs to the selected neutral molecule.
    selected = select_charge_groups(coords, [[0, 0, 0]], [[3], [2, 1], [0]], 1, selection=selection)
    assert selected.tolist() == [1, 2, 3]
    assert np.array([-1.0, -0.5, 0.5, 0.0])[selected].sum() == 0


@pytest.mark.parametrize("radius,expected", [(0.1, []), (10.0, [0, 1])])
def test_none_and_all_groups_selected(radius, expected):
    coords = np.array([[1, 0, 0], [2, 0, 0]])
    assert select_charge_groups(coords, [[0, 0, 0]], [[0], [1]], radius).tolist() == expected


def test_no_charge_sites_is_supported():
    assert build_charge_groups(2, [], atom_links=[(0, 1)]) == []
    assert build_charge_groups(0, []) == []
    selected = select_charge_groups(np.empty((0, 3)), [[0, 0, 0]], [], 1)
    assert selected.shape == (0,)
    assert selected.dtype.kind == "i"


def test_nearest_tree_matches_pairwise_reference_for_irregular_geometries():
    rng = np.random.default_rng(824)
    coords = rng.normal(size=(60, 3))
    qm_coords = rng.normal(size=(9, 3))
    groups = [np.arange(start, start + 3) for start in range(0, 60, 3)]
    selected = select_charge_groups(coords, qm_coords, groups, 0.5)
    distance = np.linalg.norm(coords[:, None, :] - qm_coords[None, :, :], axis=2).min(axis=1)
    expected = np.concatenate([group for group in groups if np.any(distance[group] < 0.5)])
    np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mm_atoms": [1, 1]},
        {"mm_atoms": [True]},
        {"mm_atoms": [1.0]},
        {"atom_links": [(0, 4)]},
        {"atom_links": [(0, 1, 2)]},
        {"atom_groups": [[]]},
        {"atom_groups": [[0, -1]]},
        {"virtual_site_parents": [[]]},
        {"virtual_site_parents": [[2.0]]},
    ],
)
def test_invalid_group_connectivity_is_rejected(kwargs):
    options = {"mm_atoms": [1, 2]}
    options.update(kwargs)
    with pytest.raises(InputError):
        build_charge_groups(3, **options)


@pytest.mark.parametrize("groups", [[], [[0]], [[0, 1], [1]], [[0, 0]], [[0.0], [1]], [[0], [-1]], [[]]])
def test_invalid_pc_partition_is_rejected(groups):
    with pytest.raises(InputError):
        select_charge_groups([[1, 0, 0], [2, 0, 0]], [[0, 0, 0]], groups, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coords": [[np.nan, 0, 0]]},
        {"coords": [[1, 0]]},
        {"qm_coords": []},
        {"qm_coords": [[np.inf, 0, 0]]},
        {"radius": 0},
        {"radius": True},
        {"selection": "unknown"},
    ],
)
def test_invalid_selection_controls_are_rejected(kwargs):
    options = {"coords": [[1, 0, 0]], "qm_coords": [[0, 0, 0]], "groups": [[0]], "radius": 2}
    options.update(kwargs)
    with pytest.raises(InputError):
        select_charge_groups(**options)
