"""Regression checks against the original finite-cluster imaging convention."""

import numpy as np
import pytest
from ase.geometry import find_mic
from test_qmmm_periodic_accuracy import _periodic_qmmm

from openmmqmmm.exceptions import InputError
from openmmqmmm.periodic_embedding import PeriodicQMGeometry


def _legacy_image(geometry, coords, box, *, place_virtual_sites=None):
    """Retain the pre-vectorization traversal, reductions, and ASE call grouping."""
    source = np.asarray(coords, dtype=float)
    imaged = source.copy()
    if geometry.parents:
        parents, children = np.asarray(geometry.parents).T
        displacements, _ = find_mic(source[children] - source[parents], box)
        for (parent, child), displacement in zip(geometry.parents, displacements, strict=True):
            imaged[child] = imaged[parent] + displacement
        first, second = geometry.bonds.T
        bond_images, _ = find_mic(source[second] - source[first], box)
        if not np.allclose(imaged[second] - imaged[first], bond_images, atol=1e-6, rtol=0):
            raise InputError("Periodic QM/MM cannot unwrap a covalent network that winds around the cell")

    if place_virtual_sites is not None:
        imaged = place_virtual_sites(imaged)

    anchor_members = next(members for members in geometry.qm_members if geometry.qmatoms[0] in members)
    anchor = imaged[anchor_members].mean(axis=0)
    for component, members in zip(geometry.components, geometry.qm_members, strict=True):
        if len(members):
            delta = imaged[members].mean(axis=0) - anchor
            nearest, _ = find_mic(delta, box)
            imaged[component] += nearest - delta
    center = imaged[geometry.qmatoms].mean(axis=0)
    mm_components = [
        component
        for component, members in zip(geometry.components, geometry.qm_members, strict=True)
        if not len(members)
    ]
    if mm_components:
        deltas = np.asarray([imaged[component].mean(axis=0) - center for component in mm_components])
        nearest, _ = find_mic(deltas, box)
        for component, shift in zip(mm_components, nearest - deltas, strict=True):
            imaged[component] += shift
    imaged -= np.floor(imaged[geometry.qmatoms[0]] @ np.linalg.inv(box)) @ box
    return imaged


def _mixed_molecules():
    rng = np.random.default_rng(801)
    # Permuting indices makes components interleave and BFS order differ from
    # atom order. The chain exercises dependencies across many traversal levels.
    components = np.split(rng.permutation(48), [33, 40, 45, 47])
    bonds = list(zip(components[0][:-1], components[0][1:], strict=True))
    ring = components[1]
    bonds.extend(zip(ring, np.roll(ring, -1), strict=True))
    bonds.extend((components[2][0], atom) for atom in components[2][1:])
    bonds.append(tuple(components[3]))
    fractional = np.empty((48, 3))
    for index, component in enumerate(components):
        fractional[component] = rng.normal(scale=0.008, size=(len(component), 3)) + np.array([0.15 * index, 0.13, 0.2])
    # Choose an anchor in a component whose root is not the first QM-containing
    # root, and retain unsorted QM indices as permitted by the geometry helper.
    ordered = sorted(components, key=lambda component: component.min())
    qmatoms = [ordered[2][-1], ordered[0][-1], ordered[2][0], ordered[1][0]]
    return fractional, bonds, qmatoms


@pytest.mark.parametrize(
    "box",
    [
        np.diag([30.0, 28.0, 26.0]),
        np.array([[30.0, 0, 0], [7, 28, 0], [4, 3, 26]]),
        np.array([[30.0, 0, 0], [24, 18, 0], [17, 8, 22]]),
        # A rotated orthorhombic cell must not be treated as a diagonal matrix.
        np.array([[18.0, 24, 0], [-22.4, 16.8, 0], [0, 0, 26]]),
    ],
    ids=["orthorhombic", "triclinic", "skew", "rotated-orthorhombic"],
)
def test_mixed_interleaved_molecules_match_legacy_across_repeated_evaluations(box):
    fractional, bonds, qmatoms = _mixed_molecules()
    geometry = PeriodicQMGeometry(len(fractional), iter(bonds), qmatoms)
    qm_roots = [
        component[0]
        for component, members in zip(geometry.components, geometry.qm_members, strict=True)
        if len(members)
    ]
    anchor_root = next(component[0] for component in geometry.components if qmatoms[0] in component)
    assert anchor_root != qm_roots[0]
    rng = np.random.default_rng(67)
    for scale in (1.0, 1.07, 0.96):
        current_box = box * scale
        coords = fractional @ current_box + rng.normal(scale=1e-3, size=fractional.shape)
        coords += rng.integers(-4, 5, size=coords.shape) @ current_box
        original = coords.copy()
        actual = geometry.image(coords, current_box)
        expected = _legacy_image(geometry, coords, current_box)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(coords, original)


@pytest.mark.parametrize("qmatoms", [[0], [3, 1], [2, 0, 3, 1]])
def test_singletons_and_all_qm_components_match_legacy(qmatoms):
    geometry = PeriodicQMGeometry(4, [], qmatoms)
    coords = np.array([[1, 2, 3], [17, 8, -19], [-11, 17, 7], [8, -5, 12]], dtype=float)
    box = np.diag([20.0, 22.0, 24.0])
    np.testing.assert_allclose(geometry.image(coords, box), _legacy_image(geometry, coords, box), rtol=0, atol=1e-12)


@pytest.mark.parametrize("return_copy", [False, True])
def test_virtual_site_callback_runs_after_unwrapping_before_component_centers(return_copy):
    box = np.diag([30.0] * 3)
    coords = np.array(
        [[26, 4, 5], [29.4, 5, 5], [0.6, 5.2, 5], [27.1, 4.1, 5], [1, 3, 4], [11, 13, 14], [29.9, 5.1, 5]]
    )
    original = coords.copy()
    geometry = PeriodicQMGeometry(7, [(0, 3), (1, 2)], [4, 0], image_links=[(6, 1), (6, 2)])
    calls = []

    def place(imaged):
        calls.append(imaged.copy())
        placed = imaged.copy() if return_copy else imaged
        placed[6] = 0.3 * imaged[1] + 0.7 * imaged[2] + [0, 0, 0.2]
        return placed

    expected = _legacy_image(geometry, coords, box, place_virtual_sites=place)
    legacy_callback_coords = calls.pop()
    actual = geometry.image(coords, box, place_virtual_sites=place)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], legacy_callback_coords, rtol=0, atol=1e-12)
    np.testing.assert_allclose(calls[0][2] - calls[0][1], [1.2, 0.2, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual[6], 0.3 * actual[1] + 0.7 * actual[2] + [0, 0, 0.2], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(coords, original)
    assert geometry.neighbors[6] == []


@pytest.mark.parametrize("offset", [-1e-8, 0, 1e-8])
@pytest.mark.parametrize("qm_fragment", [False, True])
def test_image_selection_on_both_sides_of_half_cell_boundary(offset, qm_fragment):
    box = np.diag([10.0] * 3)
    coords = np.array([[0, 0, 0], [5 + offset, 0, 0]], dtype=float)
    geometry = PeriodicQMGeometry(2, [], [0, 1] if qm_fragment else [0])
    actual = geometry.image(coords, box)
    np.testing.assert_allclose(actual, _legacy_image(geometry, coords, box), rtol=0, atol=1e-12)
    assert actual[1, 0] == pytest.approx(5 + offset if offset <= 0 else -5 + offset, rel=0, abs=1e-12)


@pytest.mark.parametrize("qm_fragments", [False, True])
def test_near_half_cell_ase_tie_preserves_singleton_qm_and_batched_mm_calls(qm_fragments):
    box = np.diag([10.0] * 3)
    coords = np.array([[0, 0, 0], [np.nextafter(-5.0, 0), 0, 0], [10 / 3] * 3])
    geometry = PeriodicQMGeometry(3, [], [0, 1, 2] if qm_fragments else [0])
    actual = geometry.image(coords, box)
    np.testing.assert_allclose(actual, _legacy_image(geometry, coords, box), rtol=0, atol=1e-12)
    # ASE's safe naive MIC retains the slightly-shorter negative vector for
    # a singleton. The other MM row forces the entire batch through general MIC,
    # whose roundoff and tie convention select the positive image instead.
    assert actual[1, 0] == pytest.approx(-5.0 if qm_fragments else 5.0, rel=0, abs=1e-12)


def test_near_half_cell_ase_tie_preserves_bond_batching_across_traversal_levels():
    box = np.diag([10.0] * 3)
    coords = np.array([[0, 0, 0], [np.nextafter(-5.0, 0), 0, 0], [10 / 3] * 3, [0, 0, 0]])
    # The long diagonal edge is one level deeper than the near-half-cell edge.
    # Splitting MIC calls by traversal level changes the selected image of 1.
    geometry = PeriodicQMGeometry(4, [(0, 1), (0, 3), (3, 2)], [0])
    actual = geometry.image(coords, box)
    np.testing.assert_allclose(actual, _legacy_image(geometry, coords, box), rtol=0, atol=1e-12)
    assert actual[1, 0] == pytest.approx(5.0, rel=0, abs=1e-12)


@pytest.mark.parametrize("qm_fragment", [False, True])
def test_component_center_reduction_preserves_bfs_mm_and_sorted_qm_order(qm_fragment):
    box = np.diag([10.0] * 3)
    order = [1, 9, 2, 8, 3, 7, 4, 6, 5]
    coords = np.zeros((10, 3))
    coords[1:, 0] = [
        5.65229427461249,
        4.6825066182047514,
        5.175635023833122,
        5.295882538276965,
        5.300879707328754,
        4.271429718630802,
        5.682987445208003,
        4.242304936725615,
        4.696079737179495,
    ]
    geometry = PeriodicQMGeometry(10, [(1, child) for child in order[1:]], [0, *order] if qm_fragment else [0])
    actual = geometry.image(coords, box)
    np.testing.assert_allclose(actual, _legacy_image(geometry, coords, box), rtol=0, atol=1e-12)
    # The nine-member sum is exactly 5 in sorted QM order but one ulp above
    # 5 in the MM component's BFS order: reordering changes the entire image.
    expected_shift = 0 if qm_fragment else -10
    np.testing.assert_allclose(actual[1:, 0], coords[1:, 0] + expected_shift, rtol=0, atol=1e-12)


@pytest.mark.parametrize("qm_fragment", [False, True])
def test_float32_callback_preserves_component_mean_precision(qm_fragment):
    coords = np.zeros((4, 3))
    coords[1:, 0] = [5, 5, 5 + np.spacing(np.float32(5))]
    box = np.diag([10.0] * 3)
    geometry = PeriodicQMGeometry(4, [(1, 2), (1, 3)], [0, 1, 2, 3] if qm_fragment else [0])

    def place(imaged):
        return imaged.astype(np.float32)

    actual = geometry.image(coords, box, place_virtual_sites=place)
    expected = _legacy_image(geometry, coords, box, place_virtual_sites=place)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert np.all(actual[1:, 0] > 0)


@pytest.mark.parametrize("image_links", [False, True])
def test_winding_cycles_still_reject_before_virtual_site_placement(image_links):
    bonds = [(0, 1), (1, 2)]
    closing_bond = [(2, 0)]
    geometry = PeriodicQMGeometry(
        3, bonds if image_links else bonds + closing_bond, [0], image_links=closing_bond if image_links else ()
    )
    calls = []

    def place(coords):
        calls.append(coords)
        return coords

    with pytest.raises(InputError, match="winds around"):
        geometry.image(np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]]), np.diag([5.0] * 3), place_virtual_sites=place)
    assert calls == []


@pytest.mark.parametrize("box", [np.diag([30.0] * 3), np.array([[30.0, 0, 0], [4, 28, 0], [2, 3, 27]])])
def test_qmmm_energy_and_gradient_match_legacy_imaging(monkeypatch, box):
    coords = np.array([[1.4, 2, 2], [29.7, 3, 2], [0.9, 3.1, 2], [28.6, 4, 2], [14, 9, 6], [0, 4.1, 2], [20, 7, 3]])
    theory, _, _ = _periodic_qmmm(
        coords,
        bonds=[(1, 2), (3, 5)],
        qmatoms=[3, 5, 6],
        charges=[0.1, -0.5, 0.4, 0, 0.3, 0, 0],
    )
    geometry = theory._periodic_geometry
    optimized_image = geometry.image

    def legacy_image(current_coords, current_box, **kwargs):
        return _legacy_image(geometry, current_coords, current_box, **kwargs)

    shifts = np.array([[1, -2, 0], [-1, 0, 2], [0, 1, 0], [2, 0, -1], [0, -1, 0], [-1, 2, 0], [0, 0, 1]])
    for current_coords in (coords, coords + shifts @ box):
        monkeypatch.setattr(geometry, "image", optimized_image)
        energy, gradient = theory.run(current_coords=current_coords, periodic_box_vectors=box, grad=True)
        gradient = gradient.copy()
        monkeypatch.setattr(geometry, "image", legacy_image)
        expected_energy, expected_gradient = theory.run(
            current_coords=current_coords, periodic_box_vectors=box, grad=True
        )
        assert energy == pytest.approx(expected_energy, rel=0, abs=1e-12)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=0, atol=1e-12)
