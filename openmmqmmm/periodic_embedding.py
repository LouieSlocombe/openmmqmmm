"""Molecule-preserving images for the finite QM/MM embedding cluster.

This constructs a reproducible finite cluster, not an Ewald sum for the QM
Hamiltonian. Image choices are discrete lattice translations, so their Cartesian
Jacobian is the identity away from image-switching surfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
from ase.geometry import find_mic

from openmmqmmm.exceptions import InputError


class PeriodicQMGeometry:
    """Unwrap topology molecules and image them consistently around the QM region."""

    def __init__(
        self,
        numatoms: int,
        bonds: Iterable[tuple[int, int]],
        qmatoms: Sequence[int],
        *,
        image_links: Iterable[tuple[int, int]] = (),
    ) -> None:
        self.qmatoms = np.asarray(qmatoms, dtype=int)
        self.neighbors: list[list[int]] = [[] for _ in range(numatoms)]
        covalent_bonds = list(bonds)
        for first, second in covalent_bonds:
            self.neighbors[first].append(int(second))
            self.neighbors[second].append(int(first))
        # Virtual charge sites must travel with their host molecule, but their
        # parent links must never be mistaken for a covalent QM/MM cap boundary.
        image_links = list(image_links)
        self.bonds = np.asarray([*covalent_bonds, *image_links], dtype=int).reshape(-1, 2)
        image_neighbors = [neighbors.copy() for neighbors in self.neighbors]
        for first, second in image_links:
            image_neighbors[first].append(second)
            image_neighbors[second].append(first)

        visited: set[int] = set()
        self.components: list[np.ndarray] = []
        self.parents: list[tuple[int, int]] = []
        depths = np.zeros(numatoms, dtype=int)
        edges_by_depth: list[list[int]] = []
        for root in range(numatoms):
            if root in visited:
                continue
            visited.add(root)
            component = [root]
            for parent in component:
                for child in image_neighbors[parent]:
                    if child not in visited:
                        visited.add(child)
                        component.append(child)
                        depth = depths[parent]
                        if depth == len(edges_by_depth):
                            edges_by_depth.append([])
                        edges_by_depth[depth].append(len(self.parents))
                        depths[child] = depth + 1
                        self.parents.append((parent, child))
            self.components.append(np.asarray(component, dtype=int))
        self.qm_members = [np.intersect1d(component, self.qmatoms) for component in self.components]
        self._parents, self._children = np.asarray(self.parents, dtype=int).reshape(-1, 2).T
        self._bond_first, self._bond_second = self.bonds.T
        # Nodes at the same depth depend only on earlier layers, even across
        # different molecules. Preserve each root-to-leaf addition order without
        # a Python iteration per bond (water boxes need just one layer).
        self._unwrap_layers = [
            (self._parents[edges], self._children[edges], np.asarray(edges, dtype=int)) for edges in edges_by_depth
        ]

        qm_components = [
            component for component, members in zip(self.components, self.qm_members, strict=True) if len(members)
        ]
        qm_members = [members for members in self.qm_members if len(members)]
        mm_components = [
            component for component, members in zip(self.components, self.qm_members, strict=True) if not len(members)
        ]
        self._qm_atoms, self._qm_rows, self._qm_counts = self._group_indices(qm_members)
        self._qm_component_atoms, self._qm_component_rows, _ = self._group_indices(qm_components)
        self._mm_atoms, self._mm_rows, self._mm_counts = self._group_indices(mm_components)
        self._anchor_row = next(index for index, members in enumerate(qm_members) if self.qmatoms[0] in members)

    @staticmethod
    def _group_indices(groups: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cache atom order, center rows and sizes for a set of components."""
        counts = np.asarray([len(group) for group in groups], dtype=int)
        atoms = np.concatenate(groups) if groups else np.empty(0, dtype=int)
        return atoms, np.repeat(np.arange(len(groups)), counts), counts

    @staticmethod
    def _centers(coords: np.ndarray, atoms: np.ndarray, rows: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Sum each component in its original order, matching its per-array mean."""
        if coords.dtype != np.float64:
            # Custom site callbacks may change dtype. Preserve their mean's
            # accumulation precision; native OpenMM sites return float64.
            stops = np.cumsum(counts)
            starts = stops - counts
            return np.asarray(
                [coords[atoms[start:stop]].mean(axis=0) for start, stop in zip(starts, stops, strict=True)]
            )
        # Unlike reduceat's pairwise reduction, bincount adds in atom order.
        # Keeping that order matters when a center is on an image-switch surface.
        sums = np.column_stack(
            [np.bincount(rows, weights=coords[atoms, axis], minlength=len(counts)) for axis in range(3)]
        )
        return sums / counts[:, None]

    def image(
        self,
        coords: np.ndarray,
        box_vectors: np.ndarray,
        *,
        place_virtual_sites: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return whole molecules near the QM region, with coordinates and box in Å."""
        box = np.asarray(box_vectors, dtype=float)
        if box.shape != (3, 3) or not np.all(np.isfinite(box)) or np.linalg.det(box) <= 1e-12:
            raise InputError("Periodic QM/MM requires finite, right-handed, nonsingular 3 x 3 box vectors in Å")
        source = np.asarray(coords, dtype=float)
        if source.shape != (len(self.neighbors), 3) or not np.all(np.isfinite(source)):
            raise InputError("Periodic QM/MM coordinates must be a finite N x 3 array matching the MM topology")
        imaged = source.copy()
        if self._parents.size:
            displacements, _lengths = find_mic(source[self._children] - source[self._parents], box)
            for parents, children, edges in self._unwrap_layers:
                imaged[children] = imaged[parents] + displacements[edges]

            # A periodic covalent network cannot be represented by one finite
            # molecule without cutting a bond. Refuse to silently select such a cut.
            first, second = self._bond_first, self._bond_second
            bond_images, _lengths = find_mic(source[second] - source[first], box)
            if not np.allclose(imaged[second] - imaged[first], bond_images, atol=1e-6, rtol=0):
                raise InputError("Periodic QM/MM cannot unwrap a covalent network that winds around the cell")

        if place_virtual_sites is not None:
            imaged = place_virtual_sites(imaged)

        qm_centers = self._centers(imaged, self._qm_atoms, self._qm_rows, self._qm_counts)
        anchor = qm_centers[self._anchor_row]
        # Each QM-containing molecule is brought near the same anchor before the
        # overall QM center is defined. A bonded QM/MM boundary moves as one molecule.
        qm_deltas = qm_centers - anchor
        qm_shifts = np.empty(qm_deltas.shape, dtype=np.result_type(qm_deltas.dtype, float))
        # Keep the singleton MIC calls: ASE can select a different numerical
        # path when a batch contains a displacement outside its safe radius.
        for index, delta in enumerate(qm_deltas):
            nearest, _length = find_mic(delta, box)
            qm_shifts[index] = nearest - delta
        imaged[self._qm_component_atoms] += qm_shifts[self._qm_component_rows]
        center = imaged[self.qmatoms].mean(axis=0)
        if self._mm_counts.size:
            deltas = self._centers(imaged, self._mm_atoms, self._mm_rows, self._mm_counts) - center
            nearest, _lengths = find_mic(deltas, box)
            imaged[self._mm_atoms] += (nearest - deltas)[self._mm_rows]

        # Equivalent images of the anchor also produce identical QM input, up to
        # roundoff, rather than relying on the QM backend's translation invariance.
        imaged -= np.floor(imaged[self.qmatoms[0]] @ np.linalg.inv(box)) @ box
        return imaged
