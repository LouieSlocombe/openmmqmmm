"""Molecule-preserving images for the finite QM/MM embedding cluster.

This constructs a reproducible finite cluster, not an Ewald sum for the QM
Hamiltonian. Image choices are discrete lattice translations, so their Cartesian
Jacobian is the identity away from image-switching surfaces.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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
                        self.parents.append((parent, child))
            self.components.append(np.asarray(component, dtype=int))
        self.qm_members = [np.intersect1d(component, self.qmatoms) for component in self.components]

    def image(self, coords: np.ndarray, box_vectors: np.ndarray) -> np.ndarray:
        """Return whole molecules near the QM region, with coordinates and box in Å."""
        box = np.asarray(box_vectors, dtype=float)
        if box.shape != (3, 3) or not np.all(np.isfinite(box)) or np.linalg.det(box) <= 1e-12:
            raise InputError("Periodic QM/MM requires finite, right-handed, nonsingular 3 x 3 box vectors in Å")
        source = np.asarray(coords, dtype=float)
        if source.shape != (len(self.neighbors), 3) or not np.all(np.isfinite(source)):
            raise InputError("Periodic QM/MM coordinates must be a finite N x 3 array matching the MM topology")
        imaged = source.copy()
        if self.parents:
            parents, children = np.asarray(self.parents).T
            displacements, _lengths = find_mic(source[children] - source[parents], box)
            for (parent, child), displacement in zip(self.parents, displacements, strict=True):
                imaged[child] = imaged[parent] + displacement

            # A periodic covalent network cannot be represented by one finite
            # molecule without cutting a bond. Refuse to silently select such a cut.
            first, second = self.bonds.T
            bond_images, _lengths = find_mic(source[second] - source[first], box)
            if not np.allclose(imaged[second] - imaged[first], bond_images, atol=1e-6, rtol=0):
                raise InputError("Periodic QM/MM cannot unwrap a covalent network that winds around the cell")

        anchor_component = next(members for members in self.qm_members if self.qmatoms[0] in members)
        anchor = imaged[anchor_component].mean(axis=0)
        # Each QM-containing molecule is brought near the same anchor before the
        # overall QM center is defined. A bonded QM/MM boundary moves as one molecule.
        for component, qm_members in zip(self.components, self.qm_members, strict=True):
            if len(qm_members):
                delta = imaged[qm_members].mean(axis=0) - anchor
                nearest, _length = find_mic(delta, box)
                imaged[component] += nearest - delta
        center = imaged[self.qmatoms].mean(axis=0)
        mm_components = [
            component
            for component, qm_members in zip(self.components, self.qm_members, strict=True)
            if not len(qm_members)
        ]
        if mm_components:
            deltas = np.asarray([imaged[component].mean(axis=0) - center for component in mm_components])
            nearest, _lengths = find_mic(deltas, box)
            for component, shift in zip(mm_components, nearest - deltas, strict=True):
                imaged[component] += shift

        # Equivalent images of the anchor also produce identical QM input, up to
        # roundoff, rather than relying on the QM backend's translation invariance.
        imaged -= np.floor(imaged[self.qmatoms[0]] @ np.linalg.inv(box)) @ box
        return imaged
