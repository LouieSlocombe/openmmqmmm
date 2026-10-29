"""Select finite embedding fields without splitting connected charge groups."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from openmmqmmm.exceptions import InputError
from openmmqmmm.utils import require_int_in_range, require_positive_finite


def _indices(values: Sequence[int], size: int, name: str, *, allow_empty: bool = False) -> list[int]:
    """Validate indices without silently converting floats or Boolean values."""
    try:
        indices = list(values)
    except TypeError:
        raise InputError(f"{name} must be a sequence of integer indices") from None
    if not indices and not allow_empty:
        raise InputError(f"{name} must not be empty")
    if any(
        isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < size
        for index in indices
    ):
        raise InputError(f"{name} must contain integer indices between 0 and {size - 1}")
    if len(set(indices)) != len(indices):
        raise InputError(f"{name} must not contain duplicate indices")
    return [int(index) for index in indices]


def build_charge_groups(
    num_atoms: int,
    mm_atoms: Sequence[int],
    *,
    atom_groups: Sequence[Sequence[int]] = (),
    atom_links: Iterable[tuple[int, int]] = (),
    virtual_site_parents: Sequence[Sequence[int]] = (),
) -> list[np.ndarray]:
    """Map connected atoms and added sites into groups of point-charge rows.

    Atom indices refer to the full QM/MM system. Connectivity through QM atoms
    is retained, so removing QM rows cannot split the remaining MM molecule.
    The point-charge order is ``mm_atoms`` followed by added virtual sites.
    Every added site is joined to all its parents, preserving boundary dipoles
    and redistributed charges along with the host molecule.
    """
    num_atoms = require_int_in_range(num_atoms, "num_atoms must be a nonnegative integer", minimum=0)
    mm_indices = _indices(mm_atoms, num_atoms, "mm_atoms", allow_empty=True)
    site_parents = [_indices(parents, num_atoms, "virtual_site_parents") for parents in virtual_site_parents]
    parents = list(range(num_atoms + len(site_parents)))
    sizes = [1] * len(parents)

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first: int, second: int) -> None:
        first, second = root(first), root(second)
        if first == second:
            return
        if sizes[first] < sizes[second]:
            first, second = second, first
        parents[second] = first
        sizes[first] += sizes[second]

    for group in atom_groups:
        members = _indices(group, num_atoms, "atom_groups")
        for index in members[1:]:
            join(members[0], index)
    for link in atom_links:
        members = _indices(link, num_atoms, "atom_links")
        if len(members) != 2:
            raise InputError("Each atom_links entry must contain two atom indices")
        join(*members)
    for offset, members in enumerate(site_parents):
        for index in members:
            join(num_atoms + offset, index)

    groups: dict[int, list[int]] = {}
    for site_index, atom_index in enumerate([*mm_indices, *range(num_atoms, len(parents))]):
        groups.setdefault(root(atom_index), []).append(site_index)
    return [np.asarray(members, dtype=int) for members in groups.values()]


def _coordinates(values: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
    try:
        coords = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a finite N x 3 coordinate array") from None
    if coords.ndim != 2 or coords.shape[1] != 3 or (not len(coords) and not allow_empty):
        raise InputError(f"{name} must be a finite N x 3 coordinate array")
    if not np.all(np.isfinite(coords)):
        raise InputError(f"{name} must contain only finite coordinates")
    return coords


def select_charge_groups(
    coords: np.ndarray,
    qm_coords: np.ndarray,
    groups: Sequence[Sequence[int]],
    radius: float,
    *,
    selection: str = "nearest",
) -> np.ndarray:
    """Select every group with a site strictly inside the requested radius.

    ``nearest`` measures distance from the closest supplied QM coordinate;
    ``centroid`` measures distance from their arithmetic mean. The caller
    chooses whether the supplied coordinates include link atoms. Output rows
    retain their order in the full point-charge field. A spatial tree avoids
    allocating a point-charge by QM-atom pair-distance matrix.
    """
    coords = _coordinates(coords, "coords", allow_empty=True)
    qm_coords = _coordinates(qm_coords, "qm_coords")
    radius = require_positive_finite(radius, "radius must be a positive finite number")
    if selection not in ("nearest", "centroid"):
        raise InputError("selection must be 'nearest' or 'centroid'")

    partition = [np.asarray(_indices(group, len(coords), "groups"), dtype=int) for group in groups]
    flattened = np.concatenate(partition) if partition else np.empty(0, dtype=int)
    if len(flattened) != len(coords) or len(np.unique(flattened)) != len(coords):
        raise InputError("groups must partition every point-charge row exactly once")
    if not len(coords):
        return np.empty(0, dtype=int)

    if selection == "nearest":
        distances, _ = cKDTree(qm_coords).query(coords, k=1)
    else:
        distances = np.linalg.norm(coords - qm_coords.mean(axis=0), axis=1)
    inside = distances < radius
    selected = np.zeros(len(coords), dtype=bool)
    for group in partition:
        if np.any(inside[group]):
            selected[group] = True
    return np.flatnonzero(selected)
