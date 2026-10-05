from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np

from openmmqmmm.exceptions import (
    InputError,
)

logger = logging.getLogger(__name__)


def cell_params_to_vectors(parameters: Sequence[float]) -> np.ndarray:
    a, b, c, alpha, beta, gamma = parameters
    rad_a = np.radians(alpha)
    rad_b = np.radians(beta)
    rad_g = np.radians(gamma)

    ax = a
    ay = 0.0
    az = 0.0

    bx = b * np.cos(rad_g)
    by = b * np.sin(rad_g)
    bz = 0.0

    cx = c * np.cos(rad_b)
    cy = c * (np.cos(rad_a) - np.cos(rad_b) * np.cos(rad_g)) / np.sin(rad_g)
    cz = np.sqrt(c**2 - cx**2 - cy**2)

    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]])


def cell_vectors_to_params(vectors: np.ndarray) -> list[float]:
    va, vb, vc = vectors[0], vectors[1], vectors[2]

    a = np.linalg.norm(va)
    b = np.linalg.norm(vb)
    c = np.linalg.norm(vc)

    alpha_rad = np.arccos(np.dot(vb, vc) / (b * c))
    beta_rad = np.arccos(np.dot(va, vc) / (a * c))
    gamma_rad = np.arccos(np.dot(va, vb) / (a * b))

    alpha = np.degrees(alpha_rad)
    beta = np.degrees(beta_rad)
    gamma = np.degrees(gamma_rad)

    return [float(a), float(b), float(c), float(alpha), float(beta), float(gamma)]


def cart_coords_to_fract(cart_coords: np.ndarray, cellvectors: np.ndarray) -> np.ndarray:
    M = np.array(cellvectors)
    return np.dot(cart_coords, np.linalg.inv(M))


def cell_volume(vectors: np.ndarray) -> float:
    a = vectors[0, :]
    b = vectors[1, :]
    c = vectors[2, :]
    return abs(np.dot(a, np.cross(b, c)))


def write_poscar_file(
    coords: np.ndarray,
    elems: Sequence[str],
    cellvectors: np.ndarray | None = None,
    celldimensions: Sequence[float] | None = None,
    filename: str | os.PathLike[str] = "POSCAR",
) -> str | os.PathLike[str]:
    if cellvectors is None and celldimensions is None:
        raise InputError("Either cellvectors or celldimensions should be provided")
    if celldimensions is not None:
        cellvectors = cell_params_to_vectors(celldimensions)

    unique_elements = []
    for e in elems:
        if e not in unique_elements:
            unique_elements.append(e)
    counts = [elems.count(e) for e in unique_elements]

    with open(filename, "w") as f:
        f.write("openmmqmmm created POSCAR file" + "\n")
        f.write("1.0" + "\n")
        f.write(f"{cellvectors[0, 0]:.4f} {cellvectors[0, 1]:.4f} {cellvectors[0, 2]:.4f} " + "\n")
        f.write(f"{cellvectors[1, 0]:.4f} {cellvectors[1, 1]:.4f} {cellvectors[1, 2]:.4f}" + "\n")
        f.write(f"{cellvectors[2, 0]:.4f} {cellvectors[2, 1]:.4f} {cellvectors[2, 2]:.4f}" + "\n")
        f.write(f"{'  '.join(unique_elements)}\n")
        f.write(f"{'  '.join(map(str, counts))}\n")
        f.write("Cartesian" + "\n")
        for target_el in unique_elements:
            for el, c in zip(elems, coords, strict=False):
                if el == target_el:
                    f.write(f"{c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}\n")
    logger.info("Wrote POSCAR file")
    return filename


def write_xsf_file(
    coords: np.ndarray,
    elems: Sequence[str],
    cellvectors: np.ndarray | None = None,
    celldimensions: Sequence[float] | None = None,
    filename: str | os.PathLike[str] = "structure.xsf",
) -> str | os.PathLike[str]:
    if cellvectors is None and celldimensions is None:
        raise InputError("Either cellvectors or celldimensions should be provided")
    if celldimensions is not None:
        cellvectors = cell_params_to_vectors(celldimensions)

    with open(filename, "w") as f:
        f.write("CRYSTAL\n")

        f.write("PRIMVEC\n")
        f.writelines(
            f"  {cellvectors[i, 0]:.10f}  {cellvectors[i, 1]:.10f}  {cellvectors[i, 2]:.10f}\n" for i in range(3)
        )

        f.write("PRIMCOORD\n")
        f.write(f"{len(elems)} 1\n")

        # XSF accepts element symbols as well as atomic numbers.
        f.writelines(f"{el}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n" for el, c in zip(elems, coords, strict=False))

    logger.info(f"Wrote XSF file: {filename}")
    return filename


def write_cif_file(
    coords: np.ndarray,
    elems: Sequence[str],
    cellvectors: np.ndarray | None = None,
    celldimensions: Sequence[float] | None = None,
    filename: str | os.PathLike[str] = "structure.cif",
) -> str | os.PathLike[str]:
    if cellvectors is None and celldimensions is None:
        raise InputError("Either cellvectors or celldimensions should be provided")
    if celldimensions is not None:
        cellvectors = cell_params_to_vectors(celldimensions)
    elif cellvectors is not None:
        celldimensions = cell_vectors_to_params(cellvectors)

    frac_coords = cart_coords_to_fract(coords, cellvectors)

    a, b, c, alpha, beta, gamma = celldimensions

    with open(filename, "w") as f:
        f.write("data_openmmqmmm_output\n")
        f.write(f"_cell_length_a    {a:.6f}\n")
        f.write(f"_cell_length_b    {b:.6f}\n")
        f.write(f"_cell_length_c    {c:.6f}\n")
        f.write(f"_cell_angle_alpha {alpha:.6f}\n")
        f.write(f"_cell_angle_beta  {beta:.6f}\n")
        f.write(f"_cell_angle_gamma {gamma:.6f}\n\n")

        f.write("_symmetry_space_group_name_H-M 'P 1'\n")
        f.write("_symmetry_Int_Tables_number 1\n\n")

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")

        for i, (el, c) in enumerate(zip(elems, frac_coords, strict=False)):
            # CIF atom-site labels must be unique.
            f.write(f"{el}{i + 1}  {el}  {c[0]:.8f}  {c[1]:.8f}  {c[2]:.8f}\n")

    logger.info(f"Wrote CIF file: {filename}")
    return filename


def align_to_standard_orientation(
    fragment_coords: np.ndarray, cell_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # QR factorizes columns, so work with the cell vectors as columns.
    H = cell_vectors.T

    # R is upper triangular, so R.T is the standard (lower-triangular) cell.
    Q, R = np.linalg.qr(H)

    # QR may return negative diagonals; flip signs so a_x, b_y and c_z are positive.
    d = np.sign(np.diag(R))
    d[d == 0] = 1

    Q = Q * d
    R = (R.T * d).T

    new_cell_vectors = R.T

    # R = Q.T @ H, so row-vector coordinates rotate as coords @ Q.
    new_coords = np.dot(fragment_coords, Q)

    return new_coords, new_cell_vectors
