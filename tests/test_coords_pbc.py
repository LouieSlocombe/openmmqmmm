import numpy as np
import pytest

from openmmqmmm.coords_pbc import (
    cart_coords_to_fract,
    cell_params_to_vectors,
    cell_vectors_to_params,
    cell_volume,
    write_cif_file,
    write_poscar_file,
    write_xsf_file,
)

# a, b, c, alpha, beta, gamma for a 10 x 20 x 30 box
ORTHORHOMBIC = [10.0, 20.0, 30.0, 90.0, 90.0, 90.0]


def test_orthorhombic_params_to_vectors():
    vectors = np.array(cell_params_to_vectors(ORTHORHOMBIC))

    assert np.allclose(vectors, np.diag([10.0, 20.0, 30.0]), atol=1e-9)


def test_params_and_vectors_roundtrip():
    triclinic = [8.0, 9.0, 10.0, 75.0, 85.0, 95.0]

    vectors = cell_params_to_vectors(triclinic)
    assert np.allclose(cell_vectors_to_params(vectors), triclinic, atol=1e-8)


def test_cell_volume_of_a_box():
    assert cell_volume(np.diag([10.0, 20.0, 30.0])) == pytest.approx(6000.0)


def test_cartesian_to_fractional():
    vectors = np.diag([10.0, 20.0, 30.0])

    fractional = cart_coords_to_fract(np.array([[5.0, 10.0, 15.0], [10.0, 20.0, 30.0]]), vectors)

    assert np.allclose(fractional[0], [0.5, 0.5, 0.5])
    assert np.allclose(fractional[1], [1.0, 1.0, 1.0])


@pytest.mark.parametrize("writer", [write_poscar_file, write_xsf_file, write_cif_file])
def test_structure_writers_produce_a_non_empty_file(tmp_path, writer):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    filename = str(tmp_path / f"structure_{writer.__name__}")

    writer(coords, ["Na", "Cl"], celldimensions=ORTHORHOMBIC, filename=filename)

    written = (tmp_path / f"structure_{writer.__name__}").read_text()
    assert "Na" in written
    assert "Cl" in written
    assert "10.0" in written or "10.00" in written
