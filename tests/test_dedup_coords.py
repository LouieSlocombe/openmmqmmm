"""Regressions and format contracts for the issue #68 geometry consolidation."""

import io
import logging

import numpy as np
import pytest

from openmmqmmm import Fragment
from openmmqmmm.coords import (
    _print_internal_coordinate_table,
    _write_coords_lines,
    angle,
    dihedral,
    read_xyzfile,
    split_multimolxyzfile,
    write_xyz_for_atoms,
)


def test_connectivity_has_no_breadth_first_depth_limit():
    coords = np.zeros((205, 3))
    coords[:, 0] = np.arange(205) * 1.4
    fragment = Fragment(elems=["C"] * len(coords), coords=coords)
    fragment.calc_connectivity(conndepth=1)
    assert fragment.connectivity == [list(range(205))]


def test_collinear_angles_remain_finite_despite_roundoff():
    rng = np.random.default_rng(68)
    vectors = rng.normal(size=(1000, 3))
    with np.errstate(invalid="ignore"):
        values = [angle(v, np.zeros(3), -v) for v in vectors]
    np.testing.assert_allclose(values, 180.0, atol=2e-6)


def test_optimization_table_uses_canonical_signed_dihedral(caplog):
    fragment = Fragment(elems=["C"] * 4, coords=[[0, 0, 0], [1.4, 0, 0], [1.4, 1.4, 0], [2, 1.4, 1.2]])
    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        _print_internal_coordinate_table(fragment)
    rows = [record.message for record in caplog.records if record.message.startswith("Dihedral")]
    assert rows
    assert float(rows[0].split()[-1].rstrip("°")) == pytest.approx(dihedral(*fragment.coords), abs=0.005)


@pytest.mark.parametrize("indices", [None, [12]])
@pytest.mark.parametrize("labels,labels2", [(None, None), (["A"], None), (["A"], ["B"])])
def test_coordinate_line_formats(indices, labels, labels2):
    output = io.StringIO()
    _write_coords_lines(output, np.array([[1.0, -2.0, 3.0]]), ["H"], indices, labels, labels2, "golden")
    expected = "   H   1.00000000   -2.00000000    3.00000000"
    if indices is not None:
        expected = "12 " + expected
    if labels is not None:
        expected += "      A"
    if labels2 is not None:
        expected += "      B"
    assert output.getvalue() == "#golden\n" + expected + "\n"


def test_xyz_subset_writers_preserve_filename_and_format(tmp_path):
    fragment = Fragment(elems=["H", "He"], coords=[[0, 0, 0], [1, 2, 3]])
    target = tmp_path / "exact.name"
    fragment.write_xyz_for_atoms(str(target), atoms=[1])
    write_xyz_for_atoms(fragment.coords, fragment.elems, [1], str(tmp_path / "module"))
    assert (
        target.read_text()
        == (tmp_path / "module.xyz").read_text()
        == ("1\ntitle\nHe       1.000000     2.000000     3.000000\n")
    )


def test_xyz_readers_accept_atomic_numbers_and_trailing_whitespace(tmp_path):
    target = tmp_path / "atoms.xyz"
    target.write_text("2\n0 1\n1 0 0 0\n8 1 2 3\n     \n")
    elems, coords = read_xyzfile(str(target))
    fragment = Fragment(xyzfile=str(target), readchargemult=True)
    assert fragment.elems == elems == ["H", "O"]
    np.testing.assert_array_equal(fragment.coords, coords)
    assert (fragment.charge, fragment.mult) == (0, 1)


def test_multiframe_xyz_titles_do_not_start_new_frames(tmp_path):
    target = tmp_path / "frames.xyz"
    target.write_text("2\n2 atoms in frame one\nH 0 0 0\nH 1 0 0\n2\nsecond\n1 0 0 1\n1 1 0 1\n")
    elems, coords, titles = split_multimolxyzfile(str(target), writexyz=True)
    assert elems == [["H", "H"], ["H", "H"]]
    assert titles == [["2", "atoms", "in", "frame", "one"], ["second"]]
    assert coords[1] == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]
    assert read_xyzfile("molecule1.xyz")[0] == ["H", "H"]
    assert len(split_multimolxyzfile(str(target), skipindex=2, return_fragments=True)) == 1
