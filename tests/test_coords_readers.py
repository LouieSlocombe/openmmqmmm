"""Coordinate-file readers in openmmqmmm.coords: GRO, Amber inpcrd, XYZ variants and the formula parser."""

from pathlib import Path

import numpy as np
import openmm.app
import openmm.unit
import parmed
import pytest

from openmmqmmm import Fragment
from openmmqmmm.coords import (
    _conv_atomtypes_elems,
    _formula_to_elem_list,
    _reformat_list_to_array,
    get_molecules_from_trajectory,
    read_ambercoordinates,
    read_gromacsfile,
    read_xyzfile,
    read_xyzfiles,
    reformat_element,
    split_multimolxyzfile,
)
from openmmqmmm.exceptions import FileFormatError, InputError

FIXTURES = Path(__file__).parent / "fixtures"
THREE_WATERS_GRO = str(FIXTURES / "three-waters.gro")
THREE_WATERS_PRMTOP = str(FIXTURES / "three-waters.prmtop")

WATER_COORDS = np.array([[0.0, 0.0, 0.0], [0.957, 0.0, 0.0], [-0.24, 0.927, 0.0]])


def _write_inpcrd(path, coords, box=None):
    """Write an Amber inpcrd: title, atom count, 6F12.7 coordinate columns, optional box line."""
    flat = np.asarray(coords, dtype=float).ravel()
    lines = ["hand-written inpcrd", f"{len(coords):5d}"]
    lines += ["".join(f"{value:12.7f}" for value in flat[i : i + 6]) for i in range(0, len(flat), 6)]
    if box is not None:
        lines.append("".join(f"{value:12.7f}" for value in box))
    Path(path).write_text("\n".join(lines) + "\n")


def _write_xyz(path, frames):
    """Write XYZ frames given as (title, elems, coords) triples."""
    text = ""
    for title, elems, coords in frames:
        text += f"{len(elems)}\n{title}\n"
        text += "".join(f"{el} {x} {y} {z}\n" for el, (x, y, z) in zip(elems, coords, strict=True))
    Path(path).write_text(text)


def test_read_gromacsfile_matches_openmm_positions_in_angstrom():
    elems, coords, box = read_gromacsfile(THREE_WATERS_GRO)

    reference = openmm.app.GromacsGroFile(THREE_WATERS_GRO)
    reference_positions = np.asarray(reference.positions.value_in_unit(openmm.unit.angstrom))
    reference_box = np.asarray(reference.getPeriodicBoxVectors().value_in_unit(openmm.unit.angstrom))
    assert elems == ["O", "H", "H"] * 3
    assert coords == pytest.approx(reference_positions)
    assert box[:3] == pytest.approx(np.diag(reference_box))
    assert box[3:] == [90.0, 90.0, 90.0]


def test_read_gromacsfile_ignores_velocity_columns_and_maps_atomtypes(tmp_path):
    grofile = tmp_path / "velocities.gro"
    grofile.write_text(
        "water with velocities\n"
        "    3\n"
        "    1SOL     OW    1   0.100   0.200   0.300  0.1000  0.2000  0.3000\n"
        "    1SOL    HW1    2   0.196   0.200   0.300 -0.1000  0.0000  0.0000\n"
        "    1SOL    HW2    3   0.076   0.293   0.300  0.0000  0.0000  0.0000\n"
        "   2.00000   2.00000   2.00000\n"
    )

    elems, coords, box = read_gromacsfile(str(grofile))

    reference = openmm.app.GromacsGroFile(str(grofile))
    assert elems == ["O", "H", "H"]
    assert coords == pytest.approx(np.array([[1.0, 2.0, 3.0], [1.96, 2.0, 3.0], [0.76, 2.93, 3.0]]))
    assert coords == pytest.approx(np.asarray(reference.positions.value_in_unit(openmm.unit.angstrom)))
    assert box == pytest.approx([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])


def test_fragment_from_grofile_takes_label_from_the_filename():
    fragment = Fragment(grofile=THREE_WATERS_GRO)

    assert fragment.label == "three-waters"
    assert fragment.numatoms == 9
    assert fragment.elems == ["O", "H", "H"] * 3
    assert fragment.coords[3] == pytest.approx([4.0, 0.0, 0.0])


def test_fragment_reports_a_missing_grofile_as_a_file_format_error():
    with pytest.raises(FileFormatError, match="not found"):
        Fragment(grofile="does-not-exist.gro")


@pytest.mark.parametrize(
    ("atomtype", "element"),
    [("OW", "O"), ("HW", "H"), ("CA", "C"), ("br", "Br")],
)
def test_conv_atomtypes_elems_maps_force_field_types_and_falls_back_to_element_symbols(atomtype, element):
    assert _conv_atomtypes_elems(atomtype) == element


def test_conv_atomtypes_elems_rejects_an_unknown_type():
    with pytest.raises(InputError, match="not recognized either as valid atomtype or element"):
        _conv_atomtypes_elems("Xq")


def test_read_ambercoordinates_matches_openmm_and_parmed_on_run_together_fields(tmp_path):
    # -100.0326085 fills its 12-column field, so it runs into the preceding value
    coords = np.vstack([WATER_COORDS, WATER_COORDS + np.array([-100.0326085, -16.3842161, 5.0]), WATER_COORDS + 10.0])
    box = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
    inpcrd = tmp_path / "waters.inpcrd"
    _write_inpcrd(inpcrd, coords, box)

    elems, read_coords, read_box = read_ambercoordinates(prmtopfile=THREE_WATERS_PRMTOP, inpcrdfile=str(inpcrd))

    openmm_positions = openmm.app.AmberInpcrdFile(str(inpcrd)).positions.value_in_unit(openmm.unit.angstrom)
    parmed_structure = parmed.load_file(str(inpcrd))
    assert elems == ["O", "H", "H"] * 3
    assert read_coords == pytest.approx(coords)
    assert read_coords == pytest.approx(np.asarray(openmm_positions))
    assert read_coords == pytest.approx(parmed_structure.coordinates[0])
    assert read_box == pytest.approx(box)
    assert read_box == pytest.approx(parmed_structure.box)


def test_read_ambercoordinates_without_box_line_returns_an_empty_box(tmp_path):
    inpcrd = tmp_path / "nobox.inpcrd"
    _write_inpcrd(inpcrd, np.vstack([WATER_COORDS, WATER_COORDS + 5.0, WATER_COORDS + 10.0]))

    _elems, coords, box = read_ambercoordinates(prmtopfile=THREE_WATERS_PRMTOP, inpcrdfile=str(inpcrd))

    assert len(coords) == 9
    assert box == []


def test_read_ambercoordinates_rejects_atom_count_mismatch_with_prmtop(tmp_path):
    inpcrd = tmp_path / "short.inpcrd"
    _write_inpcrd(inpcrd, WATER_COORDS)

    with pytest.raises(FileFormatError, match=r"Num coords \(3\) not equal to num elems \(9\)"):
        read_ambercoordinates(prmtopfile=THREE_WATERS_PRMTOP, inpcrdfile=str(inpcrd))


def test_fragment_from_amber_files_reads_elements_from_prmtop(tmp_path):
    coords = np.vstack([WATER_COORDS, WATER_COORDS + 5.0, WATER_COORDS + 10.0])
    inpcrd = tmp_path / "waters.rst7"
    _write_inpcrd(inpcrd, coords)

    fragment = Fragment(amber_inpcrdfile=str(inpcrd), amber_prmtopfile=THREE_WATERS_PRMTOP)

    assert fragment.label == "waters"
    assert fragment.elems == ["O", "H", "H"] * 3
    assert fragment.coords == pytest.approx(coords)
    assert fragment.nuccharge == 30


def test_fragment_requires_a_prmtop_with_an_inpcrd(tmp_path):
    with pytest.raises(InputError, match="amber_prmtopfile argument must be provided"):
        Fragment(amber_inpcrdfile=str(tmp_path / "x.inpcrd"))


def test_fragment_reports_missing_amber_files_as_a_file_format_error(tmp_path):
    with pytest.raises(FileFormatError, match="not found"):
        Fragment(amber_inpcrdfile=str(tmp_path / "missing.inpcrd"), amber_prmtopfile=THREE_WATERS_PRMTOP)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "Invalid XYZ-file"),
        ("3\ntitle\nH 0 0 0\nH 0 0 1\n", "does not match header line"),
        ("1\ntitle\nH 0 0 abc\n", "Invalid XYZ-file"),
    ],
)
def test_read_xyzfile_rejects_malformed_files(tmp_path, content, message):
    target = tmp_path / "bad.xyz"
    target.write_text(content)

    with pytest.raises(FileFormatError, match=message):
        read_xyzfile(str(target))


def test_fragment_rejects_a_missing_xyzfile():
    with pytest.raises(InputError, match=r"XYZ-file missing\.xyz not found"):
        Fragment(xyzfile="missing.xyz")


def test_fragment_rejects_a_non_integer_charge_mult_header(tmp_path):
    target = tmp_path / "header.xyz"
    target.write_text("1\nenergy -1.5\nH 0 0 0\n")

    with pytest.raises(FileFormatError, match="does not have a valid charge/mult"):
        Fragment(xyzfile=str(target), readchargemult=True)


def test_read_xyzfiles_sorts_naturally_and_labels_by_filename(tmp_path):
    for name, element, charge in [("mol10", "C", 2), ("mol2", "He", 1), ("mol1", "H", 0)]:
        (tmp_path / f"{name}.xyz").write_text(f"1\n{charge} 1\n{element} 0 0 0\n")
    (tmp_path / "notes.txt").write_text("ignored")

    fragments = read_xyzfiles(str(tmp_path), readchargemult=True)

    assert [fragment.label for fragment in fragments] == ["mol1.xyz", "mol2.xyz", "mol10.xyz"]
    assert [fragment.elems for fragment in fragments] == [["H"], ["He"], ["C"]]
    assert [(fragment.charge, fragment.mult) for fragment in fragments] == [(0, 1), (1, 1), (2, 1)]


def test_split_multimolxyzfile_skips_blank_lines_and_names_empty_titles_na(tmp_path):
    target = tmp_path / "frames.xyz"
    target.write_text("1\n\nH 0 0 0\n\n\n1\nsecond frame\nHe 1 1 1\n")

    elems, coords, titles = split_multimolxyzfile(str(target))

    assert elems == [["H"], ["He"]]
    assert coords == [[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]]
    assert titles == ["NA", ["second", "frame"]]


def test_split_multimolxyzfile_keeps_every_nth_frame(tmp_path):
    target = tmp_path / "seven.xyz"
    _write_xyz(target, [(f"frame {i}", ["H"], [[float(i), 0.0, 0.0]]) for i in range(7)])

    _elems, coords, titles = split_multimolxyzfile(str(target), skipindex=3)

    assert coords == [[[0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[6.0, 0.0, 0.0]]]
    assert titles == [["frame", "0"], ["frame", "3"], ["frame", "6"]]


@pytest.mark.parametrize("skipindex", [0, -1, 1.5, True])
def test_split_multimolxyzfile_rejects_non_positive_skipindex(tmp_path, skipindex):
    target = tmp_path / "one.xyz"
    _write_xyz(target, [("t", ["H"], [[0.0, 0.0, 0.0]])])

    with pytest.raises(InputError, match="skipindex must be a positive integer"):
        split_multimolxyzfile(str(target), skipindex=skipindex)


@pytest.mark.parametrize(
    ("content", "frame"),
    [
        ("-1\ntitle\n", 1),
        ("1\nfirst\nH 0 0 0\n2\nsecond\nH 0 0 0\n", 2),
        ("1\nfirst\nH 0 0 0\nabc\n", 2),
    ],
)
def test_split_multimolxyzfile_names_the_broken_frame(tmp_path, content, frame):
    target = tmp_path / "broken.xyz"
    target.write_text(content)

    with pytest.raises(FileFormatError, match=f"Invalid XYZ frame {frame} in"):
        split_multimolxyzfile(str(target))


def test_get_molecules_from_trajectory_builds_labelled_fragments_with_connectivity(tmp_path):
    target = tmp_path / "traj.xyz"
    frames = [(f"step {i}", ["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.6 + 0.01 * i]]) for i in range(4)]
    _write_xyz(target, frames)

    molecules = get_molecules_from_trajectory(str(target), skipindex=2, conncalc=True)

    assert [molecule.label for molecule in molecules] == [f"{target}_0", f"{target}_1"]
    assert [molecule.coords[1, 2] for molecule in molecules] == pytest.approx([0.6, 0.62])
    assert all(molecule.connectivity == [[0, 1]] for molecule in molecules)
    assert all(molecule.elems == ["H", "H"] for molecule in molecules)


def test_get_molecules_from_trajectory_can_write_each_frame(tmp_path):
    target = tmp_path / "traj.xyz"
    _write_xyz(target, [("a", ["He"], [[1.0, 2.0, 3.0]]), ("b", ["Ne"], [[4.0, 5.0, 6.0]])])

    molecules = get_molecules_from_trajectory(str(target), writexyz=True)

    assert len(molecules) == 2
    assert read_xyzfile("molecule2.xyz") == (["Ne"], [[4.0, 5.0, 6.0]])


@pytest.mark.parametrize(
    ("formula", "elements"),
    [
        ("H2O", ["H", "H", "O"]),
        ("CH4", ["C", "H", "H", "H", "H"]),
        ("C6H12O6", ["C"] * 6 + ["H"] * 12 + ["O"] * 6),
        ("NaCl", ["Na", "Cl"]),
        ("Fe2O3", ["Fe", "Fe", "O", "O", "O"]),
        ("HCl", ["H", "Cl"]),
        ("Cl2", ["Cl", "Cl"]),
        ("C10H22", ["C"] * 10 + ["H"] * 22),
        ("He", ["He"]),
        ("H", ["H"]),
    ],
)
def test_formula_to_elem_list_expands_counts_in_formula_order(formula, elements):
    assert _formula_to_elem_list(formula) == elements


@pytest.mark.parametrize("formula", ["", "h2o", "12"])
def test_formula_to_elem_list_yields_nothing_without_an_uppercase_symbol(formula):
    assert _formula_to_elem_list(formula) == []


@pytest.mark.parametrize(
    ("elem", "isatomnum"),
    [("Xx", False), ("", False), (999, True)],
)
def test_reformat_element_rejects_unknown_elements(elem, isatomnum):
    with pytest.raises(InputError, match="is not a valid element"):
        reformat_element(elem, isatomnum=isatomnum)


def test_reformat_element_normalises_case_and_atomic_numbers():
    assert reformat_element("cl") == "Cl"
    assert reformat_element("FE") == "Fe"
    assert reformat_element(26, isatomnum=True) == "Fe"


def test_reformat_list_to_array_rejects_flat_lists_and_other_containers():
    with pytest.raises(InputError, match="list of lists, not just a list"):
        _reformat_list_to_array([0.0, 0.0, 0.0])
    with pytest.raises(InputError, match="got tuple"):
        _reformat_list_to_array(([0.0, 0.0, 0.0],))
