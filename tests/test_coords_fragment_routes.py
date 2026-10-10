"""Fragment construction routes, attribute refresh, PDB/frag writers and the coordinate printers in coords."""

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm.app
import openmm.unit
import pytest
import rdkit.Chem  # noqa: F401 - rdkit's compiled modules segfault when loaded after the PyPI openbabel wheel

from openmmqmmm import Fragment
from openmmqmmm.coords import (
    print_coords_all,
    print_coords_for_atoms,
    print_internal_coordinate_table,
    write_coords_all,
    write_pdbfile,
)
from openmmqmmm.exceptions import FileFormatError, InputError, InternalError

TEST_DIR = Path(__file__).parent
WATER_ELEMS = ["O", "H", "H"]
WATER_COORDS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.26, 0.95, 0.0]]
HF_COORDS = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]


def _water():
    return Fragment(coords=WATER_COORDS, elems=WATER_ELEMS)


def test_fragment_coords_route_requires_matching_elems():
    with pytest.raises(InputError, match="Coords list provided but no elems list"):
        Fragment(coords=HF_COORDS)
    with pytest.raises(InputError, match=r"Coords list \(len 2\) and elems list \(1\) have different lengths"):
        Fragment(coords=HF_COORDS, elems=["H"])


@pytest.mark.parametrize(
    ("charges", "mults", "expected_charge", "expected_mult"),
    [((1, -1), (2, 2), 0, 3), ((0, 0), (1, 3), 0, 3), ((1, 0), (2, 3), 1, 4), ((0, 0), (1, 1), 0, 1)],
)
def test_fragment_from_fragments_concatenates_atoms_and_couples_spins_high_spin(
    charges, mults, expected_charge, expected_mult
):
    first = Fragment(coords=HF_COORDS, elems=["H", "F"], charge=charges[0], mult=mults[0])
    second = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS, charge=charges[1], mult=mults[1])

    combined = Fragment(fragments=[first, second])

    assert combined.elems == ["H", "F", "O", "H", "H"]
    assert combined.coords == pytest.approx(np.array(HF_COORDS + WATER_COORDS))
    assert combined.numatoms == 5
    assert (combined.charge, combined.mult) == (expected_charge, expected_mult)


def test_fragment_from_fragments_without_charges_leaves_them_unset(caplog):
    first = Fragment(coords=HF_COORDS, elems=["H", "F"])
    second = _water()

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        combined = Fragment(fragments=[first, second])

    assert (combined.charge, combined.mult) == (None, None)
    assert "Charges/multiplicities not found in inputfragments" in caplog.text


def test_fragment_from_fragments_prefers_an_explicit_charge_and_mult():
    first = Fragment(coords=HF_COORDS, elems=["H", "F"], charge=1, mult=2)
    second = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS, charge=1, mult=2)

    combined = Fragment(fragments=[first, second], charge=-1, mult=1, label="pair")

    assert (combined.charge, combined.mult, combined.label) == (-1, 1, "pair")


def test_fragment_atom_route_places_a_single_atom_at_the_origin():
    atom = Fragment(atom="He", charge=0, mult=1)

    assert atom.elems == ["He"]
    assert atom.coords == pytest.approx(np.zeros((1, 3)))
    assert atom.numatoms == 1
    assert atom.nuccharge == 2
    assert atom.formula == "He1"


@pytest.mark.parametrize("kwargs", [{"bondlength": 0.92}, {"diatomic_bondlength": 0.92}])
def test_fragment_diatomic_route_puts_the_bond_along_z(kwargs):
    diatomic = Fragment(diatomic="HF", **kwargs)

    assert diatomic.elems == ["H", "F"]
    assert diatomic.coords == pytest.approx(np.array(HF_COORDS))
    assert np.linalg.norm(diatomic.coords[1] - diatomic.coords[0]) == pytest.approx(0.92)


def test_fragment_diatomic_route_rejects_missing_bondlength_and_non_diatomic_formulas():
    with pytest.raises(InputError, match="diatomic option requires bondlength"):
        Fragment(diatomic="HF")
    with pytest.raises(InputError, match="Problem with molecular formula diatomic=H2O"):
        Fragment(diatomic="H2O", bondlength=1.0)


def test_fragment_smiles_route_builds_water_with_bonded_hydrogens():
    water = Fragment(smiles="O")

    assert water.elems == ["O", "H", "H"]
    assert water.coords.shape == (3, 3)
    for hydrogen in (1, 2):
        assert 0.9 < np.linalg.norm(water.coords[hydrogen] - water.coords[0]) < 1.1


def test_fragment_without_any_coordinate_source_is_rejected():
    with pytest.raises(InputError, match="requires some kind of valid coordinate input"):
        Fragment(charge=0, mult=1)


def test_fragment_label_argument_overrides_the_file_stem():
    fragment = Fragment(xyzfile=str(TEST_DIR / "xyzfiles" / "hi.xyz"), label="custom")

    assert fragment.label == "custom"
    assert Fragment(xyzfile=str(TEST_DIR / "xyzfiles" / "hi.xyz")).label == "hi"


def test_fragment_coordsstring_route_can_compute_connectivity():
    fragment = Fragment(coordsstring="H 0 0 0\nH 0 0 0.7\nHe 0 0 5", conncalc=True)

    assert fragment.label == "HHHe"
    assert fragment.connectivity == [[0, 1], [2]]


def test_update_attributes_rejects_empty_or_non_array_coordinates():
    with pytest.raises(InputError, match="No coordinates in fragment"):
        Fragment(coords=np.zeros((0, 3)), elems=[])

    fragment = _water()
    fragment.coords = WATER_COORDS
    with pytest.raises(InputError, match="not a numpy array"):
        fragment.update_attributes()


def test_update_attributes_pads_short_per_atom_lists_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.coords"):
        fragment = Fragment(coords=WATER_COORDS, elems=WATER_ELEMS, atomcharges=[-0.8], atomtypes=["OW"])

    assert fragment.atomcharges == [-0.8, 0.0, 0.0]
    assert fragment.atomtypes == ["OW", "None", "None"]
    assert "atomcharges list shorter than number of atoms" in caplog.text
    assert "atomtypes list shorter than number of atoms" in caplog.text

    fragment.fragmenttype_labels = [1]
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.coords"):
        fragment.update_attributes()
    assert len(fragment.fragmenttype_labels) == 3
    assert fragment.fragmenttype_labels[0] == 1
    assert "fragmenttype_labels list shorter than number of atoms" in caplog.text


def test_info_logs_the_fragment_attributes(caplog):
    fragment = _water()

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        fragment.info()

    assert "Fragment object" in caplog.text
    assert "'elems': ['O', 'H', 'H']" in caplog.text


def test_add_coords_from_string_can_recompute_connectivity():
    fragment = Fragment(coords=[[0.0, 0.0, 0.0]], elems=["H"])

    fragment.add_coords_from_string("H 0 0 0.7\nHe 0 0 5", conncalc=True)

    assert fragment.elems == ["H", "H", "He"]
    assert fragment.connectivity == [[0, 1], [2]]


def test_non_hydrogen_indices_and_centroid():
    fragment = Fragment(coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], elems=list("HCHO"))

    assert fragment.get_non_h_atomindices() == [1, 3]
    assert fragment.get_centroid() == pytest.approx([0.25, 0.5, 0.75])
    assert fragment.get_coordinate_center() == pytest.approx([0.25, 0.5, 0.75])


def test_fragment_write_pdbfile_without_pdb_names_uses_default_names(tmp_path, caplog):
    fragment = _water()

    with caplog.at_level(logging.WARNING, logger="openmmqmmm.coords"):
        path = fragment.write_pdbfile(str(tmp_path / "water.pdb"))

    assert path == str(tmp_path / "water.pdb")
    assert "holds no PDB atom/residue/chain names" in caplog.text
    pdb = openmm.app.PDBFile(path)
    assert [atom.element.symbol for atom in pdb.topology.atoms()] == WATER_ELEMS
    assert [atom.name for atom in pdb.topology.atoms()] == ["O1", "H2", "H3"]
    assert [residue.name for residue in pdb.topology.residues()] == ["DUM"]
    positions = np.asarray(pdb.positions.value_in_unit(openmm.unit.angstrom))
    assert positions == pytest.approx(np.array(WATER_COORDS), abs=1e-3)


def test_fragment_write_pdbfile_uses_stored_pdb_names(tmp_path, caplog):
    fragment = _water()
    fragment.pdb_atomnames = ["OW", "HW1", "HW2"]
    fragment.pdb_resnames = ["SOL"] * 3
    fragment.pdb_chainlabels = ["A"] * 3
    fragment.pdb_residlabels = [7] * 3
    fragment.pdb_conect_lines = ["CONECT    1    2    3\n"]

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        path = fragment.write_pdbfile(str(tmp_path / "named"))

    assert "Found PDB residue/atom/segment information" in caplog.text
    lines = Path(path).read_text().splitlines()
    atom_lines = [line for line in lines if line.startswith("ATOM")]
    assert [line[12:16].strip() for line in atom_lines] == ["OW1", "HW1", "HW2"]
    assert [line[17:20] for line in atom_lines] == ["SOL"] * 3
    assert [line[21] for line in atom_lines] == ["A"] * 3
    assert [line[22:26].strip() for line in atom_lines] == ["7"] * 3
    assert lines[-1] == "CONECT    1    2    3"


def test_write_pdbfile_takes_names_from_an_openmm_theory_like_object(tmp_path):
    fragment = _water()
    mm = SimpleNamespace(atomnames=["OH2", "H1", "H2"], resnames=["TIP3"] * 3, resids=[4] * 3, segmentnames=["WT1"] * 3)

    path = write_pdbfile(fragment, outputname=str(tmp_path / "mm"), openmmobject=mm)

    atom_lines = [line for line in Path(path).read_text().splitlines() if line.startswith("ATOM")]
    assert [line[12:16].strip() for line in atom_lines] == ["OH2", "H1", "H2"]
    assert [line[17:20] for line in atom_lines] == ["TIP"] * 3
    assert [line[22:26].strip() for line in atom_lines] == ["4"] * 3
    assert [line[72:76].strip() for line in atom_lines] == ["WT1"] * 3


def test_write_pdbfile_switches_to_hexadecimal_serials_at_one_hundred_thousand(tmp_path):
    count = 100001
    fragment = Fragment(coords=np.zeros((count, 3)), elems=["C"] * count)

    path = write_pdbfile(fragment, outputname=str(tmp_path / "big"))

    with open(path) as handle:
        serials = [line[6:11] for line in handle]
    assert serials[99998] == "99999"
    assert serials[99999] == f"{100000:x}"
    assert serials[100000] == f"{100001:x}"


def test_write_pdbfile_openmm_reuses_a_given_or_stored_topology(tmp_path, caplog):
    fragment = _water()
    topology = openmm.app.Topology()
    residue = topology.addResidue("WAT", topology.addChain())
    for name, element in [("OW", openmm.app.element.oxygen), ("HW1", openmm.app.element.hydrogen)]:
        topology.addAtom(name, element, residue)
    topology.addAtom("HW2", openmm.app.element.hydrogen, residue)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        given = fragment.write_pdbfile_openmm(filename=str(tmp_path / "given"), pdb_topology=topology)
        stored = fragment.write_pdbfile_openmm(filename=str(tmp_path / "stored.pdb"))

    assert "Using input pdb_topology" in caplog.text
    assert "Using pdbtopology found in fragment" in caplog.text
    assert fragment.pdb_topology is topology
    for path in (given, stored):
        atom_lines = [line for line in Path(path).read_text().splitlines() if line.startswith(("ATOM", "HETATM"))]
        assert [line[12:16].strip() for line in atom_lines] == ["OW", "HW1", "HW2"]
        assert [line[17:20] for line in atom_lines] == ["WAT"] * 3


def test_write_pdbfile_openmm_skip_connectivity_drops_conect_records(tmp_path):
    fragment = _water()
    fragment.define_topology()
    assert fragment.pdb_topology.getNumBonds() == 2

    path = fragment.write_pdbfile_openmm(filename=str(tmp_path / "nobonds"), skip_connectivity=True)

    assert fragment.pdb_topology.getNumBonds() == 0
    assert not any(line.startswith("CONECT") for line in Path(path).read_text().splitlines())


def test_print_system_and_read_fragment_from_file_round_trip_every_field(tmp_path):
    fragment = Fragment(
        coords=WATER_COORDS,
        elems=WATER_ELEMS,
        charge=-1,
        mult=2,
        atomcharges=[-0.834, 0.417, 0.417],
        atomtypes=["OW", "HW", "HW"],
        connectivity=[[0, 1, 2]],
    )
    fragment.fragmenttype_labels = [3, 3, 3]
    fragment.Centralmainfrag = [0, 2]
    fragment.set_energy(-76.4)
    path = str(tmp_path / "water.frag")

    fragment.print_system(path)
    restored = Fragment(fragfile=path)

    assert restored.label == "water"
    assert restored.elems == WATER_ELEMS
    assert restored.coords == pytest.approx(np.array(WATER_COORDS))
    assert (restored.charge, restored.mult) == (-1, 2)
    assert restored.atomcharges == pytest.approx([-0.834, 0.417, 0.417])
    assert restored.atomtypes == ["OW", "HW", "HW"]
    assert restored.fragmenttype_labels == [3, 3, 3]
    assert restored.connectivity == [[0, 1, 2]]
    assert restored.Centralmainfrag == [0, 2]
    text = Path(path).read_text()
    assert "charge : -1\n" in text
    assert "mult : 2\n" in text
    assert "Energy: -76.4\n" in text


def test_print_system_rejects_per_atom_lists_of_different_lengths(tmp_path):
    fragment = _water()
    fragment.atomtypes = ["OW"]

    with pytest.raises(InternalError, match="This should not have happened"):
        fragment.print_system(str(tmp_path / "broken.frag"))
    assert not (tmp_path / "broken.frag").exists()


def test_read_fragment_from_file_rejects_other_text_files(tmp_path):
    path = tmp_path / "not.frag"
    path.write_text("3\ntitle\nO 0 0 0\n")

    with pytest.raises(FileFormatError, match="not a valid fragment file"):
        Fragment(fragfile=str(path))


BOND_ROW = re.compile(r"Bond:\s+(\d+)([A-Z][a-z]?)\s+-\s+(\d+)([A-Z][a-z]?)\s+(\d+\.\d{3})$")


def _bond_rows(messages):
    rows = {}
    for message in messages:
        if match := BOND_ROW.match(message):
            index_a, elem_a, index_b, elem_b, value = match.groups()
            rows[(int(index_a), int(index_b))] = (elem_a, elem_b, float(value))
    return rows


def test_print_internal_coordinate_table_reports_each_bond_length_once(caplog):
    fragment = _water()
    coords = np.array(WATER_COORDS)

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        print_internal_coordinate_table(fragment)

    rows = _bond_rows(caplog.messages)
    assert set(rows) == {(0, 1), (0, 2)}
    assert rows[(0, 1)][:2] == ("O", "H")
    assert rows[(0, 1)][2] == pytest.approx(np.linalg.norm(coords[1] - coords[0]), abs=5e-4)
    assert rows[(0, 2)][2] == pytest.approx(np.linalg.norm(coords[2] - coords[0]), abs=5e-4)


def test_print_internal_coordinate_table_uses_full_system_indices_for_actatoms(caplog):
    shifted = [[x + 5.0, y, z] for x, y, z in WATER_COORDS]
    fragment = Fragment(coords=HF_COORDS + shifted, elems=["H", "F", *WATER_ELEMS])

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        print_internal_coordinate_table(fragment, actatoms=[2, 3, 4])

    rows = _bond_rows(caplog.messages)
    assert set(rows) == {(2, 3), (2, 4)}
    assert rows[(2, 4)][:2] == ("O", "H")
    assert rows[(2, 4)][2] == pytest.approx(np.linalg.norm(np.subtract(shifted[2], shifted[0])), abs=5e-4)


def test_print_coords_for_atoms_rejects_mismatched_labels(caplog):
    fragment = _water()

    with pytest.raises(InputError, match="Length of Labels"):
        fragment.print_coords_for_atoms([0, 1], labels=["a"])

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        print_coords_for_atoms(fragment.coords, fragment.elems, [2, 0], labels=["x", "y"])
    assert caplog.messages[-2].split() == ["x", "H", "-0.26000000", "0.95000000", "0.00000000"]
    assert caplog.messages[-1].split() == ["y", "O", "0.00000000", "0.00000000", "0.00000000"]


def test_print_coords_all_and_write_coords_all_share_one_line_format(tmp_path, caplog):
    coords = np.array([[1.0, -2.0, 3.0], [0.5, 0.5, 0.5]])
    elems = ["H", "He"]
    expected = ["   H   1.00000000   -2.00000000    3.00000000", "  He   0.50000000    0.50000000    0.50000000"]

    with caplog.at_level(logging.INFO, logger="openmmqmmm.coords"):
        print_coords_all(coords, elems)
    assert caplog.messages[-2:] == expected

    write_coords_all(coords, elems, file=str(tmp_path / "out.txt"), description="two atoms")
    assert (tmp_path / "out.txt").read_text() == "#two atoms\n" + "\n".join(expected) + "\n"
