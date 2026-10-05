from collections import defaultdict
from pathlib import Path

import numpy as np
import openmm.app
import openmm.unit
import pytest

from openmmqmmm import Fragment
from openmmqmmm.constants import BOHR_TO_ANG
from openmmqmmm.coords import write_pdbfile
from openmmqmmm.exceptions import InputError

TEST_DIR = Path(__file__).parent


def test_fragread():
    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag = Fragment(coordsstring=fragcoords)
    elems = ["H", "Cl"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]]
    HCl_frag = Fragment(elems=elems, coords=coords)
    elems2 = ["H", "Cl"]
    coords2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])
    HCl_frag_np = Fragment(elems=elems2, coords=coords2)
    HI_frag = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/hi.xyz")
    HF_frag2 = Fragment(coordsstring=fragcoords)
    elems = ["H", "Cl"]
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]
    HCl_frag.replace_coords(elems, coords)

    HCl_frag.calc_connectivity()
    print(HCl_frag.connectivity)

    assert HF_frag2.numatoms == 2, "Number of atoms is not correct"
    assert HI_frag.numatoms == 2, "Number of atoms is not correct"
    assert HF_frag.numatoms == 2, "Number of atoms is not correct"
    assert HCl_frag.numatoms == 2, "Number of atoms is not correct"
    assert HCl_frag_np.numatoms == 2, "Number of atoms is not correct"


def test_fragread_files():
    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag = Fragment(coordsstring=fragcoords)
    print("HF_frag conn", HF_frag.connectivity)
    HF_frag.print_system("HF_frag.frag")

    New_frag = Fragment(fragfile="HF_frag.frag")

    print("New_frag:", New_frag)
    print("New_frag dict:", New_frag.__dict__)

    assert New_frag.numatoms == 2, "Number of atoms is not correct"
    assert New_frag.nuccharge == 10, "Nuccharge of fragment is incorrect"


def test_read_pdb():
    PDB_frag = Fragment(pdbfile=f"{TEST_DIR}/pdbfiles/1aki.pdb", conncalc=False)
    print("PDB_frag:", PDB_frag)
    print(PDB_frag.numatoms)

    assert PDB_frag.numatoms == 1079, "Number of atoms in fragment is incorrect"


def test_add_coords_from_string_appends_and_invalidates_atom_dependent_caches():
    fragment = Fragment(coords=[[0.0, 0.0, 0.0]], elems=["H"], connectivity=[[0]])
    fragment.pdb_topology = object()
    fragment.pdb_atomnames = ["H1"]
    fragment.energy = -1.0
    fragment.hessian = np.eye(3)

    fragment.add_coords_from_string("F 0.0 0.0 1.0")

    assert fragment.elems == ["H", "F"]
    assert fragment.coords == pytest.approx(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert fragment.numatoms == 2
    assert len(fragment.atomcharges) == len(fragment.atomtypes) == len(fragment.fragmenttype_labels) == 2
    assert fragment.connectivity == []
    assert fragment.pdb_topology is None
    assert fragment.pdb_atomnames is None
    assert fragment.energy is None
    assert fragment.hessian is None


def test_replace_coords_preserves_identity_stable_connectivity_and_topology():
    fragment = Fragment(coords=[[0.0, 0.0, 0.0]], elems=["H"], connectivity=[[0]])
    topology = object()
    fragment.pdb_topology = topology
    fragment.energy = -1.0
    fragment.hessian = np.eye(3)

    fragment.replace_coords(["H"], [[1.0, 0.0, 0.0]])

    assert fragment.connectivity == [[0]]
    assert fragment.pdb_topology is topology, "Atom identity and ordering did not change"
    assert fragment.energy is None
    assert fragment.hessian is None

    fragment.replace_coords(["F"], [[1.0, 0.0, 0.0]])

    assert fragment.connectivity == []
    assert fragment.pdb_topology is None
    assert fragment.atomcharges == [0.0]
    assert fragment.atomtypes == ["None"]
    assert fragment.fragmenttype_labels == ["None"]


def test_replace_coords_recomputes_connectivity_only_when_requested():
    fragment = Fragment(
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]],
        elems=["H", "H"],
        connectivity=[[0, 1]],
    )
    fragment.pdb_topology = object()
    fragment.pdb_conect_lines = ["CONECT    1    2"]

    fragment.replace_coords(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]], conn=True)

    assert fragment.connectivity == [[0], [1]]
    assert fragment.pdb_topology is None
    assert fragment.pdb_conect_lines is None


def test_replace_coords_rejects_length_mismatch_without_mutating_fragment():
    fragment = Fragment(coords=[[0.0, 0.0, 0.0]], elems=["H"], connectivity=[[0]])

    with pytest.raises(InputError, match="different lengths"):
        fragment.replace_coords(["H", "F"], [[1.0, 0.0, 0.0]])

    assert fragment.elems == ["H"]
    assert fragment.coords == pytest.approx(np.array([[0.0, 0.0, 0.0]]))
    assert fragment.connectivity == [[0]]


def test_delete_atom_refreshes_metadata_and_invalidates_structural_caches():
    fragment = Fragment(
        coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        elems=["H", "F"],
        connectivity=[[0, 1]],
        atomcharges=[0.1, -0.1],
        atomtypes=["H1", "F1"],
    )
    fragment.pdb_topology = object()
    fragment.pdb_atomnames = ["H1", "F1"]
    fragment.energy = -1.0
    fragment.hessian = np.eye(6)
    fragment.Centralmainfrag = [0, 1]

    fragment.delete_atom(0)

    assert fragment.elems == ["F"]
    assert fragment.coords == pytest.approx(np.array([[0.0, 0.0, 1.0]]))
    assert fragment.atomcharges == [-0.1]
    assert fragment.atomtypes == ["F1"]
    assert fragment.connectivity == []
    assert fragment.pdb_topology is None
    assert fragment.pdb_atomnames is None
    assert fragment.energy is None
    assert fragment.hessian is None
    assert fragment.Centralmainfrag == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chainlabels": ["A"]},
        {"charges_column": ["0"]},
    ],
)
def test_write_pdbfile_rejects_short_per_atom_fields(tmp_path, kwargs):
    fragment = Fragment(coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], elems=["H", "F"])
    output = tmp_path / "bad"

    with pytest.raises(InputError, match="one entry per coordinate"):
        write_pdbfile(fragment, outputname=str(output), **kwargs)

    assert not output.with_suffix(".pdb").exists()


def test_write_pdbfile_rejects_element_coordinate_mismatch(tmp_path):
    fragment = Fragment(coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], elems=["H", "F"])
    fragment.elems.pop()

    with pytest.raises(InputError, match="elements=1"):
        write_pdbfile(fragment, outputname=str(tmp_path / "bad"))


# 1.2 A apart: bonded under tol=0.9 (threshold 1.52 A) or scale=2.0 (1.34 A), not under the defaults (0.72 A).
STRETCHED_H2 = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]
WATER_COORDS = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]


@pytest.mark.parametrize("kwargs", [{"tol": 0.9}, {"scale": 2.0}])
def test_calc_connectivity_defaults_scale_and_tol_independently(kwargs):
    fragment = Fragment(coords=STRETCHED_H2, elems=["H", "H"])

    fragment.calc_connectivity(**kwargs)

    assert fragment.connectivity == [[0, 1]]


def test_fragment_conncalc_honours_tol_given_without_scale():
    assert Fragment(coords=STRETCHED_H2, elems=["H", "H"], conncalc=True, tol=0.9).connectivity == [[0, 1]]
    assert Fragment(coords=STRETCHED_H2, elems=["H", "H"], conncalc=True).connectivity == [[0], [1]]


def test_define_topology_adds_each_bond_once():
    water = Fragment(coords=WATER_COORDS, elems=["O", "H", "H"])

    assert water.define_topology().getNumBonds() == 2


def test_write_pdbfile_openmm_lists_each_conect_partner_once(tmp_path):
    water = Fragment(coords=WATER_COORDS, elems=["O", "H", "H"])

    path = water.write_pdbfile_openmm(filename=str(tmp_path / "water"), calc_connectivity=True)

    assert water.pdb_topology.getNumBonds() == 2
    partners = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.startswith("CONECT"):
            atom, *bonded = line.split()[1:]
            partners[atom] += bonded
    assert {atom: sorted(bonded) for atom, bonded in partners.items()} == {"1": ["2", "3"], "2": ["1"], "3": ["1"]}


def test_write_pdbfile_round_trips_more_than_999_atoms_through_openmm(tmp_path):
    # Element-derived names overflow the 4-column field at serial 100 (Cl) and 1000 (C, N, O).
    elems = ["C"] * 1002
    elems[150] = "Cl"
    elems[1000] = "N"
    elems[1001] = "O"
    coords = 1.5 * np.array([[i % 10, (i // 10) % 10, i // 100] for i in range(len(elems))], dtype=float)
    fragment = Fragment(coords=coords, elems=elems)

    pdb = openmm.app.PDBFile(write_pdbfile(fragment, outputname=str(tmp_path / "large")))

    assert [atom.element.symbol for atom in pdb.topology.atoms()] == elems
    assert pdb.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom) == pytest.approx(coords, abs=1e-3)


def test_write_pdbfile_writes_segment_ids_in_columns_73_to_76(tmp_path):
    fragment = Fragment(coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], elems=["H", "F"])

    path = write_pdbfile(fragment, outputname=str(tmp_path / "seg"), segmentlabels=["SEGA", "SEGB"])

    atom_lines = [line for line in Path(path).read_text().splitlines() if line.startswith("ATOM")]
    assert [line[72:76] for line in atom_lines] == ["SEGA", "SEGB"]
    assert [line[76:78] for line in atom_lines] == [" H", " F"]


def test_read_chemshellfile_keeps_dots_in_directory_names(tmp_path):
    directory = tmp_path / "run.v2"
    directory.mkdir()
    path = directory / "frag.c"
    path.write_text("block = coordinates records 2\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4\nblock = connectivity records 0\n")

    fragment = Fragment(chemshellfile=str(path))

    assert fragment.elems == ["H", "H"]
    assert fragment.coords[1, 2] == pytest.approx(1.4 * BOHR_TO_ANG)


def test_read_pdbxfile_names_elementless_virtual_sites_m(tmp_path):
    topology = openmm.app.Topology()
    residue = topology.addResidue("HOH", topology.addChain())
    hydrogen = openmm.app.element.hydrogen
    for name, element in [("O", openmm.app.element.oxygen), ("H1", hydrogen), ("H2", hydrogen), ("MW", None)]:
        topology.addAtom(name, element, residue)
    positions = openmm.unit.Quantity(
        np.array([[0.0, 0.0, 0.0], [0.0957, 0.0, 0.0], [-0.024, 0.0927, 0.0], [0.003, 0.002, 0.0]]),
        openmm.unit.nanometer,
    )
    path = tmp_path / "tip4p.cif"
    with open(path, "w") as handle:
        openmm.app.PDBxFile.writeFile(topology, positions, handle)

    assert Fragment(pdbxfile=str(path)).elems == ["O", "H", "H", "M"]


@pytest.mark.parametrize(
    ("flags", "header"),
    [
        ({"write_chargemult": True, "write_energy": True}, "0 1 -1.5"),
        ({"write_chargemult": True, "write_energy": False}, "0 1"),
        ({"write_chargemult": False, "write_energy": True}, "-1.5"),
        ({"write_chargemult": False, "write_energy": False}, "title"),
    ],
)
def test_fragment_write_xyzfile_header_carries_each_requested_field(tmp_path, flags, header):
    fragment = Fragment(coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]], elems=["H", "F"], charge=0, mult=1)
    fragment.set_energy(-1.5)

    path = fragment.write_xyzfile(xyzfilename=str(tmp_path / "hf.xyz"), **flags)

    assert Path(path).read_text().splitlines()[1] == header
