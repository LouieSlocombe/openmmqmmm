from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment
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
