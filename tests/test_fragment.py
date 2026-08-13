from pathlib import Path

import numpy as np

from openmmqmmm import Fragment

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
