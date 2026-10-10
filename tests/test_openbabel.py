"""OpenBabel helpers: SMILES to 3D coordinates and XYZ to a PDB with connectivity."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import rdkit.Chem  # noqa: F401 - rdkit's compiled modules segfault when loaded after the PyPI openbabel wheel

from openmmqmmm.openbabel import smiles_to_coords, xyz_to_pdb_with_connectivity

TEST_DIR = Path(__file__).parent


@pytest.mark.parametrize(
    ("smiles", "expected_elems", "bonded_pairs"),
    [("O", ["O", "H", "H"], [(0, 1), (0, 2)]), ("C", ["C", "H", "H", "H", "H"], [(0, 1), (0, 2), (0, 3), (0, 4)])],
)
def test_smiles_to_coords_adds_hydrogens_and_places_them_at_bonding_distance(smiles, expected_elems, bonded_pairs):
    elems, coords = smiles_to_coords(smiles)

    assert elems == expected_elems
    coords = np.asarray(coords, dtype=float)
    assert coords.shape == (len(expected_elems), 3)
    for heavy, hydrogen in bonded_pairs:
        assert 0.9 < np.linalg.norm(coords[heavy] - coords[hydrogen]) < 1.2
    assert np.ptp(coords, axis=0).max() > 0.5


def test_xyz_to_pdb_with_connectivity_names_atoms_uniquely_and_keeps_bonds():
    shutil.copy(TEST_DIR / "xyzfiles" / "h2o_MeOH.xyz", "h2o_MeOH.xyz")

    pdbfile = xyz_to_pdb_with_connectivity("h2o_MeOH.xyz", resname="LIG")

    assert pdbfile == "h2o_MeOH.pdb"
    assert not Path("h2o_MeOHtemp.pdb").exists()
    lines = Path(pdbfile).read_text().splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
    assert len(atom_lines) == 9
    assert {line[17:20].strip() for line in atom_lines} == {"LIG"}
    names_by_residue = {}
    for line in atom_lines:
        names_by_residue.setdefault(line[22:26], []).append(line[12:16].strip())
    for names in names_by_residue.values():
        assert len(set(names)) == len(names)
    elements = [line[76:78].strip() for line in atom_lines]
    assert sorted(elements) == sorted(["O", "H", "H", "C", "H", "H", "H", "O", "H"])
    conect = [line for line in lines if line.startswith("CONECT")]
    bonded = {tuple(sorted((int(line[6:11]), int(field)))) for line in conect for field in line[11:].split()}
    assert len(bonded) == 7
