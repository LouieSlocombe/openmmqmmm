from __future__ import annotations

import logging
import os

from openmmqmmm.coords import reformat_element
from openmmqmmm.utils import basename

logger = logging.getLogger(__name__)


def xyz_to_pdb_with_connectivity(file: str, resname: str = "UNL") -> str:
    logger.info("xyz_to_pdb_with_connectivity function:")
    from openbabel import openbabel, pybel

    stem = basename(file)
    mol = next(pybel.readfile("xyz", file))
    mol.write(format="pdb", filename=stem + "temp.pdb", overwrite=True)
    # Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", stem + "temp.pdb"))

    os.remove(stem + "temp.pdb")

    # Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    logger.debug("Creating new atomnames for PDBfile")
    # Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        res.SetName(resname)
        for i, atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            res.SetAtomID(atom, atomname.strip() + str(i + 1))
            atomname = res.GetAtomID(atom)

    newmol.write(format="pdb", filename=stem + ".pdb", overwrite=True)
    logger.info("Wrote PDB-file: %s", stem + ".pdb")
    return stem + ".pdb"


def smiles_to_coords(smiles_string: str) -> tuple[list[str], list[list[float]]]:
    from openbabel import openbabel, pybel

    logger.info("Reading SMILES by OpenBabel")
    mol = pybel.readstring("smi", smiles_string)
    logger.info("Guessing 3D coordinates (uses MMFF94 forcefield)")
    mol.make3D()
    b_mol = mol.OBMol
    atomnums = []
    coords = []
    for atom in openbabel.OBMolAtomIter(b_mol):
        atomnums.append(atom.GetAtomicNum())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    elems = [reformat_element(atn, isatomnum=True) for atn in atomnums]
    return elems, coords
