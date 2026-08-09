import logging
import os

from openmmqmmm.exceptions import (
    MissingDependencyError,
)
from openmmqmmm.modules.module_coords import reformat_element

logger = logging.getLogger(__name__)

###################################
###################################


# Function to convert Mol file to PDB-file via OpenBabel
def mol_to_pdb(file):
    # OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "Error: mol_to_pdb requires OpenBabel library but it could not be imported\nYou can install like this:    conda install --yes -c conda-forge openbabel"
        ) from None
    mol = next(pybel.readfile("mol", file))
    mol.write(format="pdb", filename=os.path.splitext(file)[0] + ".pdb", overwrite=True)
    logger.info("Wrote PDB-file: %s", os.path.splitext(file)[0] + ".pdb")
    return os.path.splitext(file)[0] + ".pdb"


# Function to convert SDF file to PDB-file via OpenBabel
def sdf_to_pdb(file):
    # OpenBabel
    try:
        from openbabel import openbabel, pybel
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "Error: sdf_to_pdb requires OpenBabel library but it could not be imported\nYou can install like this:    conda install --yes -c conda-forge openbabel"
        ) from None
    mol = next(pybel.readfile("sdf", file))

    # Write do disk as PDB-file
    mol.write(format="pdb", filename=os.path.splitext(file)[0] + "temp.pdb", overwrite=True)
    # Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0] + "temp.pdb"))
    os.remove(os.path.splitext(file)[0] + "temp.pdb")

    # Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    logger.info("Creating new atomnames for PDBfile")
    # Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    # TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        for i, atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            res.SetAtomID(atom, atomname.strip() + str(i + 1))
            atomname = res.GetAtomID(atom)

    # Write final PDB-file
    newmol.write(format="pdb", filename=os.path.splitext(file)[0] + ".pdb", overwrite=True)
    logger.info("Wrote PDB-file: %s", os.path.splitext(file)[0] + ".pdb")
    return os.path.splitext(file)[0] + ".pdb"


# Function to read in PDB-file and write new one with CONECT lines (geometry needs to be sensible)
# NOTE: Requires OpenBabel which seems unnecessary, probably better to use OpenMM functionality instead
def writepdb_with_connectivity(file):
    # OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "Error: writepdb_with_connectivity requires OpenBabel library but it could not be imported\nYou can install like this:    conda install --yes -c conda-forge openbabel"
        ) from None
    mol = next(pybel.readfile("pdb", file))
    mol.write(format="pdb", filename=os.path.splitext(file)[0] + "_withcon.pdb", overwrite=True)
    logger.info("Wrote PDB-file: %s", os.path.splitext(file)[0] + "_withcon.pdb")
    return os.path.splitext(file)[0] + "_withcon.pdb"


# Function to read in XYZ-file (small molecule) and create PDB-file with CONECT lines (geometry needs to be sensible)
def xyz_to_pdb_with_connectivity(file, resname="UNL"):
    logger.info("xyz_to_pdb_with_connectivity function:")
    # OpenBabel
    try:
        from openbabel import openbabel, pybel
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "Error: xyz_to_pdb_with_connectivity requires OpenBabel library but it could not be imported\nYou can install OpenBabel like this:    conda install --yes -c conda-forge openbabel"
        ) from None
    # Read in XYZ-file
    mol = next(pybel.readfile("xyz", file))
    # Write do disk as PDB-file
    mol.write(format="pdb", filename=os.path.splitext(file)[0] + "temp.pdb", overwrite=True)
    # Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0] + "temp.pdb"))

    os.remove(os.path.splitext(file)[0] + "temp.pdb")

    # Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    logger.info("Creating new atomnames for PDBfile")
    # Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    # TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        # Setting residue name
        res.SetName(resname)
        for i, atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            res.SetAtomID(atom, atomname.strip() + str(i + 1))
            atomname = res.GetAtomID(atom)

    # Write final PDB-file
    newmol.write(format="pdb", filename=os.path.splitext(file)[0] + ".pdb", overwrite=True)
    logger.info("Wrote PDB-file: %s", os.path.splitext(file)[0] + ".pdb")
    return os.path.splitext(file)[0] + ".pdb"


# Function to convert PDB-file to SMILES string
def pdb_to_smiles(fname: str) -> str:
    # OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "Error: pdb_to_smiles requires OpenBabel library but it could not be imported\nYou can install like this:    conda install --yes -c conda-forge openbabel"
        ) from None
    mol = next(pybel.readfile("pdb", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()


# Function to convert SMILES string to elements and coordinates list
def smiles_to_coords(smiles_string):
    # OpenBabel
    try:
        from openbabel import openbabel, pybel
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "Error: smiles_to_coords requires OpenBabel library but it could not be imported\nYou can install like this:    conda install --yes -c conda-forge openbabel"
        ) from None
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
