import os

from openmmqmmm.functions.functions_general import ashexit
from openmmqmmm.modules.module_coords import reformat_element


###################################
# Other Openbabel functionality
###################################

# Function to convert Mol file to PDB-file via OpenBabel
def mol_to_pdb(file):
    # OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: mol_to_pdb requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("mol", file))
    mol.write(format='pdb', filename=os.path.splitext(file)[0] + '.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0] + '.pdb')
    return os.path.splitext(file)[0] + '.pdb'


# Function to convert SDF file to PDB-file via OpenBabel
def sdf_to_pdb(file):
    # OpenBabel
    try:
        from openbabel import openbabel
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: sdf_to_pdb requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("sdf", file))

    # Write do disk as PDB-file
    mol.write(format='pdb', filename=os.path.splitext(file)[0] + 'temp.pdb', overwrite=True)
    # Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0] + 'temp.pdb'))
    os.remove(os.path.splitext(file)[0] + 'temp.pdb')

    # Atomlabel = {0:'C1',1:'X',2:'C',3:'C',4:'C',5:'C',6:'C',7:'C',8:'C',9:'C',10:'C',11:'C',12:'C'}
    # Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    print("Creating new atomnames for PDBfile")
    # Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    # TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        for i, atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            # print("atomname:", atomname)
            res.SetAtomID(atom, atomname.strip() + str(i + 1))
            atomname = res.GetAtomID(atom)
            # print("atomname:", atomname)
            # res.SetAtomID(atom,Atomlabel[i])

    # Write final PDB-file
    newmol.write(format='pdb', filename=os.path.splitext(file)[0] + '.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0] + '.pdb')
    return os.path.splitext(file)[0] + '.pdb'


# Function to read in PDB-file and write new one with CONECT lines (geometry needs to be sensible)
# NOTE: Requires OpenBabel which seems unnecessary, probably better to use OpenMM functionality instead
def writepdb_with_connectivity(file):
    # OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: writepdb_with_connectivity requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("pdb", file))
    mol.write(format='pdb', filename=os.path.splitext(file)[0] + '_withcon.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0] + '_withcon.pdb')
    return os.path.splitext(file)[0] + '_withcon.pdb'


# Function to read in XYZ-file (small molecule) and create PDB-file with CONECT lines (geometry needs to be sensible)
def xyz_to_pdb_with_connectivity(file, resname="UNL"):
    print("xyz_to_pdb_with_connectivity function:")
    # OpenBabel
    try:
        from openbabel import openbabel
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: xyz_to_pdb_with_connectivity requires OpenBabel library but it could not be imported")
        print("You can install OpenBabel like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    # Read in XYZ-file
    mol = next(pybel.readfile("xyz", file))
    # Write do disk as PDB-file
    mol.write(format='pdb', filename=os.path.splitext(file)[0] + 'temp.pdb', overwrite=True)
    # Read-in again (this will create a Residue)
    newmol = next(pybel.readfile("pdb", os.path.splitext(file)[0] + 'temp.pdb'))

    os.remove(os.path.splitext(file)[0] + 'temp.pdb')

    # Atomlabel = {0:'C1',1:'X',2:'C',3:'C',4:'C',5:'C',6:'C',7:'C',8:'C',9:'C',10:'C',11:'C',12:'C'}
    # Change atomnames (AtomIDs) to something sensible (OpenBabel does not do this by default)
    print("Creating new atomnames for PDBfile")
    # Note: currently just combining element and atomindex to get a unique atomname (otherwise Modeller will not work)
    # TODO: make something better (element-specific numbering?)
    for res in pybel.ob.OBResidueIter(newmol.OBMol):
        # Setting residue name
        res.SetName(resname)
        for i, atom in enumerate(openbabel.OBResidueAtomIter(res)):
            atomname = res.GetAtomID(atom)
            # print("atomname:", atomname)
            res.SetAtomID(atom, atomname.strip() + str(i + 1))
            atomname = res.GetAtomID(atom)
            # print("atomname:", atomname)
            # res.SetAtomID(atom,Atomlabel[i])

    # Write final PDB-file
    newmol.write(format='pdb', filename=os.path.splitext(file)[0] + '.pdb', overwrite=True)
    print("Wrote PDB-file:", os.path.splitext(file)[0] + '.pdb')
    return os.path.splitext(file)[0] + '.pdb'


# Function to convert PDB-file to SMILES string
def pdb_to_smiles(fname: str) -> str:
    # OpenBabel
    try:
        from openbabel import pybel
    except ModuleNotFoundError:
        print("Error: pdb_to_smiles requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    mol = next(pybel.readfile("pdb", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()


# Function to convert SMILES string to elements and coordinates list
def smiles_to_coords(smiles_string):
    # OpenBabel
    try:
        from openbabel import pybel
        from openbabel import openbabel
    except ModuleNotFoundError:
        print("Error: smiles_to_coords requires OpenBabel library but it could not be imported")
        print("You can install like this:    conda install --yes -c conda-forge openbabel")
        ashexit()
    print("Reading SMILES by OpenBabel")
    mol = pybel.readstring("smi", smiles_string)
    print("Guessing 3D coordinates (uses MMFF94 forcefield)")
    mol.make3D()
    b_mol = mol.OBMol
    atomnums = []
    coords = []
    for atom in openbabel.OBMolAtomIter(b_mol):
        atomnums.append(atom.GetAtomicNum())
        coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])
    elems = [reformat_element(atn, isatomnum=True) for atn in atomnums]
    # frag = Fragment(elems=elems, coords=coords, charge=charge, mult=mult)
    return elems, coords
