from pathlib import Path

from openmmqmmm import Fragment, OpenMMTheory, openmm_modeller, single_point

TEST_DIR = Path(__file__).parent


# Read solvated PDB-file, create OpenMMTheory job and run MM singlepoint
def test_openmm_basic():
    # Defining fragment containing coordinates (can be read from XYZ-file, fragment or PDB-file)
    pdbfile = f"{TEST_DIR}/pdbfiles/1aki_solvated.pdb"
    fragment = Fragment(pdbfile=pdbfile)

    # Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(
        xmlfiles=["charmm36.xml", "charmm36/water.xml"],
        pdbfile=pdbfile,
        periodic=True,
        autoconstraints=None,
        rigidwater=False,
    )
    # Singlepoint MM energy
    single_point(theory=omm, fragment=fragment, grad=True)


# Read raw PDB-file, fix using pdbfixer, setup using Modeller and optimize
def test_openmm_modeller():
    pdbfile = f"{TEST_DIR}/pdbfiles/1aki.pdb"

    # Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
    _openmmobject, _ashfragment = openmm_modeller(
        pdbfile=pdbfile,
        forcefield="CHARMM36",
        watermodel="tip3p",
        ph=7.0,
        solvent_padding=10.0,
        ionicstrength=0.1,
        platform="CPU",
    )
