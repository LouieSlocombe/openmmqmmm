from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, OpenMMTheory, openmm_md, openmm_modeller, single_point
from openmmqmmm.exceptions import InputError

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
    openmmobject, fragment = openmm_modeller(
        pdbfile=pdbfile,
        forcefield="CHARMM36",
        watermodel="tip3p",
        ph=7.0,
        solvent_padding=10.0,
        ionicstrength=0.1,
        platform="CPU",
    )

    assert openmmobject is not None, "openmm_modeller should return an OpenMMTheory object"
    # The input 1aki.pdb has 1079 atoms; hydrogens, solvent and ions are added on top
    assert fragment.numatoms > 1079, "Modeller should add hydrogens, water and ions"
    assert len(fragment.elems) == fragment.numatoms
    assert "H" in fragment.elems, "Hydrogens should have been added"


def test_openmm_md_runs_and_writes_a_trajectory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )

    starting_coords = fragment.coords.copy()
    openmm_md(
        fragment=fragment,
        theory=theory,
        timestep=0.0005,
        simulation_steps=20,
        traj_frequency=10,
        temperature=300,
        integrator="LangevinMiddleIntegrator",
    )

    assert (tmp_path / "trajectory.dcd").exists(), "The DCD trajectory should have been written"
    assert (tmp_path / "trajectory_lastframe.pdb").exists(), "finalize_simulation writes the last frame"
    assert fragment.coords.shape == starting_coords.shape
    assert not np.allclose(fragment.coords, starting_coords), "MD must advance the coordinates"
    assert np.all(np.isfinite(fragment.coords))


def test_openmm_md_requires_a_run_length():
    """Neither simulation_steps nor simulation_time is an error, not a zero-length run."""
    with pytest.raises(InputError):
        openmm_md(fragment=None, theory=None)
