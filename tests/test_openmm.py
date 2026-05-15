import os

from ash import *


def test_openmm_basic():
    # Read solvated PDB-file, create OpenMMTheory job and run MM singlepoint
    pdb_file = f"./tests/pdbfiles/1aki_solvated.pdb"
    fragment = Fragment(pdbfile=pdb_file)

    # Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"],
                       pdbfile=pdb_file,
                       periodic=True,
                       autoconstraints=None,
                       rigidwater=False)

    Singlepoint(theory=omm,
                fragment=fragment,
                Grad=True)
    os.remove('ASH_SP.result')


def test_openmm_modeller():
    # Read raw PDB-file, fix using pdbfixer, setup using Modeller and optimize
    pdb_file = f"./tests/pdbfiles/1aki.pdb"

    # Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
    omm, fragment = OpenMM_Modeller(pdbfile=pdb_file,
                                    forcefield='CHARMM36',
                                    watermodel="tip3p",
                                    pH=7.0,
                                    solvent_padding=10.0,
                                    ionicstrength=0.1,
                                    platform='CPU')

    OpenMM_Opt(fragment=fragment,
               theory=omm,
               maxiter=100,
               tolerance=1000)
    os.remove('ASH_SP.result')
    os.remove('finalsystem.pdb')
    os.remove('finalsystem.xyz')
    os.remove('finalsystem.ygg')
    os.remove('frag-minimized.pdb')
    os.remove('OpenMMOpt_traj.xyz')
    os.remove('system_afterfixes.pdb')
    os.remove('system_afterfixes2.pdb')
    os.remove('system_afterH.pdb')
    os.remove('system_aftersolvent_ions.pdb')
    os.remove('system_full.xml')


def test_openmm_md():
    # Read raw PDB-file, fix using pdbfixer, setup using Modeller and run MD
    pdb_file = f"./tests/pdbfiles/1aki.pdb"

    frag = Fragment(pdbfile=pdb_file)

    # # OpenMM object
    # omm = OpenMMTheory(xmlfiles=["amber14-all.xml", "amber14/tip3pfb.xml"], pdbfile=pdb_file, periodic=True,
    #                    autoconstraints='HBonds', rigidwater=True)

    # Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
    omm, fragment = OpenMM_Modeller(pdbfile=pdb_file,
                                    forcefield='Amber14',
                                    watermodel="tip3p-fb",
                                    pH=7.0,
                                    solvent_padding=10.0,
                                    ionicstrength=0.1,
                                    platform='CPU')

    OpenMM_MD(fragment=fragment,
              theory=omm,
              simulation_steps=100,
              traj_frequency=2,
              trajectory_file_option='PDB',
              )
    os.remove('ASH_SP.result')
    os.remove('finalsystem.pdb')
    os.remove('finalsystem.xyz')
    os.remove('finalsystem.ygg')
    os.remove('system_afterfixes.pdb')
    os.remove('system_afterfixes2.pdb')
    os.remove('system_afterH.pdb')
    os.remove('system_aftersolvent_ions.pdb')
    os.remove('system_full.xml')
    os.remove('OpenMM_MD.chk')
    os.remove('OpenMM_MD_final_checkpoint.chk')
    os.remove('OpenMM_MD_final_state.xml')
    os.remove('trajectory.pdb')
    os.remove('trajectory_lastframe.pdb')
