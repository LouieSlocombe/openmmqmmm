from ash import *


def test_openmm_md():
    # Read raw PDB-file, fix using pdbfixer, setup using Modeller and run MD
    pdb_file = f"./tests/pdbfiles/1aki.pdb"
    pdb_file = f"./tests/pdbfiles/aaa.pdb"

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
              timestep=0.001,
              simulation_time=.01,
              traj_frequency=1,
              trajectory_file_option='PDB',
              integrator='RPMDIntegrator',
              )


def test_openmm_md_qmmm():
    pdb_file = f"./tests/pdbfiles/aaa.pdb"

    # Setting up new system, adding hydrogens, solvent, ions and defining forcefield, topology
    omm, fragment = OpenMM_Modeller(pdbfile=pdb_file,
                                    forcefield='Amber14',
                                    watermodel="tip3p-fb",
                                    pH=7.0,
                                    solvent_padding=10.0,
                                    ionicstrength=0.1,
                                    platform='CPU')

    qm = xTBTheory(xtbmethod="GFN2-xTB", numcores=10)

    qmatoms = [107, 106, 105]
    qmatoms = list(range(30))

    qm_mm = QMMMTheory(qm_theory=qm,
                       mm_theory=omm,
                       fragment=fragment,
                       qmatoms=qmatoms,
                       printlevel=2,
                       qm_charge=1,
                       qm_mult=1)

    OpenMM_MD(fragment=fragment,
              theory=qm_mm,
              timestep=0.001,
              simulation_time=.1,
              traj_frequency=1,
              trajectory_file_option='PDB',
              integrator='QTBIntegrator',
              )
