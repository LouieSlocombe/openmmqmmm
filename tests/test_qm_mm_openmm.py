import os

import numpy as np

from ash.interfaces.interface_ORCA import ORCATheory
from ash.interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD
from ash.interfaces.interface_pyscf import PySCFTheory
from ash.modules.module_MM import MMforcefield_read, NonBondedTheory
from ash.modules.module_QMMM import QMMMTheory
from ash.modules.module_coords import Fragment
from ash.modules.module_singlepoint import Singlepoint


def test_qm_mm_pyscf_nonbondedtheory():
    h2o_meoh = Fragment(xyzfile=f"./tests/xyzfiles/h2o_MeOH.xyz")

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qm_atoms = [3, 4, 5, 6, 7, 8]

    # Charge definitions for whole system.
    # Charges for the QM atoms are zero (since ASH will always set QM atoms to zero in elstat embedding)
    atom_charges = [-0.834, 0.417, 0.417, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Defining atomtypes for whole system
    atom_types = ['OT', 'HT', 'HT', 'CX', 'HX', 'HX', 'HX', 'OT', 'HT']

    # Read forcefield (here containing LJ-part only) from file
    mm_forcefield = MMforcefield_read(f"./tests/extra_files/MeOH_H2O-sigma.ff")

    qm = PySCFTheory(scf_type="RKS",
                     functional="PBE",
                     basis="def2-SVP",
                     densityfit=False)

    mm_part = NonBondedTheory(charges=atom_charges,
                              atomtypes=atom_types,
                              forcefield=mm_forcefield,
                              LJcombrule='geometric',
                              codeversion="py")
    # Creating QM/MM object
    qmmm_object = QMMMTheory(fragment=h2o_meoh,
                             qm_theory=qm,
                             mm_theory=mm_part,
                             qmatoms=qm_atoms,
                             embedding='elstat')

    # Single-point energy calculation of QM/MM object
    result = Singlepoint(theory=qmmm_object,
                         fragment=h2o_meoh,
                         charge=0,
                         mult=1,
                         Grad=True)

    ref_energy = -115.816202526812
    ref_gradient = np.array([[-0.0976002, 0.06207825, 0.02913397],
                             [0.02114902, -0.07293206, -0.04991815],
                             [0.07600821, 0.010929, 0.02069938],
                             [-0.00145506, -0.00227548, -0.01546497],
                             [-0.0020466, 0.00636848, 0.00799227],
                             [0.00983804, -0.0036295, 0.00482348],
                             [-0.00546189, -0.00833844, 0.00609906],
                             [-0.00195135, 0.01570807, -0.00057301],
                             [0.00151874, -0.00790899, -0.00279125]])

    assert np.isclose(result.energy, ref_energy, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, atol=1e-5), "Gradient is not correct"
    os.remove('ASH_SP.result')
    os.remove('pyscf.chk')
    os.remove('pyscf.out')


def test_qm_mm_pyscf_openmm():
    h2o_meoh = Fragment(xyzfile=f"./tests/xyzfiles/h2o_MeOH.xyz")

    # Write PDB-file for OpenMM (used for topology)
    h2o_meoh.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    pdb_file = "h2o_MeOH.pdb"

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qm_atoms = [3, 4, 5, 6, 7, 8]

    qm = PySCFTheory(scf_type="RKS",
                     functional="PBE",
                     basis="def2-SVP",
                     densityfit=False)

    mm_part = OpenMMTheory(xmlfiles=[f"./tests/extra_files/MeOH_H2O-sigma.xml"],
                           pdbfile=pdb_file,
                           autoconstraints=None,
                           rigidwater=False)

    # Creating QM/MM object
    qmmm_object = QMMMTheory(fragment=h2o_meoh,
                             qm_theory=qm,
                             mm_theory=mm_part,
                             qmatoms=qm_atoms,
                             embedding='Elstat')

    # Single-point energy calculation of QM/MM object
    result = Singlepoint(theory=qmmm_object,
                         fragment=h2o_meoh,
                         charge=0,
                         mult=1,
                         Grad=True)

    ref_energy = -115.816183777727
    ref_gradient = np.array([[-0.09756495, 0.06210685, 0.02911808],
                             [0.02114902, -0.07293212, -0.04991817],
                             [0.07600821, 0.01092899, 0.0206994],
                             [-0.00148712, -0.00230238, -0.01544885],
                             [-0.00204659, 0.00636847, 0.00799225],
                             [0.00983799, -0.00362948, 0.00482346],
                             [-0.00546187, -0.00833839, 0.00609904],
                             [-0.00195452, 0.01570642, -0.00057316],
                             [0.00151874, -0.00790902, -0.00279129]])

    assert np.isclose(result.energy, ref_energy, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, atol=1e-5), "Gradient is not correct"
    os.remove('ASH_SP.result')
    os.remove('pyscf.chk')
    os.remove('pyscf.out')
    os.remove('h2o_MeOH.pdb')


def test_qm_mm_orca_openmm():
    h2o_meoh = Fragment(xyzfile=f"./tests/xyzfiles/h2o_MeOH.xyz")

    # Write PDB-file for OpenMM (used for topology)
    h2o_meoh.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    pdb_file = "h2o_MeOH.pdb"

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qm_atoms = [3, 4, 5, 6, 7, 8]

    # ORCA
    orcasimpleinput = "! BP86 def2-SVP tightscf notrah"
    orcablocks = "%scf maxiter 200 end"

    qm = ORCATheory(orcasimpleinput=orcasimpleinput,
                    orcablocks=orcablocks)

    mm_part = OpenMMTheory(xmlfiles=[f"./tests/extra_files/MeOH_H2O-sigma.xml"],
                           pdbfile=pdb_file,
                           autoconstraints=None,
                           rigidwater=False)

    # Creating QM/MM object
    qmmm_object = QMMMTheory(fragment=h2o_meoh,
                             qm_theory=qm,
                             mm_theory=mm_part,
                             qmatoms=qm_atoms,
                             embedding='Elstat')

    # Single-point energy calculation of QM/MM object
    result = Singlepoint(theory=qmmm_object,
                         fragment=h2o_meoh,
                         charge=0,
                         mult=1,
                         Grad=True)

    ref_energy = -115.95970693488094
    ref_gradient = np.array([[-0.09752915, 0.06209886, 0.02914283],
                             [0.02113842, -0.07293119, -0.0499293],
                             [0.07594458, 0.01091787, 0.02068298],
                             [-0.00110236, -0.00293094, -0.01332718],
                             [-0.00205525, 0.00636527, 0.00796327],
                             [0.00980917, -0.00360006, 0.00472817],
                             [-0.00547724, -0.00832853, 0.00601341],
                             [-0.00224821, 0.01657924, -0.00181844],
                             [0.00152004, -0.00817051, -0.00345574]])

    assert np.isclose(result.energy, ref_energy, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, atol=1e-5), "Gradient is not correct"

    qm.cleanup()
    os.remove('ASH_SP.result')
    os.remove('h2o_MeOH.pdb')


def test_qm_mm_md():
    h2o_meoh = Fragment(xyzfile=f"./tests/xyzfiles/h2o_MeOH.xyz")

    # Write PDB-file for OpenMM (used for topology)
    h2o_meoh.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    pdb_file = "h2o_MeOH.pdb"

    frag = Fragment(pdbfile=pdb_file)

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qm_atoms = [3, 4, 5, 6, 7, 8]

    # ORCA
    orcasimpleinput = "! BP86 def2-SVP tightscf notrah"
    orcablocks = "%scf maxiter 200 end"

    qm = ORCATheory(orcasimpleinput=orcasimpleinput,
                    orcablocks=orcablocks)

    omm = OpenMMTheory(xmlfiles=[f"./tests/extra_files/MeOH_H2O-sigma.xml"],
                       pdbfile=pdb_file,
                       autoconstraints=None,
                       rigidwater=False)

    # Creating QM/MM object
    qm_mm = QMMMTheory(qm_theory=qm,
                       mm_theory=omm,
                       fragment=frag,
                       qmatoms=qm_atoms,
                       printlevel=2,
                       qm_charge=0,
                       qm_mult=1)

    OpenMM_MD(fragment=frag,
              theory=qm_mm,
              timestep=0.001,
              simulation_steps=10,
              # simulation_time=1,
              traj_frequency=1000,
              temperature=300,
              integrator='LangevinMiddleIntegrator',
              coupling_frequency=1,
              trajectory_file_option='pdb')
    qm.cleanup()


def test_qm_mm_pyscf_openmm_lysozyme():
    pdb_file = f"./tests/pdbfiles/1aki_solvated.pdb"
    fragment = Fragment(pdbfile=pdb_file)

    # Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"],
                       pdbfile=pdb_file,
                       periodic=True,
                       autoconstraints=None,
                       rigidwater=False)
    # QM
    qm_atoms = [1013, 1014, 1015, 1016, 1017, 1018]
    qm = PySCFTheory(scf_type="RKS",
                     functional="BP86",
                     basis="def2-SVP",
                     densityfit=True)
    # qm = xTBTheory()
    # Create QM/MM OBJECT
    qmmm_object = QMMMTheory(qm_theory=qm,
                             mm_theory=omm,
                             qm_charge=-1,
                             qm_mult=1,
                             fragment=fragment,
                             embedding="Elstat",
                             qmatoms=qm_atoms,
                             printlevel=2)

    Singlepoint(theory=qmmm_object,
                fragment=fragment,
                Grad=True)
    os.remove('ASH_SP.result')
    os.remove('pyscf.chk')
    os.remove('pyscf.out')
