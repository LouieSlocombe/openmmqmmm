import pytest
import shutil

from openmmqmmm import *

# QM/MM tests with OpenMMTheory as MM engine, using ORCATheory for QM-part
# Skipped when no orca binary is available in PATH
pytestmark = pytest.mark.skipif(shutil.which("orca") is None, reason="ORCA binary not found in PATH")


def test_qm_mm_orca_openmm_MeOH_H2O():
    # H2O...MeOH fragment defined. Reading XYZ file
    H2O_MeOH = Fragment(xyzfile=f"{ashpath}/tests/xyzfiles/h2o_MeOH.xyz")

    # Write PDB-file for OpenMM (used for topology)
    H2O_MeOH.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    pdbfile = "h2o_MeOH.pdb"

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qmatoms = [3, 4, 5, 6, 7, 8]

    # QM
    qm = ORCATheory(orcasimpleinput="! PBE def2-SVP NORI tightscf")

    # MM: OpenMMTheory using XML-file
    MMpart = OpenMMTheory(xmlfiles=[f"{ashpath}/tests/extra_files/MeOH_H2O-sigma.xml"], pdbfile=pdbfile,
                          autoconstraints=None, rigidwater=False)

    # Creating QM/MM object
    QMMMobject = QMMMTheory(fragment=H2O_MeOH, qm_theory=qm, mm_theory=MMpart, qmatoms=qmatoms,
                            embedding='Elstat')

    # Single-point energy calculation of QM/MM object
    result = Singlepoint(theory=QMMMobject, fragment=H2O_MeOH, charge=0, mult=1, Grad=True)

    # Determined 8 aug 2026 using ORCA 6 (PBE/def2-SVP NORI tightscf) and OpenMM 8.4
    ref_energy = -115.816207775989
    ref_gradient = np.array([[-0.09756493, 0.06210694, 0.02911785],
                             [0.02114901, -0.07293217, -0.04991809],
                             [0.07600842, 0.01092897, 0.02069975],
                             [-0.00148388, -0.00231294, -0.01543450],
                             [-0.00204777, 0.00636927, 0.00799389],
                             [0.00983886, -0.00363049, 0.00482077],
                             [-0.00546031, -0.00833997, 0.00609750],
                             [-0.00195924, 0.01572871, -0.00057065],
                             [0.00151982, -0.00791832, -0.00280650]])

    assert np.isclose(result.energy, ref_energy, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, atol=1e-5), "Gradient is not correct"


# Read Lysozyme solvated PDB-file, create OpenMMTheory job, define ORCATheory, create QM/MM and run SP
def test_qm_mm_orca_openmm_lysozyme():
    numcores = 2
    # Defining fragment containing coordinates (can be read from XYZ-file, ASH fragment or PDB-file)
    pdbfile = f"{ashpath}/tests/pdbfiles/1aki_solvated.pdb"
    fragment = Fragment(pdbfile=pdbfile)

    # Creating new OpenMM object from OpenMM full system file
    omm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile=pdbfile, periodic=True,
                       numcores=numcores, autoconstraints=None, rigidwater=False)
    # QM
    qmatomlist = [1013, 1014, 1015, 1016, 1017, 1018]
    # Distinct filename so ORCA autostart does not pick up a GBW-file from the MeOH tests above
    qm = ORCATheory(orcasimpleinput="! BP86 def2-SVP tightscf", filename="orca_lysozyme")
    # Create QM/MM OBJECT
    qmmmobject = QMMMTheory(qm_theory=qm, mm_theory=omm, qm_charge=-1, qm_mult=1,
                            fragment=fragment, embedding="Elstat", qmatoms=qmatomlist, printlevel=2)

    result = Singlepoint(theory=qmmmobject, fragment=fragment, Grad=True)

    # Sanity checks (QM/MM energy of solvated protein system)
    assert result.energy < 0.0, "QM/MM energy should be negative"
    assert np.isfinite(result.energy), "QM/MM energy should be finite"
    assert np.all(np.isfinite(result.gradient)), "QM/MM gradient should be finite"
