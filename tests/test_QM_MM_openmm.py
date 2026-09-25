from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, MolecularDynamicsEngine, OpenMMTheory, ORCATheory, QMMMTheory, single_point
from openmmqmmm.orca import find_orca

TEST_DIR = Path(__file__).parent

pytestmark = pytest.mark.skipif(
    find_orca(required=False) is None, reason="No ORCA installation found (orcadir / OPENMMQMMM_ORCADIR / PATH)"
)


def test_qm_mm_orca_openmm_meoh_h2o():
    H2O_MeOH = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")

    H2O_MeOH.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    pdbfile = "h2o_MeOH.pdb"

    # Specifying the QM atoms (3-8) by atom indices (MeOH). The other atoms (0,1,2) is the H2O and MM.
    # IMPORTANT: atom indices begin at 0.
    qmatoms = [3, 4, 5, 6, 7, 8]

    qm = ORCATheory(orcasimpleinput="! PBE def2-SVP NORI tightscf")

    MMpart = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"], pdbfile=pdbfile, autoconstraints=None, rigidwater=False
    )

    QMMMobject = QMMMTheory(fragment=H2O_MeOH, qm_theory=qm, mm_theory=MMpart, qmatoms=qmatoms, embedding="Elstat")

    result = single_point(theory=QMMMobject, fragment=H2O_MeOH, charge=0, mult=1, grad=True)

    # Determined 8 aug 2026 using ORCA 6 (PBE/def2-SVP NORI tightscf) and OpenMM 8.4
    ref_energy = -115.816207775989
    ref_gradient = np.array(
        [
            [-0.09756493, 0.06210694, 0.02911785],
            [0.02114901, -0.07293217, -0.04991809],
            [0.07600842, 0.01092897, 0.02069975],
            [-0.00148388, -0.00231294, -0.01543450],
            [-0.00204777, 0.00636927, 0.00799389],
            [0.00983886, -0.00363049, 0.00482077],
            [-0.00546031, -0.00833997, 0.00609750],
            [-0.00195924, 0.01572871, -0.00057065],
            [0.00151982, -0.00791832, -0.00280650],
        ]
    )

    # Absolute errors: 2e-6 Eh in energy and 1e-5 Eh/Bohr per gradient component.
    assert np.isclose(result.energy, ref_energy, rtol=0, atol=2e-6), "Energy is not correct"
    assert np.allclose(result.gradient, ref_gradient, rtol=0, atol=1e-5), "Gradient is not correct"

    # Second, displaced geometry moving both the QM region and the MM point-charge field.
    # Its reference must be computed before MolecularDynamicsEngine construction, which
    # switches the QMMM object into external-force mode.
    displaced_coords = H2O_MeOH.coords.copy()
    displaced_coords[0] += [0.04, -0.02, 0.03]
    displaced_coords[4] += [-0.03, 0.02, -0.04]
    displaced_fragment = Fragment(elems=H2O_MeOH.elems, coords=displaced_coords, charge=0, mult=1)
    result_displaced = single_point(theory=QMMMobject, fragment=displaced_fragment, charge=0, mult=1, grad=True)
    # At least one gradient component must change by more than 1e-4 Eh/Bohr.
    assert not np.allclose(result_displaced.gradient, result.gradient, rtol=0, atol=1e-4)

    # The same physical energy and total gradient must be assembled when OpenMM evaluates
    # the QM/MM correction from inside an RPMD bead through PythonForce, bead by bead.
    import openmm

    from openmmqmmm import constants

    MolecularDynamicsEngine(
        fragment=H2O_MeOH,
        theory=QMMMobject,
        charge=0,
        mult=1,
        timestep=0.000001,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
    )
    simulation = MMpart.create_simulation()
    MMpart.set_positions(H2O_MeOH.coords, simulation)
    displaced_nm = displaced_coords * 0.1
    simulation.integrator.setPositions(1, [openmm.Vec3(*row) for row in displaced_nm] * openmm.unit.nanometer)

    for copy, reference in enumerate((result, result_displaced)):
        state = simulation.integrator.getState(copy, getEnergy=True, getForces=True)
        rpmd_energy = (
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) / constants.HARTREE_TO_KJ_PER_MOL
        )
        rpmd_gradient = (
            -np.asarray(
                state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoules_per_mole / openmm.unit.nanometer)
            )
            / constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
        )
        # After conversion, use the same absolute Eh and Eh/Bohr tolerances as above.
        assert rpmd_energy == pytest.approx(reference.energy, rel=0, abs=2e-6), f"RPMD copy {copy} energy mismatch"
        assert rpmd_gradient == pytest.approx(reference.gradient, rel=0, abs=1e-5), (
            f"RPMD copy {copy} gradient mismatch"
        )


def test_qm_mm_orca_openmm_lysozyme():
    numcores = 2
    pdbfile = f"{TEST_DIR}/pdbfiles/1aki_solvated.pdb"
    fragment = Fragment(pdbfile=pdbfile)

    omm = OpenMMTheory(
        xmlfiles=["charmm36.xml", "charmm36/water.xml"],
        pdbfile=pdbfile,
        periodic=True,
        numcores=numcores,
        autoconstraints=None,
        rigidwater=False,
    )
    qmatomlist = [1013, 1014, 1015, 1016, 1017, 1018]
    # Distinct filename so ORCA autostart does not pick up a GBW-file from the MeOH tests above
    qm = ORCATheory(orcasimpleinput="! BP86 def2-SVP tightscf", filename="orca_lysozyme")
    qmmmobject = QMMMTheory(
        qm_theory=qm,
        mm_theory=omm,
        qm_charge=-1,
        qm_mult=1,
        fragment=fragment,
        embedding="Elstat",
        qmatoms=qmatomlist,
    )

    result = single_point(theory=qmmmobject, fragment=fragment, grad=True)

    # Smoke checks only: energy in Eh and gradient in Eh/Bohr; no reference tolerance.
    assert result.energy < 0.0, "QM/MM energy should be negative"
    assert np.isfinite(result.energy), "QM/MM energy should be finite"
    assert np.all(np.isfinite(result.gradient)), "QM/MM gradient should be finite"


# The two tests above both use electrostatic embedding on a QM region that is a whole
# molecule. That leaves mech_run and the link-atom force projection — the code both
# embeddings share — with no coverage at all.


def _meoh_water_qmmm(qmatoms, embedding, tag, unusualboundary=False):
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)

    qm = ORCATheory(orcasimpleinput="! PBE def2-SVP NORI tightscf", filename=f"orca_{tag}")
    mm = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )
    # This fixture deliberately has a nonbonded-only force-field template and
    # omits PDB connectivity. Supply the known methanol covalent graph explicitly
    # so boundary detection can use topology without inventing distance bonds.
    atoms = list(mm.topology.atoms())
    for first, second in ((3, 4), (3, 5), (3, 6), (3, 7), (7, 8)):
        mm.topology.addBond(atoms[first], atoms[second])
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=qmatoms,
        embedding=embedding,
        unusualboundary=unusualboundary,
    )
    return qmmm, fragment


def test_qm_mm_mechanical_embedding():
    """Mechanical embedding: QM and MM energies simply added, no point-charge field."""
    qmmm, fragment = _meoh_water_qmmm([3, 4, 5, 6, 7, 8], "Mech", "mech")
    result = single_point(theory=qmmm, fragment=fragment, charge=0, mult=1, grad=True)

    assert qmmm.num_linkatoms == 0, "The QM region is a whole molecule: no bond is cut"
    # Reference: ORCA 6.1.1 / OpenMM 8.4, 12 Aug 2026; absolute error at most 2e-6 Eh.
    assert np.isclose(result.energy, -115.8225732022, rtol=0, atol=2e-6)
    assert result.gradient.shape == (9, 3)
    # Mechanical embedding leaves out the QM-MM electrostatic coupling, so it must not
    # reproduce the electrostatic result: separation must exceed 1e-4 Eh.
    assert not np.isclose(result.energy, -115.816207775989, rtol=0, atol=1e-4)


def test_qm_mm_link_atom_force_projection():
    """A QM region that cuts a covalent bond gets a link atom, whose force is projected."""
    qm1, mm1 = 7, 3  # O2 (QM side of the cut bond) and C1 (MM side)
    qmmm, fragment = _meoh_water_qmmm([7, 8], "Elstat", "linkatom", unusualboundary=True)
    result = single_point(theory=qmmm, fragment=fragment, charge=0, mult=1, grad=True)

    assert qmmm.num_linkatoms == 1, "Cutting the C1-O2 bond must create exactly one link atom"
    assert result.gradient.shape == (9, 3), "The gradient covers the real atoms only, not the link atom"
    assert np.all(np.isfinite(result.gradient))

    # The projection splits the link atom's force between its two host atoms with opposite
    # signs, so their contributions very nearly cancel. Everything else is small by
    # comparison, which is what makes this visible in the total gradient. Both
    # projected and residual are in Eh/Bohr; the residual bound is intentionally 1%.
    projected = np.abs(result.gradient[[qm1, mm1]]).max()
    residual = np.abs(result.gradient[qm1] + result.gradient[mm1]).max()
    assert projected > 1.0, "The link atom's force should dominate its two host atoms"
    assert residual < 0.01 * projected, "QM1 and MM1 contributions must be equal and opposite"

    # Atoms with no link atom and no QM role have components below 1e-8 Eh/Bohr.
    assert np.abs(result.gradient[[4, 5, 6]]).max() < 1e-8, "Pure MM atoms of the capped group"
