"""Physics regressions for MM partitioning and decomposition."""

import numpy as np
import openmm
import openmm.app
import pytest
from openmm import unit

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
from openmmqmmm.constants import HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError


def _mm(system):
    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.system = system
    theory.delete_qm1_mm1_bonded = False
    return theory


def _evaluate(system, positions_nm, groups=-1):
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions_nm)
    state = context.getState(getEnergy=True, getForces=True, groups=groups)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
    del context, integrator
    return energy, forces


def _system(n):
    system = openmm.System()
    for _ in range(n):
        system.addParticle(12)
    return system


class _ZeroQM:
    numcores = 1
    theorytype = "QM"

    def set_numcores(self, numcores):
        self.numcores = numcores


def test_elstat_removes_exception_charge_but_preserves_lj():
    system = _system(3)
    nb = openmm.NonbondedForce()
    for charge in (1, -1, -0.5):
        nb.addParticle(charge, 0.3, 0.2)
    cross = nb.addException(0, 1, -0.5, 0.27, 0.6)
    mm_pair = nb.addException(1, 2, 0.15, 0.28, 0.4)
    nb.addGlobalParameter("lambda", 1)
    nb.addParticleParameterOffset("lambda", 0, 0.3, 0, 0)
    nb.addExceptionParameterOffset("lambda", cross, -0.2, 0, 0.1)
    system.addForce(nb)
    mm = _mm(system)
    mm.nonbonded_force = nb
    mm.charges = [1, -1, -0.5]
    mm.numatoms = 3
    fragment = Fragment(elems=["C"] * 3, coords=[[0, 0, 0], [5, 0, 0], [0, 6, 0]], conncalc=False)
    QMMMTheory(
        fragment=fragment,
        qm_theory=_ZeroQM(),
        mm_theory=mm,
        qmatoms=[0],
        embedding="elstat",
        qm_charge=0,
        qm_mult=1,
    )
    assert nb.getParticleParameters(0)[0]._value == 0
    assert nb.getParticleParameterOffset(0)[2] == 0
    assert nb.getExceptionParameters(cross)[2]._value == 0
    assert nb.getExceptionParameterOffset(0)[2] == 0
    assert nb.getExceptionParameters(cross)[4]._value == pytest.approx(0.6)
    assert nb.getExceptionParameterOffset(0)[4] == pytest.approx(0.1)
    assert nb.getExceptionParameters(mm_pair)[2]._value == pytest.approx(0.15)
    energy, forces = _evaluate(system, fragment.coords * 0.1)
    # Independent expected pair potential, including the surviving MM Coulomb term.
    r12 = np.linalg.norm(fragment.coords[1] - fragment.coords[2]) * 0.1
    expected = 4 * 0.7 * ((0.27 / 0.5) ** 12 - (0.27 / 0.5) ** 6)
    expected += 4 * 0.2 * ((0.3 / 0.6) ** 12 - (0.3 / 0.6) ** 6)
    expected += 138.93545764438198 * 0.15 / r12 + 4 * 0.4 * ((0.28 / r12) ** 12 - (0.28 / r12) ** 6)
    assert energy == pytest.approx(expected, abs=1e-9)
    assert np.max(abs(forces.sum(axis=0))) < 1e-10


TORSION_POSITIONS = np.array([[0, 0, 0], [0.15, 0, 0], [0.2, 0.1, 0], [0.3, 0.1, 0.1], [0.4, 0.2, 0.1]])


@pytest.mark.parametrize("qm_atoms,removed", [([0, 1, 2, 3], True), ([0, 1, 2], True), ([0, 1], False)])
def test_rb_torsion_uses_same_boundary_policy_as_periodic_torsions(qm_atoms, removed):
    system = _system(4)
    force = openmm.RBTorsionForce()
    force.addTorsion(0, 1, 2, 3, 1, 2, 3, 4, 5, 6)
    system.addForce(force)
    before, before_forces = _evaluate(system, TORSION_POSITIONS[:4])
    _mm(system).modify_bonded_forces(qm_atoms)
    after, after_forces = _evaluate(system, TORSION_POSITIONS[:4])
    assert abs(before) > 1
    assert after == pytest.approx(0 if removed else before, abs=1e-12)
    assert after_forces == pytest.approx(np.zeros((4, 3)) if removed else before_forces)


@pytest.mark.parametrize("qm_atoms,removed", [([0, 1, 2, 3, 4], True), ([0, 1, 2, 3], False)])
def test_cmap_removal_preserves_shared_maps_and_boundary_terms(qm_atoms, removed):
    system = _system(10)
    force = openmm.CMAPTorsionForce()
    force.addMap(2, [1, 2, 4, 8])
    force.addTorsion(0, 0, 1, 2, 3, 1, 2, 3, 4)
    force.addTorsion(0, 5, 6, 7, 8, 6, 7, 8, 9)
    system.addForce(force)
    positions = np.vstack([TORSION_POSITIONS, TORSION_POSITIONS + np.array([1, 0, 0])])
    before, before_forces = _evaluate(system, positions)
    _mm(system).modify_bonded_forces(qm_atoms)
    after, after_forces = _evaluate(system, positions)
    assert after == pytest.approx(before / 2 if removed else before, abs=1e-12)
    assert after_forces[:5] == pytest.approx(np.zeros((5, 3)) if removed else before_forces[:5])
    assert after_forces[5:] == pytest.approx(before_forces[5:])
    assert list(force.getMapParameters(0)[1]._value) == [1, 2, 4, 8]


@pytest.mark.parametrize("has_old_box", [False, True])
def test_explicit_triclinic_dimensions_replace_previous_cell(has_old_box):
    mm = _mm(_system(1))
    mm.topology = openmm.app.Topology()
    if has_old_box:
        mm.topology.setPeriodicBoxVectors(np.eye(3) * 3 * unit.nanometer)
    mm.forcefield = openmm.app.ForceField()
    mm.set_periodics_before_system_creation(None, None, [40, 45, 50, 90, 90, 60], False, False, False)
    vectors = np.asarray(mm.get_pbc_vectors())
    expected = [[40, 0, 0], [-17.5, 45 * np.sqrt(3) / 2, 0], [0, 0, 50]]
    assert vectors == pytest.approx(np.asarray(expected), abs=1e-12)
    assert np.linalg.det(vectors) == pytest.approx(40 * 45 * 50 * np.sqrt(3) / 2)
    assert callable(mm.topology.setUnitCellDimensions)
    mm.system.setDefaultPeriodicBoxVectors(*(vectors * 0.1))
    assert np.isfinite(_evaluate(mm.system, [[0, 0, 0]])[0])


@pytest.mark.parametrize(
    "dimensions", [[40, 45, 50], [-1, 45, 50, 90, 90, 90], [40, 45, 50, 90, 90, 0], [40, 45, 50, 5, 5, 170]]
)
def test_invalid_cell_dimensions_are_rejected(dimensions):
    mm = _mm(_system(1))
    mm.topology = openmm.app.Topology()
    mm.forcefield = openmm.app.ForceField()
    with pytest.raises(InputError, match="periodic_cell_dimensions"):
        mm.set_periodics_before_system_creation(None, None, dimensions, False, False, False)


def test_cell_update_also_reduces_the_triclinic_lattice():
    mm = _mm(_system(1))
    mm.topology = openmm.app.Topology()
    mm.update_cell(periodic_cell_dimensions=[40, 45, 50, 90, 90, 60])
    expected = np.array([[40, 0, 0], [-17.5, 45 * np.sqrt(3) / 2, 0], [0, 0, 50]])
    assert mm.get_pbc_vectors() == pytest.approx(expected, abs=1e-12)
    actual = np.asarray([v.value_in_unit(unit.angstrom) for v in mm.system.getDefaultPeriodicBoxVectors()])
    assert actual == pytest.approx(expected, abs=1e-12)
    assert np.isfinite(_evaluate(mm.system, [[0, 0, 0]])[0])


def test_lj_decomposition_includes_cross_exceptions_and_excludes_intra_region_pairs():
    system = _system(4)
    nb = openmm.NonbondedForce()
    for charge in (1, -1, 2, -2):
        nb.addParticle(charge, 0.3, 0.2)
    nb.addException(0, 2, -0.25, 0.27, 0.7)
    system.addForce(nb)
    positions = np.array([[0, 0, 0], [0, 0.5, 0], [0.4, 0, 0], [0.4, 0.6, 0]])
    before = openmm.XmlSerializer.serialize(system)
    actual = _mm(system).qmmm_lj_energy([0, 1], positions * 10) * HARTREE_TO_KJ_PER_MOL
    expected = 0
    for q in (0, 1):
        for m in (2, 3):
            sigma, epsilon = (0.27, 0.7) if (q, m) == (0, 2) else (0.3, 0.2)
            r = np.linalg.norm(positions[q] - positions[m])
            expected += 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert openmm.XmlSerializer.serialize(system) == before


def _water_dimer():
    topology = openmm.app.Topology()
    chain = topology.addChain()
    for _ in range(2):
        residue = topology.addResidue("HOH", chain)
        oxygen = topology.addAtom("O", openmm.app.element.oxygen, residue)
        for name in ("H1", "H2"):
            hydrogen = topology.addAtom(name, openmm.app.element.hydrogen, residue)
            topology.addBond(oxygen, hydrogen)
    forcefield = openmm.app.ForceField("charmm36/water.xml")
    system = forcefield.createSystem(topology, nonbondedMethod=openmm.app.NoCutoff, constraints=None, rigidWater=False)
    water = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.2399872, 0.927297, 0]])
    return system, np.vstack([water, water + np.array([4, 0, 0])])


def test_charmm_lj_decomposition_matches_isolated_custom_force():
    system, coords = _water_dimer()
    for force in system.getForces():
        if isinstance(force, (openmm.CustomNonbondedForce, openmm.CustomBondForce)):
            force.setForceGroup(1)
    expected, _ = _evaluate(system, coords * 0.1, groups=1 << 1)
    before = openmm.XmlSerializer.serialize(system)
    actual = _mm(system).qmmm_lj_energy([0, 1, 2], coords) * HARTREE_TO_KJ_PER_MOL
    assert expected < -0.1
    assert actual == pytest.approx(expected, abs=1e-12)
    assert openmm.XmlSerializer.serialize(system) == before


def test_custom_lj_exception_decomposition_selects_only_cross_pairs():
    system = _system(4)
    force = openmm.CustomBondForce("4*epsilon*((sigma/r)^12-(sigma/r)^6)")
    force.addPerBondParameter("sigma")
    force.addPerBondParameter("epsilon")
    force.addBond(0, 1, [0.3, 0.2])
    force.addBond(0, 2, [0.27, 0.7])
    force.addBond(2, 3, [0.31, 0.4])
    system.addForce(force)
    coords = np.array([[0, 0, 0], [0, 5, 0], [4, 0, 0], [4, 6, 0]])
    actual = _mm(system).qmmm_lj_energy([0, 1], coords) * HARTREE_TO_KJ_PER_MOL
    expected = 4 * 0.7 * ((0.27 / 0.4) ** 12 - (0.27 / 0.4) ** 6)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_unknown_custom_pair_potential_is_rejected_without_mutation():
    system = _system(2)
    force = openmm.CustomNonbondedForce("1/r")
    force.addParticle([])
    force.addParticle([])
    system.addForce(force)
    before = openmm.XmlSerializer.serialize(system)
    with pytest.raises(InputError, match="does not support this CustomNonbondedForce"):
        _mm(system).qmmm_lj_energy([0], [[0, 0, 0], [4, 0, 0]])
    assert openmm.XmlSerializer.serialize(system) == before
