"""Force-mutation helpers, run() option branches and the energy decomposition of OpenMMTheory."""

import logging
import re
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest
from conftest import dummy_mm, evaluate, meoh_water_mm, nonbonded_mm, use_pme

from openmmqmmm import Fragment, OpenMMTheory
from openmmqmmm.constants import HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, HARTREE_TO_KJ_PER_MOL, KCAL_TO_KJ
from openmmqmmm.exceptions import FileFormatError, InputError, InternalError
from openmmqmmm.openmm.theory import clean_up_constraints_list

TEST_DIR = Path(__file__).parent
MEOH_XML = str(TEST_DIR / "extra_files" / "MeOH_H2O-sigma.xml")
KJ_PER_NM2_PER_KCAL_PER_A2 = KCAL_TO_KJ * 100  # 1 kcal/mol/Angstrom^2 expressed in kJ/mol/nm^2
KJ = openmm.unit.kilojoule_per_mole
NM = openmm.unit.nanometer
DALTON = openmm.unit.dalton


def _chain(count, spacing=2.0):
    """Helium atoms on the x axis, `spacing` Angstrom apart."""
    return Fragment(elems=["He"] * count, coords=[[spacing * i, 0.0, 0.0] for i in range(count)], conncalc=False)


def _pair(separation=3.0):
    return Fragment(elems=["He", "He"], coords=[[0.0, 0.0, 0.0], [separation, 0.0, 0.0]], conncalc=False)


def _lj_kj(r_nm, sigma=0.3, epsilon=0.5):
    return 4 * epsilon * ((sigma / r_nm) ** 12 - (sigma / r_nm) ** 6)


def _table_row(text, label):
    match = re.search(rf"{re.escape(label)}\s+\|\s*(\S+)\s*\|\s*(\S+)", text)
    assert match is not None, label
    return [float(match.group(1)), float(match.group(2))]


def test_two_atom_constraints_take_their_distance_from_the_fragment():
    fragment = Fragment(elems=["He"] * 3, coords=[[0, 0, 0], [3, 4, 0], [6, 8, 0]], conncalc=False)

    mm = dummy_mm(fragment, bondconstraints=[[0, 1], [1, 2, 2.5]])

    assert mm.user_constraints == [[0, 1, pytest.approx(5.0)], [1, 2, 2.5]]
    assert mm.system.getNumConstraints() == 2
    i, j, distance = mm.system.getConstraintParameters(0)
    assert (i, j) == (0, 1)
    assert distance.value_in_unit(openmm.unit.angstrom) == pytest.approx(5.0)


def test_two_atom_constraints_fall_back_to_the_pdb_coordinates():
    fragment, mm = meoh_water_mm(platform="Reference", bondconstraints=[[0, 1]])
    expected = float(np.linalg.norm(fragment.coords[0] - fragment.coords[1]))

    _, _, distance = mm.system.getConstraintParameters(0)
    assert distance.value_in_unit(openmm.unit.angstrom) == pytest.approx(expected, abs=2e-3)  # PDB precision


def test_two_atom_constraints_need_a_fragment_or_a_pdbfile():
    fragment = Fragment(xyzfile=str(TEST_DIR / "xyzfiles" / "h2o_MeOH.xyz"))
    topology = openmm.app.PDBFile(fragment.write_pdbfile_openmm(filename="meoh.pdb", skip_connectivity=True)).topology

    with pytest.raises(InputError, match="requires a fragment or a PDB file"):
        OpenMMTheory(
            topoforce=True,
            topology=topology,
            forcefield=openmm.app.ForceField(MEOH_XML),
            bondconstraints=[[0, 1]],
            platform="Reference",
            autoconstraints=None,
            rigidwater=False,
        )


def test_many_constraints_are_counted_rather_than_listed(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    mm = dummy_mm(_chain(52), bondconstraints=[[i, i + 1] for i in range(51)])

    assert "51 user-defined constraints to add." in caplog.text
    assert "User-constraints to add (bond)" not in caplog.text
    assert mm.system.getNumConstraints() == 51
    assert [con[:2] for con in mm.user_constraints] == [[i, i + 1] for i in range(51)]
    assert [con[2] for con in mm.user_constraints] == pytest.approx([2.0] * 51)


def test_many_frozen_atoms_zero_masses_exclude_their_pairs_and_reduce_dof(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    mm = dummy_mm(_chain(52), frozen_atoms=list(range(50)))

    assert "50 user-defined frozen atoms to add." in caplog.text
    masses = [mm.system.getParticleMass(i).value_in_unit(DALTON) for i in range(52)]
    assert masses[:50] == [0.0] * 50
    assert all(mass > 0 for mass in masses[50:])
    assert mm.nonbonded_force.getNumExceptions() == 50 * 49 // 2
    assert any(isinstance(force, openmm.CMMotionRemover) for force in mm.system.getForces())
    assert mm.dof == 2 * 3 - 3


def test_many_restraints_build_one_named_harmonic_force(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    fragment = _chain(52)

    mm = dummy_mm(fragment, restraints=[[i, i + 1, 2.0, 1.0] for i in range(51)])

    assert "51 user-defined restraints to add." in caplog.text
    restraint = next(force for force in mm.system.getForces() if force.getName() == "OpenMMQMMM restraint")
    assert isinstance(restraint, openmm.HarmonicBondForce)
    assert restraint.getNumBonds() == 51
    i, j, length, k = restraint.getBondParameters(10)
    assert (i, j) == (10, 11)
    assert length.value_in_unit(NM) == pytest.approx(0.2)
    assert k.value_in_unit(KJ / NM**2) == pytest.approx(KJ_PER_NM2_PER_KCAL_PER_A2)
    coords = fragment.coords.copy()
    assert evaluate(mm.system, coords / 10)[0] == pytest.approx(0.0, abs=1e-12)
    coords[10, 0] += 0.1  # stretches bond 9-10 and compresses 10-11 by 0.01 nm each
    assert evaluate(mm.system, coords / 10)[0] == pytest.approx(2 * 0.5 * KJ_PER_NM2_PER_KCAL_PER_A2 * 0.01**2)


def test_clean_up_constraints_fills_missing_distances_from_the_fragment():
    fragment = Fragment(elems=["He"] * 3, coords=[[0, 0, 0], [3, 4, 0], [3, 4, 12]], conncalc=False)

    result = clean_up_constraints_list(fragment=fragment, constraints=[[0, 1], [1, 2], [0, 2, 7.5]])

    assert result == [[0, 1, pytest.approx(5.0)], [1, 2, pytest.approx(12.0)], [0, 2, 7.5]]


def test_update_lj_epsilons_changes_only_the_listed_atoms():
    mm, force = nonbonded_mm([0.1, -0.1, 0.0])

    mm.update_lj_epsilons([2, 0], [0.5 * KJ, 0.0 * KJ])

    epsilons = [force.getParticleParameters(i)[2].value_in_unit(KJ) for i in range(3)]
    assert epsilons == pytest.approx([0.0, 0.2, 0.5])
    charges = [force.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge) for i in range(3)]
    assert charges == pytest.approx([0.1, -0.1, 0.0])
    assert [epsilon.value_in_unit(KJ) for epsilon in mm.get_lj_epsilons([2, 0])] == pytest.approx([0.5, 0.0])


def test_update_lj_epsilons_rejects_a_length_mismatch():
    mm, _ = nonbonded_mm([0.1, -0.1])

    with pytest.raises(InternalError, match="size mismatch"):
        mm.update_lj_epsilons([0, 1], [0.1 * KJ])


def test_addexceptions_covers_custom_nonbonded_exclusions_and_charge_scales():
    mm, force = nonbonded_mm([0.1, -0.1, 0.2, 0.0])
    custom = openmm.CustomNonbondedForce("A1*A2/r^12-C1*C2/r^6")
    custom.addPerParticleParameter("A")
    custom.addPerParticleParameter("C")
    for _ in range(4):
        custom.addParticle([1.0, 1.0])
    custom.addExclusion(0, 1)
    mm.system.addForce(custom)
    mm._exception_charge_scales = {(0, 3): 0.5}

    mm.addexceptions([0, 1, 2])

    assert force.getNumExceptions() == 3
    excluded = set()
    for index in range(3):
        p1, p2, chargeprod, _sigma, epsilon = force.getExceptionParameters(index)
        assert chargeprod.value_in_unit(openmm.unit.elementary_charge**2) == 0
        assert epsilon.value_in_unit(KJ) == 0
        excluded.add(frozenset((p1, p2)))
    pairs = {frozenset(pair) for pair in [(0, 1), (0, 2), (1, 2)]}
    assert excluded == pairs
    assert custom.getNumExclusions() == 3
    assert {frozenset(custom.getExclusionParticles(i)) for i in range(3)} == pairs
    assert mm._exception_charge_scales == {(0, 3): 0.5, (0, 1): 0.0, (0, 2): 0.0, (1, 2): 0.0}


def test_qmmm_lj_energy_counts_only_cross_region_pairs():
    mm, _ = nonbonded_mm([0.3, -0.3, 0.1])  # conftest gives every particle sigma 0.3 nm, epsilon 0.2 kJ/mol
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    expected = (_lj_kj(0.4, 0.3, 0.2) + _lj_kj(0.8, 0.3, 0.2)) / HARTREE_TO_KJ_PER_MOL

    assert mm.qmmm_lj_energy([0], coords) == pytest.approx(expected, rel=1e-10)
    assert mm.qmmm_lj_energy([1, 2], coords) == pytest.approx(expected, rel=1e-10)


def test_qmmm_lj_energy_includes_supported_custom_bond_lj_terms():
    mm, force = nonbonded_mm([0.0, 0.0, 0.0])
    for i in range(3):
        force.setParticleParameters(i, 0.0, 0.3, 0.0)
    bond = openmm.CustomBondForce("4*epsilon*((sigma/r)^12-(sigma/r)^6)")
    bond.addPerBondParameter("sigma")
    bond.addPerBondParameter("epsilon")
    bond.addBond(0, 1, [0.3, 0.5])
    bond.addBond(1, 2, [0.3, 0.5])
    mm.system.addForce(bond)
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])

    assert mm.qmmm_lj_energy([0], coords) == pytest.approx(_lj_kj(0.4) / HARTREE_TO_KJ_PER_MOL, rel=1e-10)


def test_qmmm_lj_energy_honours_exceptions_and_parameter_offsets():
    mm, force = nonbonded_mm([0.0, 0.0, 0.0])  # sigma 0.3 nm, epsilon 0.2 kJ/mol per particle
    exception = force.addException(0, 1, 0.0, 0.25, 0.4)
    force.addGlobalParameter("lambda", 1.0)
    force.addParticleParameterOffset("lambda", 2, 0.0, 0.0, 0.3)
    force.addExceptionParameterOffset("lambda", exception, 0.0, 0.0, 0.1)
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    pair_01 = _lj_kj(0.4, sigma=0.25, epsilon=0.4 + 0.1)
    pair_02 = _lj_kj(0.8, sigma=0.3, epsilon=np.sqrt(0.2 * (0.2 + 0.3)))

    assert mm.qmmm_lj_energy([0], coords) == pytest.approx((pair_01 + pair_02) / HARTREE_TO_KJ_PER_MOL, rel=1e-10)


def test_qmmm_lj_energy_handles_the_supported_custom_nonbonded_form():
    mm, force = nonbonded_mm([0.0, 0.0, 0.0])
    for i in range(3):
        force.setParticleParameters(i, 0.0, 0.3, 0.0)
    custom = _custom_nonbonded("A1*A2/r^12-C1*C2/r^6", ["A", "C"], count=3)
    a, c = 1e-4, 1e-2
    for i in range(3):
        custom.setParticleParameters(i, [a, c])
    mm.system.addForce(custom)
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]])
    expected = sum(a * a / r**12 - c * c / r**6 for r in (0.4, 0.8))

    assert mm.qmmm_lj_energy([0], coords) == pytest.approx(expected / HARTREE_TO_KJ_PER_MOL, rel=1e-10)
    assert mm.qmmm_lj_energy([1, 2], coords) == pytest.approx(expected / HARTREE_TO_KJ_PER_MOL, rel=1e-10)


@pytest.mark.parametrize(
    ("atoms", "coords", "message"),
    [
        ([3], np.zeros((3, 3)), "outside the MM system"),
        ([-1], np.zeros((3, 3)), "outside the MM system"),
        ([0.0], np.zeros((3, 3)), "outside the MM system"),
        ([0], np.zeros((2, 3)), "finite coordinates for every MM particle"),
        ([0], np.full((3, 3), np.nan), "finite coordinates for every MM particle"),
    ],
)
def test_qmmm_lj_energy_validates_atoms_and_coordinates(atoms, coords, message):
    mm, _ = nonbonded_mm([0.0, 0.0, 0.0])

    with pytest.raises(InputError, match=message):
        mm.qmmm_lj_energy(atoms, coords)


def _custom_nonbonded(expression, parameters, count=2):
    force = openmm.CustomNonbondedForce(expression)
    for name in parameters:
        force.addPerParticleParameter(name)
    for _ in range(count):
        force.addParticle([1.0] * len(parameters))
    return force


def test_qmmm_lj_energy_rejects_unknown_custom_nonbonded_expressions():
    mm, _ = nonbonded_mm([0.0, 0.0])
    mm.system.addForce(_custom_nonbonded("r^2", []))

    with pytest.raises(InputError, match="does not support this CustomNonbondedForce expression"):
        mm.qmmm_lj_energy([0], np.zeros((2, 3)))


def test_qmmm_lj_energy_rejects_a_custom_nonbonded_parameter_name_clash():
    mm, _ = nonbonded_mm([0.0, 0.0])
    mm.system.addForce(_custom_nonbonded("A1*A2/r^12-C1*C2/r^6", ["A", "C", "qmmmLJRegion"]))

    with pytest.raises(InputError, match="conflicts with the custom force"):
        mm.qmmm_lj_energy([0], np.zeros((2, 3)))


def _custom_bond(expression, parameters):
    force = openmm.CustomBondForce(expression)
    for name in parameters:
        force.addPerBondParameter(name)
    force.addBond(0, 1, [1.0] * len(parameters))
    return force


def test_qmmm_lj_energy_rejects_unknown_custom_bond_expressions():
    mm, _ = nonbonded_mm([0.0, 0.0])
    mm.system.addForce(_custom_bond("0.5*k*(r-r0)^2", ["k", "r0"]))

    with pytest.raises(InputError, match="does not support this CustomBondForce expression"):
        mm.qmmm_lj_energy([0], np.zeros((2, 3)))


def test_qmmm_lj_energy_rejects_a_custom_bond_parameter_name_clash():
    mm, _ = nonbonded_mm([0.0, 0.0])
    mm.system.addForce(_custom_bond("4*epsilon*((sigma/r)^12-(sigma/r)^6)", ["sigma", "epsilon", "qmmmLJCross"]))

    with pytest.raises(InputError, match="conflicts with the custom force"):
        mm.qmmm_lj_energy([0], np.zeros((2, 3)))


def test_run_falls_back_to_fragment_or_internal_coordinates():
    fragment = _pair(3.0)
    mm = dummy_mm(fragment, restraints=[[0, 1, 1.0, 1.0]])
    mm.force_run = True
    expected = 0.5 * KJ_PER_NM2_PER_KCAL_PER_A2 * 0.2**2 / HARTREE_TO_KJ_PER_MOL

    assert mm.run(current_coords=fragment.coords) == pytest.approx(expected, rel=1e-10)
    assert mm.run(fragment=fragment) == pytest.approx(expected, rel=1e-10)
    with pytest.raises(FileFormatError, match="Found no coordinates"):
        mm.run()
    mm.coords = fragment.coords
    assert mm.run() == pytest.approx(expected, rel=1e-10)


def test_run_applies_a_box_override_for_the_single_point_only():
    fragment, mm = meoh_water_mm(platform="Reference")
    use_pme(mm)
    box = np.diag([25.0] * 3)

    energy = mm.run(current_coords=fragment.coords, periodic_box_vectors=box)

    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(mm.system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPeriodicBoxVectors(*(box / 10))
    context.setPositions(fragment.coords / 10)
    expected = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(KJ) / HARTREE_TO_KJ_PER_MOL
    del context, integrator
    assert energy == pytest.approx(expected, rel=1e-10)
    assert mm._cell_gradient_snapshot[1] == pytest.approx(box / 10)
    assert mm.get_pbc_vectors() == pytest.approx(np.diag([30.0] * 3))


@pytest.mark.parametrize("box", [np.eye(2), -25.0 * np.eye(3), np.full((3, 3), np.nan), np.zeros((3, 3))])
def test_run_rejects_invalid_box_overrides(box):
    fragment, mm = meoh_water_mm(platform="Reference")

    with pytest.raises(InputError, match="positive volume"):
        mm.run(current_coords=fragment.coords, periodic_box_vectors=box)


@pytest.mark.parametrize("apply_constraints", [False, True])
def test_run_can_apply_bond_constraints_before_the_energy(apply_constraints):
    fragment = _pair(1.2)
    mm = dummy_mm(
        fragment,
        bondconstraints=[[0, 1, 1.0]],
        restraints=[[0, 1, 1.0, 100.0]],
        applyconstraints_in_run=apply_constraints,
    )
    mm.force_run = True
    unconstrained = 0.5 * 100.0 * KJ_PER_NM2_PER_KCAL_PER_A2 * 0.02**2 / HARTREE_TO_KJ_PER_MOL

    energy = mm.run(current_coords=fragment.coords)

    if apply_constraints:
        assert energy == pytest.approx(0.0, abs=1e-9)
    else:
        assert energy == pytest.approx(unconstrained, rel=1e-10)


def test_run_zeroes_the_gradient_of_virtual_sites_and_transfers_it_to_parents():
    fragment = Fragment(elems=["He"] * 3, coords=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]], conncalc=False)
    mm = dummy_mm(fragment)
    mm.system.setVirtualSite(2, openmm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    mm.system.setParticleMass(2, 0)
    k = 100.0  # kJ/mol/nm^2, zero equilibrium length: E = k/8 |r1 - r0|^2
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 2, 0.0, k)
    mm.system.addForce(bond)
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [9.0, 9.0, 9.0]])  # the virtual-site row is recomputed

    energy, gradient = mm.run(current_coords=coords, grad=True)

    separation_nm = np.array([0.4, 0.0, 0.0])
    assert energy == pytest.approx(k / 8 * 0.4**2 / HARTREE_TO_KJ_PER_MOL, rel=1e-10)
    expected_on_atom1 = k / 4 * separation_nm / HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    assert gradient[1] == pytest.approx(expected_on_atom1, rel=1e-10)
    assert gradient[0] == pytest.approx(-expected_on_atom1, rel=1e-10)
    assert gradient[2] == pytest.approx(np.zeros(3), abs=0.0)


def test_run_with_force_run_logs_the_autoconstraint_incompatibility(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    fragment = _pair(3.0)
    mm = dummy_mm(fragment, rigidwater=True)

    with pytest.raises(InputError, match="force_run"):
        mm.run(current_coords=fragment.coords)

    mm.force_run = True
    assert mm.run(current_coords=fragment.coords) == pytest.approx(0.0, abs=1e-12)
    assert "not compatible with OpenMMTheory.run()" in caplog.text
    assert "force_run is True. Will continue" in caplog.text


def test_energy_decomposition_rows_sum_to_the_independent_total(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    fragment, mm = meoh_water_mm(platform="Reference", do_energy_decomposition=True)

    energy = mm.run(current_coords=fragment.coords)

    total_kj = evaluate(mm.system, fragment.coords / 10)[0]
    total_kcal = total_kj / KCAL_TO_KJ
    assert energy * HARTREE_TO_KJ_PER_MOL == pytest.approx(total_kj, rel=1e-9)
    # The table prints two decimals.
    assert _table_row(caplog.text, "NonbondedForce") == pytest.approx([total_kj, total_kcal], abs=0.01)
    assert _table_row(caplog.text, "CMMotionRemover") == pytest.approx([0.0, 0.0], abs=0.01)
    assert _table_row(caplog.text, "Sumcomponents") == pytest.approx([total_kj, total_kcal], abs=0.01)
    assert _table_row(caplog.text, "Total") == pytest.approx([total_kj, total_kcal], abs=0.01)


def test_disabling_repartitioning_skips_heavy_bonds_and_physical_hydrogens_and_updates_frozen_masses():
    topology = openmm.app.Topology()
    residue = topology.addResidue("MOL", topology.addChain())
    element = openmm.app.Element.getBySymbol
    carbon = topology.addAtom("C", element("C"), residue)
    oxygen = topology.addAtom("O", element("O"), residue)
    heavy_hydrogen = topology.addAtom("H1", element("H"), residue)
    light_hydrogen = topology.addAtom("H2", element("H"), residue)
    topology.addBond(carbon, oxygen)
    topology.addBond(heavy_hydrogen, carbon)  # hydrogen listed first
    topology.addBond(oxygen, light_hydrogen)
    physical = [atom.element.mass.value_in_unit(DALTON) for atom in topology.atoms()]
    target = 1.5 * DALTON
    transferred = target - heavy_hydrogen.element.mass

    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.topology = topology
    theory.system = openmm.System()
    theory.system.addParticle(carbon.element.mass - transferred)
    theory.system.addParticle(oxygen.element.mass)
    theory.system.addParticle(target)
    theory.system.addParticle(light_hydrogen.element.mass)
    theory.system_masses_original = [theory.system.getParticleMass(i) for i in range(4)]
    theory.hydrogenmass = target
    theory._prefreeze_masses = {}
    theory.freeze_atoms(frozen_atoms=[2])

    theory.set_simulation_parameters(integrator="RPMDIntegrator")

    masses = [theory.system.getParticleMass(i).value_in_unit(DALTON) for i in range(4)]
    assert masses == pytest.approx([physical[0], physical[1], 0.0, physical[3]])
    assert theory._prefreeze_masses[2].value_in_unit(DALTON) == pytest.approx(physical[2])
    assert [m.value_in_unit(DALTON) for m in theory.system_masses_original] == pytest.approx(physical)
    assert theory.hydrogenmass is None
    theory.unfreeze_atoms()
    assert theory.system.getParticleMass(2).value_in_unit(DALTON) == pytest.approx(physical[2])


def test_update_cell_clamps_the_cutoff_to_half_the_box_and_restores_it():
    _, mm = meoh_water_mm(platform="Reference")
    use_pme(mm)  # 3 nm cube with a 1 nm cutoff
    mm.periodic_cell_vectors = np.diag([30.0] * 3)

    mm.update_cell(periodic_cell_dimensions=[15, 15, 15, 90, 90, 90])

    assert mm.periodic_cell_vectors == pytest.approx(np.diag([15.0] * 3))
    assert mm.get_pbc_vectors() == pytest.approx(np.diag([15.0] * 3))
    assert np.array(mm.topology.getPeriodicBoxVectors().value_in_unit(NM)) == pytest.approx(np.eye(3) * 1.5)
    assert mm.nonbonded_force.getCutoffDistance().value_in_unit(NM) == pytest.approx(0.499 * 1.5)

    mm.update_cell(periodic_cell_vectors=np.diag([30.0] * 3))

    assert mm.nonbonded_force.getCutoffDistance().value_in_unit(NM) == pytest.approx(1.0)
    assert mm.get_pbc_vectors() == pytest.approx(np.diag([30.0] * 3))


def test_periodic_centerforce_uses_the_minimum_image_distance():
    fragment = Fragment(elems=["He"], coords=[[0.0, 0.0, 0.0]], conncalc=False)
    mm = dummy_mm(fragment)
    use_pme(mm)  # 30 Angstrom cube

    mm.add_centerforce(center_coords=[5.0, 5.0, 5.0], atomindices=[0], forceconstant=1.0, distance=5.0)

    expected = 0.5 * 1.0 * KCAL_TO_KJ * (7.0 - 5.0) ** 2  # 7 Angstrom from the centre, flat inside 5
    assert evaluate(mm.system, np.array([[12.0, 5.0, 5.0]]) / 10)[0] == pytest.approx(expected)
    assert evaluate(mm.system, np.array([[42.0, 5.0, 5.0]]) / 10)[0] == pytest.approx(expected)
    assert evaluate(mm.system, np.array([[8.0, 5.0, 5.0]]) / 10)[0] == pytest.approx(0.0, abs=1e-12)
