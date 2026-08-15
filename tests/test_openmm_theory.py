from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, OpenMMTheory, openmm_minimize, single_point
from openmmqmmm.exceptions import InputError

TEST_DIR = Path(__file__).parent
SOLVATED_PDB = str(TEST_DIR / "pdbfiles" / "1aki_solvated.pdb")
CHARMM_XML = ["charmm36.xml", "charmm36/water.xml"]


@pytest.fixture(scope="module")
def solvated_fragment():
    return Fragment(pdbfile=SOLVATED_PDB)


def _make_theory(**kwargs):
    options = {
        "xmlfiles": CHARMM_XML,
        "pdbfile": SOLVATED_PDB,
        "periodic": True,
        "autoconstraints": None,
        "rigidwater": False,
    }
    options.update(kwargs)
    return OpenMMTheory(**options)


def test_run_returns_energy_and_gradient(solvated_fragment):
    """The theory contract: energy alone, or (energy, gradient) when asked."""
    theory = _make_theory()

    energy = theory.run(current_coords=solvated_fragment.coords, elems=solvated_fragment.elems)
    assert np.isscalar(energy) or np.ndim(energy) == 0

    energy_again, gradient = theory.run(
        current_coords=solvated_fragment.coords, elems=solvated_fragment.elems, grad=True
    )
    assert energy_again == pytest.approx(energy, rel=1e-9), "The gradient must not change the energy"
    assert gradient.shape == (solvated_fragment.numatoms, 3), "One gradient row per atom"


def test_energy_is_translationally_invariant(solvated_fragment):
    theory = _make_theory()

    energy = theory.run(current_coords=solvated_fragment.coords, elems=solvated_fragment.elems)
    shifted = theory.run(current_coords=solvated_fragment.coords + 0.37, elems=solvated_fragment.elems)

    assert shifted == pytest.approx(energy, rel=1e-4)


def test_atom_charges_sum_to_the_system_charge():
    theory = _make_theory()

    charges = theory.getatomcharges()

    assert len(charges) == theory.numatoms
    # Lysozyme with neutralising ions: the total charge must be a whole number.
    assert sum(charges) == pytest.approx(round(sum(charges)), abs=1e-3)


def test_update_charges_zeroes_the_qm_region():
    """QMMMTheory zeroes the QM-region charges this way before embedding."""
    theory = _make_theory()
    qmatoms = [0, 1, 2, 3]

    theory.update_charges(qmatoms, [0.0] * len(qmatoms))

    charges = theory.getatomcharges()
    assert all(charges[i] == 0.0 for i in qmatoms)
    assert any(charge != 0.0 for charge in charges), "Only the listed atoms are zeroed"


def test_freezing_atoms_sets_their_mass_to_zero():
    theory = _make_theory()
    frozen = [0, 1, 2]
    original_masses = [theory.system.getParticleMass(i)._value for i in frozen]

    theory.freeze_atoms(frozen_atoms=frozen)
    assert all(theory.system.getParticleMass(i)._value == 0.0 for i in frozen)

    theory.unfreeze_atoms()
    assert [theory.system.getParticleMass(i)._value for i in frozen] == pytest.approx(original_masses)


def test_add_and_remove_force_restores_the_force_count():
    theory = _make_theory()
    before = theory.system.getNumForces()

    theory.add_custom_bond_force(0, 1, 1.0, 100.0)
    assert theory.system.getNumForces() == before + 1

    theory.remove_force(theory.system.getNumForces() - 1)
    assert theory.system.getNumForces() == before


def test_constraints_can_be_added_and_removed():
    theory = _make_theory()
    before = theory.system.getNumConstraints()

    theory.add_bondconstraints(constraints=[[0, 1, 1.0]])
    assert theory.system.getNumConstraints() == before + 1

    theory.remove_all_constraints()
    assert theory.system.getNumConstraints() == 0


def test_periodic_cell_vectors_are_available():
    theory = _make_theory()

    vectors = theory.get_pbc_vectors()

    assert np.array(vectors).shape == (3, 3)
    assert np.array(vectors).any(), "A periodic system must report a non-zero cell"


def test_energy_decomposition_covers_the_total(solvated_fragment):
    theory = _make_theory()
    total = theory.run(current_coords=solvated_fragment.coords, elems=solvated_fragment.elems)

    theory.forcegroupify()
    simulation = theory.create_simulation()
    theory.set_positions(solvated_fragment.coords, simulation)
    decomposition = theory.get_energy_decomposition(simulation.context)

    components_kj = sum(value._value for value in decomposition.values())
    # run() reports Hartree; the decomposition comes straight from OpenMM in kJ/mol.
    assert components_kj / 2625.4996394799 == pytest.approx(total, rel=1e-4)


def test_set_positions_initializes_every_rpmd_copy():
    import openmm
    import openmm.app

    num_copies = 4
    system = openmm.System()
    system.addParticle(1.0)
    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("X", chain)
    topology.addAtom("H", openmm.app.Element.getByAtomicNumber(1), residue)

    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.system = system
    theory.set_rpmd_num_copies(num_copies)
    theory.set_simulation_parameters(integrator="RPMDIntegrator")
    theory.create_integrator()
    integrator = theory.integrator
    simulation = openmm.app.Simulation(
        topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    coords = np.array([[1.25, -2.5, 3.75]])

    theory.set_positions(coords, simulation)

    expected_nm = coords * 0.1
    for copy_index in range(num_copies):
        state = integrator.getState(copy_index, getPositions=True)
        actual_nm = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
        assert actual_nm == pytest.approx(expected_nm)


def test_rpmd_rejects_constrained_system_before_context_creation():
    import openmm

    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.system = openmm.System()
    theory.system.addParticle(1.0)
    theory.system.addParticle(1.0)
    theory.system.addConstraint(0, 1, 0.1)
    theory.set_rpmd_num_copies(4)
    theory.set_simulation_parameters(integrator="RPMDIntegrator")

    with pytest.raises(InputError, match=r"RPMDIntegrator does not support constraints.*autoconstraints=None"):
        theory.create_integrator()


@pytest.mark.parametrize("integrator_name", ["RPMDIntegrator", "QTBIntegrator"])
def test_nuclear_quantum_integrators_disable_hydrogen_mass_repartitioning(integrator_name):
    import openmm
    import openmm.app

    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    carbon = topology.addAtom("C", openmm.app.Element.getByAtomicNumber(6), residue)
    hydrogen = topology.addAtom("H", openmm.app.Element.getByAtomicNumber(1), residue)
    topology.addBond(carbon, hydrogen)

    physical_carbon_mass = carbon.element.mass
    physical_hydrogen_mass = hydrogen.element.mass
    repartitioned_hydrogen_mass = 1.5 * openmm.unit.dalton
    transferred_mass = repartitioned_hydrogen_mass - physical_hydrogen_mass

    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.topology = topology
    theory.system = openmm.System()
    theory.system.addParticle(physical_carbon_mass - transferred_mass)
    theory.system.addParticle(repartitioned_hydrogen_mass)
    theory.system_masses_original = [theory.system.getParticleMass(i) for i in range(2)]
    theory.hydrogenmass = repartitioned_hydrogen_mass

    total_mass_before = sum(theory.system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(2))
    theory.set_simulation_parameters(integrator=integrator_name)

    assert theory.hydrogenmass is None
    assert theory.system.getParticleMass(0).value_in_unit(openmm.unit.dalton) == pytest.approx(
        physical_carbon_mass.value_in_unit(openmm.unit.dalton)
    )
    assert theory.system.getParticleMass(1).value_in_unit(openmm.unit.dalton) == pytest.approx(
        physical_hydrogen_mass.value_in_unit(openmm.unit.dalton)
    )
    assert sum(theory.system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(2)) == pytest.approx(
        total_mass_before
    )


def test_qtb_integrator_is_created():
    import openmm

    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.system = openmm.System()
    theory.system.addParticle(1.0)
    theory.set_simulation_parameters(integrator="QTBIntegrator")

    theory.create_integrator()

    assert isinstance(theory.integrator, openmm.QTBIntegrator)


@pytest.mark.parametrize("num_copies", [0, -1, 1.5, True, "8"])
def test_rpmd_copy_count_must_be_a_positive_integer(num_copies):
    theory = OpenMMTheory.__new__(OpenMMTheory)

    with pytest.raises(InputError, match="rpmd_num_copies must be a positive integer"):
        theory.set_rpmd_num_copies(num_copies)


def test_rpmd_force_group_contractions_reach_the_integrator():
    import openmm

    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.system = openmm.System()
    theory.system.addParticle(1.0)
    theory.set_rpmd_num_copies(4)
    theory.set_rpmd_contractions({7: 1})
    theory.set_simulation_parameters(integrator="RPMDIntegrator")
    theory.create_integrator()

    assert dict(theory.integrator.getContractions()) == {7: 1}


@pytest.mark.parametrize("contractions", [{-1: 1}, {32: 1}, {1: 0}, {1: 5}, {True: 1}, {1: True}])
def test_rpmd_force_group_contractions_are_validated(contractions):
    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.set_rpmd_num_copies(4)

    with pytest.raises(InputError, match="RPMD contraction"):
        theory.set_rpmd_contractions(contractions)


def test_write_pdbfile_uses_the_positions_it_is_given(tmp_path, solvated_fragment):
    """An explicit positions argument must win over the object's own coordinates."""
    import openmm

    theory = _make_theory()
    shifted_nm = (solvated_fragment.coords + 5.0) * 0.1
    positions = [openmm.Vec3(*row) for row in shifted_nm] * openmm.unit.nanometer

    outputname = str(tmp_path / "written")
    theory.write_pdbfile(positions=positions, outputname=outputname)

    written = (tmp_path / "written.pdb").read_text()
    assert "ATOM" in written
    first_atom_x = float(next(line for line in written.splitlines() if line.startswith("ATOM"))[30:38])
    assert first_atom_x == pytest.approx(solvated_fragment.coords[0][0] + 5.0, abs=0.01)


def test_openmm_minimize_lowers_the_energy(solvated_fragment):
    """Minimisation must not raise the energy, and must update the fragment."""
    theory = _make_theory()
    fragment = Fragment(pdbfile=SOLVATED_PDB)
    before = theory.run(current_coords=fragment.coords, elems=fragment.elems)
    starting_coords = fragment.coords.copy()

    openmm_minimize(fragment=fragment, theory=theory, maxiter=20, use_reporter=False)

    after = theory.run(current_coords=fragment.coords, elems=fragment.elems)
    assert after <= before, "Minimisation must not increase the energy"
    assert not np.allclose(fragment.coords, starting_coords), "The fragment geometry must be updated"


def test_openmm_minimize_rejects_a_non_openmm_theory():
    fragment = Fragment(pdbfile=SOLVATED_PDB)

    with pytest.raises(InputError):
        openmm_minimize(fragment=fragment, theory=None)


def test_single_point_through_the_job_function(solvated_fragment):
    theory = _make_theory()

    result = single_point(theory=theory, fragment=solvated_fragment, grad=True)

    assert result.energy is not None
    assert np.shape(result.gradient) == (solvated_fragment.numatoms, 3)
