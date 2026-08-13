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
