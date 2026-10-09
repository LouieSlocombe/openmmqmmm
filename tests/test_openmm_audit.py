"""Small physical regressions for the duplication audit (#68)."""

from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest

from openmmqmmm import Fragment, MolecularDynamicsEngine, OpenMMTheory
from openmmqmmm.constants import HARTREE_TO_KCAL_PER_MOL

FIXTURES = Path(__file__).parent / "fixtures"


def _bare_theory():
    fragment = Fragment(elems=["He"] * 4, coords=[[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 2, 2]], conncalc=False)
    theory = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    return theory, fragment


def test_multiple_restraints_have_independent_parameters():
    theory, fragment = _bare_theory()
    theory.add_custom_bond_force(0, 1, 1, 2)
    theory.add_custom_bond_force(1, 2, 1.5, 4)
    theory.add_custom_angle_force(0, 1, 2, 1, 3)
    theory.add_custom_torsion_force(0, 1, 2, 3, 0, 5)
    theory.add_centerforce([0, 0, 0], [3], 6, 1)
    energy = theory.run(current_coords=fragment.coords) * HARTREE_TO_KCAL_PER_MOL
    expected = 0.5 * 2 + 0.5 * 4 * 0.5**2 + 0.5 * 3 * (np.pi / 2 - 1) ** 2
    expected += 0.5 * 5 * (np.pi / 2) ** 2 + 0.5 * 6 * (np.sqrt(12) - 1) ** 2
    assert energy == pytest.approx(expected)
    assert not theory.system.usesPeriodicBoundaryConditions()


def test_explicit_restraints_survive_qmmm_bonded_force_removal():
    theory, fragment = _bare_theory()
    theory.add_bondrestraints([[0, 1, 1, 2]])
    before = theory.run(current_coords=fragment.coords)
    theory.modify_bonded_forces([0, 1, 2, 3])
    assert theory.run(current_coords=fragment.coords) == pytest.approx(before)


def test_topology_built_theory_keeps_every_atom_label():
    theory, _ = _bare_theory()
    assert theory.resids == [0, 0, 0, 0]
    assert theory.resnames == [res.name for res in theory.topology.residues() for _ in res.atoms()]
    assert theory.atomnames == [atom.name for atom in theory.topology.atoms()]
    topology = openmm.app.Topology()
    residue = topology.addResidue("HOH", topology.addChain())
    topology.addAtom("O", openmm.app.element.oxygen, residue)
    topology.addAtom("M", None, residue)
    theory.define_mm_elements(topology)
    assert theory.mm_elements == ["O", "M"]


def test_periodic_box_accessor_reads_system_default():
    theory, _ = _bare_theory()
    theory.system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.5))
    assert theory.get_pbc_vectors() == pytest.approx(np.eye(3) * 35)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("loader", ["charmm", "gromacs", "amber"])
def test_small_real_loader_fixtures_create_systems(loader, periodic):
    options = dict(  # noqa: C408
        platform="Reference",
        periodic=periodic,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
        periodic_cell_dimensions=[30, 30, 30, 90, 90, 90],
        periodic_nonbonded_cutoff=10,
        switching_function_distance=8,
    )
    if loader == "charmm":
        options.update(
            charmm_files=True,
            psffile=str(FIXTURES / "three-waters.psf"),
            charmmtopfile=str(FIXTURES / "three-waters.rtf"),
            charmmprmfile=str(FIXTURES / "three-waters.prm"),
        )
    elif loader == "gromacs":
        options.update(
            gromacs_files=True,
            grofile=str(FIXTURES / "three-waters.gro"),
            gromacstopfile=str(FIXTURES / "three-waters.top"),
        )
    else:
        options.update(amber_files=True, amberprmtopfile=str(FIXTURES / "three-waters.prmtop"))
    theory = OpenMMTheory(**options)
    assert theory.numatoms == 9
    coords = np.array(
        openmm.app.GromacsGroFile(str(FIXTURES / "three-waters.gro")).getPositions().value_in_unit(openmm.unit.angstrom)
    )
    energy, gradient = theory.run(current_coords=coords, grad=True)
    assert np.isfinite(energy)
    assert np.all(np.isfinite(gradient))
    assert theory.resids == [0] * 3 + [1] * 3 + [2] * 3


@pytest.mark.parametrize("periodic", [False, True])
def test_residue_template_choices_include_every_name(monkeypatch, periodic):
    topology = openmm.app.Topology()
    chain = topology.addChain()
    for name in ["FIRST", "SECOND"]:
        topology.addAtom("He", openmm.app.element.helium, topology.addResidue(name, chain))
    forcefield = openmm.app.ForceField()
    seen = {}

    def create_system(topology, **kwargs):
        seen.update(kwargs["residueTemplates"])
        system = openmm.System()
        force = openmm.NonbondedForce()
        for _ in topology.atoms():
            system.addParticle(4)
            force.addParticle(0, 1, 0)
        system.addForce(force)
        if topology.getPeriodicBoxVectors() is not None:
            system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
        return system

    monkeypatch.setattr(forcefield, "createSystem", create_system)
    OpenMMTheory(
        topoforce=True,
        topology=topology,
        forcefield=forcefield,
        platform="Reference",
        periodic=periodic,
        periodic_cell_dimensions=[30, 30, 30, 90, 90, 90],
        residuetemplate_choice={"FIRST": "A", "SECOND": "B"},
    )
    assert {res.name: template for res, template in seen.items()} == {"FIRST": "A", "SECOND": "B"}


def test_finalization_refreshes_theory_periodic_cell():
    theory, fragment = _bare_theory()
    theory.periodic = True
    theory.update_cell(periodic_cell_vectors=np.eye(3) * 30)
    engine = MolecularDynamicsEngine(fragment=fragment, theory=theory, platform="Reference")
    engine.run(simulation_steps=1)
    engine.simulation.context.setPeriodicBoxVectors(*(np.eye(3) * 4))
    engine.finalize_simulation()
    assert theory.periodic_cell_vectors == pytest.approx(np.eye(3) * 40)
    assert theory.get_pbc_vectors() == pytest.approx(np.eye(3) * 40)


def test_engine_init_and_run_accept_the_same_mixed_restraints():
    restraints = [[0, 1, 1, 2], [0, 1, 2, 1, 3], [0, 1, 2, 3, 0, 5]]
    energies = []
    for at_init in [True, False]:
        theory, fragment = _bare_theory()
        engine = MolecularDynamicsEngine(
            fragment=fragment,
            theory=theory,
            platform="Reference",
            restraints=restraints if at_init else None,
            timestep=1e-8,
        )
        engine.run(simulation_steps=1, restraints=None if at_init else restraints)
        # Evaluate the same original geometry, independent of the MD displacement.
        energies.append(theory.run(current_coords=fragment.coords))
        engine.close()
    assert energies[0] == pytest.approx(energies[1])


def test_cell_gradient_uses_last_successful_geometry_without_retaining_context():
    from openmmqmmm.constants import BOHR_TO_ANG, HARTREE_TO_KCAL_PER_MOL

    theory, fragment = _bare_theory()
    theory.periodic = True
    theory.update_cell(periodic_cell_vectors=np.eye(3) * 30)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 0.1, 836.8)  # r0=1 Angstrom; k=2 kcal/mol/Angstrom^2.
    force.setUsesPeriodicBoundaryConditions(True)
    theory.system.addForce(force)
    coords = fragment.coords.copy()
    coords[1] = [28, 0, 0]
    theory.run(current_coords=coords)
    coords[1] = [10, 0, 0]  # Caller mutation must not alter the stored evaluation.
    gradient = theory.get_cell_gradient()
    expected = np.zeros((3, 3))
    expected[0, 0] = 2 * BOHR_TO_ANG / HARTREE_TO_KCAL_PER_MOL
    assert gradient == pytest.approx(expected, rel=1e-4, abs=1e-12)
    assert not hasattr(theory, "stored_context")
    import pickle

    restored = pickle.loads(pickle.dumps(theory))
    assert restored.get_cell_gradient() == pytest.approx(gradient)


def test_cell_gradient_requires_a_periodic_single_point_first():
    from openmmqmmm.exceptions import InputError

    theory, _ = _bare_theory()
    with pytest.raises(InputError, match=r"periodic.*run"):
        theory.get_cell_gradient()


def test_remove_force_refreshes_existing_decomposition_map():
    theory, _ = _bare_theory()
    force = openmm.CustomExternalForce("1")
    force.addParticle(0, [])
    index = theory.system.addForce(force)
    theory.forcegroupify()
    count = len(theory.forcegroups)
    theory.remove_force(index)
    assert len(theory.forcegroups) == count - 1
    assert sorted(theory.forcegroups.values()) == list(range(count - 1))


def test_xml_system_loader_preserves_topology_metadata_and_energy(tmp_path):
    theory, fragment = _bare_theory()
    theory.add_custom_bond_force(0, 1, 1, 2)
    xml = tmp_path / "system.xml"
    xml.write_text(openmm.XmlSerializer.serialize(theory.system))
    theory.write_pdbfile(outputname=tmp_path / "system")
    restored = OpenMMTheory(
        xmlsystemfile=xml,
        pdbfile=str(tmp_path / "system.pdb"),
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    assert restored.atomnames == theory.atomnames
    assert restored.resids == theory.resids
    assert restored.run(current_coords=fragment.coords) == pytest.approx(theory.run(current_coords=fragment.coords))


def test_restraint_dummy_does_not_break_later_charmm_element_lookup():
    theory, _ = _bare_theory()
    theory.add_dummy_atom_to_restrain_solute(atomindices=[0])
    psf = openmm.app.CharmmPsfFile(str(FIXTURES / "three-waters.psf"))
    assert [atom.element.symbol for atom in psf.topology.atoms()] == ["O", "H", "H"] * 3
