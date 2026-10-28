"""Extended-coordinate Drude embedding conserves energy and all particle forces."""

import copy

import numpy as np
import openmm
import pytest
from conftest import central_difference_gradient, dummy_mm, evaluate, use_pme
from openmm import unit
from test_qmmm_periodic_accuracy import _CoulombQM

from openmmqmmm import Fragment, QMMMTheory, ZeroTheory
from openmmqmmm.constants import HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider, add_rpmd_python_force


def _drude_mm(*, periodic=False):
    fragment = Fragment(
        elems=["He"] * 5,
        coords=[[1, 1, 1], [6, 1.2, 1], [6.03, 1.19, 1.02], [8, 2, 1.8], [7.98, 2.03, 1.79]],
        conncalc=False,
    )
    # Shell-parent bonds are deliberately absent from the topology.
    mm = dummy_mm(fragment, bonds=[(1, 3)])
    mm.update_charges(range(5), [0, 0.6, -0.6, 0.5, -0.5])
    drude = openmm.DrudeForce()
    drude.addParticle(2, 1, -1, -1, -1, -0.6, 0.001, 1, 1)
    drude.addParticle(4, 3, -1, -1, -1, -0.5, 0.0008, 1, 1)
    drude.addScreenedPair(0, 1, 2.6)
    mm.system.addForce(drude)
    mm.system.setParticleMass(2, 0.4)
    mm.system.setParticleMass(4, 0.4)
    for first in range(1, 5):
        for second in range(first + 1, 5):
            mm.nonbonded_force.addException(first, second, 0, 1, 0)
    if periodic:
        use_pme(mm)
        drude.setUsesPeriodicBoundaryConditions(True)
    return mm, fragment


def _qmmm(mm, fragment, **kwargs):
    options = {"qmatoms": [0], "qm_charge": 0, "qm_mult": 1, "embedding": "drude"}
    options.update(kwargs)
    return QMMMTheory(fragment=fragment, qm_theory=_CoulombQM(), mm_theory=mm, **options)


@pytest.mark.parametrize("embedding", ["elstat", "drude", "polembed_drude"])
def test_drude_energy_preserves_springs_screening_and_qm_field(embedding):
    mm, fragment = _drude_mm()
    native = copy.deepcopy(mm.system)
    theory = _qmmm(mm, fragment, embedding=embedding)
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    native_energy, _forces = evaluate(native, fragment.coords * 0.1)
    qm_energy, _qm_grad, pc_grad = _CoulombQM().run(
        current_coords=fragment.coords[:1],
        current_mm_coords=fragment.coords[1:],
        mm_charges=[0.6, -0.6, 0.5, -0.5],
        grad=True,
    )
    assert energy == pytest.approx(native_energy / HARTREE_TO_KJ_PER_MOL + qm_energy, abs=1e-12)
    assert theory.QM_PC_gradient[2] == pytest.approx(pc_grad[1], abs=1e-12)
    assert np.linalg.norm(theory.QM_PC_gradient[2]) > 1e-4
    numeric = central_difference_gradient(lambda coords: theory.run(current_coords=coords), fragment.coords)
    assert gradient == pytest.approx(numeric, abs=2e-9)
    assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)


def test_drude_mechanical_embedding_retains_classical_polarization():
    mm, fragment = _drude_mm()
    mm.update_charges([0], [1.0])
    reference = evaluate(mm.system, fragment.coords * 0.1)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory(),
        mm_theory=mm,
        qmatoms=[0],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
    )
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    assert energy == pytest.approx(reference[0] / HARTREE_TO_KJ_PER_MOL, abs=1e-12)
    assert gradient == pytest.approx(-reference[1] / HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, abs=1e-12)


def test_periodic_drude_particles_follow_parents_without_topology_bonds():
    mm, fragment = _drude_mm(periodic=True)
    theory = _qmmm(mm, fragment)
    reference, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    displaced = fragment.coords + np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 1], [0, -1, 0], [1, 0, -1]]) * 30
    energy, shifted_gradient = theory.run(current_coords=displaced, grad=True)
    assert energy == pytest.approx(reference, abs=1e-11)
    assert shifted_gradient == pytest.approx(gradient, abs=1e-10)
    assert theory.pointchargecoords[1] - theory.pointchargecoords[0] == pytest.approx(
        fragment.coords[2] - fragment.coords[1], abs=1e-12
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_drude_python_force_matches_standalone_on_every_particle(periodic):
    mm, fragment = _drude_mm(periodic=periodic)
    theory = _qmmm(mm, fragment)
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    theory.openmm_externalforce = True
    provider = RPMDQMMMForceProvider(theory, fragment.elems, 0, 1, periodic=periodic)
    add_rpmd_python_force(mm.system, provider, periodic=periodic)
    integrator = openmm.DrudeLangevinIntegrator(300, 1, 1, 20, 0.0001)
    context = openmm.Context(mm.system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(fragment.coords * 0.1)
    state = context.getState(getEnergy=True, getForces=True)
    assert state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        energy * HARTREE_TO_KJ_PER_MOL, abs=1e-8
    )
    assert state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer) == pytest.approx(
        -gradient * HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, abs=1e-7
    )


@pytest.mark.parametrize("atom", [1, 2, 3, 4])
def test_drude_partition_rejection_precedes_charge_and_force_mutation(atom):
    mm, fragment = _drude_mm()
    before = openmm.XmlSerializer.serialize(mm.system)
    with pytest.raises(InputError, match="must remain entirely in the MM region"):
        _qmmm(mm, fragment, qmatoms=[atom], charges=[1] * 5)
    assert openmm.XmlSerializer.serialize(mm.system) == before


def test_drude_anisotropy_reference_must_remain_mm():
    mm, fragment = _drude_mm()
    drude = next(force for force in mm.system.getForces() if isinstance(force, openmm.DrudeForce))
    params = list(drude.getParticleParameters(0))
    params[2] = 0
    drude.setParticleParameters(0, *params)
    with pytest.raises(InputError, match="anisotropy references"):
        _qmmm(mm, fragment)


def test_drude_covalent_boundary_is_rejected():
    mm, fragment = _drude_mm()
    atoms = list(mm.topology.atoms())
    mm.topology.addBond(atoms[0], atoms[1])
    with pytest.raises(InputError, match="whole molecules"):
        _qmmm(mm, fragment)


def test_drude_shell_charge_override_is_rejected_without_mutation():
    mm, fragment = _drude_mm()
    before = openmm.XmlSerializer.serialize(mm.system)
    with pytest.raises(InputError, match="Cannot change Drude particle"):
        _qmmm(mm, fragment, charges=[0, 0.6, -0.7, 0.5, -0.5])
    assert openmm.XmlSerializer.serialize(mm.system) == before
    assert mm.charges == [0, 0.6, -0.6, 0.5, -0.5]
    _qmmm(mm, fragment, charges=mm.charges)


def test_inconsistent_drude_force_and_nonbonded_charge_is_rejected():
    mm, fragment = _drude_mm()
    mm.nonbonded_force.setParticleParameters(2, -0.7, 0.1, 0)
    with pytest.raises(InputError, match="inconsistent NonbondedForce and DrudeForce charges"):
        _qmmm(mm, fragment)


def test_alchemical_drude_shell_charge_is_rejected_without_mutation():
    mm, _fragment = _drude_mm()
    mm.nonbonded_force.addGlobalParameter("lambda", 0)
    mm.nonbonded_force.addParticleParameterOffset("lambda", 2, 0.1, 0, 0)
    before = openmm.XmlSerializer.serialize(mm.system)
    with pytest.raises(InputError, match="Charge parameter offsets on Drude particles"):
        mm.update_charges([0], [0.2])
    assert openmm.XmlSerializer.serialize(mm.system) == before


def test_swm4ndp_extra_particles_load_and_couple_to_qm_with_native_virtual_site():
    import io

    from openmmqmmm import OpenMMTheory

    topology = openmm.app.Topology()
    chain = topology.addChain()
    solute = topology.addResidue("QH", chain)
    topology.addAtom("H", openmm.app.element.hydrogen, solute)
    water = topology.addResidue("HOH", chain)
    oxygen = topology.addAtom("O", openmm.app.element.oxygen, water)
    hydrogens = [topology.addAtom(name, openmm.app.element.hydrogen, water) for name in ["H1", "H2"]]
    for hydrogen in hydrogens:
        topology.addBond(oxygen, hydrogen)
    solute_xml = io.StringIO("""<ForceField>
      <AtomTypes><Type name="QH" class="QH" element="H" mass="1.008"/></AtomTypes>
      <Residues><Residue name="QH"><Atom name="H" type="QH"/></Residue></Residues>
      <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
        <Atom type="QH" charge="0" sigma="1" epsilon="0"/>
      </NonbondedForce></ForceField>""")
    forcefield = openmm.app.ForceField("swm4ndp.xml", solute_xml)
    modeller = openmm.app.Modeller(
        topology, np.array([[0, 0, 0], [0.5, 0, 0], [0.59572, 0, 0], [0.476, 0.0927, 0]]) * unit.nanometer
    )
    modeller.addExtraParticles(forcefield)
    mm = OpenMMTheory(
        topoforce=True,
        topology=modeller.topology,
        forcefield=forcefield,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    fragment = Fragment(
        elems=mm.mm_elements,
        coords=np.asarray(modeller.positions.value_in_unit(unit.angstrom)),
        conncalc=False,
    )
    drude = next(force for force in mm.system.getForces() if isinstance(force, openmm.DrudeForce))
    shell, parent, *_rest = drude.getParticleParameters(0)
    fragment.coords[shell] = fragment.coords[parent] + [0.02, -0.01, 0.01]
    assert fragment.elems.count("M") == 2
    assert not mm.system.isVirtualSite(shell)
    theory = _qmmm(mm, fragment)
    _energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    numeric = central_difference_gradient(lambda coords: theory.run(current_coords=coords), fragment.coords)
    assert gradient == pytest.approx(numeric, abs=3e-8)
    assert np.linalg.norm(theory.QM_PC_gradient[shell]) > 1e-4
    virtual = next(i for i in range(mm.numatoms) if mm.system.isVirtualSite(i))
    assert gradient[virtual] == pytest.approx(np.zeros(3), abs=1e-14)
