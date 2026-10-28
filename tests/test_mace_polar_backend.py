"""Opt-in checks against a local pretrained PolarMACE checkpoint; never download weights."""

import os

import numpy as np
import openmm
import pytest
from ase import Atoms, units
from conftest import evaluate
from openmm import app

from openmmqmmm import Fragment, MLMMTheory, OpenMMTheory
from openmmqmmm.constants import HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, HARTREE_TO_KJ_PER_MOL


@pytest.fixture(scope="module")
def polar_checkpoint():
    model = os.environ.get("OPENMMQMMM_MACE_POLAR_MODEL")
    if model is None:
        pytest.skip("Set OPENMMQMMM_MACE_POLAR_MODEL to run pretrained MACE-POLAR checks")
    torch = pytest.importorskip("torch")
    pytest.importorskip("mace.calculators")
    pytest.importorskip("graph_longrange")
    pytest.importorskip("openmmml")
    previous_threads, previous_dtype = torch.get_num_threads(), torch.get_default_dtype()
    torch.set_num_threads(1)
    yield model
    torch.set_default_dtype(previous_dtype)
    torch.set_num_threads(previous_threads)


def _water_topology(count, periodic):
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(count):
        residue = topology.addResidue("HOH", chain)
        atoms = [
            topology.addAtom(name, element, residue)
            for name, element in (("O", app.element.oxygen), ("H1", app.element.hydrogen), ("H2", app.element.hydrogen))
        ]
        topology.addBond(atoms[0], atoms[1])
        topology.addBond(atoms[0], atoms[2])
    if periodic:
        topology.setPeriodicBoxVectors(np.eye(3) * 3 * openmm.unit.nanometer)
    return topology


@pytest.mark.parametrize("periodic", [False, True])
def test_pretrained_polar_energy_and_forces_match_ase_and_mm(polar_checkpoint, periodic):
    from mace.calculators import mace_polar

    coords = np.array([[0, 0, 0], [0.96, 0, 0], [-0.25, 0.93, 0], [4, 0.2, 0.1], [4.96, 0.2, 0.1], [3.75, 1.13, 0.1]])
    fragment = Fragment(elems=["O", "H", "H"] * 2, coords=coords, conncalc=False)
    forcefield = app.ForceField("tip3p.xml")
    source = OpenMMTheory(
        topoforce=True,
        topology=_water_topology(2, periodic),
        forcefield=forcefield,
        fragment=fragment,
        periodic=periodic,
        periodic_nonbonded_cutoff=10,
        dispersion_correction=False,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
        platform="Reference",
    )
    theory = MLMMTheory(
        mm_theory=source,
        mlatoms=[0, 1, 2],
        potential="mace-polar",
        potential_kwargs={"modelPath": polar_checkpoint},
        system_kwargs={"device": "cpu"},
    )
    calculator = mace_polar(model=polar_checkpoint, device="cpu", default_dtype="float64")
    atoms = Atoms("OHH", positions=coords[:3], pbc=periodic, cell=np.eye(3) * 30, calculator=calculator)
    atoms.info.update(charge=0, spin=1)
    ml_energy = atoms.get_potential_energy() / (units.kJ / units.mol)
    ml_forces = atoms.get_forces() * 10 / (units.kJ / units.mol)

    isolated_mm = forcefield.createSystem(
        _water_topology(1, periodic),
        nonbondedMethod=app.PME if periodic else app.NoCutoff,
        nonbondedCutoff=1 * openmm.unit.nanometer,
        constraints=None,
        rigidWater=False,
    )
    for force in isolated_mm.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.setUseDispersionCorrection(False)
    full_energy, full_forces = evaluate(source.system, coords * 0.1)
    isolated_energy, isolated_forces = evaluate(isolated_mm, coords[:3] * 0.1)
    expected_energy = full_energy - isolated_energy + ml_energy
    expected_forces = full_forces.copy()
    expected_forces[:3] += ml_forces - isolated_forces

    energy, gradient = theory.run(current_coords=coords, grad=True)
    assert energy * HARTREE_TO_KJ_PER_MOL == pytest.approx(expected_energy, rel=0, abs=1e-6)
    np.testing.assert_allclose(-gradient * HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, expected_forces, rtol=1e-10, atol=1e-7)
    simulation = theory.create_simulation()
    theory.set_positions(coords, simulation)
    simulation.context.setVelocitiesToTemperature(10 * openmm.unit.kelvin, 12345)
    simulation.step(2)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    assert np.isfinite(state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole))
    assert np.isfinite(state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)).all()
