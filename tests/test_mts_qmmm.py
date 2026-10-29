"""Independent force identities, integration accuracy and cost checks for the MTS prototype."""

import copy

import numpy as np
import openmm
import pytest
from conftest import central_difference_gradient, dummy_mm, use_pme
from openmm import unit

from openmmqmmm import Fragment, QMMMTheory, export_mts_potential
from openmmqmmm.constants import ANG_TO_BOHR, HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError


class _EmbeddedQuadratic:
    numcores = 1
    theorytype = "QM"

    def __init__(self, strength=0.01):
        self.strength = strength
        self.calls = []

    def run(self, *, current_coords, current_mm_coords=None, mm_charges=None, grad=False, pc=False, **kwargs):
        qm = np.asarray(current_coords) * ANG_TO_BOHR
        energy = self.strength * np.sum(qm**2) / 2
        gradient = self.strength * qm
        if pc:
            mm = np.asarray(current_mm_coords).reshape(-1, 3) * ANG_TO_BOHR
            delta = qm[:, None] - mm[None]
            pairs = self.strength * np.asarray(mm_charges)[None, :, None] * delta
            energy += np.sum(pairs * delta) / 2
            gradient += pairs.sum(axis=1)
            pc_gradient = -pairs.sum(axis=0)
        self.calls.append(np.array(current_coords))
        if not grad:
            return float(energy)
        return (float(energy), gradient, pc_gradient) if pc else (float(energy), gradient)


def _theory(*, boundary=False, virtual=False, method="shift", dipoles=True, strength=0.01, periodic=False):
    if boundary:
        coords = [[0.2, -0.4, 0.3], [1.5, 0.1, 0.7], [2.8, 0.9, 0.4]]
        elems = ["C"] * 3
        bonds = [(0, 1), (1, 2)]
        charges = [0, 0.4, -0.1]
        if virtual:
            coords += [[2.15, 0.5, 0.55]]
            elems += ["He"]
            charges += [-0.2]
    else:
        coords, elems, bonds, charges = [[0.5, 0, 0]], ["H"], [], [0]
    fragment = Fragment(elems=elems, coords=coords, conncalc=False, charge=0, mult=1)
    mm = dummy_mm(fragment, bonds=bonds)
    if periodic:
        use_pme(mm)
    for index in reversed(range(mm.system.getNumForces())):
        if isinstance(mm.system.getForce(index), openmm.CMMotionRemover):
            mm.system.removeForce(index)
    if virtual:
        mm.system.setParticleMass(3, 0)
        mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(1, 2, 0.5, 0.5))
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_EmbeddedQuadratic(strength),
        mm_theory=mm,
        charges=charges,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        chargeboundary_method=method,
        dipole_correction=dipoles,
    )
    if boundary:
        bond = openmm.HarmonicBondForce()
        bond.addBond(1, 2, 0.12, 200)
        bond.setUsesPeriodicBoundaryConditions(periodic)
        bond.setForceGroup(7)
        mm.system.addForce(bond)
    return theory


def _simulation(export):
    simulation = openmm.app.Simulation(
        export.modeller.topology, export.system, export.integrator, openmm.Platform.getPlatformByName("Reference")
    )
    simulation.context.setPositions(export.modeller.positions)
    simulation.context.computeVirtualSites()
    simulation.context.setVelocities(np.zeros((export.system.getNumParticles(), 3)))
    return simulation


def _energy_force(simulation, coords, groups=-1):
    simulation.context.setPositions(coords * 0.1)
    simulation.context.computeVirtualSites()
    state = simulation.context.getState(getEnergy=True, getForces=True, groups=groups)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJ_PER_MOL
    gradient = (
        -state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        / HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    )
    return energy, gradient


@pytest.mark.parametrize(("method", "dipoles"), [("shift", False), ("shift", True), ("rcd", False)])
@pytest.mark.parametrize("virtual", [False, True])
@pytest.mark.parametrize("periodic", [False, True])
def test_full_mts_potential_matches_high_level_and_displaced_derivatives(method, dipoles, virtual, periodic):
    theory = _theory(boundary=True, virtual=virtual, method=method, dipoles=dipoles, periodic=periodic)
    reference = _EmbeddedQuadratic(0.007)
    displacement = np.array([[0.015, 0.01, -0.02], [-0.01, 0.02, 0.01], [0.03, 0, -0.01], [0, 0, 0]])
    coords = theory.fragment.coords + displacement[: len(theory.elems)]
    expected_energy, expected_gradient = theory.run(current_coords=coords, grad=True)
    expected_gradient = expected_gradient.copy()
    numeric = central_difference_gradient(lambda x: theory.run(current_coords=x), coords)
    assert expected_gradient == pytest.approx(numeric, abs=2e-9)

    export = export_mts_potential(theory=theory, reference_qm_theory=reference)
    simulation = _simulation(export)
    energy, gradient = _energy_force(simulation, coords)
    real_atoms = [0, 1, 2]
    assert energy == pytest.approx(expected_energy, abs=1e-12)
    assert gradient[real_atoms] == pytest.approx(expected_gradient[real_atoms], abs=2e-11)
    assert gradient[real_atoms] == pytest.approx(numeric[real_atoms], abs=2e-9)
    fast_energy, fast_gradient = _energy_force(simulation, coords, 1 << export.fast_group)
    correction_energy, correction_gradient = _energy_force(simulation, coords, 1 << export.slow_group)
    assert abs(correction_energy) > 1e-6
    assert abs(fast_energy) > abs(correction_energy)
    assert np.linalg.norm(fast_gradient) > 0
    assert np.linalg.norm(correction_gradient) > 0
    assert export.reference_provider.theory.mm_theory is theory.mm_theory
    assert export.reference_provider.theory.pointcharges == pytest.approx(theory.pointcharges)
    assert export.reference_provider.theory.pointchargecoords == pytest.approx(theory.pointchargecoords)
    assert export.reference_provider.theory.pointcharges is not theory.pointcharges
    assert reference.qmatoms == theory.qmatoms


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_mts_force_cadence_reduces_high_evaluations_without_duplicate_reference_calls(ratio):
    theory = _theory()
    export = export_mts_potential(
        theory=theory, reference_qm_theory=_EmbeddedQuadratic(0.009), inner_steps=ratio, inner_timestep=0.0001
    )
    simulation = _simulation(export)
    simulation.step(24 // ratio)
    assert export.reference_provider.evaluation_count == 25
    assert export.high_provider.evaluation_count == 24 // ratio + 1
    assert len(theory.qm_theory.calls) == export.high_provider.evaluation_count
    assert len(export.reference_provider.theory.qm_theory.calls) == 25
    before = export.high_provider.evaluation_count
    simulation.context.getState(getEnergy=True, getForces=True)
    assert export.high_provider.evaluation_count == before
    assert export.reference_provider.evaluation_count == 25


def _oscillator_errors(inner_timestep, ratio=4):
    theory = _theory()
    export = export_mts_potential(
        theory=theory,
        reference_qm_theory=_EmbeddedQuadratic(0.009),
        inner_timestep=inner_timestep,
        inner_steps=ratio,
    )
    simulation = _simulation(export)
    mass = export.system.getParticleMass(0).value_in_unit(unit.dalton)
    spring = 0.01 * ANG_TO_BOHR * 10 * HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    omega = np.sqrt(spring / mass)
    initial_energy = 0.5 * spring * 0.05**2
    elapsed = 0.04
    outer_steps = round(elapsed / (inner_timestep * ratio))
    energies = []
    for _ in range(outer_steps):
        simulation.step(1)
        state = simulation.context.getState(getEnergy=True)
        energies.append((state.getPotentialEnergy() + state.getKineticEnergy()).value_in_unit(unit.kilojoule_per_mole))
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    exact_x = 0.05 * np.cos(omega * elapsed)
    return abs(positions[0, 0] - exact_x), max(abs(np.asarray(energies) - initial_energy)) / initial_energy


def test_mts_converges_to_independent_harmonic_trajectory_and_conserves_nve_energy():
    coarse_position, coarse_drift = _oscillator_errors(0.0005)
    fine_position, fine_drift = _oscillator_errors(0.00025)
    assert fine_position < coarse_position / 3
    assert fine_drift < coarse_drift / 3
    assert fine_position < 2e-5
    assert fine_drift < 5e-4


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_inner_steps_do_not_change_standalone_theory(bad):
    theory = _theory()
    before = openmm.XmlSerializer.serialize(theory.mm_theory.system)
    with pytest.raises(InputError, match="inner_steps"):
        export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic(), inner_steps=bad)
    assert openmm.XmlSerializer.serialize(theory.mm_theory.system) == before
    assert not theory.openmm_externalforce and not theory.exit_after_customexternalforce_update


@pytest.mark.parametrize("bad", [0, -1, True, float("inf"), float("nan")])
def test_invalid_inner_timestep_is_rejected(bad):
    theory = _theory()
    with pytest.raises(InputError, match="inner_timestep"):
        export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic(), inner_timestep=bad)
    assert not theory.openmm_externalforce


@pytest.mark.parametrize("option", ["truncated_pc", "update_qm_region_charges", "openmm_externalforce"])
def test_unsupported_history_and_already_bound_theories_are_rejected(option):
    theory = _theory()
    setattr(theory, option, True)
    before = openmm.XmlSerializer.serialize(theory.mm_theory.system)
    with pytest.raises(InputError):
        export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic())
    assert openmm.XmlSerializer.serialize(theory.mm_theory.system) == before


def test_reference_backend_identity_and_orca_scratch_must_be_independent(fake_orca_dir):
    from openmmqmmm import ORCATheory

    theory = _theory()
    with pytest.raises(InputError, match="independent objects"):
        export_mts_potential(theory=theory, reference_qm_theory=theory.qm_theory)
    theory.qm_theory = ORCATheory(orcasimpleinput="! HF def2-SVP", filename="same")
    reference = ORCATheory(orcasimpleinput="! HF STO-3G", filename="./same")
    before = openmm.XmlSerializer.serialize(theory.mm_theory.system)
    with pytest.raises(InputError, match="distinct filenames"):
        export_mts_potential(theory=theory, reference_qm_theory=reference)
    assert openmm.XmlSerializer.serialize(theory.mm_theory.system) == before
    assert not theory.openmm_externalforce


@pytest.mark.parametrize(
    "force",
    [openmm.AndersenThermostat(300, 1), openmm.MonteCarloBarostat(1, 300), openmm.DrudeForce()],
)
def test_nve_export_rejects_thermostats_barostats_and_drude(force):
    theory = _theory()
    theory.mm_theory.system.addForce(copy.deepcopy(force))
    before = openmm.XmlSerializer.serialize(theory.mm_theory.system)
    with pytest.raises(InputError, match="Experimental NVE"):
        export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic())
    assert openmm.XmlSerializer.serialize(theory.mm_theory.system) == before
    assert not theory.openmm_externalforce


def test_reciprocal_groups_are_reserved_and_all_native_groups_are_fast():
    theory = _theory()
    theory.mm_theory.nonbonded_force.setReciprocalSpaceForceGroup(31)
    export = export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic())
    assert 31 not in (export.fast_group, export.slow_group)
    expressions = [export.integrator.getComputationStep(i)[2] for i in range(export.integrator.getNumComputations())]
    assert sum("f31/m" in expression for expression in expressions) == 2 * export.inner_steps

    crowded = _theory()
    for group in range(30):
        force = openmm.CustomExternalForce("0")
        force.setForceGroup(group)
        crowded.mm_theory.system.addForce(force)
    crowded.mm_theory.nonbonded_force.setReciprocalSpaceForceGroup(31)
    before = openmm.XmlSerializer.serialize(crowded.mm_theory.system)
    with pytest.raises(InputError, match="two unused force groups"):
        export_mts_potential(theory=crowded, reference_qm_theory=_EmbeddedQuadratic())
    assert openmm.XmlSerializer.serialize(crowded.mm_theory.system) == before


def test_second_export_is_rejected_before_force_mutation():
    theory = _theory()
    export = export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic())
    count = export.system.getNumForces()
    with pytest.raises(InputError, match="fresh standalone"):
        export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic())
    assert export.system.getNumForces() == count


def test_periodic_box_changes_reach_both_levels_and_invalidate_both_caches():
    theory = _theory(boundary=True, periodic=True)
    export = export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic(0.007))
    simulation = _simulation(export)
    coords = theory.fragment.coords
    for edge in (3.0, 4.0):
        simulation.context.setPeriodicBoxVectors(*(np.eye(3) * edge))
        _energy_force(simulation, coords)
        assert theory._current_periodic_box_vectors == pytest.approx(np.eye(3) * edge * 10)
        assert export.reference_provider.theory._current_periodic_box_vectors == pytest.approx(np.eye(3) * edge * 10)
    assert export.high_provider.evaluation_count == export.reference_provider.evaluation_count == 2
    simulation.context.setPeriodicBoxVectors(*(np.eye(3) * 3.0))
    _energy_force(simulation, coords)
    assert export.high_provider.evaluation_count == export.reference_provider.evaluation_count == 2


def test_nonconservative_link_projection_is_rejected_without_mutation():
    theory = _theory(boundary=True)
    theory.linkatom_forceproj_method = "none"
    before = openmm.XmlSerializer.serialize(theory.mm_theory.system)
    with pytest.raises(InputError, match="link-atom force projection"):
        export_mts_potential(theory=theory, reference_qm_theory=_EmbeddedQuadratic())
    assert openmm.XmlSerializer.serialize(theory.mm_theory.system) == before
    assert not theory.openmm_externalforce


def test_partial_force_attachment_rolls_back_and_retains_standalone_energy(monkeypatch):
    theory = _theory(boundary=True)
    reference = _EmbeddedQuadratic()
    reference.qmatoms = [99]
    system = theory.mm_theory.system
    before = openmm.XmlSerializer.serialize(system)
    expected = theory.run(current_coords=theory.fragment.coords)
    original_add = system.addForce
    calls = 0

    def fail_second(force):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("attachment rejected")
        return original_add(force)

    monkeypatch.setattr(system, "addForce", fail_second)
    with pytest.raises(RuntimeError, match="attachment rejected"):
        export_mts_potential(theory=theory, reference_qm_theory=reference)
    assert openmm.XmlSerializer.serialize(system) == before
    assert reference.qmatoms == [99]
    assert not theory.openmm_externalforce and not theory.exit_after_customexternalforce_update
    assert theory.run(current_coords=theory.fragment.coords) == pytest.approx(expected, abs=1e-12)
