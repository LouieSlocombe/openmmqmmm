"""Native virtual-site forces agree between standalone and OpenMM callbacks."""

import copy
import pickle

import numpy as np
import openmm
import pytest
from openmm import unit

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR, HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, HARTREE_TO_KJ_PER_MOL
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider, add_rpmd_python_force
from openmmqmmm.virtual_sites import NativeVirtualSites


class _CoulombQM:
    numcores = 1

    def run(self, *, current_coords, current_mm_coords, mm_charges, grad=False, **kwargs):
        delta = (np.asarray(current_coords)[:, None, :] - np.asarray(current_mm_coords)[None, :, :]) * ANG_TO_BOHR
        radii = np.linalg.norm(delta, axis=-1)
        charges = np.asarray(mm_charges)
        energy = np.sum(charges / radii)
        if not grad:
            return energy
        pairs = -charges[None, :, None] * delta / radii[:, :, None] ** 3
        return energy, pairs.sum(axis=1), -pairs.sum(axis=0)


def _theory(*, periodic):
    fragment = Fragment(
        elems=["He"] * 5,
        coords=[[1, 1, 1], [6, 1.2, 1], [8, 1, 1.1], [99, -99, 99], [12, 2, 1.8]],
        conncalc=False,
    )
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    atoms = list(mm.topology.atoms())
    mm.topology.addBond(atoms[1], atoms[2])
    mm.system.setParticleMass(3, 0)
    mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(1, 2, 0.4, 0.6))
    if periodic:
        mm.periodic = True
        mm.system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.0))
        mm.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        mm.nonbonded_force.setCutoffDistance(1.0)
    mm.update_charges(list(range(5)), [0, 0, 0, -1, 0.4])
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_CoulombQM(),
        mm_theory=mm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
    )
    return theory, fragment


@pytest.mark.parametrize("periodic", [False, True])
def test_native_callback_matches_standalone_without_double_projection(periodic):
    theory, fragment = _theory(periodic=periodic)
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    assert gradient[3] == pytest.approx([0, 0, 0], abs=1e-14)

    # The MM virtual-site force is nonzero as well as the QM embedding force,
    # so projecting an already redistributed native MM contribution fails this.
    h = 1e-5
    for atom, axis in [(1, 0), (2, 1), (4, 2)]:
        plus, minus = fragment.coords.copy(), fragment.coords.copy()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        numeric = (theory.run(current_coords=plus) - theory.run(current_coords=minus)) / (2 * h * ANG_TO_BOHR)
        assert gradient[atom, axis] == pytest.approx(numeric, abs=1e-8)

    theory.openmm_externalforce = True
    provider = RPMDQMMMForceProvider(theory, fragment.elems, 0, 1, periodic=periodic)
    add_rpmd_python_force(theory.mm_theory.system, provider, periodic=periodic)
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(theory.mm_theory.system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(fragment.coords * 0.1)
    context.computeVirtualSites()
    state = context.getState(getEnergy=True, getForces=True)
    callback_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / HARTREE_TO_KJ_PER_MOL
    callback_gradient = (
        -state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        / HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    )

    assert callback_energy == pytest.approx(energy, abs=1e-11)
    assert callback_gradient[[0, 1, 2, 4]] == pytest.approx(gradient[[0, 1, 2, 4]], abs=1e-10)
    # OpenMM retains the raw site force; the callback must leave it available
    # for the native distribution step, unlike the standalone return value.
    assert np.linalg.norm(callback_gradient[3]) > 1e-5


def _nested_sites():
    system = openmm.System()
    for mass in [1, 1, 1, 0, 0]:
        system.addParticle(mass)
    # Site 3 depends on the later-indexed site 4, exercising dependency order.
    system.setVirtualSite(3, openmm.TwoParticleAverageSite(4, 2, 0.7, 0.3))
    system.setVirtualSite(4, openmm.TwoParticleAverageSite(0, 1, 0.25, 0.75))
    return NativeVirtualSites(system)


def test_nested_native_sites_place_and_project_in_dependency_order():
    helper = _nested_sites()
    coords = np.array([[1, 2, 3], [4, 5, 6], [7, 9, 11], [100, 100, 100], [-100, -100, -100]], dtype=float)
    placed = helper.place(coords)
    expected4 = 0.25 * coords[0] + 0.75 * coords[1]
    expected3 = 0.7 * expected4 + 0.3 * coords[2]
    assert placed[3] == pytest.approx(expected3, abs=1e-12)
    assert placed[4] == pytest.approx(expected4, abs=1e-12)

    gradient = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [1, 2, 3], [-2, -3, -4]])
    projected = helper.project(placed, gradient)
    accumulated4 = gradient[4] + 0.7 * gradient[3]
    expected = gradient.copy()
    expected[0] += 0.25 * accumulated4
    expected[1] += 0.75 * accumulated4
    expected[2] += 0.3 * gradient[3]
    expected[[3, 4]] = 0
    assert projected == pytest.approx(expected, abs=1e-12)

    # A coordinate-linear energy gives an independent finite-difference check
    # of all nested chain contributions while supplied site rows are ignored.
    h = 1e-5
    for atom in range(3):
        for axis in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[atom, axis] += h
            minus[atom, axis] -= h
            numeric = np.sum((helper.place(plus) - helper.place(minus)) * gradient) / (2 * h)
            assert projected[atom, axis] == pytest.approx(numeric, abs=1e-9)


@pytest.mark.parametrize("copier", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_native_site_helper_recreates_context_after_copying(copier):
    helper = _nested_sites()
    coords = np.arange(15, dtype=float).reshape(5, 3)
    placed = helper.place(coords)
    gradient = np.linspace(-1, 1, 15).reshape(5, 3)
    projected = helper.project(placed, gradient)

    copied = copier(helper)
    assert copied._context is None
    assert copied._force is None
    assert copied._integrator is None
    assert copied.place(coords) == pytest.approx(placed, abs=1e-12)
    assert copied.project(coords, gradient) == pytest.approx(projected, abs=1e-12)
    assert copied._context is not helper._context


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("qmatoms", [[3], [4], [1], [2]])
@pytest.mark.parametrize("explicit_charges", [False, True])
def test_native_partition_rejection_is_persistent_and_does_not_mutate_mm(periodic, qmatoms, explicit_charges):
    initial, fragment = _theory(periodic=periodic)
    mm = initial.mm_theory
    # Real host 1 is reached from site 3 only through site 4. Checking every
    # direct dependency edge must reject this transitive partition as well.
    mm.system.setParticleMass(4, 0)
    mm.system.setVirtualSite(4, openmm.TwoParticleAverageSite(1, 2, 0.25, 0.75))
    mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(4, 2, 0.7, 0.3))
    system_before = openmm.XmlSerializer.serialize(mm.system)
    charges_before = mm.charges.copy()
    rejected = QMMMTheory.__new__(QMMMTheory)

    with pytest.raises(InputError, match=r"[Nn]ative OpenMM virtual sites"):
        rejected.__init__(
            fragment=fragment,
            qm_theory=_CoulombQM(),
            mm_theory=mm,
            qmatoms=qmatoms,
            qm_charge=0,
            qm_mult=1,
            charges=[0.11, 0.12, 0.13, 0.14, 0.15] if explicit_charges else None,
        )

    assert mm.charges == charges_before
    assert openmm.XmlSerializer.serialize(mm.system) == system_before
    # Failed validation must not publish a cached None that bypasses the guard.
    for _ in range(2):
        with pytest.raises(InputError, match=r"[Nn]ative OpenMM virtual sites"):
            rejected._get_native_virtual_sites()
