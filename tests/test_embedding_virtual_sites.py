"""Physical energy derivatives with native OpenMM virtual charge sites."""

import numpy as np
import openmm
import pytest
from test_qmmm_periodic_accuracy import _CoulombQM

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR


def _native_site_theory(kind, *, periodic=False):
    coords = np.array([[1.0, 1, 1], [6.0, 1, 1], [8.0, 1, 1], [6.0, 3, 1], [13.0, 11, 9]])
    fragment = Fragment(elems=["He"] * 5, coords=coords, conncalc=False)
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
    mm.topology.addBond(atoms[1], atoms[3])
    sites = {
        "two": openmm.TwoParticleAverageSite(1, 2, 0.3, 0.7),
        "three": openmm.ThreeParticleAverageSite(1, 2, 3, 0.2, 0.3, 0.5),
        "outofplane": openmm.OutOfPlaneSite(1, 2, 3, 0.3, 0.4, 1.7),
        "local": openmm.LocalCoordinatesSite(
            [1, 2, 3], [1, 0, 0], [-1, 1, 0], [-1, 0, 1], openmm.Vec3(0.05, 0.07, 0.04)
        ),
    }
    mm.system.setParticleMass(4, 0)
    mm.system.setVirtualSite(4, sites[kind])
    mm.update_charges(range(5), [0, 0, 0, 0, -1])
    if periodic:
        mm.periodic = True
        mm.system.setDefaultPeriodicBoxVectors(*(np.eye(3) * 3.0))
        mm.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
        mm.nonbonded_force.setCutoffDistance(1.0)
    theory = QMMMTheory(fragment=fragment, qm_theory=_CoulombQM(), mm_theory=mm, qmatoms=[0], qm_charge=0, qm_mult=1)
    return theory, fragment


@pytest.mark.parametrize("kind", ["two", "three", "outofplane", "local"])
@pytest.mark.parametrize("periodic", [False, True])
def test_native_site_gradients_differentiate_energy_in_independent_host_coordinates(kind, periodic):
    theory, fragment = _native_site_theory(kind, periodic=periodic)
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    assert theory.QM_PC_gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    step = 1e-5
    for atom in range(5):
        for axis in range(3):
            plus, minus = fragment.coords.copy(), fragment.coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical = (theory.run(current_coords=plus) - theory.run(current_coords=minus)) / (2 * step * ANG_TO_BOHR)
            assert gradient[atom, axis] == pytest.approx(numerical, abs=3e-8)
    assert gradient[4] == pytest.approx(np.zeros(3), abs=1e-14)
    if not periodic:
        assert gradient.sum(axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    # The independent input virtual-site row is ignored, even for distant images.
    altered = fragment.coords.copy()
    altered[4] += [1234.0, -2233, 899]
    assert theory.run(current_coords=altered) == pytest.approx(energy, abs=1e-12)


def test_periodic_virtual_site_placement_uses_unwrapped_hosts():
    theory, fragment = _native_site_theory("two", periodic=True)
    coords = fragment.coords.copy()
    coords[1:4, 0] += 22
    reference, gradient = theory.run(current_coords=coords, grad=True)
    gradient = gradient.copy()
    wrapped = coords.copy()
    wrapped[2, 0] -= 30
    energy, wrapped_gradient = theory.run(current_coords=wrapped, grad=True)
    assert energy == pytest.approx(reference, abs=1e-12)
    assert wrapped_gradient == pytest.approx(gradient, abs=1e-12)


def test_standalone_mm_recomputes_virtual_sites_and_zeros_dependent_rows():
    theory, fragment = _native_site_theory("local")
    mm = theory.mm_theory
    # Introduce a charged real MM atom so native MM site forces are nonzero too.
    mm.update_charges([3], [0.5])
    energy, gradient = mm.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    step = 1e-5
    plus, minus = fragment.coords.copy(), fragment.coords.copy()
    plus[1, 2] += step
    minus[1, 2] -= step
    numerical = (mm.run(current_coords=plus) - mm.run(current_coords=minus)) / (2 * step * ANG_TO_BOHR)
    assert gradient[1, 2] == pytest.approx(numerical, abs=1e-8)
    assert gradient[4] == pytest.approx(np.zeros(3), abs=1e-14)
    changed_site = fragment.coords.copy()
    changed_site[4] += 5
    assert mm.run(current_coords=changed_site) == pytest.approx(energy, abs=1e-12)
