"""Periodic image invariance and energy-derivative checks for finite QM/MM clusters."""

import numpy as np
import openmm
import pytest
from openmm import unit

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError
from openmmqmmm.periodic_embedding import PeriodicQMGeometry


class _CoulombQM:
    numcores = 1
    theorytype = "QM"

    def run(self, *, current_coords, current_mm_coords, mm_charges, grad=False, **_kwargs):
        qm = np.asarray(current_coords) * ANG_TO_BOHR
        mm = np.asarray(current_mm_coords) * ANG_TO_BOHR
        charges = np.asarray(mm_charges)
        delta = qm[:, None, :] - mm[None, :, :]
        radii = np.linalg.norm(delta, axis=-1)
        energy = np.sum(charges / radii)
        pairs = -charges[None, :, None] * delta / radii[:, :, None] ** 3
        if grad:
            return energy, pairs.sum(axis=1), -pairs.sum(axis=0)
        return energy


def _periodic_qmmm(coords, bonds=(), *, elems=None, qmatoms=(0,), charges=None):
    fragment = Fragment(elems=elems or ["He"] * len(coords), coords=coords, conncalc=False)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    atoms = list(mm.topology.atoms())
    for first, second in bonds:
        mm.topology.addBond(atoms[first], atoms[second])
    box = np.diag([30.0] * 3)
    mm.periodic = True
    mm.system.setDefaultPeriodicBoxVectors(*(box * 0.1))
    force = mm.nonbonded_force
    force.setNonbondedMethod(openmm.NonbondedForce.PME)
    force.setCutoffDistance(1.0)
    charges = [0, -1] if charges is None else charges
    for index, charge in enumerate(charges):
        force.setParticleParameters(index, charge, 0.1, 0)
    mm.charges = list(charges)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_CoulombQM(),
        mm_theory=mm,
        qmatoms=qmatoms,
        qm_charge=0,
        qm_mult=1,
    )
    return theory, fragment, box


def test_equivalent_point_charge_images_have_identical_energy_and_gradient():
    theory, fragment, box = _periodic_qmmm([[1, 0, 0], [29, 0, 0]])
    reference_energy, reference_gradient = theory.run(current_coords=fragment.coords, grad=True)
    assert theory.QMenergy == pytest.approx(-1 / (2 * ANG_TO_BOHR))
    for lattice_shifts in ([[1, -2, 0], [0, 0, 1]], [[0, 0, 0], [-1, 0, 0]]):
        shifted = fragment.coords + np.asarray(lattice_shifts) @ box
        energy, gradient = theory.run(current_coords=shifted, grad=True)
        assert energy == pytest.approx(reference_energy, abs=1e-12)
        assert gradient == pytest.approx(reference_gradient, abs=1e-12)


@pytest.mark.parametrize("box", [np.diag([30.0] * 3), np.array([[30, 0, 0], [5, 28, 0], [3, 2, 26]])])
def test_triclinic_images_keep_entire_molecules_and_multiple_qm_fragments_together(box):
    coords = np.array([[1, 1, 1], [2.4, 1, 1], [0, 3, 1], [15, 2, 1], [16.4, 2.1, 1]])
    geometry = PeriodicQMGeometry(5, [(0, 1), (3, 4)], [0, 2])
    baseline = geometry.image(coords, box)
    shifts = np.array([[2, 0, -1], [-2, 1, 0], [1, -1, 0], [0, 2, 0], [-1, 0, 1]])
    shifted = geometry.image(coords + shifts @ box, box)
    assert shifted == pytest.approx(baseline, abs=1e-12)
    assert shifted[1] - shifted[0] == pytest.approx(coords[1] - coords[0])
    assert shifted[4] - shifted[3] == pytest.approx(coords[4] - coords[3])


def test_topology_boundary_across_box_and_projected_gradients_match_finite_differences():
    theory, fragment, box = _periodic_qmmm(
        [[0.7, 1, 1], [29.3, 1, 1], [27.9, 1.3, 1]],
        bonds=[(0, 1), (1, 2)],
        elems=["C"] * 3,
        charges=[0, 0.2, -0.3],
    )
    assert theory.boundaryatoms == {0: [1]}
    assert theory.MMboundarydict == {1: [2]}
    energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
    gradient = gradient.copy()
    assert theory.num_linkatoms == 1
    shifted = fragment.coords + np.array([[1, 0, 0], [-1, 0, 1], [0, 2, 0]]) @ box
    shifted_energy, shifted_gradient = theory.run(current_coords=shifted, grad=True)
    assert shifted_energy == pytest.approx(energy, abs=1e-11)
    assert shifted_gradient == pytest.approx(gradient, abs=1e-11)
    h = 1e-5
    for atom, axis in [(0, 0), (0, 1), (1, 1), (2, 1)]:
        plus = fragment.coords.copy()
        minus = fragment.coords.copy()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        numeric = (theory.run(current_coords=plus) - theory.run(current_coords=minus)) / (2 * h * ANG_TO_BOHR)
        assert numeric == pytest.approx(gradient[atom, axis], abs=1e-7)


def test_explicit_current_box_reaches_both_qm_images_and_mm_context():
    theory, fragment, original_box = _periodic_qmmm([[1, 1, 1], [24, 1, 1]])
    theory.run(current_coords=fragment.coords)
    old_qm = theory.QMenergy
    theory.run(current_coords=fragment.coords, periodic_box_vectors=np.diag([25.0] * 3))
    assert old_qm == pytest.approx(-1 / (7 * ANG_TO_BOHR))
    assert theory.QMenergy == pytest.approx(-1 / (2 * ANG_TO_BOHR))
    explicit_mm_energy = theory.MMenergy
    # A separate MM call at the requested box must use exactly the same cell.
    assert theory.mm_theory.run(
        current_coords=fragment.coords, periodic_box_vectors=np.diag([25.0] * 3)
    ) == pytest.approx(explicit_mm_energy, abs=1e-12)
    assert np.asarray(
        unit.Quantity(theory.mm_theory.system.getDefaultPeriodicBoxVectors()).value_in_unit(unit.angstrom)
    ) == pytest.approx(original_box)


def test_winding_covalent_network_is_rejected_instead_of_cut_silently():
    geometry = PeriodicQMGeometry(3, [(0, 1), (1, 2), (2, 0)], [0])
    with pytest.raises(InputError, match="winds around"):
        geometry.image(np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0]]), np.diag([5.0] * 3))


def test_virtual_charge_sites_share_host_image_without_creating_covalent_boundaries():
    coords = np.array([[0, 1, 1], [14.8, 1, 1], [15.1, 1, 1], [15.05, 1, 1]])
    geometry = PeriodicQMGeometry(4, [(1, 2)], [0], image_links=[(3, 1), (3, 2)])
    assert geometry.neighbors == [[], [2], [1], []]
    imaged = geometry.image(coords, np.diag([30.0] * 3))
    assert imaged[3] - imaged[1] == pytest.approx(coords[3] - coords[1])
    shifted = coords.copy()
    shifted[3, 0] += 30
    assert geometry.image(shifted, np.diag([30.0] * 3)) == pytest.approx(imaged)


@pytest.mark.parametrize("box", [np.zeros((3, 3)), np.eye(2), np.diag([-1, 1, 1]), np.full((3, 3), np.nan)])
def test_invalid_periodic_box_is_rejected(box):
    with pytest.raises(InputError, match="box vectors"):
        PeriodicQMGeometry(1, [], [0]).image(np.zeros((1, 3)), box)
