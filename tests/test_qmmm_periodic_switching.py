"""Record finite-cluster jumps separately from smooth periodic MM interactions.

The analytic QM stand-in is a Coulomb probe, not a periodic electronic-structure
reference. These checks deliberately preserve a known limitation: selecting a
different image of a neutral molecule can change the QM energy and both its QM
and MM forces by a finite amount. No assertion requires continuity at that switch.
"""

import numpy as np
import openmm
import pytest
from test_qmmm_periodic_accuracy import _periodic_qmmm

from openmmqmmm.constants import ANG_TO_BOHR


@pytest.fixture(params=["orthorhombic", "triclinic"])
def switch_system(request):
    box = np.diag([30.0, 28.0, 26.0])
    if request.param == "triclinic":
        box = np.array([[30.0, 0, 0], [7, 28, 0], [4, 3, 26]])
    lattice_step = box[1]
    normal = lattice_step / np.linalg.norm(lattice_step)
    tangent = np.array([normal[1], -normal[0], 0])
    qm = np.array([[3.2, 4.0, 4.0], [4.4, 4.5, 4.2], [4.1, 3.7, 5.1]])
    center = qm.mean(axis=0)
    # This is the Cartesian Voronoi face between images separated by b. Its
    # normal is oblique in the triclinic case; it is not a fractional-axis test.
    moving_center = center + lattice_step / 2 + 1.7 * tangent + [0, 0, -0.8]
    moving = moving_center + np.array([[-0.35, -0.5, 0.2], [0.35, 0.5, -0.2]])
    fixed = center + np.array([-4, 5, 1.5]) + np.array([[-0.2, 0.3, -0.1], [0.2, -0.3, 0.1]])
    surface = np.vstack([qm, moving, fixed])
    charges = np.array([0, 0, 0, 0.7, -0.7, 0.4, -0.4])
    theory, _fragment, _box = _periodic_qmmm(
        surface,
        bonds=[(0, 1), (1, 2), (3, 4), (5, 6)],
        qmatoms=(0, 1, 2),
        charges=charges,
    )
    # A stationary second molecule gives PME a nonconstant interaction as the
    # first molecule crosses the QM imaging face. There are no covalent caps.
    assert not theory.linkatoms
    assert theory.mm_theory.nonbonded_force.getNonbondedMethod() == openmm.NonbondedForce.PME
    return theory, surface, box, normal, charges


def _displace_molecule(surface, normal, displacement):
    coords = surface.copy()
    coords[3:5] += displacement * normal
    return coords


def _branch_reference(coords, charges, lattice_step, *, after_switch):
    """Evaluate explicit image branches without calling the imaging algorithm."""
    imaged = coords.copy()
    if after_switch:
        imaged[3:5] -= lattice_step
    # Use a scalar pair sum independently of the vectorized probe backend.
    energy = 0.0
    forces = np.zeros_like(imaged)
    for qm_atom in range(3):
        for mm_atom in range(3, len(coords)):
            delta = (imaged[qm_atom] - imaged[mm_atom]) * ANG_TO_BOHR
            radius = np.sqrt(np.dot(delta, delta))
            energy += charges[mm_atom] / radius
            pair_force = charges[mm_atom] * delta / radius**3
            forces[qm_atom] += pair_force
            forces[mm_atom] -= pair_force
    return energy, forces, imaged


def _sample(theory, coords, box):
    energy, gradient = theory.run(current_coords=coords, periodic_box_vectors=box, grad=True)
    return {
        "energy": energy,
        "forces": -gradient.copy(),
        "qm_energy": theory.QMenergy,
        "qm_forces": -theory.QM_PC_gradient.copy(),
        "mm_energy": theory.MMenergy,
        "mm_forces": -theory.MMgradient.copy(),
        "images": theory.pointchargecoords.copy(),
    }


def test_neutral_molecule_image_switch_has_finite_energy_and_force_jumps(switch_system):
    theory, surface, box, normal, charges = switch_system
    left_limit = _branch_reference(surface, charges, box[1], after_switch=False)
    right_limit = _branch_reference(surface, charges, box[1], after_switch=True)
    energy_jump_limit = right_limit[0] - left_limit[0]
    force_jump_limit = right_limit[1] - left_limit[1]
    # Nonsymmetric QM sites and molecular orientation prevent an accidental
    # cancellation from concealing the finite-cluster discontinuity.
    assert abs(energy_jump_limit) > 1e-4
    assert np.linalg.norm(force_jump_limit[:3]) > 1e-6
    assert np.linalg.norm(force_jump_limit[3:5]) > 1e-6

    energy_errors, force_errors, mm_energy_gaps, mm_force_gaps = [], [], [], []
    crossing_secants = []
    for epsilon in (1e-3, 1e-4, 1e-5):  # Angstrom; never evaluate the ambiguous tie.
        samples = []
        for sign in (-1, 1):
            coords = _displace_molecule(surface, normal, sign * epsilon)
            sample = _sample(theory, coords, box)
            expected_energy, expected_forces, expected_images = _branch_reference(
                coords, charges, box[1], after_switch=sign > 0
            )
            assert sample["qm_energy"] == pytest.approx(expected_energy, rel=0, abs=1e-13)
            np.testing.assert_allclose(sample["qm_forces"], expected_forces, rtol=0, atol=1e-13)
            np.testing.assert_allclose(sample["images"], expected_images[3:], rtol=0, atol=1e-12)
            np.testing.assert_allclose(sample["images"][1] - sample["images"][0], surface[4] - surface[3], atol=1e-12)
            assert sample["energy"] == pytest.approx(sample["qm_energy"] + sample["mm_energy"], abs=1e-13)
            np.testing.assert_allclose(sample["forces"], sample["qm_forces"] + sample["mm_forces"], atol=1e-13)
            samples.append(sample)

        left, right = samples
        energy_jump = right["qm_energy"] - left["qm_energy"]
        force_jump = right["qm_forces"] - left["qm_forces"]
        energy_errors.append(abs(energy_jump - energy_jump_limit))
        force_errors.append(np.linalg.norm(force_jump - force_jump_limit))
        mm_energy_gaps.append(abs(right["mm_energy"] - left["mm_energy"]))
        mm_force_gaps.append(np.linalg.norm(right["mm_forces"] - left["mm_forces"]))
        crossing_secants.append(energy_jump / (2 * epsilon * ANG_TO_BOHR))
        # The observable total jumps approach the same limit: PME adds only a
        # smooth variation with the physical coordinates, not the QM image jump.
        assert right["energy"] - left["energy"] == pytest.approx(energy_jump_limit, rel=2e-3, abs=1e-8)
        np.testing.assert_allclose(right["forces"] - left["forces"], force_jump_limit, rtol=0, atol=epsilon * 1e-3)

    for errors in (energy_errors, force_errors, mm_energy_gaps, mm_force_gaps):
        assert errors[1] < 0.2 * errors[0]
        assert errors[2] < 0.2 * errors[1]
    assert energy_errors[-1] < 1e-5 * abs(energy_jump_limit)
    assert force_errors[-1] < 1e-5 * np.linalg.norm(force_jump_limit)
    assert mm_energy_gaps[-1] < 1e-8
    assert mm_force_gaps[-1] < 1e-8

    # A central difference straddling the face includes jump/(2*h); shrinking
    # the step therefore diverges instead of validating an analytic gradient.
    assert np.abs(crossing_secants[1:]) == pytest.approx(10 * np.abs(crossing_secants[:-1]), rel=1e-3)
    branch_slopes = [-np.sum(branch[1][3:5] @ normal) for branch in (left_limit, right_limit)]
    assert abs(crossing_secants[-1]) > 1e4 * max(np.abs(branch_slopes))


def test_local_derivatives_remain_valid_on_either_side_of_image_switch(switch_system):
    theory, surface, box, normal, _charges = switch_system
    h = 1e-5  # Angstrom; all finite-difference points stay on the same branch.
    directions = []
    qm_direction = np.zeros_like(surface)
    qm_direction[0, 0] = 1
    directions.append(qm_direction)
    molecule_direction = np.zeros_like(surface)
    molecule_direction[3:5] = normal
    directions.append(molecule_direction)

    for offset in (-2e-3, 2e-3):
        coords = _displace_molecule(surface, normal, offset)
        reference = _sample(theory, coords, box)
        for direction in directions:
            minus = _sample(theory, coords - h * direction, box)
            plus = _sample(theory, coords + h * direction, box)
            for component in ("qm_", "mm_", ""):
                numeric = (plus[f"{component}energy"] - minus[f"{component}energy"]) / (2 * h * ANG_TO_BOHR)
                analytic = -np.sum(reference[f"{component}forces"] * direction)
                assert numeric == pytest.approx(analytic, rel=0, abs=2e-8)
