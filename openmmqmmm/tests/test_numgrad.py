"""Tests for the numerical-gradient wrapper.

NumGrad had no coverage, including the 1.0.1 fix that made `run` honour `grad=`.
A theory with an analytic energy makes the finite-difference result checkable
against the exact derivative, with no external program involved.
"""

import numpy as np
import pytest

from openmmqmmm import NumGrad, ZeroTheory
from openmmqmmm.exceptions import InputError


class HarmonicPairTheory:
    """A toy theory: two atoms on a spring, E = 0.5*k*(r - r0)**2.

    Its analytic gradient is known exactly, so it pins the finite-difference
    machinery — displacement in bohr, the stencil, and the assembly of the
    (natoms, 3) array — rather than just its shape.
    """

    def __init__(self, force_constant=0.5, equilibrium=1.0):
        self.theorytype = "QM"
        self.theorynamelabel = "HarmonicPair"
        self.numcores = 1
        self.force_constant = force_constant
        self.equilibrium = equilibrium

    def set_numcores(self, numcores):
        self.numcores = numcores

    def analytic_gradient(self, coords):
        coords = np.asarray(coords, dtype=float)
        separation = coords[1] - coords[0]
        distance = np.linalg.norm(separation)
        magnitude = self.force_constant * (distance - self.equilibrium)
        direction = separation / distance
        return np.array([-magnitude * direction, magnitude * direction])

    def run(self, current_coords=None, elems=None, charge=None, mult=None, grad=False, **kwargs):
        coords = np.asarray(current_coords, dtype=float)
        distance = np.linalg.norm(coords[1] - coords[0])
        energy = 0.5 * self.force_constant * (distance - self.equilibrium) ** 2
        if grad:
            return energy, self.analytic_gradient(coords)
        return energy


STRETCHED_PAIR = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])


def test_numerical_gradient_matches_the_analytic_one():
    """Finite differences must reproduce the exact derivative of a known potential."""
    theory = HarmonicPairTheory()
    numgrad = NumGrad(theory=theory)

    _energy, gradient = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=True)

    # Coordinates are in Angstrom and gradients in Eh/bohr, so convert the reference.
    ang2bohr = 1.88972612546
    expected = theory.analytic_gradient(STRETCHED_PAIR) / ang2bohr
    assert np.allclose(gradient, expected, atol=1e-6)


def test_gradient_sums_to_zero():
    """Newton's third law: an isolated pair feels no net force."""
    numgrad = NumGrad(theory=HarmonicPairTheory())

    _energy, gradient = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=True)

    assert np.allclose(gradient.sum(axis=0), 0.0, atol=1e-8)


def test_run_honours_the_grad_flag():
    """grad=False returns the energy alone.

    NumGrad.run used to ignore grad= and always return a tuple, so callers expecting
    a scalar energy silently got one.
    """
    numgrad = NumGrad(theory=HarmonicPairTheory())

    energy = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=False)

    assert np.isscalar(energy) or np.ndim(energy) == 0, f"Expected a scalar energy, got {energy!r}"


@pytest.mark.parametrize(
    ("npoint", "tolerance"),
    # The forward difference is first-order accurate, the central one second-order.
    [(1, 1e-3), (2, 1e-6)],
)
def test_both_stencils_reach_their_expected_accuracy(npoint, tolerance):
    numgrad = NumGrad(theory=HarmonicPairTheory(), npoint=npoint)

    _energy, gradient = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=True)

    ang2bohr = 1.88972612546
    expected = HarmonicPairTheory().analytic_gradient(STRETCHED_PAIR) / ang2bohr
    assert np.allclose(gradient, expected, atol=tolerance)


def test_rejects_unknown_npoint():
    """An unsupported stencil must fail loudly.

    It used to skip gradient assembly and return an all-zero gradient, which an
    optimizer reads as an already-converged structure.
    """
    with pytest.raises(InputError):
        NumGrad(theory=ZeroTheory(), npoint=3)


def test_flat_surface_gives_zero_gradient():
    numgrad = NumGrad(theory=ZeroTheory())

    energy, gradient = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=True)

    assert energy == 0.0
    assert gradient.shape == (2, 3), "One gradient row per atom"
    assert np.allclose(gradient, 0.0)
