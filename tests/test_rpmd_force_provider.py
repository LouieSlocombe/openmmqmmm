"""The PythonForce providers validate the external theory's answers and cache by geometry (and box)."""

import numpy as np
import openmm
import pytest

from openmmqmmm.exceptions import InputError, InternalError
from openmmqmmm.openmm.rpmd_force import RPMDExternalQMForceProvider, add_rpmd_python_force

HARTREE_KJ_PER_MOL = 2625.499639
HARTREE_PER_BOHR_KJ_PER_MOL_NM = HARTREE_KJ_PER_MOL / 0.052917721


class _Theory:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def run(self, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def _state(positions_nm, box_nm=None):
    system = openmm.System()
    for _ in positions_nm:
        system.addParticle(1.0)
    if box_nm is not None:
        system.setDefaultPeriodicBoxVectors(*np.diag([box_nm] * 3))
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(np.asarray(positions_nm, dtype=float))
    return context.getState(getPositions=True)


def _provider(theory, periodic=False, cache_size=64):
    return RPMDExternalQMForceProvider(theory, ["H", "H"], 0, 1, periodic=periodic, cache_size=cache_size)


def test_energy_and_gradient_are_converted_to_openmm_units_with_forces_negated():
    gradient = np.array([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]])
    provider = _provider(_Theory(result=(-1.5, gradient)))

    energy, forces = provider(_state([[0, 0, 0], [0.1, 0, 0]]))

    assert energy == pytest.approx(-1.5 * HARTREE_KJ_PER_MOL, rel=1e-7)
    np.testing.assert_allclose(forces, -gradient * HARTREE_PER_BOHR_KJ_PER_MOL_NM, rtol=1e-7)


def test_positions_of_the_wrong_size_are_rejected_before_the_theory_runs():
    theory = _Theory(result=(0.0, np.zeros((2, 3))))
    with pytest.raises(InternalError, match=r"positions with shape \(1, 3\); expected \(2, 3\)"):
        _provider(theory)(_state([[0, 0, 0]]))
    assert theory.calls == 0


def test_a_failing_theory_is_reported_with_the_atom_count():
    provider = _provider(_Theory(error=ValueError("scf did not converge")))
    with pytest.raises(RuntimeError, match="failed for 2 atoms: scf did not converge"):
        provider(_state([[0, 0, 0], [0.1, 0, 0]]))


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ((0.0, np.zeros((3, 3))), r"gradient with shape \(3, 3\)"),
        ((float("nan"), np.zeros((2, 3))), "non-finite"),
        ((0.0, np.full((2, 3), np.inf)), "non-finite"),
    ],
)
def test_malformed_theory_answers_are_rejected(result, message):
    with pytest.raises(InternalError, match=message):
        _provider(_Theory(result=result))(_state([[0, 0, 0], [0.1, 0, 0]]))


@pytest.mark.parametrize("result", [0.0, (0.0, np.zeros((2, 3)), None)])
def test_a_theory_not_returning_an_energy_gradient_pair_is_reported_as_an_evaluation_failure(result):
    with pytest.raises(RuntimeError, match=r"failed for 2 atoms: .*\(energy, gradient\) pair"):
        _provider(_Theory(result=result))(_state([[0, 0, 0], [0.1, 0, 0]]))


def test_cache_holds_only_the_configured_number_of_geometries():
    theory = _Theory(result=(0.0, np.zeros((2, 3))))
    provider = _provider(theory, cache_size=1)
    first, second = _state([[0, 0, 0], [0.1, 0, 0]]), _state([[0, 0, 0], [0.2, 0, 0]])

    for state in (first, second, first):
        provider(state)
    assert (theory.calls, provider.evaluation_count, provider.cache_hits) == (3, 3, 0)

    roomy = _provider(_Theory(result=(0.0, np.zeros((2, 3)))), cache_size=2)
    for state in (first, second, first):
        roomy(state)
    assert (roomy.evaluation_count, roomy.cache_hits) == (2, 1)
    roomy.clear_cache()
    roomy(first)
    assert (roomy.evaluation_count, roomy.cache_hits) == (3, 1)


def test_periodic_cache_keys_include_the_box():
    theory = _Theory(result=(0.0, np.zeros((2, 3))))
    provider = _provider(theory, periodic=True)
    positions = [[0, 0, 0], [0.1, 0, 0]]

    provider(_state(positions, box_nm=2.0))
    provider(_state(positions, box_nm=2.5))
    provider(_state(positions, box_nm=2.0))

    assert (provider.evaluation_count, provider.cache_hits) == (2, 1)


def test_python_force_needs_a_free_force_group():
    system = openmm.System()
    system.addParticle(1.0)
    for group in range(32):
        force = openmm.CustomExternalForce("0")
        force.setForceGroup(group)
        system.addForce(force)

    with pytest.raises(InputError, match="all 32 groups"):
        add_rpmd_python_force(system, _provider(_Theory()))
