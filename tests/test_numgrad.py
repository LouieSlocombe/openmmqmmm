import numpy as np
import pytest

from openmmqmmm import Fragment, NumGrad, QMMMTheory, ZeroTheory
from openmmqmmm.exceptions import InputError


class HarmonicPairTheory:
    """A toy theory: two atoms on a spring, E = 0.5*k*(r - r0)**2."""

    def __init__(self, force_constant=0.5, equilibrium=1.0):
        self.theorytype = "QM"
        self.theorynamelabel = "HarmonicPair"
        self.numcores = 1
        self.force_constant = force_constant
        self.equilibrium = equilibrium
        self.calls = []

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
        self.calls.append(coords.copy())
        distance = np.linalg.norm(coords[1] - coords[0])
        energy = 0.5 * self.force_constant * (distance - self.equilibrium) ** 2
        if grad:
            return energy, self.analytic_gradient(coords)
        return energy


STRETCHED_PAIR = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])


def test_numerical_gradient_matches_the_analytic_one():
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
    theory = HarmonicPairTheory()
    numgrad = NumGrad(theory=theory)

    energy = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=False)

    assert np.isscalar(energy) or np.ndim(energy) == 0, f"Expected a scalar energy, got {energy!r}"
    assert len(theory.calls) == 1
    assert numgrad.gradient is None


def test_standalone_wrapper_does_not_forward_absent_qmmm_options():
    class StrictTheory:
        def run(self, *, current_coords, elems, grad, label, charge, mult):
            assert elems == ["H", "H"]
            assert grad is False
            return float(np.sum(np.asarray(current_coords) ** 2))

    energy = NumGrad(StrictTheory()).run(
        current_coords=STRETCHED_PAIR,
        elems=["H", "H"],
        charge=0,
        mult=1,
    )

    assert energy == pytest.approx(np.sum(STRETCHED_PAIR**2))


def test_qm_element_names_are_accepted_for_mechanical_qmmm_calls():
    theory = HarmonicPairTheory()

    energy, gradient = NumGrad(theory=theory).run(
        current_coords=STRETCHED_PAIR,
        qm_elems=["H", "H"],
        charge=0,
        mult=1,
        grad=True,
    )

    assert np.isscalar(energy)
    assert gradient.shape == (2, 3)


def test_point_charge_gradient_request_fails_before_energy_calls():
    theory = HarmonicPairTheory()

    with pytest.raises(InputError, match="point-charge gradients"):
        NumGrad(theory=theory).run(
            current_coords=STRETCHED_PAIR,
            current_mm_coords=np.array([[5.0, 0.0, 0.0]]),
            mm_charges=[-0.5],
            qm_elems=["H", "H"],
            charge=0,
            mult=1,
            grad=True,
            pc=True,
        )

    assert theory.calls == []


def test_numgrad_can_serve_as_a_mechanical_qmmm_component():
    fragment = Fragment(
        coords=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        elems=["H", "H"],
        charge=0,
        mult=1,
        conncalc=False,
    )
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=NumGrad(ZeroTheory()),
        mm_theory=None,
        qmatoms=[0],
        charges=[0.0, 0.0],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
    )

    energy, gradient = qmmm.run(
        current_coords=fragment.coords,
        elems=fragment.elems,
        grad=True,
        charge=0,
        mult=1,
    )

    assert energy == 0.0
    assert gradient == pytest.approx(np.zeros((2, 3)))


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


@pytest.mark.parametrize("npoint", [0, 3, 1.0, True])
def test_rejects_unknown_npoint(npoint):
    with pytest.raises(InputError, match="npoint"):
        NumGrad(theory=ZeroTheory(), npoint=npoint)


def test_rejects_an_object_without_a_theory_interface():
    with pytest.raises(InputError, match="callable run"):
        NumGrad(theory=object())


@pytest.mark.parametrize("runmode", ["concurrent", "", None])
def test_rejects_unknown_runmode(runmode):
    with pytest.raises(InputError, match="runmode"):
        NumGrad(theory=ZeroTheory(), runmode=runmode)


@pytest.mark.parametrize("displacement", [0.0, -0.1, np.nan, np.inf, -np.inf, "not-a-number", True])
def test_rejects_nonpositive_or_nonfinite_displacement(displacement):
    with pytest.raises(InputError, match="positive finite"):
        NumGrad(theory=ZeroTheory(), displacement=displacement)


@pytest.mark.parametrize("numcores", [0, -1, 1.5, True])
def test_rejects_invalid_numcores(numcores):
    with pytest.raises(InputError, match="positive integer"):
        NumGrad(theory=ZeroTheory(), numcores=numcores)


def test_set_numcores_validates_the_new_value():
    numgrad = NumGrad(theory=ZeroTheory())

    with pytest.raises(InputError, match="positive integer"):
        numgrad.set_numcores(0)

    assert numgrad.numcores == 1


@pytest.mark.parametrize(
    "coords",
    [
        None,
        np.array([0.0, 0.0, 0.0]),
        np.zeros((2, 2)),
        np.zeros((0, 3)),
        np.array([[0.0, 0.0, np.nan], [0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, np.inf], [0.0, 0.0, 1.0]]),
    ],
)
def test_rejects_invalid_coordinates_before_calling_theory(coords):
    theory = HarmonicPairTheory()

    with pytest.raises(InputError, match="current_coords"):
        NumGrad(theory=theory).run(current_coords=coords, elems=["H", "H"], charge=0, mult=1, grad=True)

    assert theory.calls == []


@pytest.mark.parametrize("elems", [None, ["H"], ["H", "H", "H"]])
def test_rejects_missing_or_mismatched_elements_before_calling_theory(elems):
    theory = HarmonicPairTheory()

    with pytest.raises(InputError, match="element"):
        NumGrad(theory=theory).run(current_coords=STRETCHED_PAIR, elems=elems, charge=0, mult=1, grad=True)

    assert theory.calls == []


@pytest.mark.parametrize(("npoint", "expected_calls"), [(1, 7), (2, 13)])
def test_serial_stencils_evaluate_each_geometry_once(npoint, expected_calls):
    theory = HarmonicPairTheory()

    NumGrad(theory=theory, npoint=npoint).run(
        current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=True
    )

    assert len(theory.calls) == expected_calls
    assert sum(np.array_equal(call, STRETCHED_PAIR) for call in theory.calls) == 1


def test_read_only_caller_coordinates_are_not_mutated_while_building_displacements():
    coords = STRETCHED_PAIR.copy()
    coords.setflags(write=False)

    NumGrad(theory=HarmonicPairTheory(), npoint=1).run(
        current_coords=coords, elems=["H", "H"], charge=0, mult=1, grad=True
    )

    assert coords == pytest.approx(STRETCHED_PAIR)


def test_parallel_forward_difference_schedules_one_reference_geometry(monkeypatch):
    captured = {}

    def fake_job_parallel(**kwargs):
        fragments = kwargs["fragments"]
        captured["labels"] = [fragment.label for fragment in fragments]
        captured["numcores"] = kwargs["numcores"]
        energies = {}
        for fragment in fragments:
            distance = np.linalg.norm(fragment.coords[1] - fragment.coords[0])
            energies[fragment.label] = 0.25 * (distance - 1.0) ** 2
        return type("ParallelResult", (), {"energies_dict": energies})()

    monkeypatch.setattr("openmmqmmm.parallel.job_parallel", fake_job_parallel)

    _energy, gradient = NumGrad(theory=HarmonicPairTheory(), npoint=1, runmode="parallel", numcores=2).run(
        current_coords=STRETCHED_PAIR,
        elems=["H", "H"],
        charge=0,
        mult=1,
        grad=True,
        numcores=3,
    )

    assert captured["labels"][0] == "orig"
    assert captured["labels"].count("orig") == 1
    assert "Originalgeo" not in captured["labels"]
    assert len(captured["labels"]) == 7
    assert captured["numcores"] == 3
    assert gradient.shape == (2, 3)


def test_parallel_run_rejects_invalid_numcores_override(monkeypatch):
    monkeypatch.setattr(
        "openmmqmmm.parallel.job_parallel", lambda **_kwargs: pytest.fail("parallel work should not be submitted")
    )

    with pytest.raises(InputError, match="positive integer"):
        NumGrad(theory=HarmonicPairTheory(), runmode="parallel").run(
            current_coords=STRETCHED_PAIR,
            elems=["H", "H"],
            charge=0,
            mult=1,
            grad=True,
            numcores=0,
        )


def test_flat_surface_gives_zero_gradient():
    numgrad = NumGrad(theory=ZeroTheory())

    energy, gradient = numgrad.run(current_coords=STRETCHED_PAIR, elems=["H", "H"], charge=0, mult=1, grad=True)

    assert energy == 0.0
    assert gradient.shape == (2, 3), "One gradient row per atom"
    assert np.allclose(gradient, 0.0)
