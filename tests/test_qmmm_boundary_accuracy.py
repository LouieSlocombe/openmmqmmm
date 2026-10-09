"""Energy-derivative checks for covalent caps and truncated charge fields."""

import numpy as np
import pytest
from conftest import central_difference_gradient

from openmmqmmm import Fragment, QMMMTheory, ZeroTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError
from openmmqmmm.qmmm import _linkatom_force_chainrule


class _BoundaryMM:
    numatoms = 3

    def update_charges(self, *_args):
        pass


class _CapEnergyQM:
    """An anisotropic cap bond and cap/charge couplings with exact derivatives."""

    numcores = 1
    theorytype = "QM"

    def run(self, *, current_coords, current_mm_coords=None, mm_charges=None, grad=False, pc=False, **_kwargs):
        coords = np.asarray(current_coords) * ANG_TO_BOHR
        weights = np.array([1.3, 0.7, 1.1])
        displacement = coords[-1] - coords[0]
        energy = 0.5 * np.dot(displacement, weights * displacement)
        qm_gradient = np.zeros_like(coords)
        qm_gradient[-1] = weights * displacement
        qm_gradient[0] = -qm_gradient[-1]
        if pc:
            pc_coords = np.asarray(current_mm_coords).reshape(-1, 3) * ANG_TO_BOHR
            displacement = coords[-1] - pc_coords
            weighted_displacement = np.asarray(mm_charges)[:, None] * weights * displacement
            energy += 0.5 * np.sum(displacement * weighted_displacement)
            qm_gradient[-1] += np.sum(weighted_displacement, axis=0)
            pc_gradient = -weighted_displacement
        if not grad:
            return float(energy)
        if pc:
            return float(energy), qm_gradient, pc_gradient
        return float(energy), qm_gradient


def _boundary_theory(*, qm_theory=None, **options):
    fragment = Fragment(
        elems=["C", "C", "C"],
        coords=[[0.2, -0.4, 0.3], [1.5, 0.1, 0.7], [2.8, 0.9, 0.4]],
        charge=0,
        mult=1,
    )
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm_theory if qm_theory is not None else _CapEnergyQM(),
        mm_theory=_BoundaryMM(),
        qmatoms=[0],
        charges=[0.0, 0.4, -0.1],
        qm_charge=0,
        qm_mult=1,
        **options,
    )
    return theory, fragment


def _finite_difference(theory, fragment, step=1.0e-5):
    return central_difference_gradient(
        lambda coords: theory.run(current_coords=coords, elems=fragment.elems, grad=False),
        fragment.coords,
        step=step,
    )


@pytest.mark.parametrize("placement", ["simple", "ratio"])
@pytest.mark.parametrize("projection", ["adv", "lever", "chain"])
def test_linkatom_projection_matches_placement_derivative(placement, projection):
    theory, fragment = _boundary_theory(
        embedding="mech", linkatom_method=placement, linkatom_forceproj_method=projection
    )
    _energy, gradient = theory.run(current_coords=fragment.coords, elems=fragment.elems, grad=True)
    gradient = gradient.copy()

    assert gradient == pytest.approx(_finite_difference(theory, fragment), abs=2.0e-9)
    assert np.sum(gradient, axis=0) == pytest.approx(np.zeros(3), abs=1.0e-14)


def test_ratio_placement_uses_exact_derivative_with_default_projection():
    theory, fragment = _boundary_theory(embedding="mech", linkatom_method="ratio", linkatom_ratio=0.45)
    _energy, gradient = theory.run(current_coords=fragment.coords, elems=fragment.elems, grad=True)
    gradient = gradient.copy()

    assert gradient == pytest.approx(_finite_difference(theory, fragment), abs=2.0e-9)


def test_fixed_distance_projection_obeys_chain_rule_and_conserves_cap_force():
    qm_coords = np.array([0.2, -0.4, 0.3])
    mm_coords = np.array([1.5, 0.1, 0.7])
    cap_gradient = np.array([0.4, -0.7, 0.9])

    def cap_position(qm, mm):
        return qm + 1.09 * (mm - qm) / np.linalg.norm(mm - qm)

    qm_gradient, mm_gradient = _linkatom_force_chainrule(
        qm_coords, mm_coords, cap_position(qm_coords, mm_coords), cap_gradient
    )
    numerical = np.zeros((2, 3))
    step = 1.0e-5
    for atom in range(2):
        for axis in range(3):
            plus = np.array([qm_coords, mm_coords])
            minus = plus.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            numerical[atom, axis] = np.dot(cap_gradient, cap_position(*plus) - cap_position(*minus)) / (2 * step)

    assert np.array([qm_gradient, mm_gradient]) == pytest.approx(numerical, abs=2.0e-9)
    assert np.asarray(qm_gradient) + mm_gradient == pytest.approx(cap_gradient, abs=1.0e-14)


@pytest.mark.parametrize("placement", ["simple", "ratio"])
@pytest.mark.parametrize("boundary,dipoles", [("shift", False), ("shift", True), ("rcd", False)])
@pytest.mark.parametrize("radius", [0.1, 1.4])
def test_truncated_full_refresh_includes_cap_and_virtual_charge_gradients(placement, boundary, dipoles, radius):
    options = {
        "embedding": "elstat",
        "linkatom_method": placement,
        "chargeboundary_method": boundary,
        "dipole_correction": dipoles,
    }
    theory, fragment = _boundary_theory(
        **options, truncated_pc=True, truncated_pc_radius=radius, truncated_pc_recalc_iter=1
    )
    reference, _fragment = _boundary_theory(**options)
    reference_energy, reference_gradient = reference.run(
        current_coords=fragment.coords, elems=fragment.elems, grad=True
    )
    energy, gradient = theory.run(current_coords=fragment.coords, elems=fragment.elems, grad=True)
    gradient = gradient.copy()

    assert theory.original_QMcorrection_gradient.shape == (len(theory.qmatoms) + theory.num_linkatoms, 3)
    assert energy == pytest.approx(reference_energy, abs=1.0e-14)
    assert gradient == pytest.approx(reference_gradient, abs=1.0e-14)
    assert gradient == pytest.approx(_finite_difference(theory, fragment), abs=2.0e-9)
    assert np.sum(gradient, axis=0) == pytest.approx(np.zeros(3), abs=1.0e-14)


@pytest.mark.parametrize("embedding", ["mech", "elstat"])
@pytest.mark.parametrize("grad", [False, True])
def test_zero_theory_keeps_link_atom_gradient_rows(embedding, grad):
    theory, fragment = _boundary_theory(qm_theory=ZeroTheory(), embedding=embedding)
    result = theory.run(current_coords=fragment.coords, grad=grad)
    assert theory.linkatoms is True
    assert theory.num_linkatoms == 1
    if grad:
        energy, gradient = result
        assert energy == 0.0
        assert gradient == pytest.approx(np.zeros_like(fragment.coords))
        assert theory.QMgradient.shape == (2, 3)
    else:
        assert result == 0.0


@pytest.mark.parametrize("grad", [False, True])
def test_zero_theory_supports_truncated_pc_full_refresh(grad):
    theory, fragment = _boundary_theory(
        qm_theory=ZeroTheory(),
        embedding="elstat",
        truncated_pc=True,
        truncated_pc_radius=0.1,
        truncated_pc_recalc_iter=1,
    )
    result = theory.run(current_coords=fragment.coords, grad=grad)
    assert theory.linkatoms is True
    assert theory.truncPC_E_correction == 0.0
    if grad:
        energy, gradient = result
        assert energy == 0.0
        assert gradient == pytest.approx(np.zeros_like(fragment.coords))
        assert theory.PCgradient.shape == (4, 3)
    else:
        assert result == 0.0


@pytest.mark.parametrize("method,dipoles", [("shift", False), ("shift", True), ("rcd", False)])
def test_boundary_charge_positions_equal_gradient_mappings(method, dipoles):
    theory, fragment = _boundary_theory(chargeboundary_method=method, dipole_correction=dipoles)
    for displacement in (np.zeros((3, 3)), np.array([[0.01, 0.02, 0], [0.03, 0, -0.01], [0, 0.04, 0.02]])):
        coords = fragment.coords + displacement
        theory.run(current_coords=coords)
        sites = theory.pointchargecoords[len(theory.mmatoms) :]
        expected = np.asarray(
            [
                sum(weight * coords[index] for index, weight in mapping)
                for mapping in theory._virtual_site_gradient_mappings
            ]
        ).reshape(-1, 3)
        assert sites == pytest.approx(expected, abs=1e-15)


def test_unknown_charge_boundary_method_is_rejected_before_first_run():
    with pytest.raises(InputError, match="chargeboundary_method"):
        _boundary_theory(chargeboundary_method="invalid")
