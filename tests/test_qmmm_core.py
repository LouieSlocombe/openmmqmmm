from types import SimpleNamespace

import numpy as np
import pytest

import openmmqmmm.qmmm as qmmm_module
from openmmqmmm import Fragment, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError, InternalError


class _AnalyticQM:
    """Deterministic QM stand-in exposing both QM and point-charge gradients."""

    def __init__(self):
        self.numcores = 1
        self.theorytype = "QM"

    def set_numcores(self, numcores):
        self.numcores = numcores

    def run(self, *, current_coords=None, current_mm_coords=None, grad=False, pc=False, **_kwargs):
        energy = 1.25
        qm_gradient = np.tile([1.0, 2.0, 3.0], (len(current_coords), 1))
        num_pointcharges = 0 if current_mm_coords is None else len(current_mm_coords)
        pc_gradient = np.tile([4.0, 5.0, 6.0], (num_pointcharges, 1))
        if not grad:
            return energy
        if pc:
            return energy, qm_gradient, pc_gradient
        return energy, qm_gradient


def _minimal_qmmm(
    qmatoms,
    *,
    embedding="mech",
    linkatom_forceproj_method="adv",
    chargeboundary_method="shift",
    mm_theory=None,
):
    fragment = Fragment(
        elems=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        charge=0,
        mult=1,
        conncalc=False,
    )
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_AnalyticQM(),
        mm_theory=mm_theory,
        qmatoms=qmatoms,
        charges=[0.25, -0.25],
        embedding=embedding,
        qm_charge=0,
        qm_mult=1,
        linkatom_forceproj_method=linkatom_forceproj_method,
        chargeboundary_method=chargeboundary_method,
    )
    return theory, fragment


@pytest.mark.parametrize(
    ("embedding", "expected_gradient"),
    [
        ("mech", [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]),
        ("elstat", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    ],
)
def test_qmmm_gradient_without_an_mm_theory(embedding, expected_gradient):
    theory, fragment = _minimal_qmmm([0], embedding=embedding)

    energy, gradient = theory.run(
        current_coords=fragment.coords,
        elems=fragment.elems,
        grad=True,
        charge=0,
        mult=1,
    )

    assert energy == pytest.approx(1.25)
    assert np.allclose(gradient, expected_gradient)


def test_no_mm_qmmm_set_numcores_updates_the_wrapper_and_qm_theory():
    theory, _fragment = _minimal_qmmm([0])

    theory.set_numcores(4)

    assert theory.numcores == 4
    assert theory.qm_theory.numcores == 4


def test_updating_qm_region_charges_requires_a_mechanical_mm_theory():
    fragment = Fragment(
        elems=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        charge=0,
        mult=1,
        conncalc=False,
    )

    with pytest.raises(InputError, match="requires mechanical embedding and an MM theory"):
        QMMMTheory(
            fragment=fragment,
            qm_theory=_AnalyticQM(),
            mm_theory=None,
            qmatoms=[0],
            charges=[0.0, 0.0],
            embedding="mech",
            qm_charge=0,
            qm_mult=1,
            update_qm_region_charges=True,
        )


def test_rcd_without_a_covalent_boundary_uses_only_real_mm_sites():
    theory, fragment = _minimal_qmmm([0], embedding="elstat", chargeboundary_method="rcd")

    energy, gradient = theory.run(
        current_coords=fragment.coords,
        elems=fragment.elems,
        grad=True,
        charge=0,
        mult=1,
    )

    assert energy == pytest.approx(1.25)
    assert np.allclose(gradient, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert theory._virtual_site_gradient_mappings == []


class _BoundaryMM:
    def __init__(self, numatoms):
        self.numatoms = numatoms
        self.updates = []

    def update_charges(self, atom_indices, charges):
        self.updates.append((list(atom_indices), list(charges)))


def test_electrostatic_boundary_rejects_an_mm1_atom_without_mm_side_neighbours():
    fragment = Fragment(elems=["C", "C"], coords=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

    with pytest.raises(InputError, match="no MM-side neighbours"):
        QMMMTheory(
            fragment=fragment,
            qm_theory=_AnalyticQM(),
            mm_theory=_BoundaryMM(fragment.numatoms),
            qmatoms=[0],
            charges=[0.0, 0.3],
            embedding="elstat",
            qm_charge=0,
            qm_mult=1,
        )


def test_mm_boundary_neighbours_exclude_every_qm_atom(monkeypatch):
    theory = object.__new__(QMMMTheory)
    theory.boundaryatoms = {0: [2], 1: [2]}
    theory.qmatoms = [0, 1]
    theory.coords = np.zeros((4, 3))
    theory.elems = ["C"] * 4
    theory.embedding = "elstat"
    monkeypatch.setattr(
        qmmm_module.openmmqmmm.coords,
        "get_connected_atoms",
        lambda _coords, _elems, _scale, _tol, _atom: [0, 1, 3],
    )

    theory.get_mm_boundary(scale=1.0, tol=0.1)

    assert theory.MMboundarydict == {2: [3]}
    assert theory.MMboundary_indices == [2]
    assert theory.MMboundary_counts.tolist() == [1]


def test_openmm_qmmm_keeps_an_independent_snapshot_of_original_charges():
    class OpenMMTheory:
        def __init__(self):
            self.numatoms = 2
            self.charges = [-1.0, 1.0]

        def remove_constraints_for_atoms(self, _atoms):
            pass

        def modify_bonded_forces(self, _atoms):
            pass

        def addexceptions(self, _atoms):
            pass

        def delete_exceptions(self, _atoms):
            pass

        def update_charges(self, atom_indices, charges):
            for atom_index, charge in zip(atom_indices, charges, strict=True):
                self.charges[atom_index] = charge

    fragment = Fragment(
        elems=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        charge=-1,
        mult=1,
        conncalc=False,
    )
    mm_theory = OpenMMTheory()

    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_AnalyticQM(),
        mm_theory=mm_theory,
        qmatoms=[0],
        embedding="elstat",
        qm_charge=-1,
        qm_mult=1,
    )

    assert theory.charges == [-1.0, 1.0]
    assert theory.charges_qmregionzeroed == [0.0, 1.0]
    assert mm_theory.charges == [0.0, 1.0]
    assert theory.charges is not mm_theory.charges


class _PointChargeHarmonicQM(_AnalyticQM):
    """Coordinate-dependent point-charge energy with exact analytic gradients."""

    def run(
        self,
        *,
        current_coords=None,
        current_mm_coords=None,
        mm_charges=None,
        grad=False,
        pc=False,
        **_kwargs,
    ):
        qm_coords = np.asarray(current_coords, dtype=float)
        pc_coords_bohr = np.asarray(current_mm_coords, dtype=float) * ANG_TO_BOHR
        charges = np.asarray(mm_charges, dtype=float)
        energy = 0.5 * np.sum(charges[:, None] * pc_coords_bohr**2)
        if not grad:
            return energy

        qm_gradient = np.zeros_like(qm_coords)
        pc_gradient = charges[:, None] * pc_coords_bohr
        if pc:
            return energy, qm_gradient, pc_gradient
        return energy, qm_gradient


def test_truncated_pc_energy_then_gradient_initializes_both_corrections():
    fragment = Fragment(
        elems=["H", "H"],
        coords=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        charge=0,
        mult=1,
        conncalc=False,
    )
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_PointChargeHarmonicQM(),
        mm_theory=None,
        qmatoms=[0],
        charges=[0.0, -0.25],
        embedding="elstat",
        qm_charge=0,
        qm_mult=1,
        truncated_pc=True,
        truncated_pc_radius=1.0,
        truncated_pc_recalc_iter=1,
    )
    expected_energy = 0.5 * -0.25 * (5.0 * ANG_TO_BOHR) ** 2

    energy_only = theory.run(
        current_coords=fragment.coords,
        elems=fragment.elems,
        grad=False,
        charge=0,
        mult=1,
    )
    energy_with_gradient, gradient = theory.run(
        current_coords=fragment.coords,
        elems=fragment.elems,
        grad=True,
        charge=0,
        mult=1,
    )

    assert energy_only == pytest.approx(expected_energy)
    assert energy_with_gradient == pytest.approx(expected_energy)
    assert gradient[1] == pytest.approx([-0.25 * 5.0 * ANG_TO_BOHR, 0.0, 0.0])
    assert hasattr(theory, "original_QMcorrection_gradient")
    assert hasattr(theory, "original_PCcorrection_gradient")


def _boundary_qmmm(chargeboundary_method):
    fragment = Fragment(
        elems=["C", "C", "C", "C"],
        coords=[[-1.4, 0.0, 0.0], [0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.4, 0.0]],
        charge=0,
        mult=1,
    )
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=_PointChargeHarmonicQM(),
        mm_theory=_BoundaryMM(fragment.numatoms),
        qmatoms=[0],
        charges=[0.0, 0.4, -0.1, -0.3],
        embedding="elstat",
        qm_charge=0,
        qm_mult=1,
        chargeboundary_method=chargeboundary_method,
        dipole_correction=chargeboundary_method == "shift",
        linkatom_forceproj_method="none",
    )
    return theory, fragment


def _finite_difference_qmmm_gradient(theory, fragment, step=1.0e-5):
    coords = fragment.coords.copy()
    gradient_per_angstrom = np.zeros_like(coords)
    for atom_index in range(len(coords)):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom_index, axis] += step
            minus[atom_index, axis] -= step
            plus_energy = theory.run(current_coords=plus, elems=fragment.elems, grad=False, charge=0, mult=1)
            minus_energy = theory.run(current_coords=minus, elems=fragment.elems, grad=False, charge=0, mult=1)
            gradient_per_angstrom[atom_index, axis] = (plus_energy - minus_energy) / (2 * step)
    return gradient_per_angstrom / ANG_TO_BOHR


@pytest.mark.parametrize("chargeboundary_method", ["rcd", "shift"])
def test_virtual_charge_gradients_match_finite_difference(chargeboundary_method):
    theory, fragment = _boundary_qmmm(chargeboundary_method)
    _energy, analytic_gradient = theory.run(
        current_coords=fragment.coords,
        elems=fragment.elems,
        grad=True,
        charge=0,
        mult=1,
    )

    finite_difference_gradient = _finite_difference_qmmm_gradient(theory, fragment)

    assert analytic_gradient == pytest.approx(finite_difference_gradient, abs=1.0e-8)
    assert isinstance(theory.charges, list)


def test_rcd_shifting_uses_full_indices_before_compacting_to_mm_atoms():
    theory = object.__new__(QMMMTheory)
    theory.charges = [0.0, 0.3, -0.3]
    theory.MMboundary_indices = [1]
    theory.MMboundary_counts = np.array([1])
    theory.MMboundarydict = {1: [2]}
    theory.mmatoms = np.array([1, 2])

    pointcharges, additional_charges = theory.rcd_shifting_prep([0.0, 0.3, -0.3])
    pointchargecoords = theory.rcd_shifting_update(
        used_mmcoords=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        fullcoords=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )

    assert pointcharges == pytest.approx([0.0, -0.6, 0.6])
    assert additional_charges == pytest.approx([0.6])
    assert np.allclose(pointchargecoords, [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 0.0, 0.0]])


@pytest.mark.parametrize("qmatoms", [[-1], [2], [0.0], [True], []])
def test_qmmm_rejects_invalid_qm_atom_indices(qmatoms):
    with pytest.raises(InputError, match=r"QM atom|qmatoms"):
        _minimal_qmmm(qmatoms)


def test_qmmm_accepts_numpy_integer_indices_and_removes_duplicates():
    theory, _fragment = _minimal_qmmm([np.int64(1), 1])

    assert theory.qmatoms == [1]


def test_none_linkatom_projection_is_normalized_to_a_noop():
    theory, fragment = _minimal_qmmm([0], linkatom_forceproj_method=None)
    theory.linkatoms_dict = {(0, 1): np.array([0.5, 0.0, 0.0])}
    theory.linkatom_indices = [1]
    theory.QMgradient = np.array([[0.0, 0.0, 0.0], [7.0, 8.0, 9.0]])
    gradient = np.zeros((2, 3))

    theory._add_linkatom_force_projection(
        gradient,
        used_qmcoords=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        current_coords=fragment.coords,
    )

    assert theory.linkatom_forceproj_method == "none"
    assert np.allclose(gradient, np.zeros((2, 3)))


def test_linkatom_projection_name_is_normalized_and_validated():
    theory, _fragment = _minimal_qmmm([0], linkatom_forceproj_method="ADV")
    assert theory.linkatom_forceproj_method == "adv"

    with pytest.raises(InputError, match="linkatom_forceproj_method"):
        _minimal_qmmm([0], linkatom_forceproj_method="unknown")


class _RecordingMM:
    def __init__(self):
        self.numatoms = 2
        self.updates = []

    def update_charges(self, atom_indices, charges):
        self.updates.append((list(atom_indices), list(charges)))


def test_mechanical_embedding_logs_original_mm_region_charges_at_debug(caplog):
    with caplog.at_level("DEBUG", logger="openmmqmmm.qmmm"):
        _minimal_qmmm([0], embedding="mech", mm_theory=_RecordingMM())

    assert "QM atom 0 (H) charge: 0.25" in caplog.text
    assert "MM atom 1 (H) charge: -0.25" in caplog.text


class _DecompositionMM:
    def __init__(self):
        self.epsilons = [0.75]

    def qmmm_lj_energy(self, _atom_indices, _coords):
        return 0.5


class _DecompositionQMMM:
    def __init__(self, **kwargs):
        self.mm_theory = kwargs.get("mm_theory", _DecompositionMM())
        self.qm_theory = object()
        self.qmatoms = [0]
        self.qm_charge = 0
        self.qm_mult = 1
        self.unusualboundary = False
        self.excludeboundaryatomlist = []
        self.embedding = "elstat"
        self.pc = True
        self.runcalls = 4
        self.openmm_externalforce = False
        self.linkatom_ratio = 0.42

    def _image_periodic_coords(self, coords):
        return coords


def test_energy_decomposition_never_mutates_lj_parameters_on_failure(monkeypatch):
    theory = _DecompositionQMMM()
    monkeypatch.setattr(qmmm_module, "QMMMTheory", _DecompositionQMMM)
    calls = iter(
        [
            SimpleNamespace(energy=3.0, qm_energy=1.0, mm_energy=2.0),
            RuntimeError("QM failure"),
        ]
    )

    def fake_single_point(**_kwargs):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(qmmm_module.openmmqmmm, "single_point", fake_single_point)

    with pytest.raises(RuntimeError, match="QM failure"):
        qmmm_module.compute_decomposed_qm_mm_energy(fragment=SimpleNamespace(coords=np.zeros((1, 3))), theory=theory)

    assert theory.mm_theory.epsilons == [0.75]
    assert theory.embedding == "elstat"
    assert theory.pc is True
    assert theory.runcalls == 4


def test_energy_decomposition_requires_a_compatible_mm_theory(monkeypatch):
    theory = _DecompositionQMMM(mm_theory=None)
    monkeypatch.setattr(qmmm_module, "QMMMTheory", _DecompositionQMMM)

    with pytest.raises(InputError, match="isolated Lennard-Jones"):
        qmmm_module.compute_decomposed_qm_mm_energy(fragment=object(), theory=theory)


def test_energy_decomposition_rejects_negative_residual(monkeypatch):
    theory = _DecompositionQMMM()
    monkeypatch.setattr(qmmm_module, "QMMMTheory", _DecompositionQMMM)
    calls = iter(
        [
            SimpleNamespace(energy=-100.0, qm_energy=1.0, mm_energy=2.0),
            SimpleNamespace(qm_energy=0.5),
        ]
    )
    monkeypatch.setattr(qmmm_module.openmmqmmm, "single_point", lambda **_kwargs: next(calls))

    with pytest.raises(InternalError, match="residual=-103"):
        qmmm_module.compute_decomposed_qm_mm_energy(fragment=SimpleNamespace(coords=np.zeros((1, 3))), theory=theory)

    assert theory.mm_theory.epsilons == [0.75]


def test_energy_decomposition_preserves_custom_cap_definition(monkeypatch):
    theory = _DecompositionQMMM()
    monkeypatch.setattr(qmmm_module, "QMMMTheory", _DecompositionQMMM)
    evaluated_theories = []

    def fake_single_point(*, theory, **_kwargs):
        evaluated_theories.append(theory)
        return SimpleNamespace(energy=3.0, qm_energy=1.0, mm_energy=2.0)

    monkeypatch.setattr(qmmm_module.openmmqmmm, "single_point", fake_single_point)
    qmmm_module.compute_decomposed_qm_mm_energy(fragment=SimpleNamespace(coords=np.zeros((1, 3))), theory=theory)
    assert evaluated_theories[0] is theory
    assert evaluated_theories[1] is not theory
    assert evaluated_theories[1].linkatom_ratio == 0.42
    assert evaluated_theories[1].embedding == "mech"
    assert theory.embedding == "elstat"
