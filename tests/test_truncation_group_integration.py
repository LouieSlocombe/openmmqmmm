"""Charge selection preserves physical groups at the QM-backend interface."""

import numpy as np
import openmm
import pytest
from conftest import dummy_mm, use_pme
from test_drude_qmmm import _drude_mm
from test_qmmm_boundary_accuracy import _boundary_theory

from openmmqmmm import Fragment, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError


class _RecordingQM:
    numcores = 1
    theorytype = "QM"

    def __init__(self):
        self.fields = []

    def run(self, *, current_coords, current_mm_coords, mm_charges, grad=False, **_kwargs):
        qm_coords = np.asarray(current_coords)
        pc_coords = np.asarray(current_mm_coords).reshape(-1, 3)
        charges = np.asarray(mm_charges)
        self.fields.append((pc_coords.copy(), charges.copy()))
        delta = (qm_coords[:, None, :] - pc_coords[None, :, :]) * ANG_TO_BOHR
        weighted = delta * charges[None, :, None]
        energy = 0.5 * np.sum(delta * weighted)
        if grad:
            return energy, weighted.sum(axis=1), -weighted.sum(axis=0)
        return energy


def _water_theory(**options):
    fragment = Fragment(
        elems=["He", "O", "H", "H", "O", "H", "H"],
        coords=[[0, 0, 0], [2.8, 0, 0], [3.5, 0.6, 0], [3.5, -0.6, 0], [10, 0, 0], [10.7, 0.6, 0], [10.7, -0.6, 0]],
        conncalc=False,
    )
    mm = dummy_mm(fragment, bonds=[(1, 2), (1, 3), (4, 5), (4, 6)])
    mm.update_charges(range(7), [0, -0.8, 0.4, 0.4, -0.8, 0.4, 0.4])
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=3.0,
        **options,
    )
    return theory, fragment, qm


def test_topology_water_is_complete_when_only_oxygen_is_inside_radius():
    theory, fragment, qm = _water_theory()
    theory.run(current_coords=fragment.coords)
    selected_coords, selected_charges = qm.fields[0]
    assert selected_coords == pytest.approx(fragment.coords[1:4])
    assert selected_charges == pytest.approx([-0.8, 0.4, 0.4])
    assert np.sum(selected_charges) == pytest.approx(0.0, abs=1e-14)
    assert len(qm.fields[-1][1]) == 6


def test_explicit_groups_merge_with_complete_topology_molecules():
    # Linking the two oxygens also includes both waters' hydrogens.
    theory, fragment, qm = _water_theory(truncated_pc_groups=[[1, 4]])
    theory.run(current_coords=fragment.coords)
    selected_coords, selected_charges = qm.fields[0]
    assert selected_coords == pytest.approx(fragment.coords[1:])
    assert selected_charges == pytest.approx([-0.8, 0.4, 0.4, -0.8, 0.4, 0.4])


def test_changing_explicit_groups_reselects_before_reusing_a_cached_field():
    theory, fragment, qm = _water_theory()
    theory.run(current_coords=fragment.coords)
    calls_before = len(qm.fields)
    theory.truncated_pc_groups = [[1, 4]]
    theory.run(current_coords=fragment.coords)
    assert qm.fields[calls_before][0] == pytest.approx(fragment.coords[1:])
    assert theory.truncated_pc_recalc_flag is True


@pytest.mark.parametrize("method,dipoles,radius", [("shift", True, 2.38663), ("rcd", False, 2.0)])
def test_boundary_corrective_charges_remain_with_their_parent_group(method, dipoles, radius):
    options = {"chargeboundary_method": method, "dipole_correction": dipoles}
    reference, fragment = _boundary_theory(**options)
    reference.run(current_coords=fragment.coords)
    full_coords = reference.pointchargecoords.copy()
    full_charges = np.asarray(reference.pointcharges).copy()
    capped_qm = np.vstack([fragment.coords[0], *reference.linkatoms_dict.values()])
    raw_selection = np.linalg.norm(full_coords - capped_qm.mean(axis=0), axis=1) < radius
    # Both examples would have split a boundary charge group under sitewise selection.
    assert raw_selection.any()
    assert not raw_selection.all()

    theory, _fragment = _boundary_theory(
        **options,
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=radius,
        truncated_pc_selection="centroid",
    )
    theory.run(current_coords=fragment.coords)
    assert theory.pointchargecoords == pytest.approx(full_coords)
    assert theory.pointcharges == pytest.approx(full_charges)
    assert np.sum(theory.pointcharges) == pytest.approx(0.3, abs=1e-14)


def test_nearest_selection_uses_real_qm_atoms_without_link_caps():
    theory, fragment = _boundary_theory(truncated_pc=True, truncated_pc_radius=0.5, truncated_pc_recalc_iter=50)
    theory.run(current_coords=fragment.coords)
    cap_coords = np.asarray(list(theory.linkatoms_dict.values()))
    assert np.linalg.norm(cap_coords[0] - fragment.coords[1]) < 0.5
    assert np.linalg.norm(fragment.coords[0] - fragment.coords[1]) > 0.5
    assert len(theory.pointcharges) == 0


@pytest.mark.parametrize("selection,expected_count", [(None, 1), ("nearest", 1), ("centroid", 0)])
def test_nearest_selection_reaches_the_edge_of_an_extended_qm_region(selection, expected_count):
    fragment = Fragment(
        elems=["He"] * 4,
        coords=[[-10, 0, 0], [10, 0, 0], [11, 0, 0], [20, 0, 0]],
        conncalc=False,
    )
    qm = _RecordingQM()
    options = {} if selection is None else {"truncated_pc_selection": selection}
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        qmatoms=[0, 1],
        charges=[0, 0, 0.25, -0.1],
        qm_charge=0,
        qm_mult=1,
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=2.0,
        **options,
    )
    theory.run(current_coords=fragment.coords)
    selected_coords, selected_charges = qm.fields[0]
    assert len(selected_charges) == expected_count
    if expected_count:
        assert selected_coords == pytest.approx(np.array([[11, 0, 0]]))
        assert selected_charges == pytest.approx([0.25])
    else:
        calls_before = len(qm.fields)
        theory.truncated_pc_selection = "nearest"
        theory.run(current_coords=fragment.coords)
        assert qm.fields[calls_before][1] == pytest.approx([0.25])
        assert theory.truncated_pc_recalc_flag is True


@pytest.mark.parametrize("periodic", [False, True])
def test_native_virtual_site_dependencies_complete_the_group_without_topology_bonds(periodic):
    fragment = Fragment(
        elems=["He"] * 5,
        coords=[[1, 1, 1], [4.8, 1, 1], [6.2, 1, 1], [5.5, 1, 1], [12, 1, 1]],
        conncalc=False,
    )
    mm = dummy_mm(fragment)
    mm.system.setParticleMass(3, 0)
    mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(1, 2, 0.5, 0.5))
    mm.update_charges(range(5), [0, 0.3, -0.1, -0.2, 0.5])
    if periodic:
        use_pme(mm)
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=4.0,
    )
    reference_energy = theory.run(current_coords=fragment.coords)
    selected_coords, selected_charges = qm.fields[0]
    assert selected_coords == pytest.approx(fragment.coords[1:4])
    assert selected_charges == pytest.approx([0.3, -0.1, -0.2])
    assert np.sum(selected_charges) == pytest.approx(0.0, abs=1e-14)

    if periodic:
        wrapped = fragment.coords.copy()
        wrapped[2, 0] -= 30
        wrapped[3] += [60, -30, 30]
        assert theory.run(current_coords=wrapped) == pytest.approx(reference_energy, abs=1e-10)
        assert qm.fields[-1][0] == pytest.approx(selected_coords, abs=1e-12)


@pytest.mark.parametrize("periodic", [False, True])
def test_drude_shells_remain_with_their_parent_molecule(periodic):
    mm, fragment = _drude_mm(periodic=periodic)
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        embedding="drude",
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=5.01,
    )
    distances = np.linalg.norm(fragment.coords[1:] - fragment.coords[0], axis=1)
    assert np.flatnonzero(distances < 5.01).tolist() == [0]
    theory.run(current_coords=fragment.coords)
    selected_coords, selected_charges = qm.fields[0]
    assert selected_coords == pytest.approx(fragment.coords[1:])
    assert selected_charges == pytest.approx([0.6, -0.6, 0.5, -0.5])
    assert np.sum(selected_charges) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    "options",
    [
        {"truncated_pc_selection": "unknown"},
        {"truncated_pc_selection": None},
        {"truncated_pc_groups": [[]]},
        {"truncated_pc_groups": [[2]]},
        {"truncated_pc_groups": [[1, 1]]},
        {"truncated_pc_groups": [[True]]},
        {"truncated_pc_groups": [[1.0]]},
        {"truncated_pc_groups": [1]},
        {"truncated_pc_groups": "12"},
    ],
)
def test_invalid_selection_configuration_is_rejected_before_mm_charge_mutation(options):
    fragment = Fragment(elems=["He", "He"], coords=[[0, 0, 0], [5, 0, 0]], conncalc=False)
    mm = dummy_mm(fragment)
    original_charges = mm.charges.copy()
    original_system = openmm.XmlSerializer.serialize(mm.system)
    qm = _RecordingQM()
    with pytest.raises(InputError, match=next(iter(options))):
        QMMMTheory(
            fragment=fragment,
            qm_theory=qm,
            mm_theory=mm,
            qmatoms=[0],
            qm_charge=0,
            qm_mult=1,
            charges=[0.2, 0.3],
            truncated_pc=True,
            **options,
        )
    assert qm.fields == []
    assert mm.charges == original_charges
    assert openmm.XmlSerializer.serialize(mm.system) == original_system


@pytest.mark.parametrize("interval", [1, 50])
def test_an_entirely_qm_system_accepts_an_empty_embedding_field(interval):
    fragment = Fragment(elems=["He", "He"], coords=[[0, 0, 0], [5, 0, 0]], conncalc=False)
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        qmatoms=[0, 1],
        qm_charge=0,
        qm_mult=1,
        charges=[0, 0],
        truncated_pc=True,
        truncated_pc_recalc_iter=interval,
    )
    for _ in range(2):
        assert theory.run(current_coords=fragment.coords) == 0.0
    assert all(coords.shape == (0, 3) and charges.shape == (0,) for coords, charges in qm.fields)
    assert theory.truncated_pc_diagnostics["selected_count"] == 0
    assert theory.truncated_pc_diagnostics["full_count"] == 0
    if interval == 1:
        energy, gradient = theory.run(current_coords=fragment.coords, grad=True)
        assert energy == 0.0
        assert gradient == pytest.approx(np.zeros_like(fragment.coords))


@pytest.mark.parametrize("groups,expected_count", [(None, 1), ([[1, 2]], 2)])
def test_explicit_groups_supply_missing_molecular_metadata(groups, expected_count):
    fragment = Fragment(elems=["He"] * 3, coords=[[0, 0, 0], [2, 0, 0], [5, 0, 0]], conncalc=False)
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        charges=[0, 0.5, -0.5],
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=3.0,
        truncated_pc_groups=groups,
    )
    theory.run(current_coords=fragment.coords)
    assert len(qm.fields[0][1]) == expected_count


def test_openmm_topology_takes_precedence_over_fragment_connectivity():
    fragment = Fragment(elems=["He"] * 3, coords=[[0, 0, 0], [2, 0, 0], [5, 0, 0]], conncalc=False)
    fragment.connectivity = [[0, 1, 2]]
    mm = dummy_mm(fragment)  # Authoritative topology contains three unbonded particles.
    mm.update_charges(range(3), [0, 0.5, -0.5])
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=3.0,
    )
    theory.run(current_coords=fragment.coords)
    assert qm.fields[0][1] == pytest.approx([0.5])
    assert theory.linkatoms is False


def test_qm_atoms_do_not_split_the_remaining_molecular_charge_group():
    fragment = Fragment(
        elems=["C"] * 5,
        coords=[[-2.8, 0, 0], [-1.4, 0, 0], [0, 0, 0], [1.8, 0, 0], [3.2, 0, 0]],
        conncalc=False,
    )
    mm = dummy_mm(fragment, bonds=[(0, 1), (1, 2), (2, 3), (3, 4)])
    mm.update_charges(range(5), [0.1, 0.2, 0, -0.1, -0.2])
    qm = _RecordingQM()
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[2],
        qm_charge=0,
        qm_mult=1,
        dipole_correction=False,
        truncated_pc=True,
        truncated_pc_recalc_iter=50,
        truncated_pc_radius=1.5,
    )
    theory.run(current_coords=fragment.coords)
    assert qm.fields[0][0] == pytest.approx(fragment.coords[[0, 1, 3, 4]])
    assert np.sum(qm.fields[0][1]) == pytest.approx(0.0, abs=1e-14)
