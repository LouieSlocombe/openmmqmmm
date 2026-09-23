"""Regression checks for covalent topology and charge-boundary selection."""

import numpy as np
import openmm
import pytest

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
from openmmqmmm.coords import get_boundary_atoms
from openmmqmmm.exceptions import InputError


class _QM:
    numcores = 1
    theorytype = "QM"


def _mm(fragment, bonds, *, periodic=False):
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
    if periodic:
        mm.periodic = True
        mm.system.setDefaultPeriodicBoxVectors(*np.diag([3.0] * 3))
    return mm


def _qmmm(fragment, mm, *, qmatoms=(0,), **kwargs):
    return QMMMTheory(
        fragment=fragment,
        qm_theory=_QM(),
        mm_theory=mm,
        qmatoms=qmatoms,
        qm_charge=0,
        qm_mult=1,
        **kwargs,
    )


def test_nonperiodic_topology_retains_stretched_boundary_and_recipient_bonds():
    fragment = Fragment(elems=["C"] * 3, coords=[[0, 0, 0], [2, 0, 0], [4, 0, 0]], conncalc=False)
    theory = _qmmm(fragment, _mm(fragment, [(0, 1), (1, 2)]))

    assert theory.boundaryatoms == {0: [1]}
    assert theory.MMboundarydict == {1: [2]}
    assert theory.linkatoms is True


def test_nonperiodic_close_contact_does_not_create_a_topology_boundary():
    fragment = Fragment(elems=["C"] * 3, coords=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]], conncalc=False)
    theory = _qmmm(fragment, _mm(fragment, [(1, 2)]))

    assert theory.boundaryatoms == {}
    assert theory.linkatoms is False


def test_nonperiodic_charge_recipients_follow_bonds_instead_of_close_contacts():
    fragment = Fragment(elems=["C"] * 4, coords=[[0, 0, 0], [1.4, 0, 0], [3.4, 0, 0], [1.4, 1.2, 0]], conncalc=False)
    theory = _qmmm(fragment, _mm(fragment, [(0, 1), (1, 2)]))

    assert theory.MMboundarydict == {1: [2]}


@pytest.mark.parametrize("periodic", [False, True])
def test_virtual_site_parent_links_do_not_become_covalent_boundary_recipients(periodic):
    fragment = Fragment(
        elems=["C", "C", "C", "He"],
        coords=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [2.1, 0, 0]],
        conncalc=False,
    )
    mm = _mm(fragment, [(0, 1), (1, 2)], periodic=periodic)
    mm.system.setParticleMass(3, 0)
    mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(1, 2, 0.5, 0.5))
    theory = _qmmm(fragment, mm)

    assert theory.boundaryatoms == {0: [1]}
    assert theory.MMboundarydict == {1: [2]}
    assert theory._topology_neighbors[3] == []
    if periodic:
        assert any(3 in component and 1 in component for component in theory._periodic_geometry.components)


@pytest.mark.parametrize("method", ["shift", "rcd"])
def test_adjacent_mm1_atoms_without_other_recipients_are_rejected(method):
    fragment = Fragment(elems=["C"] * 4, coords=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0]], conncalc=False)
    mm = _mm(fragment, [(0, 1), (1, 2), (2, 3)])
    with pytest.raises(InputError, match="complete MM1 boundary"):
        _qmmm(fragment, mm, qmatoms=[0, 3], chargeboundary_method=method)


def test_mechanical_boundary_does_not_require_charge_recipients():
    fragment = Fragment(elems=["C"] * 4, coords=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0]], conncalc=False)
    theory = _qmmm(fragment, _mm(fragment, [(0, 1), (1, 2), (2, 3)]), qmatoms=[0, 3], embedding="mech")

    assert theory.boundaryatoms == {0: [1], 3: [2]}
    assert theory.MMboundarydict == {1: [], 2: []}


@pytest.mark.parametrize("method", ["shift", "rcd"])
def test_adjacent_mm1_atoms_redistribute_outward_preserving_charge_and_dipole(method):
    fragment = Fragment(
        elems=["C"] * 6,
        coords=[[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0], [1.4, 1.4, 0], [2.8, -1.4, 0]],
        conncalc=False,
    )
    charges = np.array([0, 0.3, 0.4, 0, -0.2, -0.5])
    mm = _mm(fragment, [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)])
    theory = _qmmm(fragment, mm, qmatoms=[0, 3], charges=charges, chargeboundary_method=method)
    theory.runprep(fragment.coords)
    if method == "rcd":
        positions = theory.rcd_shifting_update(fragment.coords[theory.mmatoms], fragment.coords)
    else:
        positions = np.vstack((fragment.coords[theory.mmatoms], theory.dipole_coords))

    assert theory.MMboundarydict == {1: [4], 2: [5]}
    assert theory.pointcharges[:2] == pytest.approx([0, 0])
    assert np.sum(theory.pointcharges) == pytest.approx(np.sum(charges), abs=1e-14)
    assert np.asarray(theory.pointcharges) @ positions == pytest.approx(charges @ fragment.coords, abs=1e-14)


@pytest.mark.parametrize("elems", [["O", "C", "C"], ["C", "C", "O"]])
def test_each_inferred_boundary_pair_requires_nonpolar_cut_or_explicit_override(elems):
    coords = np.array([[0, 0, 0], [1.4, 0, 0], [-1.4, 0, 0]])
    with pytest.raises(InputError, match="unusualboundary=True"):
        get_boundary_atoms([0], coords, elems, 1.0, 0.3)
    assert get_boundary_atoms([0], coords, elems, 1.0, 0.3, unusualboundary=True) == {0: [1, 2]}
    assert get_boundary_atoms([0], coords, elems, 1.0, 0.3, excludeboundaryatomlist=[0]) == {}


@pytest.mark.parametrize("elems", [["O", "C", "C"], ["C", "C", "O"]])
def test_each_topology_boundary_pair_requires_nonpolar_cut_or_explicit_override(elems):
    fragment = Fragment(elems=elems, coords=[[0, 0, 0], [1.4, 0, 0], [-1.4, 0, 0]], conncalc=False)
    mm = _mm(fragment, [(0, 1), (0, 2)])
    with pytest.raises(InputError, match="unusualboundary=True"):
        _qmmm(fragment, mm, embedding="mech")
    theory = _qmmm(fragment, mm, embedding="mech", unusualboundary=True)
    assert theory.boundaryatoms == {0: [1, 2]}
