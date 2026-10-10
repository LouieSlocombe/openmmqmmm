"""Option branches of openmm_minimize on a cheap harmonically restrained helium pair."""

import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest
from conftest import dummy_mm, evaluate, use_pme

from openmmqmmm import Fragment, openmm_minimize
from openmmqmmm.exceptions import InputError

RESTRAINT = [0, 1, 1.0, 100.0]  # atoms, Angstrom, kcal/mol/Angstrom^2
TRAJECTORY = "OpenMMOpt_traj.xyz"


def _restrained_pair(**overrides):
    fragment = Fragment(elems=["He", "He"], coords=[[-2.0, 0.0, 0.0], [-0.5, 0.0, 0.0]], conncalc=False)
    return fragment, dummy_mm(fragment, restraints=[RESTRAINT], **overrides)


def _energy_kj(mm, coords_angstrom):
    return evaluate(mm.system, np.asarray(coords_angstrom, dtype=float) / 10)[0]


def _distance(coords):
    coords = np.asarray(coords, dtype=float)
    return float(np.linalg.norm(coords[1] - coords[0]))


def _xyz_frames(path=TRAJECTORY):
    lines = Path(path).read_text().splitlines()
    frames = []
    index = 0
    while index < len(lines):
        natoms = int(lines[index])
        rows = [row.split() for row in lines[index + 2 : index + 2 + natoms]]
        frames.append(([row[0] for row in rows], np.array([[float(x) for x in row[1:4]] for row in rows])))
        index += natoms + 2
    return frames


def _pdb_coords(path="frag-minimized.pdb"):
    return np.asarray(openmm.app.PDBFile(path).positions.value_in_unit(openmm.unit.angstrom))


def test_minimize_with_reporter_reaches_the_restraint_minimum_and_writes_files():
    fragment, mm = _restrained_pair()
    energy_before = _energy_kj(mm, fragment.coords)

    result = openmm_minimize(fragment=fragment, theory=mm, maxiter=200, tolerance=1e-4, traj_frequency=1)

    assert result is fragment
    assert _distance(fragment.coords) == pytest.approx(1.0, abs=1e-3)
    assert _energy_kj(mm, fragment.coords) < energy_before
    assert _pdb_coords() == pytest.approx(fragment.coords, abs=1e-3)
    frames = _xyz_frames()
    assert len(frames) >= 1
    assert all(elements == ["He", "He"] for elements, _ in frames)
    assert _energy_kj(mm, frames[-1][1]) <= _energy_kj(mm, frames[0][1])


def test_minimize_stops_at_maxiter_and_logs_it(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    fragment, mm = _restrained_pair()

    openmm_minimize(fragment=fragment, theory=mm, maxiter=1, traj_frequency=1)

    assert "Max iterations reached" in caplog.text
    assert len(_xyz_frames()) == 1


def test_minimize_removes_a_stale_trajectory_before_writing():
    Path(TRAJECTORY).write_text("STALE\n")
    fragment, mm = _restrained_pair()

    openmm_minimize(fragment=fragment, theory=mm, maxiter=50, traj_frequency=1)

    text = Path(TRAJECTORY).read_text()
    assert "STALE" not in text
    assert text.splitlines()[0] == "2"


def test_minimize_traj_frequency_writes_only_the_first_frame_of_a_short_run():
    fragment, mm = _restrained_pair()

    openmm_minimize(fragment=fragment, theory=mm, maxiter=200, traj_frequency=100)

    assert len(_xyz_frames()) == 1
    assert _distance(fragment.coords) == pytest.approx(1.0, abs=1e-3)


def test_minimize_without_reporter_writes_no_trajectory():
    fragment, mm = _restrained_pair()

    openmm_minimize(fragment=fragment, theory=mm, maxiter=200, use_reporter=False)

    assert not Path(TRAJECTORY).exists()
    assert _pdb_coords() == pytest.approx(fragment.coords, abs=1e-3)
    assert _distance(fragment.coords) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("enforce_periodic_box", [True, False])
def test_minimize_periodic_pdb_wrapping_follows_enforce_periodic_box(enforce_periodic_box):
    fragment, mm = _restrained_pair()
    use_pme(mm)  # 30 Angstrom cube; both atoms sit at negative x

    openmm_minimize(fragment=fragment, theory=mm, maxiter=200, enforce_periodic_box=enforce_periodic_box)

    assert fragment.coords[:, 0].max() < 0.0
    shift = 30.0 if enforce_periodic_box else 0.0
    assert _pdb_coords()[:, 0] == pytest.approx(fragment.coords[:, 0] + shift, abs=1e-3)
    assert _distance(fragment.coords) == pytest.approx(1.0, abs=1e-3)


def test_minimize_requires_a_fragment():
    _, mm = _restrained_pair()

    with pytest.raises(InputError, match="No fragment object"):
        openmm_minimize(fragment=None, theory=mm)


def test_minimize_keeps_frozen_atoms_fixed_and_warns_about_constraint_conflicts(caplog):
    caplog.set_level(logging.WARNING, logger="openmmqmmm")
    fragment, mm = _restrained_pair(frozen_atoms=[0], rigidwater=True)
    start = fragment.coords.copy()

    openmm_minimize(fragment=fragment, theory=mm, maxiter=200, use_reporter=False)

    assert "Autoconstraints have not been set" in caplog.text
    assert "Frozen_atoms options selected but there are general constraints" in caplog.text
    assert fragment.coords[0] == pytest.approx(start[0], abs=1e-9)
    assert _distance(fragment.coords) == pytest.approx(1.0, abs=1e-3)
