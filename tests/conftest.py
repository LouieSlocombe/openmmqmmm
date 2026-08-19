import stat

import numpy as np
import pytest

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory

# What a real ORCA binary prints when invoked with no arguments; find_orca probes
# for this to tell the quantum chemistry program from unrelated `orca` binaries.
ORCA_PROBE_OUTPUT = "This program requires the name of a parameterfile"


class _AnalyticQM:
    """Small deterministic QM stand-in used to exercise PythonForce without ORCA."""

    def __init__(self, force_constant=0.01):
        self.numcores = 1
        self.theorytype = "QM"
        self.theorynamelabel = "AnalyticQM"
        self.force_constant = force_constant
        self.calls = []

    def run(self, *, current_coords=None, current_mm_coords=None, grad=False, pc=False, **_kwargs):
        from openmmqmmm import constants

        qm_bohr = np.asarray(current_coords) * constants.ANG_TO_BOHR
        mm_bohr = (
            np.asarray(current_mm_coords) * constants.ANG_TO_BOHR if current_mm_coords is not None else np.zeros((0, 3))
        )
        energy = 0.5 * self.force_constant * (np.sum(qm_bohr * qm_bohr) + np.sum(mm_bohr * mm_bohr))
        qm_gradient = self.force_constant * qm_bohr
        mm_gradient = self.force_constant * mm_bohr
        self.calls.append(np.asarray(current_coords).copy())
        if not grad:
            return energy
        if pc:
            return energy, qm_gradient, mm_gradient
        return energy, qm_gradient


def _make_analytic_qmmm(embedding="mech", coords=None, **kwargs):
    if embedding == "mech":
        if coords is None:
            coords = [[-0.5, 0, 0], [0.5, 0, 0]]
        fragment = Fragment(elems=["H", "H"], coords=coords, charge=0, mult=1)
        qmatoms = [0, 1]
    else:
        fragment = Fragment(elems=["H", "H"], coords=[[1.0, 0, 0], [5.0, 0, 0]], charge=0, mult=1, conncalc=False)
        qmatoms = [0]
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    qm = _AnalyticQM()
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=qmatoms,
        embedding=embedding,
        qm_charge=0,
        qm_mult=1,
        dipole_correction=False,
        **kwargs,
    )
    return qmmm, fragment, qm


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def make_fake_orca_install():
    def _make(directory, with_helpers=True, output=ORCA_PROBE_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
        orca = directory / "orca"
        orca.write_text(f"#!/bin/sh\necho '{output}'\nexit 2\n")
        orca.chmod(orca.stat().st_mode | stat.S_IXUSR)
        if with_helpers:
            for helper in ("orca_scf", "orca_gtoint"):
                (directory / helper).write_text("")
        return directory

    return _make


@pytest.fixture
def fake_orca_dir(tmp_path, monkeypatch, make_fake_orca_install):
    orca_dir = make_fake_orca_install(tmp_path / "fake_orca")
    monkeypatch.setenv("OPENMMQMMM_ORCADIR", str(orca_dir))
    return orca_dir
