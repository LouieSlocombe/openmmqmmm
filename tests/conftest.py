import logging
import stat
from pathlib import Path

import numpy as np
import openmm
import pytest

from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory, ZeroTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.orca import find_orca

# What a real ORCA binary prints when invoked with no arguments; find_orca probes
# for this to tell the quantum chemistry program from unrelated `orca` binaries.
ORCA_PROBE_OUTPUT = "This program requires the name of a parameterfile"


requires_orca = pytest.mark.skipif(
    find_orca(required=False) is None,
    reason="No ORCA installation found (orcadir / OPENMMQMMM_ORCADIR / PATH)",
)


def dummy_mm(fragment, bonds=(), **overrides):
    """Build an unconstrained dummy MM theory; each call owns an independent System."""
    options = {
        "dummysystem": True,
        "platform": "Reference",
        "autoconstraints": None,
        "rigidwater": False,
        "hydrogenmass": None,
    }
    options.update(overrides)
    mm = OpenMMTheory(fragment=fragment, **options)
    atoms = list(mm.topology.atoms())
    for first, second in bonds:
        mm.topology.addBond(atoms[first], atoms[second])
    return mm


def use_pme(mm, edge_nm=3.0):
    mm.periodic = True
    mm.system.setDefaultPeriodicBoxVectors(*np.diag([edge_nm] * 3))
    mm.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    mm.nonbonded_force.setCutoffDistance(1.0)


def make_subregion_qmmm(qm_theory=None, **kwargs):
    """Two separated H atoms; callers must supply the QM-region electronic state."""
    fragment = Fragment(elems=["H", "H"], coords=[[1.0, 0, 0], [5.0, 0, 0]], charge=0, mult=1, conncalc=False)
    options = {"embedding": "elstat", "dipole_correction": False}
    options.update(kwargs)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory() if qm_theory is None else qm_theory,
        mm_theory=dummy_mm(fragment),
        qmatoms=[0],
        **options,
    )
    return theory, fragment


def meoh_water_mm(**mm_kwargs):
    test_dir = Path(__file__).parent
    fragment = Fragment(xyzfile=str(test_dir / "xyzfiles/h2o_MeOH.xyz"))
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    options = {
        "xmlfiles": [str(test_dir / "extra_files/MeOH_H2O-sigma.xml")],
        "pdbfile": "h2o_MeOH.pdb",
        "autoconstraints": None,
        "rigidwater": False,
    }
    options.update(mm_kwargs)
    return fragment, OpenMMTheory(**options)


def central_difference(energy_fn, coords, atom, axis, step=1e-5):
    """Independent test reference, converting an Angstrom displacement to Eh/Bohr."""
    plus, minus = np.array(coords, dtype=float, copy=True), np.array(coords, dtype=float, copy=True)
    plus[atom, axis] += step
    minus[atom, axis] -= step
    return (energy_fn(plus) - energy_fn(minus)) / (2 * step * ANG_TO_BOHR)


def central_difference_gradient(energy_fn, coords, step=1e-5):
    return np.asarray(
        [[central_difference(energy_fn, coords, atom, axis, step) for axis in range(3)] for atom in range(len(coords))]
    )


def evaluate(system, positions_nm, groups=-1, derivatives=False):
    """Independent Reference Context evaluation in OpenMM native energy/force units."""
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions_nm)
    state = context.getState(getEnergy=True, getForces=True, groups=groups, getParameterDerivatives=derivatives)
    energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoules_per_mole / openmm.unit.nanometer)
    result = (energy, forces, dict(state.getEnergyParameterDerivatives())) if derivatives else (energy, forces)
    del context, integrator
    return result


def bare_mm(system):
    """A minimal MM object for tests of force mutation, without creating a Context."""
    mm = OpenMMTheory.__new__(OpenMMTheory)
    mm.system = system
    mm.delete_qm1_mm1_bonded = False
    return mm


def nonbonded_mm(charges):
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    for charge in charges:
        system.addParticle(12)
        nonbonded.addParticle(charge, 0.3, 0.2)
    system.addForce(nonbonded)
    mm = bare_mm(system)
    mm.nonbonded_force = nonbonded
    mm.charges = list(charges)
    mm.numatoms = len(charges)
    return mm, nonbonded


@pytest.fixture
def cpu_mm():
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [5, 0, 0]], conncalc=False)
    return dummy_mm(fragment, platform="CPU", numcores=1, properties={"DeterministicForces": "true"})


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
    qm = _AnalyticQM()
    if embedding == "mech":
        if coords is None:
            coords = [[-0.5, 0, 0], [0.5, 0, 0]]
        fragment = Fragment(elems=["H", "H"], coords=coords, charge=0, mult=1)
        qmmm = QMMMTheory(
            fragment=fragment,
            qm_theory=qm,
            mm_theory=dummy_mm(fragment),
            qmatoms=[0, 1],
            embedding=embedding,
            qm_charge=0,
            qm_mult=1,
            dipole_correction=False,
            **kwargs,
        )
    else:
        qmmm, fragment = make_subregion_qmmm(qm, embedding=embedding, qm_charge=0, qm_mult=1, **kwargs)
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


@pytest.fixture
def isolated_package_logger():
    """Restore package logging after tests that exercise global logger state."""
    package_logger = logging.getLogger("openmmqmmm")
    configured_loggers = (package_logger, logging.getLogger("geometric"))
    original_states = {
        configured_logger: (
            list(configured_logger.handlers),
            configured_logger.level,
            configured_logger.propagate,
            configured_logger.disabled,
        )
        for configured_logger in configured_loggers
    }
    yield package_logger
    handlers_to_close = set()
    for configured_logger, (original_handlers, _level, _propagate, _disabled) in original_states.items():
        for handler in list(configured_logger.handlers):
            configured_logger.removeHandler(handler)
            if handler not in original_handlers:
                handlers_to_close.add(handler)
    for handler in handlers_to_close:
        handler.close()
    for configured_logger, (original_handlers, level, propagate, disabled) in original_states.items():
        for handler in original_handlers:
            configured_logger.addHandler(handler)
        configured_logger.setLevel(level)
        configured_logger.propagate = propagate
        configured_logger.disabled = disabled
