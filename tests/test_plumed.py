import importlib.util
import shutil
import subprocess
import sys
from types import SimpleNamespace

import openmm
import pytest
from conftest import meoh_water_mm

from openmmqmmm import openmm_md_plumed


@pytest.fixture
def plumed_forces(monkeypatch):
    forces = []

    class _FakePlumedForce(openmm.CustomExternalForce):
        """Contributes no energy; records what the engine configures on it."""

        def __init__(self, script):
            super().__init__("0")
            self.script = script
            self.temperature = None
            forces.append(self)

        def setTemperature(self, temperature):  # noqa: N802
            self.temperature = temperature

    monkeypatch.setitem(sys.modules, "openmmplumed", SimpleNamespace(PlumedForce=_FakePlumedForce))
    return forces


def test_openmm_md_plumed_passes_the_run_temperature(plumed_forces):
    """PlumedForce defaults to -1 K, which PLUMED reads as "no kT" and every kT-derived value breaks."""
    fragment, theory = meoh_water_mm()

    openmm_md_plumed(
        fragment=fragment,
        theory=theory,
        timestep=0.0005,
        simulation_steps=2,
        traj_frequency=2,
        temperature=350,
        plumed_input_string="d1: DISTANCE ATOMS=1,4\n",
    )

    assert len(plumed_forces) == 1, "openmm_md_plumed should add exactly one PlumedForce"
    assert plumed_forces[0].temperature == 350, "The run temperature has to reach PLUMED, or kT is undefined there"


def test_openmm_md_plumed_applies_each_restraint_once(plumed_forces):
    """A restraint added a second time doubles its force constant."""
    fragment, theory = meoh_water_mm()
    restrained_pair = {0, 3}

    openmm_md_plumed(
        fragment=fragment,
        theory=theory,
        timestep=0.0005,
        simulation_steps=1,
        traj_frequency=1,
        restraints=[[0, 3, 2.5, 10.0]],
        plumed_input_string="d1: DISTANCE ATOMS=1,4\n",
    )

    assert len(plumed_forces) == 1
    restraint_terms = [
        type(force).__name__
        for force in theory.system.getForces()
        if isinstance(force, openmm.HarmonicBondForce | openmm.CustomBondForce)
        for i in range(force.getNumBonds())
        if set(force.getBondParameters(i)[:2]) == restrained_pair
    ]
    assert len(restraint_terms) == 1, f"The restraint was applied {len(restraint_terms)} times: {restraint_terms}"


def _plumed_has_opes():
    """conda-forge's PLUMED build omits opes; only a source build (see build_tools/) has it."""
    plumed = shutil.which("plumed")
    if plumed is None:
        return False
    probe = subprocess.run(
        [plumed, "--no-mpi", "config", "-q", "module", "opes"],
        check=False,
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.mark.skipif(
    importlib.util.find_spec("openmmplumed") is None or not _plumed_has_opes(),
    reason="needs the openmm-plumed plugin and a PLUMED built with the opes module",
)
def test_openmm_md_plumed_opes_default_biasfactor(tmp_path):
    """OPES leaves BIASFACTOR at BARRIER/kT, so the value it records shows whether PLUMED got a kT.

    With no temperature the default is infinite, and OPES either aborts outright (adaptive SIGMA)
    or silently biases towards a uniform target. SIGMA is explicit here only so that kernels — and
    with them the biasfactor header — are written without waiting out the adaptive-SIGMA warmup.
    """
    fragment, theory = meoh_water_mm()

    openmm_md_plumed(
        fragment=fragment,
        theory=theory,
        timestep=0.0005,
        simulation_steps=10,
        traj_frequency=10,
        temperature=300,
        plumed_input_string=(
            "d1: DISTANCE ATOMS=1,4\n"
            "opes: OPES_METAD ARG=d1 PACE=5 BARRIER=10 SIGMA=0.05 FILE=KERNELS\n"
            "PRINT ARG=d1,opes.bias FILE=COLVAR STRIDE=5\n"
        ),
    )

    kernels = (tmp_path / "KERNELS").read_text()
    header = next((line for line in kernels.splitlines() if "biasfactor" in line), None)
    assert header is not None, f"OPES recorded no biasfactor: {kernels!r}"
    # BARRIER/kT at 300 K, the value that is infinite when PLUMED has no temperature.
    assert float(header.split()[-1]) == pytest.approx(4.009, abs=0.01)
