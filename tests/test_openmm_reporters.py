"""Reporter factories preserve format, restart and checkpoint contracts."""

from functools import partial
from pathlib import Path
from sys import stdout
from types import SimpleNamespace

import numpy as np
import openmm
import pytest

from openmmqmmm.openmm import md


def _reporter(kind, filename, interval, **kwargs):
    return SimpleNamespace(kind=kind, filename=filename, interval=interval, options=kwargs)


@pytest.mark.parametrize("rpmd", [False, True])
@pytest.mark.parametrize("restart,existing", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "format_name,extension", [("PDB", "pdb"), ("DCD", "dcd"), ("NetCDFReporter", "nc"), ("HDF5Reporter", "lh5")]
)
def test_trajectory_reporters_preserve_format_and_restart(monkeypatch, rpmd, restart, existing, format_name, extension):
    for kind in ["PDB", "DCD", "Checkpoint", "StateData"]:
        monkeypatch.setattr(openmm.app, kind + "Reporter", partial(_reporter, kind))
    for kind in ["NetCDF", "HDF5"]:
        monkeypatch.setattr(md.mdtraj.reporters, kind + "Reporter", partial(_reporter, kind + "Reporter"))
    monkeypatch.setattr(md, "ForceReporter", partial(_reporter, "Force"))
    if existing:
        Path("trajectory.dcd").touch()
    engine = md.MolecularDynamicsEngine.__new__(md.MolecularDynamicsEngine)
    engine.__dict__.update(
        trajectory_file_option=format_name,
        trajfilename="trajectory",
        traj_frequency=7,
        restartfile_frequency=11,
        enforce_periodic_box=False,
        force_file_option=True,
        atomic_units_force_reporter=True,
        energy_file_option=None,
        theory_runtype="MM",
        dataoutputoption=stdout,
        datafilename=None,
        volume=False,
        density=False,
        rpmd_report_copy=0,
        openmmobject=SimpleNamespace(dof=3),
        _rpmd_reporters=[],
        _rpmd_reporter_owner=None,
        _simulation_reporters=[],
        _simulation_reporter_owner=None,
    )
    integrator = openmm.RPMDIntegrator(4, 300, 1, 0.001) if rpmd else openmm.VerletIntegrator(0.001)
    simulation = SimpleNamespace(integrator=integrator, reporters=[])
    engine.set_sim_reporters(simulation, restart=restart)
    reporters = engine._rpmd_reporters if rpmd else simulation.reporters
    trajectory = next(reporter for reporter in reporters if getattr(reporter, "kind", None) == format_name)
    assert trajectory.filename == f"trajectory.{extension}"
    assert trajectory.interval == 7
    if format_name == "DCD":
        assert trajectory.options["append"] == (restart and existing)
    force = next(reporter for reporter in reporters if getattr(reporter, "kind", None) == "Force")
    assert force.filename == "trajectory_force.txt"
    assert force.options == {"atomic_units": True}


@pytest.mark.parametrize("kind", ["MM", "QMMM"])
def test_classical_restart_pair_uses_restart_frequency(kind):
    from conftest import _make_analytic_qmmm, dummy_mm

    from openmmqmmm import Fragment

    if kind == "MM":
        fragment = Fragment(elems=["He", "He"], coords=[[0, 0, 0], [3, 0, 0]], conncalc=False)
        theory = dummy_mm(fragment)
    else:
        theory, fragment, _ = _make_analytic_qmmm()
    engine = md.MolecularDynamicsEngine(
        fragment=fragment, theory=theory, platform="Reference", timestep=1e-8, traj_frequency=7, restartfile_frequency=2
    )
    engine.run(simulation_steps=2)
    engine.close()
    assert Path("OpenMM_MD_checkpoint.chk").is_file()
    assert Path("OpenMM_MD_state.xml").is_file()
    assert not Path("OpenMM_MD.chk").exists()


@pytest.mark.parametrize("kind", ["QMMM", "QM"])
def test_classical_output_events_batch_steps_share_states_and_continue_on_restart(monkeypatch, kind):
    from conftest import _make_analytic_qmmm

    qmmm, fragment, qm = _make_analytic_qmmm()
    engine = md.MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm if kind == "QMMM" else qm,
        platform="Reference",
        integrator="VerletIntegrator",
        timestep=1e-7,
        traj_frequency=6,
        restartfile_frequency=5,
        trajectory_file_option="XYZ",
        energy_file_option="energies.txt",
        specialatoms=[0],
        specialtraj_frequency=4,
    )
    batches, states, extra_steps = [], [], []

    class ExtraReporter:
        def describeNextReport(self, simulation):  # noqa: N802
            return {"steps": 7 - simulation.currentStep % 7, "include": []}

        def report(self, simulation, state):
            extra_steps.append(simulation.currentStep)

    def instrument(current_engine):
        simulation = current_engine.simulation
        simulation.context.setVelocities(np.zeros((2, 3)))
        step = simulation.integrator.step
        get_state = simulation.context.getState

        def record_step(count):
            batches.append(count)
            return step(count)

        def record_state(*args, **kwargs):
            if simulation.currentStep:
                states.append((simulation.currentStep, kwargs))
            return get_state(*args, **kwargs)

        monkeypatch.setattr(simulation.integrator, "step", record_step)
        monkeypatch.setattr(simulation.context, "getState", record_state)

    engine.run(simulation_steps=13, extra_reporters=[ExtraReporter()], pre_dynamics_hook=instrument)

    assert sum(batches) == 13 and max(batches) > 1
    assert {step for step, _ in states} == {4, 5, 6, 7, 8, 10, 12}
    for step in (4, 8):
        requests = [options for report_step, options in states if report_step == step]
        assert len(requests) == 1
        assert requests[0].get("positions") and not requests[0].get("energy")
    for step in (6, 12):
        requests = [options for report_step, options in states if report_step == step]
        assert len(requests) == 1, "XYZ, energy and state reporters must share one State"
        assert requests[0].get("positions") and requests[0].get("energy")
    assert extra_steps == [7]
    assert Path("OpenMM_MD_checkpoint.chk").is_file()
    assert Path("OpenMM_MD_state.xml").is_file()

    engine.run(simulation_steps=6, restart=True)
    assert engine.simulation.currentStep == 19
    assert extra_steps == [7, 14]
    assert Path("OpenMMMD_traj.xyz").read_text().splitlines().count("2") == 3
    assert Path("wrapped_special_traj.xyz").read_text().splitlines().count("1") == 4
    assert len(Path("energies.txt").read_text().splitlines()) == 3
    final_positions = engine.simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    engine.finalize_simulation()
    np.testing.assert_allclose(fragment.coords, final_positions.value_in_unit(openmm.unit.angstrom))
    assert Path("OpenMM_MD_final_checkpoint.chk").is_file()


@pytest.mark.parametrize("option", ["special_wrapping", "special_wrapping_updatepos"])
def test_mm_rejects_unimplemented_wrapping(option):
    from conftest import dummy_mm

    from openmmqmmm import Fragment
    from openmmqmmm.exceptions import InputError

    fragment = Fragment(elems=["He"], coords=[[0, 0, 0]], conncalc=False)
    with pytest.raises(InputError, match="MM dynamics does not support special_wrapping"):
        md.MolecularDynamicsEngine(fragment=fragment, theory=dummy_mm(fragment), **{option: True})
