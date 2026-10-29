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
        rpmd_energy_frequency=7,
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


@pytest.mark.parametrize("frequency,diagnostic_steps", [(None, [2, 4, 6]), (3, [3, 6]), (0, [])])
def test_contracted_rpmd_reports_energy_once_at_its_own_cadence(monkeypatch, frequency, diagnostic_steps):
    from conftest import _make_analytic_qmmm

    qmmm, fragment, _ = _make_analytic_qmmm()
    engine = md.MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        integrator="RPMDIntegrator",
        rpmd_num_copies=8,
        rpmd_qm_num_copies=1,
        rpmd_energy_frequency=frequency,
        timestep=1e-6,
        traj_frequency=2,
        restartfile_frequency=0,
        datafilename="state.csv",
    )
    diagnostics, state_requests = [], []

    def instrument(current_engine):
        simulation = current_engine.simulation
        integrator = simulation.integrator
        for index in range(8):
            coords = np.array([[0.05 * index - 0.05, 0, 0], [0.05 * index + 0.05, 0, 0]])
            integrator.setPositions(index, coords * openmm.unit.nanometer)
            integrator.setVelocities(index, np.zeros((2, 3)))
        total_energy = integrator.getTotalEnergy
        get_state = integrator.getState

        def record_total_energy():
            before = current_engine.rpmd_force_provider.evaluation_count
            value = total_energy()
            diagnostics.append((simulation.currentStep, current_engine.rpmd_force_provider.evaluation_count - before))
            return value

        def record_state(*args, **kwargs):
            before = current_engine.rpmd_force_provider.evaluation_count
            value = get_state(*args, **kwargs)
            state_requests.append(
                (simulation.currentStep, kwargs, current_engine.rpmd_force_provider.evaluation_count - before)
            )
            return value

        monkeypatch.setattr(integrator, "getTotalEnergy", record_total_energy)
        monkeypatch.setattr(integrator, "getState", record_state)

    engine.run(simulation_steps=6, pre_dynamics_hook=instrument)
    engine.finalize_simulation()

    assert diagnostics == [(step, 8) for step in diagnostic_steps], "Log and file must share one full-ring evaluation"
    assert all(extra_evaluations == 0 for _, _, extra_evaluations in state_requests)
    for step, options, _ in state_requests:
        if step not in diagnostic_steps:
            assert not options.get("getEnergy") and not options.get("getForces")
    rows = Path("state.csv").read_text().splitlines()
    assert len(rows) == (1 + len(diagnostic_steps) if diagnostic_steps else 0)
    assert [int(row.split(",")[0]) for row in rows[1:]] == diagnostic_steps
    trajectory = md.mdtraj.load("trajectory.dcd", top="trajectory_firstframe.pdb")
    assert trajectory.n_frames == 3


def test_rpmd_disabled_diagnostics_preserve_explicit_force_reports(monkeypatch):
    from conftest import _make_analytic_qmmm

    qmmm, fragment, _ = _make_analytic_qmmm()
    engine = md.MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
        rpmd_qm_num_copies=1,
        rpmd_energy_frequency=0,
        timestep=1e-6,
        traj_frequency=1,
        force_file_option="enabled",
    )

    def reject_full_ring(current_engine):
        def unexpected():
            pytest.fail("Explicit bead force output must not request full-ring diagnostics")

        monkeypatch.setattr(current_engine.simulation.integrator, "getTotalEnergy", unexpected)

    engine.run(simulation_steps=1, pre_dynamics_hook=reject_full_ring)
    state = engine.simulation.integrator.getState(0, getEnergy=True, getForces=True)
    expected = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
    engine.close()
    lines = Path("trajectory_force.txt").read_text().splitlines()
    assert np.array([[float(value) for value in line.split()] for line in lines[-2:]]) == pytest.approx(
        expected, rel=1e-5
    )


@pytest.mark.parametrize("frequency", [-1, True, 1.5])
def test_rpmd_energy_frequency_rejects_invalid_intervals_before_mutating_theory(frequency):
    from conftest import _make_analytic_qmmm

    from openmmqmmm.exceptions import InputError

    qmmm, fragment, _ = _make_analytic_qmmm()
    with pytest.raises(InputError, match="rpmd_energy_frequency"):
        md.MolecularDynamicsEngine(
            fragment=fragment, theory=qmmm, integrator="RPMDIntegrator", rpmd_energy_frequency=frequency
        )
    assert not qmmm.openmm_externalforce


def test_rpmd_diagnostic_cadence_works_without_trajectory_output():
    from conftest import _make_analytic_qmmm

    qmmm, fragment, _ = _make_analytic_qmmm()
    engine = md.MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
        rpmd_energy_frequency=2,
        traj_frequency=0,
        timestep=1e-6,
        datafilename="state.csv",
    )
    engine.run(simulation_steps=3)
    engine.close()
    assert [int(row.split(",")[0]) for row in Path("state.csv").read_text().splitlines()[1:]] == [2]
    assert not Path("trajectory.dcd").exists()


@pytest.mark.parametrize("option", ["special_wrapping", "special_wrapping_updatepos"])
def test_mm_rejects_unimplemented_wrapping(option):
    from conftest import dummy_mm

    from openmmqmmm import Fragment
    from openmmqmmm.exceptions import InputError

    fragment = Fragment(elems=["He"], coords=[[0, 0, 0]], conncalc=False)
    with pytest.raises(InputError, match="MM dynamics does not support special_wrapping"):
        md.MolecularDynamicsEngine(fragment=fragment, theory=dummy_mm(fragment), **{option: True})
