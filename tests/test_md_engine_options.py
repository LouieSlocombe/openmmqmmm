"""MD engine option branches: thermostat/barostat wiring, restart files, special outputs, warm-up and NPT drivers."""

import io
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm
import openmm.app
import pytest
from conftest import _make_analytic_qmmm, dummy_mm, evaluate, meoh_water_mm, use_pme

from openmmqmmm import Fragment, MolecularDynamicsEngine, gentle_warmup_md, openmm_box_equilibration, openmm_md
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm import md
from openmmqmmm.openmm.md import (
    RPMD_FINAL_RESTART_FILENAME,
    RPMD_RESTART_FORMAT_VERSION,
    _LoggerWriter,
    _RPMDStateDataReporter,
    read_npt_statefile,
)

NM = openmm.unit.nanometer
MD_LOGGER = "openmmqmmm.openmm.md"


def _helium_pair(**mm_options):
    fragment = Fragment(elems=["He", "He"], coords=[[0, 0, 0], [3, 0, 0]], conncalc=False)
    return fragment, dummy_mm(fragment, **mm_options)


def _force_names(system):
    return [force.__class__.__name__ for force in system.getForces()]


def _positions_nm(context):
    return context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(NM)


def _minimal_rpmd_simulation(num_copies):
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce("0.5*x*x")
    force.addParticle(0, [])
    system.addForce(force)
    topology = openmm.app.Topology()
    residue = topology.addResidue("X", topology.addChain())
    topology.addAtom("H", openmm.app.Element.getByAtomicNumber(1), residue)
    integrator = openmm.RPMDIntegrator(num_copies, 300 * openmm.unit.kelvin, 1 / openmm.unit.picosecond, 0.001)
    simulation = openmm.app.Simulation(topology, system, integrator, openmm.Platform.getPlatformByName("Reference"))
    for copy_index in range(num_copies):
        simulation.integrator.setPositions(copy_index, np.array([[0.1 * (copy_index + 1), 0, 0]]) * NM)
    engine = MolecularDynamicsEngine.__new__(MolecularDynamicsEngine)
    engine.simulation = simulation
    engine.openmmobject = SimpleNamespace(system=system)
    engine.rpmd_report_copy = 0
    return engine


def test_read_npt_statefile_returns_step_volume_and_density_columns():
    Path("npt.csv").write_text(
        '#"Step","Time (ps)","Potential Energy (kJ/mole)","Box Volume (nm^3)","Density (g/mL)"\n'
        "100,0.1,-5.0,27.5,0.99\n200,0.2,-6.0,27.0,1.01\n"
    )

    data = read_npt_statefile("npt.csv")

    assert data["steps"].tolist() == ["100", "200"]
    np.testing.assert_allclose(data["volume"], [27.5, 27.0])
    np.testing.assert_allclose(data["density"], [0.99, 1.01])


def test_logger_writer_flush_emits_the_buffered_partial_line_once(caplog):
    writer = _LoggerWriter(logging.getLogger(MD_LOGGER), level=logging.DEBUG)
    assert writer.writable()
    with caplog.at_level(logging.DEBUG, logger=MD_LOGGER):
        writer.write("tail without newline")
        assert not caplog.records
        writer.flush()
        writer.flush()
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.DEBUG, "tail without newline")
    ]


def test_rpmd_state_report_without_degrees_of_freedom_reports_nan_temperature_and_no_header_on_append():
    engine = _minimal_rpmd_simulation(2)
    output = io.StringIO()
    reporter = _RPMDStateDataReporter(output, copy_index=0, degrees_of_freedom=None, append=True)

    reporter.report(engine.simulation, engine.simulation.integrator.getState(0, getEnergy=True, getVelocities=True))

    lines = output.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].split(",")[5] == "nan"


def _saved_restart_with(engine, **overrides):
    engine._save_rpmd_restart("restart.npz")
    with np.load("restart.npz") as saved:
        payload = dict(saved)
    payload.update(overrides)
    np.savez("restart.npz", **payload)
    return "restart.npz"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"format_version": 99}, "format version 99"),
        ({"num_copies": 3}, "contains 3 copies"),
        ({"positions_nm": np.zeros((2, 5, 3))}, "incompatible dimensions"),
    ],
)
def test_rpmd_restart_rejects_inconsistent_payloads(override, message):
    engine = _minimal_rpmd_simulation(2)
    filename = _saved_restart_with(engine, **override)
    with pytest.raises(InputError, match=message):
        engine._load_rpmd_restart(filename)


def test_rpmd_restart_rejects_a_truncated_or_missing_file():
    engine = _minimal_rpmd_simulation(2)
    np.savez("partial.npz", format_version=RPMD_RESTART_FORMAT_VERSION)
    with pytest.raises(InputError, match="Invalid RPMD restart file"):
        engine._load_rpmd_restart("partial.npz")
    with pytest.raises(InputError, match="Invalid RPMD restart file"):
        engine._load_rpmd_restart("missing.npz")


def test_anderson_thermostat_is_added_once_and_forces_the_verlet_integrator():
    fragment, mm = _helium_pair()
    for _ in range(2):
        MolecularDynamicsEngine(fragment=fragment, theory=mm, anderson_thermostat=True).close()
    assert _force_names(mm.system).count("AndersenThermostat") == 1
    assert mm.integrator_name == "VerletIntegrator"


def test_stale_barostat_and_thermostat_are_both_removed_for_a_plain_run():
    fragment, mm = _helium_pair()
    use_pme(mm)
    original = _force_names(mm.system)
    mm.system.addForce(openmm.AndersenThermostat(300, 1))
    mm.system.addForce(openmm.MonteCarloBarostat(1, 300))

    MolecularDynamicsEngine(fragment=fragment, theory=mm).close()

    assert _force_names(mm.system) == original


def test_existing_barostat_is_reused_and_switches_volume_reporting_on():
    fragment, mm = _helium_pair()
    use_pme(mm)
    MolecularDynamicsEngine(fragment=fragment, theory=mm, barostat="MonteCarloBarostat").close()
    engine = MolecularDynamicsEngine(fragment=fragment, theory=mm, barostat="MonteCarloBarostat", pressure=2)
    engine.close()

    assert _force_names(mm.system).count("MonteCarloBarostat") == 1
    assert engine.integrator == "LangevinMiddleIntegrator"
    assert engine.volume is True
    assert engine.density is True


def test_centerforce_with_explicit_atoms_only_acts_beyond_the_flat_bottom():
    fragment, mm = _helium_pair()
    MolecularDynamicsEngine(
        fragment=fragment,
        theory=mm,
        add_centerforce=True,
        centerforce_atoms=[1],
        centerforce_center=[0.0, 0.0, 0.0],
        centerforce_constant=2.0,
        centerforce_distance=1.0,
    ).close()

    force = next(force for force in mm.system.getForces() if force.getName() == "OpenMMQMMM restraint")
    assert force.getNumParticles() == 1
    assert force.getParticleParameters(0)[0] == 1
    energy, forces = evaluate(mm.system, fragment.coords / 10)
    assert energy == pytest.approx(0.5 * 2.0 * 4.184 * (3.0 - 1.0) ** 2)
    assert forces[0] == pytest.approx([0, 0, 0])
    assert forces[1][0] < 0


def test_centerforce_defaults_to_the_qm_atoms_and_the_fragment_centre():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    MolecularDynamicsEngine(fragment=fragment, theory=qmmm, add_centerforce=True, centerforce_distance=0.0).close()

    force = next(force for force in qmmm.mm_theory.system.getForces() if force.getName() == "OpenMMQMMM restraint")
    assert [force.getParticleParameters(i)[0] for i in range(force.getNumParticles())] == qmmm.qmatoms
    centre_nm = np.mean(fragment.coords, axis=0) / 10
    np.testing.assert_allclose(force.getParticleParameters(0)[1][2:], centre_nm, atol=1e-12)


def test_dummy_atom_restraint_requires_solute_indices():
    fragment, mm = _helium_pair()
    with pytest.raises(InputError, match="solute_indices"):
        MolecularDynamicsEngine(fragment=fragment, theory=mm, dummyatomrestraint=True)


def test_restraint_of_an_unsupported_length_is_rejected():
    fragment, mm = _helium_pair()
    with pytest.raises(InputError, match="has 3 entries"):
        MolecularDynamicsEngine(fragment=fragment, theory=mm, restraints=[[0, 1, 2.0]])


def test_qm_copy_count_is_only_accepted_for_rpmd_dynamics():
    fragment, mm = _helium_pair()
    with pytest.raises(InputError, match="only valid for QM/MM or external-QM RPMD"):
        MolecularDynamicsEngine(fragment=fragment, theory=mm, rpmd_qm_num_copies=2)


def test_classical_restart_pair_can_be_written_on_demand():
    fragment, mm = _helium_pair()
    engine = MolecularDynamicsEngine(fragment=fragment, theory=mm, timestep=1e-6)
    engine.run(simulation_steps=0)
    engine.write_state_and_chk_files(5)
    engine.close()

    assert Path("OpenMM_MD_state.xml").is_file()
    assert Path("OpenMM_MD_checkpoint.chk").is_file()


def test_qmmm_run_writes_xyz_energy_and_special_atom_trajectories_at_their_frequencies():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        timestep=1e-7,
        traj_frequency=2,
        specialatoms=[0],
        specialtraj_frequency=3,
        trajectory_file_option="XYZ",
        energy_file_option="energies.txt",
        restartfile_frequency=100,
    )
    engine.run(simulation_steps=6)
    engine.close()

    xyz_lines = Path("OpenMMMD_traj.xyz").read_text().splitlines()
    assert xyz_lines.count("2") == 3
    energies = [float(value) for value in Path("energies.txt").read_text().split()]
    assert len(energies) == 3
    expected_energy = 0.5 * 0.01 * np.sum((np.asarray(fragment.coords) * ANG_TO_BOHR) ** 2)
    assert energies == pytest.approx([expected_energy] * 3, rel=1e-3)
    special_lines = Path("wrapped_special_traj.xyz").read_text().splitlines()
    assert special_lines.count("1") == 2
    assert all(line.split()[0] == "H" for line in special_lines if line.startswith("H"))


@pytest.mark.parametrize("restart_option", ["chkfile", "statefile"])
def test_restart_file_given_at_construction_restores_the_saved_positions(restart_option):
    fragment, mm = _helium_pair()
    original_coords = np.array(fragment.coords, copy=True)
    engine = MolecularDynamicsEngine(fragment=fragment, theory=mm, timestep=1e-6)
    engine.run(simulation_steps=1)
    shifted_nm = _positions_nm(engine.simulation.context) + np.array([0.1, 0.0, 0.0])
    engine.simulation.context.setPositions(shifted_nm * NM)
    engine.finalize_simulation()
    fragment.coords = original_coords
    filename = {"chkfile": "OpenMM_MD_final_checkpoint.chk", "statefile": "OpenMM_MD_final_state.xml"}[restart_option]

    restarted = MolecularDynamicsEngine(fragment=fragment, theory=mm, timestep=1e-6, **{restart_option: filename})
    restarted.run(simulation_steps=0)
    restored_nm = _positions_nm(restarted.simulation.context)
    restarted.close()

    np.testing.assert_allclose(restored_nm, shifted_nm, atol=1e-6)
    assert not np.allclose(restored_nm, original_coords / 10)


def test_rpmd_restart_file_given_as_checkpoint_restores_every_bead():
    fragment, theory = meoh_water_mm(rpmd_num_copies=2)
    engine = MolecularDynamicsEngine(fragment=fragment, theory=theory, integrator="RPMDIntegrator", timestep=1e-6)
    engine.run(simulation_steps=0)
    bead_positions = []
    for copy_index in range(2):
        positions = np.array(fragment.coords) / 10 + [0.01 * (copy_index + 1), 0, 0]
        engine.simulation.integrator.setPositions(copy_index, positions * NM)
        bead_positions.append(positions)
    engine.finalize_simulation()

    restarted = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        integrator="RPMDIntegrator",
        timestep=1e-6,
        chkfile=RPMD_FINAL_RESTART_FILENAME,
    )
    restarted.run(simulation_steps=0)
    for copy_index in range(2):
        state = restarted.simulation.integrator.getState(copy_index, getPositions=True)
        np.testing.assert_allclose(
            state.getPositions(asNumpy=True).value_in_unit(NM), bead_positions[copy_index], atol=1e-6
        )
    restarted.close()


def test_run_rejects_an_unrecognised_theory_runtype():
    fragment, mm = _helium_pair()
    engine = MolecularDynamicsEngine(fragment=fragment, theory=mm)
    engine.theory_runtype = "Hybrid"
    with pytest.raises(InputError, match="Unrecognized theory runtype"):
        engine.run(simulation_steps=1)


def test_openmm_md_accepts_a_simulation_time_and_writes_the_final_frame():
    fragment, mm = _helium_pair()
    openmm_md(fragment=fragment, theory=mm, timestep=1e-6, simulation_time=3e-6, trajfilename="timed")
    assert Path("timed_lastframe.pdb").is_file()
    assert Path("timed.dcd").is_file()


def test_gentle_warmup_runs_every_default_stage_and_survives_mdtraj_failures(caplog):
    fragment, mm = _helium_pair()
    with caplog.at_level(logging.WARNING, logger=MD_LOGGER):
        gentle_warmup_md(theory=mm, fragment=fragment, trajfilename="warm", maxoptsteps=2)

    for cycle in range(3):
        assert Path(f"warm_cycle{cycle}.dcd").is_file()
        assert Path(f"warm_cycle{cycle}_lastframe.pdb").is_file()
    assert caplog.text.count("MDTraj trajectory analysis failed") == 3


def test_gentle_warmup_reports_clashing_atoms_and_continues_past_a_failed_minimization(monkeypatch, caplog):
    fragment, mm = _helium_pair()
    mm.add_custom_bond_force(0, 1, 0.5, 1000.0)

    def failing_minimize(**_kwargs):
        raise RuntimeError("minimizer exploded")

    monkeypatch.setattr(md, "openmm_minimize", failing_minimize)
    with caplog.at_level(logging.INFO, logger=MD_LOGGER):
        gentle_warmup_md(
            theory=mm,
            fragment=fragment,
            time_steps=[1e-7],
            steps=[2],
            temperatures=[1],
            traj_frequencies=[1],
            gradient_threshold=1.0,
            use_mdtraj=False,
            trajfilename="clash",
        )

    assert "Number of atoms with large forces: 2" in caplog.text
    assert "minimizer exploded" in caplog.text
    assert Path("clash_cycle0.dcd").is_file()


def test_gentle_warmup_can_skip_every_optional_stage():
    fragment, mm = _helium_pair()
    gentle_warmup_md(
        theory=mm,
        fragment=fragment,
        time_steps=[1e-7],
        steps=[1],
        temperatures=[1],
        traj_frequencies=[1],
        check_gradient_first=False,
        initial_opt=False,
        use_mdtraj=False,
        trajfilename="bare",
    )
    assert Path("bare_cycle0.dcd").is_file()
    assert not Path("frag-minimized.pdb").exists()


def test_gentle_warmup_requires_theory_and_fragment():
    fragment, _mm = _helium_pair()
    with pytest.raises(InputError, match="requires theory"):
        gentle_warmup_md(fragment=fragment)


def test_box_equilibration_stops_once_volume_and_density_settle_and_returns_the_box(caplog):
    fragment, mm = _helium_pair()
    use_pme(mm)
    with caplog.at_level(logging.INFO, logger=MD_LOGGER):
        vectors = openmm_box_equilibration(
            fragment=fragment,
            theory=mm,
            numsteps_per_npt=4,
            traj_frequency=2,
            max_npt_cycles=3,
            timestep=1e-6,
            barostat_frequency=1,
            volume_threshold=1e9,
            density_threshold=1e9,
            use_mdtraj=False,
        )

    assert "Equilibration of periodic box finished after 4" in caplog.text
    data = read_npt_statefile("nptsim.csv")
    assert data["steps"].tolist() == ["2", "4"]
    edge_nm = vectors[0][0].value_in_unit(NM)
    assert edge_nm**3 == pytest.approx(data["volume"][-1], rel=1e-6)
    assert mm.system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(NM) == pytest.approx(edge_nm)
    assert Path("equilibration_NPT.dcd").is_file()
    assert Path("equilibration_NPT_lastframe.pdb").is_file()


def test_box_equilibration_warns_at_the_cycle_limit_with_frozen_atoms_and_an_unimageable_format(caplog):
    # Three atoms: OpenMM's StateDataReporter divides by the mobile degrees of freedom, zero for one free atom.
    fragment = Fragment(elems=["He", "He", "He"], coords=[[0, 0, 0], [3, 0, 0], [0, 3, 0]], conncalc=False)
    mm = dummy_mm(fragment, frozen_atoms=[0])
    use_pme(mm)
    with caplog.at_level(logging.WARNING, logger=MD_LOGGER):
        openmm_box_equilibration(
            fragment=fragment,
            theory=mm,
            numsteps_per_npt=2,
            traj_frequency=1,
            max_npt_cycles=2,
            timestep=1e-6,
            barostat_frequency=1,
            volume_threshold=0.0,
            trajectory_file_option="XYZ",
        )

    assert "frozen atoms defined" in caplog.text
    assert "Max NPT cycles reached (2)" in caplog.text
    assert "MDTraj reimaging failed" in caplog.text
    assert len(read_npt_statefile("nptsim.csv")["steps"]) == 4


def test_box_equilibration_validates_its_inputs_before_building_an_engine():
    fragment, mm = _helium_pair()
    with pytest.raises(InputError, match="Fragment and theory required"):
        openmm_box_equilibration(fragment=fragment)
    with pytest.raises(InputError, match="numsteps_per_npt"):
        openmm_box_equilibration(fragment=fragment, theory=mm, numsteps_per_npt=5, traj_frequency=10)
