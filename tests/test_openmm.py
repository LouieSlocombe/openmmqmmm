from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openmmqmmm import (
    Fragment,
    MolecularDynamicsEngine,
    OpenMMTheory,
    QMMMTheory,
    openmm_md,
    openmm_modeller,
    single_point,
)
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.systemsetup import _normalise_modeller_solvent_name

TEST_DIR = Path(__file__).parent


def _make_minimal_rpmd_simulation(num_copies):
    import openmm
    import openmm.app

    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce("0.5*x*x")
    force.addParticle(0, [])
    system.addForce(force)

    topology = openmm.app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("X", chain)
    topology.addAtom("H", openmm.app.Element.getByAtomicNumber(1), residue)
    integrator = openmm.RPMDIntegrator(
        num_copies,
        300 * openmm.unit.kelvin,
        1 / openmm.unit.picosecond,
        0.001 * openmm.unit.picoseconds,
    )
    simulation = openmm.app.Simulation(
        topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    return simulation, system


@pytest.mark.parametrize(
    ("watermodel", "expected"),
    [
        (None, "tip3p"),
        ("TIP3P", "tip3p"),
        ("tip3pfb", "tip3p"),
        ("tip3p-fb", "tip3p"),
        ("SPCE", "spce"),
        ("tip4pew", "tip4pew"),
    ],
)
def test_modeller_solvent_name_is_defined_for_every_water_model(watermodel, expected):
    assert _normalise_modeller_solvent_name(watermodel) == expected


def test_openmm_basic():
    pdbfile = f"{TEST_DIR}/pdbfiles/1aki_solvated.pdb"
    fragment = Fragment(pdbfile=pdbfile)

    omm = OpenMMTheory(
        xmlfiles=["charmm36.xml", "charmm36/water.xml"],
        pdbfile=pdbfile,
        periodic=True,
        autoconstraints=None,
        rigidwater=False,
    )
    single_point(theory=omm, fragment=fragment, grad=True)


@pytest.mark.parametrize("forcefield_route", ["name", "xmlfile", "object"])
def test_openmm_modeller(forcefield_route):
    import openmm.app

    pdbfile = f"{TEST_DIR}/pdbfiles/1aki.pdb"
    if forcefield_route == "name":
        forcefield_kwargs = {"forcefield": "CHARMM36", "watermodel": "tip3p"}
    elif forcefield_route == "xmlfile":
        forcefield_kwargs = {"xmlfile": "charmm36.xml", "waterxmlfile": "charmm36/water.xml"}
    else:
        forcefield_kwargs = {"forcefield_object": openmm.app.ForceField("charmm36.xml", "charmm36/water.xml")}

    openmmobject, fragment = openmm_modeller(
        pdbfile=pdbfile,
        ph=7.0,
        solvent_padding=10.0,
        ionicstrength=0.1,
        platform="CPU",
        **forcefield_kwargs,
    )

    assert openmmobject is not None, "openmm_modeller should return an OpenMMTheory object"
    # The input 1aki.pdb has 1079 atoms; hydrogens, solvent and ions are added on top
    assert fragment.numatoms > 1079, "Modeller should add hydrogens, water and ions"
    assert len(fragment.elems) == fragment.numatoms
    assert "H" in fragment.elems, "Hydrogens should have been added"


def test_openmm_md_runs_and_writes_a_trajectory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )

    starting_coords = fragment.coords.copy()
    openmm_md(
        fragment=fragment,
        theory=theory,
        timestep=0.0005,
        simulation_steps=20,
        traj_frequency=10,
        temperature=300,
        integrator="LangevinMiddleIntegrator",
    )

    assert (tmp_path / "trajectory.dcd").exists(), "The DCD trajectory should have been written"
    assert (tmp_path / "trajectory_lastframe.pdb").exists(), "finalize_simulation writes the last frame"
    assert fragment.coords.shape == starting_coords.shape
    assert not np.allclose(fragment.coords, starting_coords), "MD must advance the coordinates"
    assert np.all(np.isfinite(fragment.coords))


def test_openmm_md_requires_a_run_length():
    """Neither simulation_steps nor simulation_time is an error, not a zero-length run."""
    with pytest.raises(InputError):
        openmm_md(fragment=None, theory=None)


def test_md_engine_can_override_the_rpmd_copy_count():
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )

    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        integrator="RPMDIntegrator",
        rpmd_num_copies=6,
    )

    assert engine.rpmd_num_copies == 6
    assert theory.rpmd_num_copies == 6


def test_md_engine_rejects_rpmd_with_barostat_before_adding_force():
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )

    with pytest.raises(InputError, match="RPMDIntegrator cannot be used with a barostat"):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=theory,
            integrator="RPMDIntegrator",
            barostat="MonteCarloBarostat",
        )

    force_names = {force.__class__.__name__ for force in theory.system.getForces()}
    assert "MonteCarloBarostat" not in force_names


def test_md_engine_rejects_qmmm_rpmd_before_mutating_qmmm_object():
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    qmmm = QMMMTheory.__new__(QMMMTheory)

    with pytest.raises(InputError, match="QM/MM and external-QM forces cannot be applied independently"):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=qmmm,
            charge=0,
            mult=1,
            integrator="RPMDIntegrator",
        )

    assert not hasattr(qmmm, "openmm_externalforce")
    assert not hasattr(qmmm, "exit_after_customexternalforce_update")


def test_rpmd_restart_round_trip_preserves_every_copy(tmp_path):
    import openmm

    num_copies = 3
    simulation, system = _make_minimal_rpmd_simulation(num_copies)
    expected_positions = []
    expected_velocities = []
    for copy_index in range(num_copies):
        positions = np.array([[0.1 + copy_index, 0.2, 0.3]])
        velocities = np.array([[0.01, 0.02 + copy_index, 0.03]])
        simulation.integrator.setPositions(copy_index, [openmm.Vec3(*xyz) for xyz in positions] * openmm.unit.nanometer)
        simulation.integrator.setVelocities(
            copy_index,
            [openmm.Vec3(*xyz) for xyz in velocities] * (openmm.unit.nanometer / openmm.unit.picosecond),
        )
        expected_positions.append(positions)
        expected_velocities.append(velocities)
    simulation.currentStep = 17

    engine = MolecularDynamicsEngine.__new__(MolecularDynamicsEngine)
    engine.simulation = simulation
    engine.openmmobject = SimpleNamespace(system=system)
    engine.rpmd_report_copy = 0
    restart_file = tmp_path / "rpmd_restart.npz"
    engine._save_rpmd_restart(restart_file)

    restored_simulation, restored_system = _make_minimal_rpmd_simulation(num_copies)
    restored_engine = MolecularDynamicsEngine.__new__(MolecularDynamicsEngine)
    restored_engine.simulation = restored_simulation
    restored_engine.openmmobject = SimpleNamespace(system=restored_system)
    restored_engine.rpmd_report_copy = 0
    restored_engine._load_rpmd_restart(restart_file)

    assert restored_simulation.currentStep == 17
    for copy_index in range(num_copies):
        state = restored_simulation.integrator.getState(copy_index, getPositions=True, getVelocities=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer)
        velocities = state.getVelocities(asNumpy=True).value_in_unit(openmm.unit.nanometer / openmm.unit.picosecond)
        assert positions == pytest.approx(expected_positions[copy_index])
        assert velocities == pytest.approx(expected_velocities[copy_index])


def test_rpmd_state_report_labels_copy_and_ring_polymer_energy():
    import io

    import openmm

    from openmmqmmm.openmm.md import _RPMDStateDataReporter

    simulation, _system = _make_minimal_rpmd_simulation(2)
    for copy_index in range(2):
        simulation.integrator.setPositions(copy_index, [openmm.Vec3(0.1 + copy_index, 0, 0)] * openmm.unit.nanometer)
        simulation.integrator.setVelocities(
            copy_index,
            [openmm.Vec3(0, 0, 0)] * (openmm.unit.nanometer / openmm.unit.picosecond),
        )

    output = io.StringIO()
    reporter = _RPMDStateDataReporter(output, copy_index=1, degrees_of_freedom=3)
    state = simulation.integrator.getState(1, getEnergy=True, getVelocities=True)
    reporter.report(simulation, state)

    report = output.getvalue()
    assert "RPMD Copy" in report
    assert "Ring Polymer Total Energy" in report
    assert report.splitlines()[1].split(",")[2] == "1"


def test_rpmd_reporters_bypass_simulation_context_reporting(tmp_path):
    import io

    import openmm

    simulation, _system = _make_minimal_rpmd_simulation(2)
    for copy_index in range(2):
        simulation.integrator.setPositions(copy_index, [openmm.Vec3(0.1 + copy_index, 0, 0)] * openmm.unit.nanometer)
        simulation.integrator.setVelocities(
            copy_index,
            [openmm.Vec3(0, 0, 0)] * (openmm.unit.nanometer / openmm.unit.picosecond),
        )

    data_output = io.StringIO()
    engine = MolecularDynamicsEngine.__new__(MolecularDynamicsEngine)
    engine.openmmobject = SimpleNamespace(dof=3)
    engine.simulation = simulation
    engine.rpmd_report_copy = 0
    engine._rpmd_reporters = []
    engine.dataoutputoption = data_output
    engine.trajectory_file_option = "DCD"
    engine.trajfilename = str(tmp_path / "rpmd_trajectory")
    engine.traj_frequency = 1
    engine.enforce_periodic_box = False
    engine.force_file_option = None
    engine.atomic_units_force_reporter = False

    engine.set_sim_reporters(simulation)
    assert simulation.reporters == []
    engine._report_rpmd_state()

    assert "Ring Polymer Total Energy" in data_output.getvalue()
    assert (tmp_path / "rpmd_trajectory.dcd").exists()
    engine._rpmd_reporters.clear()


def test_rpmd_run_finalization_writes_only_bead_complete_restart():
    from openmmqmmm.openmm.md import RPMD_FINAL_RESTART_FILENAME

    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
        rpmd_num_copies=2,
    )
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        integrator="RPMDIntegrator",
        trajectory_file_option="DCD",
    )

    engine.run(simulation_steps=0)
    engine.finalize_simulation()

    assert Path(RPMD_FINAL_RESTART_FILENAME).exists()
    assert not Path("OpenMM_MD_final_state.xml").exists()
    assert not Path("OpenMM_MD_final_checkpoint.chk").exists()
