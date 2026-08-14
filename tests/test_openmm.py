import importlib.util
import shutil
import sys
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
from openmmqmmm.exceptions import InputError, MissingDependencyError
from openmmqmmm.openmm.systemsetup import _normalise_modeller_solvent_name

TEST_DIR = Path(__file__).parent


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


def _make_analytic_qmmm(embedding="mech", **kwargs):
    if embedding == "mech":
        fragment = Fragment(elems=["H", "H"], coords=[[-0.5, 0, 0], [0.5, 0, 0]], charge=0, mult=1)
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


def _write_methanol_pdb():
    fragment = Fragment(
        elems=["C", "O", "H", "H", "H", "H"],
        coords=[
            [-0.046, 0.662, 0.000],
            [-0.046, -0.755, 0.000],
            [-1.086, 0.976, 0.000],
            [0.438, 1.071, 0.890],
            [0.438, 1.071, -0.890],
            [0.860, -1.057, 0.000],
        ],
        charge=0,
        mult=1,
    )
    return fragment.write_pdbfile_openmm(filename="methanol.pdb", resname="LIG")


@pytest.mark.skipif(
    importlib.util.find_spec("forcefill") is None or shutil.which("antechamber") is None,
    reason="needs forcefill and AmberTools",
)
def test_openmm_modeller_parameterize_nonstandard():
    pdbfile = _write_methanol_pdb()

    openmmobject, fragment = openmm_modeller(
        pdbfile=pdbfile,
        forcefield="Amber14",
        watermodel="tip3p",
        parameterize_nonstandard=True,
        net_charges={"LIG": 0},
        solvent_boxdims=[30.0, 30.0, 30.0],
    )

    assert openmmobject is not None, "openmm_modeller should return an OpenMMTheory object"
    assert Path("nonstandard_ff.xml").is_file(), "forcefill should write the generated ligand XML to CWD"
    assert fragment.numatoms > 6, "Solvation should add water around the methanol"


def test_openmm_modeller_parameterize_rejects_forcefield_object():
    import openmm.app

    pdbfile = _write_methanol_pdb()
    with pytest.raises(InputError, match="parameterize_nonstandard"):
        openmm_modeller(
            pdbfile=pdbfile,
            forcefield_object=openmm.app.ForceField("amber14-all.xml", "amber14/tip3p.xml"),
            parameterize_nonstandard=True,
        )


def test_openmm_modeller_parameterize_missing_forcefill(monkeypatch):
    monkeypatch.setitem(sys.modules, "forcefill", None)
    pdbfile = _write_methanol_pdb()
    with pytest.raises(MissingDependencyError, match="forcefill"):
        openmm_modeller(
            pdbfile=pdbfile,
            forcefield="Amber14",
            watermodel="tip3p",
            parameterize_nonstandard=True,
        )


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


def test_qmmm_rpmd_uses_pythonforce_instead_of_shared_external_parameters():
    import openmm

    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
    )

    force_names = [force.__class__.__name__ for force in qmmm.mm_theory.system.getForces()]
    assert force_names.count("PythonForce") == 1
    assert "CustomExternalForce" not in force_names
    assert engine.rpmd_force_provider is not None

    simulation = qmmm.mm_theory.create_simulation()
    for copy, half_separation_nm in enumerate((0.05, 0.15)):
        positions = [
            openmm.Vec3(-half_separation_nm, 0, 0),
            openmm.Vec3(half_separation_nm, 0, 0),
        ]
        simulation.integrator.setPositions(copy, positions * openmm.unit.nanometer)
        simulation.integrator.setVelocities(
            copy,
            [openmm.Vec3(0, 0, 0), openmm.Vec3(0, 0, 0)] * openmm.unit.nanometer / openmm.unit.picosecond,
        )

    forces = []
    for copy in range(2):
        state = simulation.integrator.getState(copy, getForces=True, groups={engine.rpmd_external_force_group})
        forces.append(
            state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoules_per_mole / openmm.unit.nanometer)[0, 0]
        )
    assert forces[0] != pytest.approx(forces[1])


def test_qmmm_rpmd_pythonforce_system_round_trips_through_xml():
    import openmm

    qmmm, fragment, _qm = _make_analytic_qmmm()
    MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
    )

    serialized = openmm.XmlSerializer.serialize(qmmm.mm_theory.system)
    restored = openmm.XmlSerializer.deserialize(serialized)
    assert [force.__class__.__name__ for force in restored.getForces()].count("PythonForce") == 1


def test_qmmm_rpmd_step_evaluates_and_caches_each_final_bead():
    import openmm

    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        timestep=0.000001,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
    )
    simulation = qmmm.mm_theory.create_simulation()
    for copy, half_separation_nm in enumerate((0.05, 0.15)):
        simulation.integrator.setPositions(
            copy,
            [openmm.Vec3(-half_separation_nm, 0, 0), openmm.Vec3(half_separation_nm, 0, 0)] * openmm.unit.nanometer,
        )
        simulation.integrator.setVelocities(
            copy,
            [openmm.Vec3(0, 0, 0), openmm.Vec3(0, 0, 0)] * openmm.unit.nanometer / openmm.unit.picosecond,
        )

    provider = engine.rpmd_force_provider
    provider.clear_cache()
    evaluations_before = provider.evaluation_count
    simulation.integrator.step(1)
    assert provider.evaluation_count - evaluations_before == 4, "RPMD evaluates the force twice on every bead"

    evaluations_after_step = provider.evaluation_count
    cache_hits_before = provider.cache_hits
    simulation.integrator.getState(0, getEnergy=True, getForces=True)
    assert provider.evaluation_count == evaluations_after_step
    assert provider.cache_hits == cache_hits_before + 1, "Reporting should reuse the final force evaluation"


def test_qmmm_rpmd_engine_run_reports_and_restarts_all_beads():
    from openmmqmmm.openmm.md import RPMD_FINAL_RESTART_FILENAME, RPMD_RESTART_FILENAME

    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        timestep=0.000001,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
        traj_frequency=1,
        restartfile_frequency=1,
        trajectory_file_option="DCD",
    )
    engine.run(simulation_steps=1)
    assert engine.simulation.currentStep == 1
    assert Path(RPMD_RESTART_FILENAME).exists()
    with np.load(RPMD_RESTART_FILENAME) as restart:
        assert restart["positions_nm"].shape == (2, fragment.numatoms, 3)

    engine.finalize_simulation()
    assert Path(RPMD_FINAL_RESTART_FILENAME).exists()


def test_qmmm_rpmd_electrostatic_pythonforce_returns_physical_qm_energy():
    import openmm

    from openmmqmmm import constants

    qmmm, fragment, _qm = _make_analytic_qmmm("elstat")
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
    )
    simulation = qmmm.mm_theory.create_simulation()
    qmmm.mm_theory.set_positions(fragment.coords, simulation)

    state = simulation.integrator.getState(0, getEnergy=True, groups={engine.rpmd_external_force_group})
    energy_hartree = (
        state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole) / constants.HARTREE_TO_KJ_PER_MOL
    )
    assert energy_hartree == pytest.approx(qmmm.QMenergy)
    assert energy_hartree > 0, "The legacy CustomExternalForce correction would give the wrong sign"


def test_qmmm_rpmd_can_contract_only_the_qm_force_to_the_centroid():
    import openmm

    qmmm, fragment, qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        timestep=0.000001,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
        rpmd_qm_num_copies=1,
    )
    assert qmmm.mm_theory.rpmd_contractions == {engine.rpmd_external_force_group: 1}

    simulation = qmmm.mm_theory.create_simulation()
    for copy, center_nm in enumerate((0.1, 0.3)):
        simulation.integrator.setPositions(
            copy,
            [openmm.Vec3(center_nm - 0.05, 0, 0), openmm.Vec3(center_nm + 0.05, 0, 0)] * openmm.unit.nanometer,
        )
        simulation.integrator.setVelocities(
            copy,
            [openmm.Vec3(0, 0, 0), openmm.Vec3(0, 0, 0)] * openmm.unit.nanometer / openmm.unit.picosecond,
        )

    qm.calls.clear()
    simulation.integrator.step(1)
    assert len(qm.calls) == 2
    assert np.mean(qm.calls[0][:, 0]) == pytest.approx(2.0), "The contracted QM force sees the centroid geometry"


def test_external_qm_rpmd_uses_the_same_bead_specific_force_path():
    import openmm

    fragment = Fragment(elems=["H", "H"], coords=[[-0.5, 0, 0], [0.5, 0, 0]], charge=0, mult=1)
    qm = _AnalyticQM()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qm,
        charge=0,
        mult=1,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
        platform="Reference",
        constraints=None,
    )
    assert engine.theory_runtype == "QM"
    assert isinstance(engine.rpmd_python_force, openmm.PythonForce)

    simulation = engine.openmmobject.create_simulation()
    for copy, half_separation_nm in enumerate((0.05, 0.15)):
        simulation.integrator.setPositions(
            copy,
            [openmm.Vec3(-half_separation_nm, 0, 0), openmm.Vec3(half_separation_nm, 0, 0)] * openmm.unit.nanometer,
        )
        simulation.integrator.setVelocities(
            copy,
            [openmm.Vec3(0, 0, 0), openmm.Vec3(0, 0, 0)] * openmm.unit.nanometer / openmm.unit.picosecond,
        )
    simulation.integrator.step(1)
    assert engine.rpmd_force_provider.evaluation_count == 4


@pytest.mark.parametrize(
    ("qmmm_kwargs", "engine_kwargs", "message"),
    [
        ({"truncated_pc": True}, {}, "does not support truncated_pc"),
        ({"update_qm_region_charges": True}, {}, "does not support update_qm_region_charges"),
        ({}, {"special_wrapping": True}, "does not support special_wrapping"),
    ],
)
def test_qmmm_rpmd_rejects_state_shared_across_beads(qmmm_kwargs, engine_kwargs, message):
    qmmm, fragment, _qm = _make_analytic_qmmm(**qmmm_kwargs)
    with pytest.raises(InputError, match=message):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=qmmm,
            charge=0,
            mult=1,
            integrator="RPMDIntegrator",
            rpmd_num_copies=2,
            **engine_kwargs,
        )


@pytest.mark.parametrize("num_copies", [0, -1, 3, 1.5, True])
def test_qmmm_rpmd_qm_copy_count_is_validated(num_copies):
    qmmm, fragment, _qm = _make_analytic_qmmm()
    with pytest.raises(InputError, match="rpmd_qm_num_copies must be a positive integer"):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=qmmm,
            charge=0,
            mult=1,
            integrator="RPMDIntegrator",
            rpmd_num_copies=2,
            rpmd_qm_num_copies=num_copies,
        )


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
