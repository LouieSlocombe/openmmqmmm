import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import _AnalyticQM, _make_analytic_qmmm

from openmmqmmm import (
    Fragment,
    MolecularDynamicsEngine,
    OpenMMTheory,
    export_rpmd_potential,
    modeller_from_topology,
    openmm_md,
    openmm_md_plumed,
    openmm_modeller,
    single_point,
)
from openmmqmmm.exceptions import InputError, MissingDependencyError
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


def _write_plumed_test_system():
    """The h2o_MeOH system used by the MD tests, written to the current directory."""
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )
    return fragment, theory


def test_openmm_md_plumed_passes_the_run_temperature(tmp_path, monkeypatch):
    """PlumedForce defaults to -1 K, which PLUMED reads as "no kT" and every kT-derived value breaks."""
    import openmm

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
    monkeypatch.chdir(tmp_path)
    fragment, theory = _write_plumed_test_system()

    openmm_md_plumed(
        fragment=fragment,
        theory=theory,
        timestep=0.0005,
        simulation_steps=2,
        traj_frequency=2,
        temperature=350,
        plumed_input_string="d1: DISTANCE ATOMS=1,4\n",
    )

    assert len(forces) == 1, "openmm_md_plumed should add exactly one PlumedForce"
    assert forces[0].temperature == 350, "The run temperature has to reach PLUMED, or kT is undefined there"


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
def test_openmm_md_plumed_opes_default_biasfactor(tmp_path, monkeypatch):
    """OPES leaves BIASFACTOR at BARRIER/kT, so the value it records shows whether PLUMED got a kT.

    With no temperature the default is infinite, and OPES either aborts outright (adaptive SIGMA)
    or silently biases towards a uniform target. SIGMA is explicit here only so that kernels — and
    with them the biasfactor header — are written without waiting out the adaptive-SIGMA warmup.
    """
    monkeypatch.chdir(tmp_path)
    fragment, theory = _write_plumed_test_system()

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


@pytest.mark.parametrize("kind", ["qmmm-mech", "qmmm-elstat", "external-qm"])
def test_rpmd_each_bead_gets_the_analytic_force_of_its_own_geometry(kind):
    import openmm

    from openmmqmmm import constants

    bead_positions_nm = np.array(
        [
            [[-0.05, 0.01, 0.02], [0.05, -0.03, 0.04]],
            [[-0.15, 0.02, -0.01], [0.16, 0.03, -0.02]],
            [[0.07, -0.06, 0.05], [-0.08, 0.09, -0.04]],
        ]
    )
    engine_kwargs = {"charge": 0, "mult": 1, "integrator": "RPMDIntegrator", "rpmd_num_copies": 3}
    if kind == "external-qm":
        fragment = Fragment(elems=["H", "H"], coords=[[-0.5, 0, 0], [0.5, 0, 0]], charge=0, mult=1)
        qm = _AnalyticQM()
        engine = MolecularDynamicsEngine(
            fragment=fragment, theory=qm, platform="Reference", constraints=None, **engine_kwargs
        )
        simulation = engine.openmmobject.create_simulation()
        qm_rows = [0, 1]
    else:
        embedding = "mech" if kind == "qmmm-mech" else "elstat"
        qmmm, fragment, qm = _make_analytic_qmmm(embedding)
        engine = MolecularDynamicsEngine(fragment=fragment, theory=qmmm, **engine_kwargs)
        simulation = qmmm.mm_theory.create_simulation()
        qm_rows = [0, 1] if embedding == "mech" else [0]

    zero_velocities = [openmm.Vec3(0, 0, 0), openmm.Vec3(0, 0, 0)] * (openmm.unit.nanometer / openmm.unit.picosecond)
    for copy, positions in enumerate(bead_positions_nm):
        simulation.integrator.setPositions(copy, [openmm.Vec3(*row) for row in positions] * openmm.unit.nanometer)
        simulation.integrator.setVelocities(copy, zero_velocities)

    provider = engine.rpmd_force_provider
    group = engine.rpmd_external_force_group
    for copy, positions_nm in enumerate(bead_positions_nm):
        provider.clear_cache()
        qm.calls.clear()
        state = simulation.integrator.getState(copy, getEnergy=True, getForces=True, groups={group})

        coords_bohr = positions_nm * 10.0 * constants.ANG_TO_BOHR
        expected_energy = 0.5 * qm.force_constant * np.sum(coords_bohr**2) * constants.HARTREE_TO_KJ_PER_MOL
        expected_forces = -qm.force_constant * coords_bohr * constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM

        assert qm.calls, "Requesting a bead's state must evaluate the QM theory"
        for call in qm.calls:
            assert call == pytest.approx(positions_nm[qm_rows] * 10.0, abs=1e-10), "QM saw another bead's geometry"
        energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoules_per_mole / openmm.unit.nanometer)
        assert energy == pytest.approx(expected_energy, rel=1e-9)
        assert forces == pytest.approx(expected_forces, rel=1e-9, abs=1e-9)


def test_qmmm_rpmd_second_engine_on_the_same_theory_is_rejected():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine_kwargs = {
        "fragment": fragment,
        "theory": qmmm,
        "charge": 0,
        "mult": 1,
        "integrator": "RPMDIntegrator",
        "rpmd_num_copies": 2,
    }
    MolecularDynamicsEngine(**engine_kwargs)
    with pytest.raises(InputError, match="already carries"):
        MolecularDynamicsEngine(**engine_kwargs)
    force_names = [force.__class__.__name__ for force in qmmm.mm_theory.system.getForces()]
    assert force_names.count("PythonForce") == 1


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


def test_export_rpmd_potential_wires_pythonforce():
    import openmm

    qmmm, fragment, _qm = _make_analytic_qmmm()
    export = export_rpmd_potential(theory=qmmm, num_beads=4)

    assert export.system is qmmm.mm_theory.system
    force_names = [force.__class__.__name__ for force in export.system.getForces()]
    assert force_names.count("PythonForce") == 1
    assert export.python_force.getForceGroup() == export.force_group
    assert export.num_beads == 4
    assert export.provider.cache_size == 2 * 4 + 4
    assert qmmm.openmm_externalforce is True
    assert qmmm.exit_after_customexternalforce_update is True
    assert export.modeller.topology.getNumAtoms() == 2
    positions_nm = np.asarray(export.modeller.positions.value_in_unit(openmm.unit.nanometer))
    assert positions_nm == pytest.approx(np.asarray(fragment.coords) * 0.1)


def test_export_rpmd_potential_energy_matches_analytic():
    import openmm
    import openmm.app

    from openmmqmmm import constants

    qmmm, _fragment, qm = _make_analytic_qmmm()
    export = export_rpmd_potential(theory=qmmm, num_beads=2)

    integrator = openmm.RPMDIntegrator(
        2, 300 * openmm.unit.kelvin, 1 / openmm.unit.picosecond, 0.0005 * openmm.unit.picoseconds
    )
    simulation = openmm.app.Simulation(
        export.modeller.topology,
        export.system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    bead_positions_nm = np.array(
        [
            [[-0.05, 0.01, 0.02], [0.05, -0.03, 0.04]],
            [[-0.15, 0.02, -0.01], [0.16, 0.03, -0.02]],
        ]
    )
    for copy, positions in enumerate(bead_positions_nm):
        simulation.integrator.setPositions(copy, [openmm.Vec3(*row) for row in positions] * openmm.unit.nanometer)

    for copy, positions_nm in enumerate(bead_positions_nm):
        state = simulation.integrator.getState(copy, getEnergy=True, groups={export.force_group})
        coords_bohr = positions_nm * 10.0 * constants.ANG_TO_BOHR
        expected_energy = 0.5 * qm.force_constant * np.sum(coords_bohr**2) * constants.HARTREE_TO_KJ_PER_MOL
        energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
        assert energy == pytest.approx(expected_energy, rel=1e-9)


def test_export_rpmd_potential_rejects_second_export_and_engine():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    export_rpmd_potential(theory=qmmm, num_beads=2)

    with pytest.raises(InputError, match="already carries"):
        export_rpmd_potential(theory=qmmm, num_beads=2)
    with pytest.raises(InputError, match="already carries"):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=qmmm,
            charge=0,
            mult=1,
            integrator="RPMDIntegrator",
            rpmd_num_copies=2,
        )
    force_names = [force.__class__.__name__ for force in qmmm.mm_theory.system.getForces()]
    assert force_names.count("PythonForce") == 1


_WATER_DIMER_PDB = """CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H
ATOM      3  H2  HOH A   1      -0.240   0.927   0.000  1.00  0.00           H
ATOM      4  O   HOH A   2       3.000   0.000   0.000  1.00  0.00           O
ATOM      5  H1  HOH A   2       3.957   0.000   0.000  1.00  0.00           H
ATOM      6  H2  HOH A   2       2.760   0.927   0.000  1.00  0.00           H
END
"""


def _make_water_dimer_qmmm(**mm_kwargs):
    from openmmqmmm import QMMMTheory

    Path("dimer.pdb").write_text(_WATER_DIMER_PDB)
    fragment = Fragment(pdbfile="dimer.pdb")
    mm = OpenMMTheory(
        xmlfiles=["amber14-all.xml", "amber14/tip3p.xml"],
        pdbfile="dimer.pdb",
        platform="Reference",
        **mm_kwargs,
    )
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=_AnalyticQM(),
        mm_theory=mm,
        qmatoms=[0, 1, 2],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
        dipole_correction=False,
    )
    return qmmm, mm


def test_export_rpmd_potential_restores_physical_hydrogen_masses():
    import openmm.unit

    qmmm, mm = _make_water_dimer_qmmm(autoconstraints=None, rigidwater=False, hydrogenmass=1.5)
    repartitioned = [mm.system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(6)]
    assert repartitioned == pytest.approx([15.015, 1.5, 1.5] * 2, abs=0.01), "the build must start repartitioned"

    export_rpmd_potential(theory=qmmm, num_beads=4)

    restored = [mm.system.getParticleMass(i).value_in_unit(openmm.unit.dalton) for i in range(6)]
    assert restored == pytest.approx([15.999, 1.008, 1.008] * 2, abs=0.01), (
        "export must hand external NQE drivers physical nuclear masses"
    )


def test_export_rpmd_potential_rejects_constrained_system_for_beads():
    qmmm, mm = _make_water_dimer_qmmm(autoconstraints=None, rigidwater=True, hydrogenmass=None)
    assert mm.system.getNumConstraints() == 3, "the MM water's rigid-water constraints must survive"

    with pytest.raises(InputError, match="does not support constraints"):
        export_rpmd_potential(theory=qmmm, num_beads=4)

    force_names = [force.__class__.__name__ for force in mm.system.getForces()]
    assert force_names.count("PythonForce") == 0, "the rejection must fire before the force is attached"

    export = export_rpmd_potential(theory=qmmm, num_beads=1)
    assert export.num_beads == 1


def test_export_rpmd_potential_rejects_non_qmmm_theory():
    with pytest.raises(InputError, match="requires a QMMMTheory"):
        export_rpmd_potential(theory=_AnalyticQM(), num_beads=2)


def test_export_rpmd_potential_validates_num_beads():
    qmmm, _fragment, _qm = _make_analytic_qmmm()
    with pytest.raises(InputError, match="num_beads must be a positive integer"):
        export_rpmd_potential(theory=qmmm, num_beads=0)
    force_names = [force.__class__.__name__ for force in qmmm.mm_theory.system.getForces()]
    assert "PythonForce" not in force_names


def test_modeller_from_topology_converts_angstrom_to_nm():
    import openmm

    qmmm, fragment, _qm = _make_analytic_qmmm()
    modeller = modeller_from_topology(topology=qmmm.mm_theory.topology, coords_angstrom=fragment.coords)

    positions_nm = np.asarray(modeller.positions.value_in_unit(openmm.unit.nanometer))
    assert positions_nm == pytest.approx(np.asarray(fragment.coords) * 0.1)
    with pytest.raises(InputError, match="shape"):
        modeller_from_topology(topology=qmmm.mm_theory.topology, coords_angstrom=[[0.0, 0.0, 0.0]])


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


class _RecordingReporter:
    """Minimal OpenMM-protocol reporter that records the step of every report call."""

    def __init__(self):
        self.reports = []

    def describeNextReport(self, simulation):  # noqa: N802 - OpenMM reporter API, do not rename
        return (1, False, False, False, False)

    def report(self, simulation, state):
        self.reports.append(simulation.currentStep)


def test_rpmd_run_invokes_pre_dynamics_hook_and_extra_reporters():
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
    )
    recorder = _RecordingReporter()
    hook_steps = []

    def seed_hook(md):
        assert md is engine
        assert md.simulation is not None
        hook_steps.append(md.simulation.currentStep)

    engine.run(simulation_steps=2, extra_reporters=[recorder], pre_dynamics_hook=seed_hook)

    assert hook_steps == [0], "The hook must run once, before any dynamics"
    assert recorder in engine._rpmd_reporters
    assert recorder.reports == [1, 2], "The RPMD event loop must drive extra reporters at traj_frequency"


def test_classical_run_attaches_extra_reporters_natively():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        timestep=0.000001,
        traj_frequency=1,
    )
    recorder = _RecordingReporter()

    engine.run(simulation_steps=1, extra_reporters=[recorder])

    assert recorder in engine.simulation.reporters
    assert recorder.reports, "Native OpenMM scheduling must call the extra reporter"


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
    ("engine_kwargs", "message"),
    [
        ({"special_wrapping": True}, "does not support special_wrapping"),
        ({"special_wrapping_updatepos": True}, "does not support special_wrapping"),
        ({"dummyatomrestraint": True}, "does not support dummyatomrestraint"),
    ],
)
def test_external_qm_rpmd_rejects_options_the_pythonforce_path_ignores(engine_kwargs, message):
    fragment = Fragment(elems=["H", "H"], coords=[[-0.5, 0, 0], [0.5, 0, 0]], charge=0, mult=1)
    qm = _AnalyticQM()
    with pytest.raises(InputError, match=message):
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=qm,
            charge=0,
            mult=1,
            integrator="RPMDIntegrator",
            rpmd_num_copies=2,
            platform="Reference",
            constraints=None,
            **engine_kwargs,
        )


@pytest.mark.parametrize(
    ("qmmm_kwargs", "engine_kwargs", "message"),
    [
        ({"truncated_pc": True}, {}, "does not support truncated_pc"),
        ({"update_qm_region_charges": True}, {}, "does not support update_qm_region_charges"),
        ({}, {"special_wrapping": True}, "does not support special_wrapping"),
        ({}, {"special_wrapping_updatepos": True}, "does not support special_wrapping"),
        ({}, {"dummyatomrestraint": True}, "does not support dummyatomrestraint"),
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
