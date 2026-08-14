from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, MolecularDynamicsEngine, OpenMMTheory, openmm_md, openmm_modeller, single_point
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.systemsetup import _normalise_modeller_solvent_name

TEST_DIR = Path(__file__).parent


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
