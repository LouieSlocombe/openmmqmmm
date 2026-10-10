"""Loader, periodicity and constructor-option branches of OpenMMTheory."""

import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest
from conftest import dummy_mm, meoh_water_mm

from openmmqmmm import Fragment, OpenMMTheory
from openmmqmmm.exceptions import FileFormatError, InputError

TEST_DIR = Path(__file__).parent
FIXTURES = TEST_DIR / "fixtures"
MEOH_XML = str(TEST_DIR / "extra_files" / "MeOH_H2O-sigma.xml")
MEOH_PDB = "h2o_MeOH.pdb"
CELL = [30, 30, 30, 90, 90, 90]
BASE = {"platform": "Reference", "autoconstraints": None, "rigidwater": False, "hydrogenmass": None}


def _water_coords():
    gro = openmm.app.GromacsGroFile(str(FIXTURES / "three-waters.gro"))
    return np.array(gro.getPositions().value_in_unit(openmm.unit.angstrom))


def _loader_options(loader, periodic, **extra):
    options = dict(BASE, periodic=periodic, **extra)
    if periodic:
        options.update(periodic_cell_dimensions=CELL, periodic_nonbonded_cutoff=10, switching_function_distance=8)
    if loader == "gromacs":
        options.update(
            gromacs_files=True,
            grofile=str(FIXTURES / "three-waters.gro"),
            gromacstopfile=str(FIXTURES / "three-waters.top"),
        )
    elif loader == "charmm":
        options.update(
            charmm_files=True,
            psffile=str(FIXTURES / "three-waters.psf"),
            charmmtopfile=str(FIXTURES / "three-waters.rtf"),
            charmmprmfile=str(FIXTURES / "three-waters.prm"),
        )
    else:
        options.update(amber_files=True, amberprmtopfile=str(FIXTURES / "three-waters.prmtop"))
    return options


def _write_meoh_pdb():
    fragment = Fragment(xyzfile=str(TEST_DIR / "xyzfiles" / "h2o_MeOH.xyz"))
    fragment.write_pdbfile_openmm(filename=MEOH_PDB, skip_connectivity=True)
    return fragment


def _write_pdb_with_box(edge_angstrom, filename="boxed.pdb"):
    pdb = openmm.app.PDBFile(MEOH_PDB)
    vectors = [openmm.Vec3(edge_angstrom, 0, 0), openmm.Vec3(0, edge_angstrom, 0), openmm.Vec3(0, 0, edge_angstrom)]
    pdb.topology.setPeriodicBoxVectors(vectors * openmm.unit.angstrom)
    with open(filename, "w") as handle:
        openmm.app.PDBFile.writeFile(pdb.topology, pdb.positions, handle)
    return filename


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("loader", ["gromacs", "charmm", "amber"])
def test_parmed_loaders_reproduce_the_openmm_loader_energy(loader, periodic):
    coords = _water_coords()
    reference = OpenMMTheory(**_loader_options(loader, periodic)).run(current_coords=coords)

    theory = OpenMMTheory(**_loader_options(loader, periodic, use_parmed=True))

    assert theory.numatoms == 9
    assert theory.mm_elements == ["O", "H", "H"] * 3
    assert theory.resids == [0] * 3 + [1] * 3 + [2] * 3
    assert theory.run(current_coords=coords) == pytest.approx(reference, abs=1e-9)
    if loader == "charmm":
        assert theory.segmentnames == ["WAT"] * 9
        assert theory.atomtypes == ["OT", "HT", "HT"] * 3
    if periodic:
        assert theory.get_pbc_vectors() == pytest.approx(np.diag([30.0] * 3))


def test_gromacs_include_dir_resolves_itp_includes():
    head, tail = (FIXTURES / "three-waters.top").read_text().split("[ moleculetype ]")
    molecule, system_section = tail.split("[ system ]")
    include_dir = Path("gmx_include")
    include_dir.mkdir()
    (include_dir / "water.itp").write_text("[ moleculetype ]" + molecule)
    Path("split.top").write_text(head + '#include "water.itp"\n\n[ system ]' + system_section)
    coords = _water_coords()
    reference = OpenMMTheory(**_loader_options("gromacs", False)).run(current_coords=coords)

    theory = OpenMMTheory(
        **_loader_options("gromacs", False, gromacstopfile="split.top", gromacstopdir=str(include_dir))
    )

    assert theory.numatoms == 9
    assert theory.run(current_coords=coords) == pytest.approx(reference, abs=1e-12)


@pytest.mark.parametrize("use_parmed", [False, True])
def test_gromacs_periodic_box_defaults_to_the_gro_file_cell(caplog, use_parmed):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    options = {**_loader_options("gromacs", False, use_parmed=use_parmed), "periodic": True}  # no cell given

    theory = OpenMMTheory(**options)

    assert "Found PBC information in topology object" in caplog.text
    assert theory.get_pbc_vectors() == pytest.approx(np.diag([30.0] * 3))  # the .gro box line is 3 nm
    assert theory.nonbonded_force.getNonbondedMethod() == openmm.NonbondedForce.PME


def test_write_pdbfile_falls_back_to_the_fragment_coordinates():
    fragment = Fragment(elems=["He", "Ne"], coords=[[1.25, -2.5, 0.75], [4.0, 0.0, -1.0]], conncalc=False)
    theory = dummy_mm(fragment)

    theory.write_pdbfile(outputname="from_fragment")

    pdb = openmm.app.PDBFile("from_fragment.pdb")
    assert [atom.element.symbol for atom in pdb.topology.atoms()] == ["He", "Ne"]
    assert np.asarray(pdb.positions.value_in_unit(openmm.unit.angstrom)) == pytest.approx(fragment.coords, abs=1e-3)


def test_topology_forcefield_route_reads_the_topology_from_the_pdb():
    fragment, reference = meoh_water_mm(platform="Reference")
    energy = reference.run(current_coords=fragment.coords)

    theory = OpenMMTheory(topoforce=True, forcefield=openmm.app.ForceField(MEOH_XML), pdbfile=MEOH_PDB, **BASE)

    assert theory.atomnames == reference.atomnames
    assert theory.mm_elements == fragment.elems
    assert theory.run(current_coords=fragment.coords) == pytest.approx(energy, abs=1e-12)


def test_periodic_box_can_come_from_the_pdb_cryst1_record(caplog):
    caplog.set_level(logging.WARNING, logger="openmmqmmm")
    _write_meoh_pdb()
    pdbfile = _write_pdb_with_box(20.0)

    theory = OpenMMTheory(xmlfiles=[MEOH_XML], pdbfile=pdbfile, periodic=True, **BASE)

    assert "using periodic-box information from the PDB topology" in caplog.text
    assert theory.get_pbc_vectors() == pytest.approx(np.diag([20.0] * 3))
    # The default 12 Angstrom cutoff exceeds OpenMM's half-box limit for a 20 Angstrom cell.
    assert theory.nonbonded_force.getCutoffDistance().value_in_unit(openmm.unit.angstrom) == pytest.approx(0.499 * 20.0)


def test_pdbx_input_matches_the_pdb_route():
    fragment, reference = meoh_water_mm(platform="Reference")
    pdb = openmm.app.PDBFile(MEOH_PDB)
    with open("h2o_MeOH.cif", "w") as handle:
        openmm.app.PDBxFile.writeFile(pdb.topology, pdb.positions, handle)

    theory = OpenMMTheory(xmlfiles=[MEOH_XML], pdbxfile="h2o_MeOH.cif", **BASE)

    assert theory.atomnames == reference.atomnames
    assert theory.run(current_coords=fragment.coords) == pytest.approx(
        reference.run(current_coords=fragment.coords), abs=1e-12
    )


def test_xml_forcefield_route_needs_a_structure_file():
    with pytest.raises(InputError, match="No pdbfile or pdbxfile"):
        OpenMMTheory(xmlfiles=[MEOH_XML], **BASE)


def test_periodic_request_without_any_box_information_is_rejected():
    _write_meoh_pdb()

    with pytest.raises(FileFormatError, match="Found no PBC information"):
        OpenMMTheory(xmlfiles=[MEOH_XML], pdbfile=MEOH_PDB, periodic=True, **BASE)


def test_set_periodics_on_an_existing_system_updates_box_and_cell_vectors():
    _, theory = meoh_water_mm(platform="Reference")

    theory.set_periodics_before_system_creation(np.diag([20.0] * 3), None, None, False, False, False)

    assert theory.get_pbc_vectors() == pytest.approx(np.diag([20.0] * 3))
    assert theory.periodic_cell_vectors == pytest.approx(np.diag([20.0] * 3))
    topology_box = np.array(theory.topology.getPeriodicBoxVectors().value_in_unit(openmm.unit.angstrom))
    assert topology_box == pytest.approx(np.diag([20.0] * 3))


def test_charmm_periodic_cell_dimensions_keyword_is_rejected():
    fragment = Fragment(elems=["He"], coords=[[0.0, 0.0, 0.0]], conncalc=False)

    with pytest.raises(InputError, match="charmm_periodic_cell_dimensions is deprecated"):
        dummy_mm(fragment, charmm_periodic_cell_dimensions=CELL)


def test_pbc_vectors_keyword_still_sets_the_box_with_a_warning(caplog):
    caplog.set_level(logging.WARNING, logger="openmmqmmm")

    _, theory = meoh_water_mm(platform="Reference", periodic=True, pbc_vectors=np.diag([30.0] * 3))

    assert "PBCvectors keyword is on its way out" in caplog.text
    assert theory.get_pbc_vectors() == pytest.approx(np.diag([30.0] * 3))


def test_constraints_keyword_is_an_alias_for_bondconstraints(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    _, theory = meoh_water_mm(platform="Reference", constraints=[[0, 1, 0.96]])

    assert "constraints keyword specified is deprecated" in caplog.text
    assert theory.system.getNumConstraints() == 1
    i, j, distance = theory.system.getConstraintParameters(0)
    assert (i, j) == (0, 1)
    assert distance.value_in_unit(openmm.unit.angstrom) == pytest.approx(0.96)
    assert theory.user_constraints == [[0, 1, 0.96]]


@pytest.mark.parametrize(
    ("option", "expected"),
    [
        ("None", None),
        (None, None),
        ("HBonds", openmm.app.HBonds),
        ("AllBonds", openmm.app.AllBonds),
        ("HAngles", openmm.app.HAngles),
    ],
)
def test_autoconstraints_option_maps_to_the_openmm_constant(option, expected):
    fragment = Fragment(elems=["He"], coords=[[0.0, 0.0, 0.0]], conncalc=False)

    assert dummy_mm(fragment, autoconstraints=option).autoconstraints is expected


def test_unknown_autoconstraints_option_is_rejected():
    fragment = Fragment(elems=["He"], coords=[[0.0, 0.0, 0.0]], conncalc=False)

    with pytest.raises(InputError, match="Unknown autoconstraints"):
        dummy_mm(fragment, autoconstraints="Bogus")


@pytest.mark.parametrize(
    ("method", "message"),
    [("CutoffPeriodic", "CutoffPeriodic not currently allowed"), ("Bogus", "Unknown non-periodic nonbonded method")],
)
def test_invalid_nonperiodic_nonbonded_methods_are_rejected(method, message):
    with pytest.raises(InputError, match=message):
        meoh_water_mm(platform="Reference", nonbonded_method_no_pbc=method)


def test_cutoff_nonperiodic_method_sets_the_requested_cutoff():
    _, theory = meoh_water_mm(
        platform="Reference", nonbonded_method_no_pbc="CutoffNonPeriodic", nonbonded_cutoff_no_pbc=5.0
    )

    assert theory.nonbonded_force.getNonbondedMethod() == openmm.NonbondedForce.CutoffNonPeriodic
    assert theory.nonbonded_force.getCutoffDistance().value_in_unit(openmm.unit.nanometer) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("PME", openmm.NonbondedForce.PME),
        ("Ewald", openmm.NonbondedForce.Ewald),
        ("LJPME", openmm.NonbondedForce.LJPME),
        ("CutoffPeriodic", openmm.NonbondedForce.CutoffPeriodic),
    ],
)
def test_periodic_nonbonded_method_names_map_to_openmm_methods(method, expected):
    _, theory = meoh_water_mm(
        platform="Reference", periodic=True, periodic_cell_dimensions=CELL, nonbonded_method_pbc=method
    )

    assert theory.nonbonded_force.getNonbondedMethod() == expected


def test_unknown_periodic_nonbonded_method_is_rejected():
    with pytest.raises(InputError, match="Unknown nonbonded method"):
        meoh_water_mm(platform="Reference", periodic=True, periodic_cell_dimensions=CELL, nonbonded_method_pbc="Bogus")


def test_pme_parameters_and_dispersion_correction_reach_the_nonbonded_force():
    _, theory = meoh_water_mm(
        platform="Reference",
        periodic=True,
        periodic_cell_dimensions=CELL,
        pme_parameters=(3.0, 24, 24, 24),
        dispersion_correction=False,
    )

    alpha, nx, ny, nz = theory.nonbonded_force.getPMEParameters()
    assert alpha.value_in_unit(openmm.unit.nanometer**-1) == pytest.approx(3.0)
    assert (nx, ny, nz) == (24, 24, 24)
    assert theory.nonbonded_force.getUseDispersionCorrection() is False


def test_debug_logging_lists_each_defined_constraint(caplog):
    caplog.set_level(logging.DEBUG, logger="openmmqmmm")

    meoh_water_mm(platform="Reference", bondconstraints=[[0, 1, 0.96], [0, 2, 0.96]])

    assert caplog.text.count("Defined constraints:") == 2


def test_energy_decomposition_option_assigns_distinct_force_groups():
    _, theory = meoh_water_mm(platform="Reference", do_energy_decomposition=True)

    groups = [force.getForceGroup() for force in theory.system.getForces()]
    assert sorted(groups) == list(range(theory.system.getNumForces()))
    assert sorted(theory.forcegroups.values()) == groups


def test_write_pdbfile_needs_positions_or_a_fragment():
    _, theory = meoh_water_mm(platform="Reference")

    with pytest.raises(InputError, match="Can not write PDB-file"):
        theory.write_pdbfile()


def test_unknown_integrator_name_is_rejected():
    theory = OpenMMTheory.__new__(OpenMMTheory)
    theory.system = openmm.System()
    theory.set_simulation_parameters(integrator="LeapfrogIntegrator")

    with pytest.raises(InputError, match="Valid integrator keywords are: VerletIntegrator, VariableVerletIntegrator"):
        theory.create_integrator()


def test_cleanup_is_a_logged_no_op(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    _, theory = meoh_water_mm(platform="Reference")
    forces_before = theory.system.getNumForces()

    theory.cleanup()

    assert "Cleanup for OpenMMTheory called" in caplog.text
    assert theory.system.getNumForces() == forces_before
