import importlib.util
import shutil
import sys
from pathlib import Path

import openmm
import pytest

from openmmqmmm import Fragment, openmm_modeller
from openmmqmmm.exceptions import InputError, MissingDependencyError
from openmmqmmm.openmm.systemsetup import _normalise_modeller_solvent_name

TEST_DIR = Path(__file__).parent


def _write_dipeptide_pdb():
    """Residues 1-2 (LYS-VAL) of 1aki, heavy atoms only; PDBFixer adds the terminal OXT."""
    lines = [
        line
        for line in (TEST_DIR / "pdbfiles" / "1aki.pdb").read_text().splitlines()
        if line.startswith("ATOM") and int(line[22:26]) <= 2
    ]
    Path("dipeptide.pdb").write_text("\n".join([*lines, "END"]) + "\n")
    return "dipeptide.pdb"


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


@pytest.mark.parametrize(
    "forcefield_route", ["name", "xmlfile", "object"], ids=["name-protein", "xmlfile-dipeptide", "object-dipeptide"]
)
def test_openmm_modeller(forcefield_route):
    import openmm.app

    if forcefield_route == "name":
        pdbfile = TEST_DIR / "pdbfiles" / "1aki.pdb"
        forcefield_kwargs = {"forcefield": "CHARMM36", "watermodel": "tip3p"}
    else:
        pdbfile = Path(_write_dipeptide_pdb())
        if forcefield_route == "xmlfile":
            forcefield_kwargs = {"xmlfile": "charmm36.xml", "waterxmlfile": "charmm36/water.xml"}
        else:
            forcefield_kwargs = {"forcefield_object": openmm.app.ForceField("charmm36.xml", "charmm36/water.xml")}

    openmmobject, fragment = openmm_modeller(
        pdbfile=str(pdbfile),
        ph=7.0,
        solvent_padding=10.0,
        ionicstrength=0.1 if forcefield_route == "name" else 0.5,  # Include salt pairs even in the small box.
        platform="CPU",
        **forcefield_kwargs,
    )

    topology = openmmobject.topology
    residues = list(topology.residues())
    protein = [
        residue for residue in residues if any(atom.element == openmm.app.element.carbon for atom in residue.atoms())
    ]
    assert len(protein) == (129 if forcefield_route == "name" else 2)
    assert all(any(atom.element == openmm.app.element.hydrogen for atom in residue.atoms()) for residue in protein)
    expected_disulfides = {
        frozenset((int(line[17:21]), int(line[31:35])))
        for line in pdbfile.read_text().splitlines()
        if line.startswith("SSBOND")
    }
    disulfides = {
        frozenset((int(first.residue.id), int(second.residue.id)))
        for first, second in topology.bonds()
        if first.element == second.element == openmm.app.element.sulfur
    }
    assert disulfides == expected_disulfides
    waters = [residue for residue in residues if residue.name == "HOH"]
    assert waters
    assert {len(list(residue.atoms())) for residue in waters} == {3}
    elements = {atom.element for atom in topology.atoms()}
    assert {openmm.app.element.sodium, openmm.app.element.chlorine} <= elements
    assert fragment.numatoms == topology.getNumAtoms() == openmmobject.system.getNumParticles()
    assert len(fragment.elems) == fragment.numatoms


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


@pytest.mark.parametrize("implicit_solvent_xmlfile", [None, "implicit/gbn2.xml"])
def test_implicit_solvent_puts_a_gb_force_in_the_system(implicit_solvent_xmlfile):
    openmmobject, _ = openmm_modeller(
        pdbfile=_write_dipeptide_pdb(),
        forcefield="Amber14",
        implicit=True,
        implicit_solvent_xmlfile=implicit_solvent_xmlfile,
    )

    forces = openmmobject.system.getForces()
    assert any(isinstance(force, (openmm.GBSAOBCForce, openmm.CustomGBForce)) for force in forces)


def test_a_solvent_box_uses_the_requested_water_model():
    _, fragment = openmm_modeller(
        pdbfile=_write_dipeptide_pdb(),
        forcefield="Amber14",
        watermodel="tip4pew",
        waterxmlfile="amber14/tip4pew.xml",
        solvent_boxdims=[25.0, 25.0, 25.0],
    )

    topology = openmm.app.PDBFile("finalsystem.pdb").topology
    water_sizes = {len(list(residue.atoms())) for residue in topology.residues() if residue.name == "HOH"}
    assert water_sizes == {4}
    assert fragment.numatoms == topology.getNumAtoms()


@pytest.mark.parametrize("format_name", ["PDB", "PDBx"])
def test_topology_writers_preserve_single_frame_bytes(tmp_path, format_name):
    import io

    from openmmqmmm.openmm import systemsetup

    topology = openmm.app.Topology()
    residue = topology.addResidue("HOH", topology.addChain())
    topology.addAtom("O", openmm.app.element.oxygen, residue)
    positions = [openmm.Vec3(1.25, 2.5, -0.5)] * openmm.unit.angstrom
    writer = getattr(openmm.app, format_name + "File")
    expected = io.StringIO()
    writer.writeHeader(topology, expected)
    writer.writeModel(topology, positions, expected)
    if format_name == "PDB":
        writer.writeFooter(topology, expected)
    output = tmp_path / "structure"
    function = (
        systemsetup.write_pdbfile_openmm_topology
        if format_name == "PDB"
        else systemsetup.write_pdbxfile_openmm_topology
    )
    function(topology, positions, output)
    assert output.read_bytes() == expected.getvalue().encode()
