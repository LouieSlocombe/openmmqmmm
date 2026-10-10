"""PDB-text, forcefield-selection and modeller-helper branches of openmmqmmm.openmm.systemsetup."""

import logging
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest

from openmmqmmm import Fragment, openmm_modeller
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.systemsetup import (
    FORCEFIELD_XMLFILES,
    _build_forcefield_object,
    _log_output_files_and_usage,
    _log_residue_table,
    _parameterize_nonstandard_residues,
    _resolve_named_forcefield,
    find_alternate_locations_residues,
    merge_pdb_files,
    write_pdbfile_openmm_topology,
)

TEST_DIR = Path(__file__).parent
OPENMM_DATA_DIR = Path(openmm.app.forcefield.__file__).parent / "data"

ALTLOC_PDB_LINES = [
    "REMARK   1 handwritten",
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N",
    "ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.60 10.00           C",
    "ATOM      3  CA BALA A   1       1.100   0.000   0.000  0.40 10.00           C",
    "ATOM      4  C   ALA A   1       2.000   0.000   0.000  1.00 10.00           C",
    "ATOM      5  N   SER A   2       3.000   0.000   0.000  1.00 10.00           N",
    "ATOM      6  OG ASER A   2       4.000   0.000   0.000  0.30 10.00           O",
    "ATOM      7  OG BSER A   2       4.100   0.000   0.000  0.70 10.00           O",
    "HETATM    8  O  AHOH B   3       5.000   0.000   0.000  0.50 10.00           O",
    "HETATM    9  O  BHOH B   3       5.100   0.000   0.000  0.50 10.00           O",
    "END",
]

# Highest occupancy wins, a tie keeps the first location, and the altloc column is blanked.
ALTLOC_RESOLVED_LINES = [
    "REMARK   1 handwritten",
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N",
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  0.60 10.00           C",
    "ATOM      4  C   ALA A   1       2.000   0.000   0.000  1.00 10.00           C",
    "ATOM      5  N   SER A   2       3.000   0.000   0.000  1.00 10.00           N",
    "ATOM      7  OG  SER A   2       4.100   0.000   0.000  0.70 10.00           O",
    "HETATM    8  O   HOH B   3       5.000   0.000   0.000  0.50 10.00           O",
    "END",
]

LIG_XML = """<ForceField>
<AtomTypes>
<Type name="LC" class="LC" element="C" mass="12.011"/>
</AtomTypes>
<Residues>
<Residue name="LIG">
<Atom name="C1" type="LC"/>
</Residue>
</Residues>
<NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
<Atom type="LC" charge="0.0" sigma="0.34" epsilon="0.45"/>
</NonbondedForce>
</ForceField>
"""

METHANOL_COORDS = [
    [-0.046, 0.662, 0.000],
    [-0.046, -0.755, 0.000],
    [-1.086, 0.976, 0.000],
    [0.438, 1.071, 0.890],
    [0.438, 1.071, -0.890],
    [0.860, -1.057, 0.000],
]


def _write_altloc_pdb(path="alt.pdb"):
    Path(path).write_text("\n".join(ALTLOC_PDB_LINES) + "\n")
    return path


def _write_methanol_pdb():
    fragment = Fragment(elems=["C", "O", "H", "H", "H", "H"], coords=METHANOL_COORDS, charge=0, mult=1)
    return fragment.write_pdbfile_openmm(filename="methanol.pdb", resname="LIG")


def _write_tripeptide_pdb():
    """ARG-HIS-GLY (residues 14-16 of 1aki) renumbered 1-3, heavy atoms only."""
    lines = [
        line[:22] + f"{int(line[22:26]) - 13:4d}" + line[26:]
        for line in (TEST_DIR / "pdbfiles" / "1aki.pdb").read_text().splitlines()
        if line.startswith("ATOM") and 14 <= int(line[22:26]) <= 16
    ]
    Path("tripeptide.pdb").write_text("\n".join([*lines, "END"]) + "\n")
    return "tripeptide.pdb"


def _single_residue_topology(resname, atom_names, chain_id="A"):
    topology = openmm.app.Topology()
    residue = topology.addResidue(resname, topology.addChain(chain_id))
    for name in atom_names:
        topology.addAtom(name, openmm.app.Element.getBySymbol(name[0]), residue)
    return topology


def test_altloc_pdb_is_rejected_unless_higher_occupancy_is_requested():
    with pytest.raises(InputError, match="use_higher_occupancy=True"):
        find_alternate_locations_residues(_write_altloc_pdb())
    assert not Path("system_afteratlocfixes.pdb").exists()


def test_altloc_pdb_keeps_the_higher_occupancy_location(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    written = find_alternate_locations_residues(_write_altloc_pdb(), use_higher_occupancy=True)

    assert written == "system_afteratlocfixes.pdb"
    assert Path(written).read_text().splitlines() == ALTLOC_RESOLVED_LINES
    topology = openmm.app.PDBFile(written).topology
    assert [atom.name for atom in topology.atoms()] == ["N", "CA", "C", "N", "OG", "O"]
    assert "Chain A:" in caplog.text
    assert "Chain B:" in caplog.text
    for residue in ("ALA1", "SER2", "HOH3"):
        assert residue in caplog.text
    assert "Alternate locations for atom: A_ALA_1_CA" in caplog.text
    assert "Choosing line with occupancy 0.7." in caplog.text


def test_pdb_without_altlocs_is_returned_untouched():
    lines = [line for line in ALTLOC_PDB_LINES if line[16:17] == " " or not line.startswith(("ATOM", "HETATM"))]
    Path("clean.pdb").write_text("\n".join(lines) + "\n")

    assert find_alternate_locations_residues("clean.pdb") == "clean.pdb"
    assert find_alternate_locations_residues("clean.pdb", use_higher_occupancy=True) == "clean.pdb"
    assert not Path("system_afteratlocfixes.pdb").exists()


def test_merge_pdb_files_concatenates_residues_and_positions():
    water = _single_residue_topology("HOH", ["O", "H1", "H2"], chain_id="A")
    water_positions = [openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(0.96, 0.0, 0.0), openmm.Vec3(-0.24, 0.93, 0.0)]
    ligand = _single_residue_topology("LIG", ["C1", "O1"], chain_id="B")
    ligand_positions = [openmm.Vec3(5.0, 5.0, 5.0), openmm.Vec3(6.4, 5.0, 5.0)]
    with open("water.pdb", "w") as handle:
        openmm.app.PDBFile.writeFile(water, water_positions * openmm.unit.angstrom, handle)
    with open("ligand.pdb", "w") as handle:
        openmm.app.PDBFile.writeFile(ligand, ligand_positions * openmm.unit.angstrom, handle)

    merged = merge_pdb_files("water.pdb", "ligand.pdb", outputname="complex.pdb")

    assert merged == "complex.pdb"
    pdb = openmm.app.PDBFile("complex.pdb")
    assert [residue.name for residue in pdb.topology.residues()] == ["HOH", "LIG"]
    assert pdb.topology.getNumChains() == 2
    assert [atom.name for atom in pdb.topology.atoms()] == ["O", "H1", "H2", "C1", "O1"]
    expected = np.array([[*vec] for vec in water_positions + ligand_positions])
    assert np.asarray(pdb.positions.value_in_unit(openmm.unit.angstrom)) == pytest.approx(expected, abs=1e-3)


def test_topology_writer_adds_requested_bonds_as_conect_records():
    topology = _single_residue_topology("MOL", ["C1", "H1", "H2"])
    positions = [openmm.Vec3(0, 0, 0), openmm.Vec3(1.09, 0, 0), openmm.Vec3(0, 1.09, 0)] * openmm.unit.angstrom

    write_pdbfile_openmm_topology(topology, positions, "bonded.pdb", connectivity_dict={0: [1, 2]})

    assert topology.getNumBonds() == 2
    text = Path("bonded.pdb").read_text()
    assert text.count("CONECT") == 3
    assert {frozenset((a.index, b.index)) for a, b in openmm.app.PDBFile("bonded.pdb").topology.bonds()} == {
        frozenset((0, 1)),
        frozenset((0, 2)),
    }


@pytest.mark.parametrize(("name", "xmlfile"), sorted(FORCEFIELD_XMLFILES.items()))
def test_every_forcefield_shorthand_points_at_a_shipped_xml(name, xmlfile):
    assert (OPENMM_DATA_DIR / xmlfile).is_file(), name


@pytest.mark.parametrize(
    ("forcefield", "watermodel", "waterxmlfile", "expected"),
    [
        ("CHARMM36", None, None, ("charmm36.xml", "tip3p", "charmm36/water.xml")),
        ("CHARMM36", "TIP3P", None, ("charmm36.xml", "TIP3P", "charmm36/water.xml")),
        ("CHARMM36", "tip4pew", "charmm36/tip4pew.xml", ("charmm36.xml", "tip4pew", "charmm36/tip4pew.xml")),
        ("Amber14", None, None, ("amber14-all.xml", "tip3pfb", "amber14/tip3pfb.xml")),
        ("Amber14", "TIP3P-FB", None, ("amber14-all.xml", "TIP3P-FB", "amber14/tip3pfb.xml")),
        ("Amber14", "tip3p", None, ("amber14-all.xml", "tip3p", "amber14/tip3p.xml")),
        ("Amber99sb", "tip3p", None, ("amber99sb.xml", "tip3p", "tip3p.xml")),
        ("Amber14", "spce", None, ("amber14-all.xml", "spce", None)),
        ("Amoeba2013", None, None, ("amoeba2013.xml", None, None)),
    ],
)
def test_named_forcefield_resolves_xml_and_water_files(forcefield, watermodel, waterxmlfile, expected):
    assert _resolve_named_forcefield(forcefield, watermodel, waterxmlfile) == expected
    for xml in expected[::2]:
        if xml is not None:
            assert (OPENMM_DATA_DIR / xml).is_file()


@pytest.mark.parametrize("forcefield", ["OPLS", None, ["Amber14"]])
def test_unknown_forcefield_name_is_rejected(forcefield):
    with pytest.raises(InputError, match="Unknown forcefield"):
        _resolve_named_forcefield(forcefield, None, None)


def test_forcefield_object_requires_a_name_object_or_xmlfile():
    with pytest.raises(InputError, match="forcefield name, forcefieldobject or xmlfile"):
        _build_forcefield_object(
            xmlfile=None, forcefield_object=None, extraxmlfile=None, waterxmlfile=None, watermodel=None
        )


def test_supplied_forcefield_object_is_returned_and_water_xml_ignored(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    supplied = openmm.app.ForceField("tip3p.xml")

    result = _build_forcefield_object(
        xmlfile=None,
        forcefield_object=supplied,
        extraxmlfile=None,
        waterxmlfile="amber14/tip3p.xml",
        watermodel="tip3p",
    )

    assert result is supplied
    assert "Ignoring waterxmlfile" in caplog.text
    assert "solvent box geometry used by Modeller: tip3p" in caplog.text


def test_missing_extra_xmlfile_is_rejected():
    with pytest.raises(InputError, match=r"nowhere\.xml cannot be found"):
        _build_forcefield_object(
            xmlfile="tip3p.xml", forcefield_object=None, extraxmlfile="nowhere.xml", waterxmlfile=None, watermodel=None
        )


def test_extra_xmlfile_templates_are_loaded_alongside_the_main_xml():
    Path("lig.xml").write_text(LIG_XML)

    forcefield = _build_forcefield_object(
        xmlfile="tip3p.xml", forcefield_object=None, extraxmlfile="lig.xml", waterxmlfile=None, watermodel=None
    )

    assert {"HOH", "LIG"} <= set(forcefield._templates)


def test_residue_table_marks_requested_variants_per_chain(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    topology = openmm.app.Topology()
    chain_a = topology.addChain("A")
    chain_b = topology.addChain("B")
    residues = [
        topology.addResidue("HIS", chain_a, id="1"),
        topology.addResidue("HIS", chain_a, id="2"),
        topology.addResidue("HIS", chain_b, id="2"),
    ]

    states = _log_residue_table(residues, {"A": {2: "HIP"}, "B": {}})

    assert states == [None, "HIP", None]
    assert caplog.text.count("This residue will be changed to: HIP") == 1
    assert caplog.text.count("--" * 30) == 1


@pytest.mark.parametrize("with_extras", [True, False])
def test_usage_summary_lists_every_xml_and_template_choice(caplog, with_extras):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    _log_output_files_and_usage(
        systemxmlfile="system_full.xml",
        xmlfile="a.xml",
        waterxmlfile="w.xml",
        extraxmlfile="e.xml" if with_extras else None,
        nonstandard_xmlfile="n.xml" if with_extras else None,
        residuetemplate_choice={"FER": "FE2"} if with_extras else None,
        periodic=True,
    )

    if with_extras:
        assert 'xmlfiles=["a.xml", "w.xml", "e.xml", "n.xml"]' in caplog.text
        assert "n.xml (forcefill-generated ligand parameters" in caplog.text
        assert "residuetemplate_choice={'FER': 'FE2'}" in caplog.text
    else:
        assert 'xmlfiles=["a.xml", "w.xml"]' in caplog.text
        assert "forcefill-generated" not in caplog.text
        assert "residuetemplate_choice option was provided" not in caplog.text


def _fake_forcefill(monkeypatch, outcome):
    """Install a stand-in forcefill module whose build_forcefield_xml returns or raises `outcome`."""
    calls = []

    def build_forcefield_xml(pdbfile, output_xml, **kwargs):
        calls.append((pdbfile, output_xml, kwargs))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    module = types.ModuleType("forcefill")
    module.build_forcefield_xml = build_forcefield_xml
    monkeypatch.setitem(sys.modules, "forcefill", module)
    return calls


def _forcefill_result(forcefield_xml, parameterized=(), skipped=None):
    return types.SimpleNamespace(
        forcefield_xml=forcefield_xml, parameterized=list(parameterized), skipped=skipped or {}
    )


def test_parameterize_nonstandard_reports_skipped_residues_and_returns_none(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    _fake_forcefill(monkeypatch, _forcefill_result(None, skipped={"LIG": "no heavy atoms"}))

    result = _parameterize_nonstandard_residues(
        "x.pdb",
        forcefield_obj=openmm.app.ForceField("tip3p.xml"),
        xmlfile="amber14-all.xml",
        waterxmlfile="amber14/tip3p.xml",
        extraxmlfile=None,
        ligand_files=None,
        net_charges={"LIG": 0},
        ligand_backend="gaff",
    )

    assert result is None
    assert "forcefill skipped LIG: no heavy atoms" in caplog.text
    assert "nothing to parameterize" in caplog.text


def test_parameterize_nonstandard_loads_the_generated_xml_into_the_forcefield(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    Path("lig.xml").write_text(LIG_XML)
    calls = _fake_forcefill(monkeypatch, _forcefill_result("lig.xml", parameterized=["LIG"]))
    forcefield = openmm.app.ForceField("tip3p.xml")

    result = _parameterize_nonstandard_residues(
        "x.pdb",
        forcefield_obj=forcefield,
        xmlfile="amber14-all.xml",
        waterxmlfile="amber14/tip3p.xml",
        extraxmlfile="extra.xml",
        ligand_files={"LIG": "lig.sdf"},
        net_charges=None,
        ligand_backend="openff",
    )

    assert result == "lig.xml"
    assert "LIG" in forcefield._templates
    assert calls == [
        (
            "x.pdb",
            "nonstandard_ff.xml",
            {
                "base_forcefield": ["amber14-all.xml", "extra.xml", "amber14/tip3p.xml"],
                "residue_files": {"LIG": "lig.sdf"},
                "net_charges": None,
                "backend": "openff",
                "workdir": "forcefill_files",
            },
        )
    ]
    assert "forcefill parameterized ['LIG'] -> lig.xml" in caplog.text


@pytest.mark.parametrize("error", [ValueError("antechamber failed"), RuntimeError("timeout")])
def test_parameterize_nonstandard_wraps_forcefill_errors(monkeypatch, error):
    _fake_forcefill(monkeypatch, error)

    with pytest.raises(InputError, match=rf"(?s)could not parameterize.*{re.escape(str(error))}"):
        _parameterize_nonstandard_residues(
            "x.pdb",
            forcefield_obj=openmm.app.ForceField("tip3p.xml"),
            xmlfile="amber14-all.xml",
            waterxmlfile=None,
            extraxmlfile=None,
            ligand_files=None,
            net_charges=None,
            ligand_backend="gaff",
        )


def test_modeller_requires_a_pdbfile():
    with pytest.raises(InputError, match="pdbfile keyword"):
        openmm_modeller(forcefield="Amber14")


def test_modeller_translates_add_hydrogens_failure_for_unknown_residues():
    pdbfile = _write_methanol_pdb()

    with pytest.raises(InputError, match="Read the OpenMM documentation"):
        openmm_modeller(pdbfile=pdbfile, forcefield="Amber14", watermodel="tip3p", use_pdbfixer=False)

    assert Path("system_afterfixes2.pdb").is_file()
    assert not Path("system_afterfixes.pdb").exists()
    assert not Path("system_afterH.pdb").exists()


def test_modeller_translates_ambiguous_template_errors(monkeypatch):
    pdbfile = _write_methanol_pdb()
    forcefield = openmm.app.ForceField("amber14-all.xml", "amber14/tip3p.xml")

    def ambiguous(*_args, **_kwargs):
        raise ValueError("Multiple non-identical matching templates found for residue")

    monkeypatch.setattr(forcefield, "getUnmatchedResidues", ambiguous)

    with pytest.raises(InputError, match="residuetemplate_choice"):
        openmm_modeller(pdbfile=pdbfile, forcefield_object=forcefield, use_pdbfixer=False)


@pytest.mark.parametrize(
    ("variant", "present", "absent"),
    [("HIP", {"HD1", "HE2"}, set()), ("HIE", {"HE2"}, {"HD1"}), ("HID", {"HD1"}, {"HE2"})],
)
def test_modeller_applies_residue_variants_by_chain_and_resid(caplog, variant, present, absent):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    _, fragment = openmm_modeller(
        pdbfile=_write_tripeptide_pdb(),
        forcefield="Amber14",
        implicit=True,
        residue_variants={"A": {2: variant}},
    )

    assert f"This residue will be changed to: {variant}" in caplog.text
    topology = openmm.app.PDBFile("finalsystem.pdb").topology
    histidine = next(residue for residue in topology.residues() if residue.name == "HIS")
    names = {atom.name for atom in histidine.atoms()}
    assert present <= names
    assert not (absent & names)
    assert fragment.numatoms == topology.getNumAtoms()
    assert os.path.isfile("finalsystem.cif")
