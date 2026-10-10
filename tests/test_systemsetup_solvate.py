"""solvate_small_molecule on a methanol solute with hand-written forcefield XML files."""

import logging
from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest
import rdkit.Chem  # noqa: F401  # must load before openbabel (used by the bonded path) or rdkit segfaults later

from openmmqmmm import Fragment
from openmmqmmm.exceptions import InputError
from openmmqmmm.openmm.systemsetup import solvate_small_molecule

TEST_DIR = Path(__file__).parent
BOX = [12.0, 12.0, 12.0]

METHANOL_COORDS = [
    [-0.046, 0.662, 0.000],
    [-0.046, -0.755, 0.000],
    [-1.086, 0.976, 0.000],
    [0.438, 1.071, 0.890],
    [0.438, 1.071, -0.890],
    [0.860, -1.057, 0.000],
]
METHANOL_BONDS = {("C1", "O2"), ("C1", "H3"), ("C1", "H4"), ("C1", "H5"), ("O2", "H6")}

ATOM_TYPES = """<AtomTypes>
<Type name="LC" class="LC" element="C" mass="12.011"/>
<Type name="LO" class="LO" element="O" mass="15.999"/>
<Type name="LH" class="LH" element="H" mass="1.008"/>
</AtomTypes>
"""

# Atom names follow write_pdbfile's element+"Y"+index scheme used by the bond-free path.
LIG_RESIDUE = """<Residues>
<Residue name="LIG">
<Atom name="CY0" type="LC"/>
<Atom name="OY1" type="LO"/>
<Atom name="HY2" type="LH"/>
<Atom name="HY3" type="LH"/>
<Atom name="HY4" type="LH"/>
<Atom name="HY5" type="LH"/>
</Residue>
</Residues>
"""

# Atom names follow openbabel's element+index scheme used by the bonded path.
UNL_RESIDUE = """<Residues>
<Residue name="UNL">
<Atom name="C1" type="LC"/>
<Atom name="O2" type="LO"/>
<Atom name="H3" type="LH"/>
<Atom name="H4" type="LH"/>
<Atom name="H5" type="LH"/>
<Atom name="H6" type="LH"/>
<Bond atomName1="C1" atomName2="O2"/>
<Bond atomName1="C1" atomName2="H3"/>
<Bond atomName1="C1" atomName2="H4"/>
<Bond atomName1="C1" atomName2="H5"/>
<Bond atomName1="O2" atomName2="H6"/>
</Residue>
</Residues>
<HarmonicBondForce>
<Bond class1="LC" class2="LO" length="0.141" k="267776"/>
</HarmonicBondForce>
"""

AMBER_NONBONDED = """<NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">
<Atom type="LC" charge="0.1" sigma="0.34" epsilon="0.45"/>
<Atom type="LO" charge="-0.6" sigma="0.31" epsilon="0.71"/>
<Atom type="LH" charge="0.125" sigma="0.25" epsilon="0.06"/>
</NonbondedForce>
"""

CHARMM_NONBONDED = """<NonbondedForce coulomb14scale="1.0" lj14scale="1.0">
<Atom type="LC" charge="0.1" sigma="0.0" epsilon="0.0"/>
<Atom type="LO" charge="-0.6" sigma="0.0" epsilon="0.0"/>
<Atom type="LH" charge="0.125" sigma="0.0" epsilon="0.0"/>
</NonbondedForce>
<LennardJonesForce lj14scale="1.0">
<Atom type="LC" sigma="0.34" epsilon="0.45"/>
<Atom type="LO" sigma="0.31" epsilon="0.71"/>
<Atom type="LH" sigma="0.25" epsilon="0.06"/>
</LennardJonesForce>
"""

XML_SECTIONS = {
    "amber": LIG_RESIDUE + AMBER_NONBONDED,
    "amber-bonded": UNL_RESIDUE + AMBER_NONBONDED,
    "charmm": LIG_RESIDUE + CHARMM_NONBONDED,
}
EXPECTED_LOG = {
    "amber": ("Found Amber-style scaling parameter", "amber14/tip3p.xml"),
    "amber-bonded": ("Found Amber-style scaling parameter", "amber14/tip3p.xml"),
    "charmm": ("Found CHARMM-style format", "charmm36/water.xml"),
}


def _write_xml(style):
    path = f"solute_{style}.xml"
    Path(path).write_text("<ForceField>\n" + ATOM_TYPES + XML_SECTIONS[style] + "</ForceField>\n")
    return path


def _methanol():
    return Fragment(elems=["C", "O", "H", "H", "H", "H"], coords=METHANOL_COORDS, charge=0, mult=1)


def _count_water(topology):
    return sum(1 for residue in topology.residues() if residue.name == "HOH")


@pytest.mark.parametrize("style", ["amber", "charmm", "amber-bonded"])
def test_solvation_adds_tip3p_waters_around_the_solute_in_the_requested_box(caplog, style):
    caplog.set_level(logging.INFO, logger="openmmqmmm")
    fragment = _methanol()

    forcefield, topology, solvated = solvate_small_molecule(
        fragment=fragment, charge=0, mult=1, watermodel="tip3p", solvent_boxdims=BOX, xmlfile=_write_xml(style)
    )

    n_water = _count_water(topology)
    assert n_water > 0
    assert topology.getNumAtoms() == 6 + 3 * n_water
    assert all(len(list(residue.atoms())) == 3 for residue in topology.residues() if residue.name == "HOH")
    assert solvated.numatoms == topology.getNumAtoms()
    assert solvated.elems[:6] == fragment.elems
    assert solvated.coords[:6] == pytest.approx(fragment.coords, abs=1e-3)
    box = np.array(topology.getPeriodicBoxVectors().value_in_unit(openmm.unit.angstrom))
    assert box == pytest.approx(np.diag(BOX))
    solute_name = "UNL" if style == "amber-bonded" else "LIG"
    assert solute_name in forcefield._templates
    assert next(topology.residues()).name == solute_name
    assert forcefield.getUnmatchedResidues(topology) == []
    written = openmm.app.PDBFile("system_aftersolvent.pdb").topology
    assert written.getNumAtoms() == topology.getNumAtoms()
    assert len(Path("system_aftersolvent.xyz").read_text().splitlines()) == topology.getNumAtoms() + 2
    for message in EXPECTED_LOG[style]:
        assert message in caplog.text


def test_bonded_xml_routes_the_solute_through_openbabel_connectivity(caplog):
    caplog.set_level(logging.INFO, logger="openmmqmmm")

    _, topology, _ = solvate_small_molecule(
        fragment=_methanol(),
        charge=0,
        mult=1,
        watermodel="tip3p",
        solvent_boxdims=BOX,
        xmlfile=_write_xml("amber-bonded"),
    )

    assert "XML-file contains bonded parameters" in caplog.text
    assert "CONECT" in Path("smallmol.pdb").read_text()
    solute = next(topology.residues())
    solute_bonds = {frozenset((a.name, b.name)) for a, b in topology.bonds() if a.residue is solute}
    assert solute_bonds == {frozenset(bond) for bond in METHANOL_BONDS}


def test_bond_free_xml_writes_the_solute_with_indexed_dummy_names():
    solvate_small_molecule(
        fragment=_methanol(), charge=0, mult=1, watermodel="tip3p", solvent_boxdims=BOX, xmlfile=_write_xml("amber")
    )

    solute = openmm.app.PDBFile("smallmol.pdb").topology
    assert [atom.name for atom in solute.atoms()] == ["CY0", "OY1", "HY2", "HY3", "HY4", "HY5"]
    assert [residue.name for residue in solute.residues()] == ["LIG"]
    assert solute.getNumBonds() == 0


def test_solvation_requires_a_fragment():
    with pytest.raises(InputError, match="No fragment object"):
        solvate_small_molecule(xmlfile="solute.xml")


def test_solvation_requires_an_xmlfile():
    with pytest.raises(InputError, match="No xmlfile was provided"):
        solvate_small_molecule(fragment=_methanol(), charge=0, mult=1, watermodel="tip3p")


def test_solvation_rejects_an_xml_of_unknown_lj_style():
    with pytest.raises(InputError, match="Unknown LJ14 scaling type"):
        solvate_small_molecule(
            fragment=_methanol(),
            charge=0,
            mult=1,
            watermodel="tip3p",
            xmlfile=str(TEST_DIR / "extra_files" / "MeOH_H2O-sigma.xml"),
        )


@pytest.mark.parametrize("watermodel", ["spce", "tip4pew", None])
def test_solvation_supports_only_tip3p(watermodel):
    with pytest.raises(InputError, match="Only TIP3P water supported"):
        solvate_small_molecule(
            fragment=_methanol(), charge=0, mult=1, watermodel=watermodel, xmlfile=_write_xml("amber")
        )


def test_skipping_the_xmlfile_requires_an_explicit_lj_treatment():
    with pytest.raises(InputError, match=r"Unsupported LJ_treatment \(None\)"):
        solvate_small_molecule(fragment=_methanol(), charge=0, mult=1, watermodel="tip3p", skip_xmlfile=True)
