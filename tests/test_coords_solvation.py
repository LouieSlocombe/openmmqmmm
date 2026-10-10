"""Solute insertion into a solvent box and the dummy-topology builder in openmmqmmm.coords."""

from pathlib import Path

import numpy as np
import openmm.app
import openmm.unit
import pytest

from openmmqmmm import Fragment, insert_solute_into_solvent
from openmmqmmm.coords import define_dummy_topology, read_xyzfile
from openmmqmmm.exceptions import InputError

H2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def _h2_solvent(centres):
    coords = np.vstack([H2 + centre for centre in centres])
    return Fragment(coords=coords, elems=["H"] * len(coords))


def test_insert_two_solutes_removes_solvent_clashing_with_either_solute():
    # The two H2 solutes are stacked 2 A apart along z (centre z = 1.35); the solvent centre is
    # (15, 15, 15.35), so the solutes land at z = 14.0, 14.7, 16.0 and 16.7. Molecules A and B
    # sit 0.8 A below the first solute and above the second; the other four are 10 A away.
    molecule_a = [[15.0, 15.0, 12.5]]
    molecule_b = [[15.0, 15.0, 17.5]]
    far = [[5.0, 5.0, 5.0], [25.0, 25.0, 25.0], [5.0, 25.0, 15.0], [25.0, 5.0, 15.0]]
    solvent = _h2_solvent(molecule_a + molecule_b + far)
    solute = Fragment(coords=H2.copy(), elems=["H", "H"])
    solute2 = Fragment(coords=H2.copy(), elems=["H", "H"])

    solution = insert_solute_into_solvent(solute=solute, solute2=solute2, solvent=solvent)

    assert solution.numatoms == 12
    assert solution.coords[:4] == pytest.approx(
        np.array([[15.0, 15.0, 14.0], [15.0, 15.0, 14.7], [15.0, 15.0, 16.0], [15.0, 15.0, 16.7]])
    )
    assert solution.coords[4:] == pytest.approx(_h2_solvent(far).coords)
    assert read_xyzfile("solution-pre.xyz")[1] == pytest.approx(np.vstack([solution.coords[:4], solvent.coords]))
    assert read_xyzfile("solution.xyz")[1] == pytest.approx(solution.coords)


def test_insert_solute_requires_pdb_inputs_when_writing_a_pdb():
    solute = Fragment(coords=H2.copy(), elems=["H", "H"])

    with pytest.raises(InputError, match="write_pdb is active but no input solute_pdb or solvent_pdb"):
        insert_solute_into_solvent(solute=solute, solvent=solute, write_pdb=True)


def _write_water_pdb(path, centres, resname, box_nm=None):
    """Write bonded water residues at the given centres through OpenMM, so the file is independently valid."""
    topology = openmm.app.Topology()
    chain = topology.addChain()
    positions = []
    for centre in centres:
        residue = topology.addResidue(resname, chain)
        oxygen = topology.addAtom("O", openmm.app.element.oxygen, residue)
        h1 = topology.addAtom("H1", openmm.app.element.hydrogen, residue)
        h2 = topology.addAtom("H2", openmm.app.element.hydrogen, residue)
        topology.addBond(oxygen, h1)
        topology.addBond(oxygen, h2)
        positions.extend(WATER + centre)
    if box_nm is not None:
        topology.setPeriodicBoxVectors(np.diag([box_nm] * 3) * openmm.unit.nanometer)
    with open(path, "w") as handle:
        openmm.app.PDBFile.writeFile(topology, np.array(positions) * openmm.unit.angstrom, handle)
    return str(path)


# Solvent centres averaging to (10, 10, 10), where the solute water is placed on top of the first one
SOLVENT_CENTRES = [[10.0, 10.0, 10.0], [2.0, 2.0, 2.0], [18.0, 18.0, 18.0], [2.0, 18.0, 10.0], [18.0, 2.0, 10.0]]


@pytest.mark.parametrize("write_solute_connectivity", [True, False])
def test_insert_solute_from_pdb_files_writes_a_pdb_with_residues_box_and_bonds(tmp_path, write_solute_connectivity):
    solute_pdb = _write_water_pdb(tmp_path / "solute.pdb", [[0.0, 0.0, 0.0]], "LIG")
    solvent_pdb = _write_water_pdb(tmp_path / "solvent.pdb", SOLVENT_CENTRES, "HOH", box_nm=2.0)
    outputname = str(tmp_path / "solution.pdb")

    solution = insert_solute_into_solvent(
        solute_pdb=solute_pdb,
        solvent_pdb=solvent_pdb,
        write_pdb=True,
        outputname=outputname,
        write_solute_connectivity=write_solute_connectivity,
    )

    assert solution.numatoms == 15
    assert solution.coords[:3] == pytest.approx(WATER + 10.0)
    assert solution.coords[3:] == pytest.approx(np.vstack([WATER + centre for centre in SOLVENT_CENTRES[1:]]))

    written = openmm.app.PDBFile(outputname)
    assert [residue.name for residue in written.topology.residues()] == ["LIG"] + ["HOH"] * 4
    assert written.topology.getNumAtoms() == 15
    positions = np.asarray(written.positions.value_in_unit(openmm.unit.angstrom))
    assert positions == pytest.approx(solution.coords, abs=1e-3)
    box = np.asarray(written.topology.getPeriodicBoxVectors().value_in_unit(openmm.unit.angstrom))
    assert box == pytest.approx(np.diag([20.0, 20.0, 20.0]))
    conect = [line for line in Path(outputname).read_text().splitlines() if line.startswith("CONECT")]
    assert bool(conect) is write_solute_connectivity
    solute_bonds = [bond for bond in written.topology.bonds() if bond[0].residue.name == "LIG"]
    assert len(solute_bonds) == (2 if write_solute_connectivity else 0)


def test_insert_two_solutes_from_pdb_files_keeps_both_solute_residues(tmp_path):
    # Centres still average to (10, 10, 10); the first water now sits 1 A below the solute only,
    # since a solvent molecule touching both solutes would bridge them.
    centres = [[10.0, 10.0, 8.0], [2.0, 2.0, 4.0], [18.0, 18.0, 18.0], [2.0, 18.0, 10.0], [18.0, 2.0, 10.0]]
    solute_pdb = _write_water_pdb(tmp_path / "solute.pdb", [[0.0, 0.0, 0.0]], "LIG")
    solute2_pdb = _write_water_pdb(tmp_path / "solute2.pdb", [[0.0, 0.0, 0.0]], "LI2")
    solvent_pdb = _write_water_pdb(tmp_path / "solvent.pdb", centres, "HOH")
    outputname = str(tmp_path / "solution.pdb")

    solution = insert_solute_into_solvent(
        solute_pdb=solute_pdb,
        solute2_pdb=solute2_pdb,
        solvent_pdb=solvent_pdb,
        write_pdb=True,
        outputname=outputname,
        write_pbc_info=False,
    )

    # The second water is lifted 2 A along z before both are centred on the solvent centre
    assert solution.numatoms == 18
    assert solution.coords[:3] == pytest.approx(WATER + np.array([10.0, 10.0, 9.0]))
    assert solution.coords[3:6] == pytest.approx(WATER + np.array([10.0, 10.0, 11.0]))
    assert solution.coords[6:] == pytest.approx(np.vstack([WATER + centre for centre in centres[1:]]))
    written = openmm.app.PDBFile(outputname)
    assert [residue.name for residue in written.topology.residues()] == ["LIG", "LI2"] + ["HOH"] * 4
    assert written.topology.getPeriodicBoxVectors() is None
    positions = np.asarray(written.positions.value_in_unit(openmm.unit.angstrom))
    assert positions == pytest.approx(solution.coords, abs=1e-3)


def test_define_dummy_topology_numbers_atoms_per_element_in_one_residue():
    topology = define_dummy_topology(["O", "H", "H", "C"], resname="WAT")

    assert topology.getNumChains() == 1
    residues = list(topology.residues())
    assert [residue.name for residue in residues] == ["WAT"]
    atoms = list(topology.atoms())
    assert [atom.name for atom in atoms] == ["O1", "H1", "H2", "C1"]
    assert [atom.element.symbol for atom in atoms] == ["O", "H", "H", "C"]
    assert topology.getNumBonds() == 0
