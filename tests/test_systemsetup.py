from pathlib import Path

import openmm
import pytest

from openmmqmmm import openmm_modeller

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
