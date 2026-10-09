"""Residue selection uses the same topology interpretation as OpenMM."""

from pathlib import Path

import numpy as np
import pytest
from openmm.app import CharmmPsfFile, PDBFile

from openmmqmmm import Fragment
from openmmqmmm.qmmm import define_active_region, read_charges_from_psf

FIXTURES = Path(__file__).parent


def test_active_region_ignores_atom_mentions_in_pdb_remarks():
    pdbfile = FIXTURES / "pdbfiles" / "1aki.pdb"
    pdb = PDBFile(str(pdbfile))
    expected = [atom.index for atom in pdb.topology.atoms() if atom.residue.index == 0]
    assert define_active_region(pdbfile=str(pdbfile), originatom=0, radius=0.01) == expected


def test_active_region_psf_does_not_treat_angle_records_as_atoms():
    psffile = FIXTURES / "fixtures" / "three-waters.psf"
    assert len(list(CharmmPsfFile(str(psffile)).topology.atoms())) == 9
    fragment = Fragment(elems=["O", "H", "H"] * 3, coords=np.arange(27).reshape(9, 3), conncalc=False)
    assert define_active_region(fragment=fragment, psffile=psffile, originatom=0, radius=100) == list(range(9))
    assert read_charges_from_psf(psffile) == pytest.approx([-0.834, 0.417, 0.417] * 3)
