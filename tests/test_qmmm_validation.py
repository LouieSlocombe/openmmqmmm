"""QM/MM input validation, active-region selection and the DEBUG-level gradient dumps."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm.app
import pytest
from conftest import _make_analytic_qmmm, dummy_mm, make_subregion_qmmm

from openmmqmmm import Fragment, QMMMTheory, ZeroTheory
from openmmqmmm.exceptions import InputError
from openmmqmmm.qmmm import compute_decomposed_qm_mm_energy, define_active_region
from openmmqmmm.utils import read_intlist_from_file

QMMM_LOGGER = "openmmqmmm.qmmm"


def _pair():
    return Fragment(elems=["H", "H"], coords=[[0, 0, 0], [3, 0, 0]], charge=0, mult=1, conncalc=False)


def _bare_mm(fragment):
    return dummy_mm(fragment)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"qm_theory": None}, "requires defining"),
        ({"qmatoms": None}, "requires defining"),
        ({"qm_theory": ZeroTheory(), "qmatoms": [0], "fragment": None}, "fragment= keyword"),
        ({"linkatom_forceproj_method": 5}, "must be one of: adv, lever, chain, none"),
        ({"embedding": "pbcmm-elstat"}, "not supported in this distribution"),
        ({"embedding": "sideways"}, "Unknown embedding"),
        ({"charges": [0.1]}, "Number of charges not matching"),
        ({"mm_theory": None, "charges": None}, "requires either a charges list"),
    ],
)
def test_constructor_rejects_incomplete_or_inconsistent_input(options, message):
    fragment = _pair()
    arguments = {"qm_theory": ZeroTheory(), "qmatoms": [0], "fragment": fragment, "mm_theory": _bare_mm(fragment)}
    arguments.update(options)
    with pytest.raises(InputError, match=message):
        QMMMTheory(**arguments)


def test_an_mm_theory_without_any_charges_is_rejected():
    fragment = _pair()
    mm = _bare_mm(fragment)
    mm.charges = []
    with pytest.raises(InputError, match="No charges present"):
        QMMMTheory(qm_theory=ZeroTheory(), qmatoms=[0], fragment=fragment, mm_theory=mm)


def test_run_rejects_an_embedding_it_cannot_dispatch():
    qmmm, fragment = make_subregion_qmmm(qm_charge=0, qm_mult=1)
    qmmm.embedding = "polarized"
    with pytest.raises(InputError, match="Unknown embedding 'polarized'"):
        qmmm.run(current_coords=fragment.coords, elems=fragment.elems)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"radius": None, "originatom": 0}, "requires radius and originatom"),
        ({"radius": 2.0, "originatom": None}, "requires radius and originatom"),
        ({"radius": 2.0, "originatom": 0, "fragment": None}, "fragment or pdbfile"),
        ({"radius": 2.0, "originatom": 0, "mmtheory": None}, "residue topology information"),
        ({"radius": 2.0, "originatom": 0, "mmtheory": SimpleNamespace(resids=[])}, "resids list is empty"),
    ],
)
def test_active_region_definition_checks_its_inputs(options, message):
    arguments = {"fragment": _pair(), "mmtheory": SimpleNamespace(resids=[0, 1])}
    arguments.update(options)
    with pytest.raises(InputError, match=message):
        define_active_region(**arguments)


def _three_residue_chain():
    coords = [[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0], [20, 0, 0], [21, 0, 0]]
    return Fragment(elems=["C"] * 6, coords=coords, charge=0, mult=1, conncalc=False), [0, 0, 1, 1, 2, 2]


def test_active_region_keeps_whole_residues_within_the_radius_and_writes_them_out():
    fragment, resids = _three_residue_chain()

    active = define_active_region(fragment=fragment, mmtheory=SimpleNamespace(resids=resids), radius=5.5, originatom=0)

    assert active == [0, 1, 2, 3]
    assert read_intlist_from_file("active_atoms") == [0, 1, 2, 3]
    assert Path("ActiveRegion.xyz").read_text().splitlines()[0] == "4"


def test_active_region_can_take_coordinates_and_residues_from_a_pdb_file():
    fragment, resids = _three_residue_chain()
    topology = openmm.app.Topology()
    chain = topology.addChain()
    residues = [topology.addResidue("RES", chain) for _ in range(3)]
    for atom, resid in enumerate(resids):
        topology.addAtom(f"C{atom}", openmm.app.Element.getBySymbol("C"), residues[resid])
    with open("chain.pdb", "w") as pdb:
        openmm.app.PDBFile.writeFile(topology, fragment.coords * openmm.unit.angstrom, pdb)

    active = define_active_region(pdbfile="chain.pdb", radius=1.5, originatom=5)

    assert active == [4, 5]


def test_energy_decomposition_requires_a_charged_electrostatic_standalone_qmmm_theory():
    with pytest.raises(InputError, match="Please provide a QMMMTheory"):
        compute_decomposed_qm_mm_energy(theory=ZeroTheory())

    uncharged, _ = make_subregion_qmmm()
    with pytest.raises(InputError, match="qm_charge and qm_mult"):
        compute_decomposed_qm_mm_energy(theory=uncharged)

    mechanical, _ = make_subregion_qmmm(embedding="mech", qm_charge=0, qm_mult=1)
    with pytest.raises(InputError, match="requires electrostatic embedding"):
        compute_decomposed_qm_mm_energy(theory=mechanical)

    external, _ = make_subregion_qmmm(qm_charge=0, qm_mult=1, openmm_externalforce=True)
    with pytest.raises(InputError, match="standalone QM/MM theory"):
        compute_decomposed_qm_mm_energy(theory=external)


def _gradient_rows(filename):
    return [line for line in Path(filename).read_text().splitlines()[1:] if line.strip()]


def test_debug_logging_dumps_every_gradient_component_of_an_electrostatic_run(caplog):
    qmmm, fragment = make_subregion_qmmm(qm_charge=0, qm_mult=1)

    with caplog.at_level(logging.DEBUG, logger=QMMM_LOGGER):
        _energy, gradient = qmmm.run(current_coords=fragment.coords, elems=fragment.elems, grad=True, label="dbg")

    assert gradient.shape == (2, 3)
    for stem in ("QM+PCgradient", "MMgradient", "QM_MMgradient"):
        assert len(_gradient_rows(f"{stem}_dbg")) == len(fragment.elems)
    for stem in ("QMgradient-without-linkatoms", "QMgradient-with-linkatoms", "PCgradient"):
        assert len(_gradient_rows(f"{stem}_dbg")) == 1


def test_debug_logging_reports_qm_region_charges_of_a_mechanical_run(caplog):
    qmmm, fragment, _qm = _make_analytic_qmmm()

    with caplog.at_level(logging.DEBUG, logger=QMMM_LOGGER):
        qmmm.run(current_coords=fragment.coords, elems=fragment.elems, grad=True, label="mech")

    assert "QM atom 0 has charge" in caplog.text
    assert "QM atom 1 has charge" in caplog.text
    assert not Path("PCgradient_mech").exists()
    assert len(_gradient_rows("QM_MMgradient_mech")) == 2
    np.testing.assert_allclose(np.loadtxt("QM_MMgradient_mech", skiprows=1, usecols=(2, 3, 4)).shape, (2, 3))
