"""Guard branches of the small modules: results serialization, NumGrad, utils, ASE, virtual sites, PLUMED."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm.app
import pytest
from ase import Atoms
from ase.calculators.calculator import CalculatorSetupError, PropertyNotImplementedError
from conftest import dummy_mm
from test_ase_calculator import StubQMMMTheory

from openmmqmmm import Fragment, NumGrad, OpenMMQMMMCalculator, Results, ZeroTheory, openmm_md_plumed
from openmmqmmm.coords_pbc import align_to_standard_orientation, cell_vectors_to_params, write_cif_file
from openmmqmmm.exceptions import FileFormatError, InputError
from openmmqmmm.openmm.nqe_export import modeller_from_topology
from openmmqmmm.results import read_results_from_file
from openmmqmmm.utils import pygrep2, read_intlist_from_file
from openmmqmmm.virtual_sites import NativeVirtualSites, _copy_site


def _round_trip(results):
    results.write_to_disk(filename="results.json")
    return read_results_from_file("results.json")


def test_results_write_paths_as_strings():
    restored = _round_trip(Results(label="paths", properties={"output": Path("runs") / "a.out"}))
    assert restored.properties == {"output": "runs/a.out"}


@pytest.mark.parametrize(
    ("field", "value", "warning"),
    [
        ("energy", np.float64("nan"), "Non-finite value found in energy"),
        ("energy", float("inf"), "Non-finite value found in energy"),
        ("energy", np.complex128(1 + 2j), "complex NumPy values"),
        ("properties", {"frag": Fragment(elems=["H"], coords=[[0, 0, 0]], conncalc=False)}, "Fragment objects"),
    ],
)
def test_results_omit_fields_json_cannot_hold_and_say_which(field, value, warning, caplog):
    with caplog.at_level(logging.WARNING, logger="openmmqmmm.results"):
        restored = _round_trip(Results(label="kept", **{field: value}))
    assert restored.label == "kept"
    assert getattr(restored, field) is None
    assert warning in caplog.text


def test_results_file_must_hold_a_json_object():
    Path("list.json").write_text("[1, 2]")
    with pytest.raises(FileFormatError, match="JSON object, not list"):
        read_results_from_file("list.json")


class _RecordingTheory:
    def __init__(self):
        self.theorytype = "QM"
        self.theorynamelabel = "Recording"
        self.numcores = 1
        self.calls = []

    def set_numcores(self, numcores):
        self.numcores = numcores

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return 0.0


def test_numgrad_rejects_non_numeric_coordinates_and_hessian_requests():
    numgrad = NumGrad(theory=_RecordingTheory())
    with pytest.raises(InputError, match="numeric N x 3"):
        numgrad.run(current_coords=[["a", "b", "c"]], elems=["H"])
    with pytest.raises(InputError, match="does not compute Hessians"):
        numgrad.run(current_coords=[[0, 0, 0]], elems=["H"], hessian=True)


def test_numgrad_forwards_point_charge_options_only_when_given(caplog):
    theory = _RecordingTheory()
    numgrad = NumGrad(theory=theory)
    mm_coords = np.array([[5.0, 0, 0]])

    numgrad.run(
        current_coords=[[0, 0, 0]], qm_elems=["H"], pc=True, current_mm_coords=mm_coords, mm_charges=[-0.5], numcores=2
    )

    forwarded = theory.calls[0]
    assert forwarded["pc"] is True
    assert forwarded["numcores"] == 2
    assert forwarded["mm_charges"] == [-0.5]
    np.testing.assert_array_equal(forwarded["current_mm_coords"], mm_coords)
    assert forwarded["qm_elems"] == ["H"]
    with caplog.at_level(logging.DEBUG, logger="openmmqmmm.numgrad"):
        numgrad.cleanup()
    assert "nothing to remove" in caplog.text


def test_pygrep2_can_echo_its_matches_to_the_log(caplog):
    Path("notes.txt").write_text("alpha 1\nbeta 2\nalpha 3\n")
    with caplog.at_level(logging.INFO, logger="openmmqmmm.utils"):
        matches = pygrep2("alpha", "notes.txt", print_output=True)
    assert matches == ["alpha 1\n", "alpha 3\n"]
    assert "alpha 1\nalpha 3" in caplog.text


def test_reading_an_absent_integer_list_is_a_file_error():
    with pytest.raises(FileFormatError, match="does not exist"):
        read_intlist_from_file("absent.txt")


def test_ase_calculator_requires_a_qmmm_theory_and_finite_positions():
    with pytest.raises(CalculatorSetupError, match=r"must be an openmmqmmm\.QMMMTheory"):
        OpenMMQMMMCalculator(ZeroTheory())

    calculator = OpenMMQMMMCalculator(StubQMMMTheory())
    atoms = Atoms("H2", positions=[[0, 0, 0], [np.nan, 0, 0]])
    with pytest.raises(CalculatorSetupError, match="finite"):
        calculator.calculate(atoms, ["energy"], [])
    with pytest.raises(PropertyNotImplementedError, match="stress"):
        calculator.calculate(Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]]), ["stress"], [])
    with pytest.raises(CalculatorSetupError, match="Atoms object is required"):
        OpenMMQMMMCalculator(StubQMMMTheory()).calculate(None, ["energy"], [])


def test_ase_calculator_computes_energy_and_forces_by_default():
    theory = StubQMMMTheory()
    calculator = OpenMMQMMMCalculator(theory)

    calculator.calculate(Atoms("H2", positions=[[0, 0, 0], [0.7, 0, 0]]))

    assert set(calculator.results) == {"energy", "forces"}
    assert theory.calls[0]["grad"] is True


def test_unknown_virtual_site_types_and_cyclic_parents_are_rejected():
    with pytest.raises(InputError, match="Unsupported OpenMM virtual site: SimpleNamespace"):
        _copy_site(SimpleNamespace(getNumParticles=lambda: 0, getParticle=lambda i: i))

    sites = NativeVirtualSites.__new__(NativeVirtualSites)
    sites.indices = [0, 1]
    sites.parents = {0: 1, 1: 0}
    with pytest.raises(InputError, match="cyclic parent"):
        sites.seed(np.zeros((2, 3)))


def test_modeller_from_topology_rejects_non_finite_coordinates():
    topology = openmm.app.Topology()
    residue = topology.addResidue("X", topology.addChain())
    topology.addAtom("H", openmm.app.Element.getBySymbol("H"), residue)
    with pytest.raises(InputError, match="finite"):
        modeller_from_topology(topology=topology, coords_angstrom=[[np.nan, 0, 0]])


def test_plumed_driver_requires_an_input_string():
    fragment = Fragment(elems=["He"], coords=[[0, 0, 0]], conncalc=False)
    with pytest.raises(InputError, match="plumed_input_string is required"):
        openmm_md_plumed(fragment=fragment, theory=dummy_mm(fragment), simulation_steps=1)


def test_cif_writer_derives_cell_parameters_from_vectors_and_needs_a_cell():
    cell = np.array([[10.0, 0, 0], [1.0, 10.0, 0], [0.5, 0.5, 10.0]])
    write_cif_file(np.array([[0.0, 0, 0], [1.0, 0, 0]]), ["O", "H"], cellvectors=cell, filename="cell.cif")

    text = Path("cell.cif").read_text()
    a, _b, _c, _alpha, _beta, gamma = cell_vectors_to_params(cell)
    parameters = {line.split()[0]: float(line.split()[1]) for line in text.splitlines() if line.startswith("_cell_")}
    assert parameters["_cell_length_a"] == pytest.approx(a)
    assert parameters["_cell_length_b"] == pytest.approx(np.linalg.norm(cell[1]))
    assert parameters["_cell_angle_gamma"] == pytest.approx(gamma)
    with pytest.raises(InputError, match="cellvectors or celldimensions"):
        write_cif_file(np.zeros((1, 3)), ["O"], filename="none.cif")


def test_standard_orientation_makes_the_cell_lower_triangular_and_keeps_geometry():
    angle = 0.9
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    cell = np.array([[10.0, 0, 0], [1.0, 10.0, 0], [0.5, 0.5, 10.0]]) @ rotation
    coords = np.array([[0.0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]) @ rotation

    aligned_coords, aligned_cell = align_to_standard_orientation(coords, cell)

    np.testing.assert_allclose(np.triu(aligned_cell, 1), 0, atol=1e-12)
    assert np.all(np.diag(aligned_cell) > 0)
    np.testing.assert_allclose(cell_vectors_to_params(aligned_cell), cell_vectors_to_params(cell))
    original_distances = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.testing.assert_allclose(
        np.linalg.norm(aligned_coords[:, None] - aligned_coords[None], axis=-1), original_distances
    )
    np.testing.assert_allclose(coords @ np.linalg.inv(cell), aligned_coords @ np.linalg.inv(aligned_cell), atol=1e-12)
