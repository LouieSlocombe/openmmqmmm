"""geomeTRIC adapter: periodic-cell optimization steps, bond orders, active-region evaluation and option wiring."""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import _make_analytic_qmmm

from openmmqmmm import Fragment, ZeroTheory, optimize_geometry
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.coords_pbc import cell_vectors_to_params
from openmmqmmm.exceptions import FileFormatError, InputError
from openmmqmmm.geometric import GeometricArgs, GeometricEngine, GeometricOptimizer
from openmmqmmm.numgrad import NumGrad

GEOMETRIC_LOGGER = "openmmqmmm.geometric"
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
LOWER_CELL = np.array([[10.0, 0.0, 0.0], [1.0, 10.0, 0.0], [0.5, 0.5, 10.0]])


def _water():
    return Fragment(elems=["O", "H", "H"], coords=WATER.copy(), charge=0, mult=1)


class _PeriodicTheory:
    """Zero-energy periodic theory recording the cells and geometries it is handed."""

    def __init__(self, cell, gradient=None, cell_gradient=None):
        self.periodic = True
        self.periodic_cell_vectors = np.array(cell, dtype=float)
        self.theorytype = "QM"
        self.numcores = 1
        self.theorynamelabel = "PeriodicZero"
        self.gradient = gradient
        self.cell_gradient = np.zeros((3, 3)) if cell_gradient is None else np.asarray(cell_gradient, dtype=float)
        self.cells = []
        self.geometries = []

    def update_cell(self, vectors):
        self.periodic_cell_vectors = np.array(vectors, dtype=float)
        self.cells.append(self.periodic_cell_vectors.copy())

    def get_cell_gradient(self):
        return self.cell_gradient.copy()

    def run(self, current_coords=None, elems=None, grad=False, charge=None, mult=None, **_kwargs):
        self.geometries.append(np.array(current_coords, dtype=float))
        gradient = np.zeros((len(current_coords), 3)) if self.gradient is None else self.gradient
        return (0.0, gradient) if grad else 0.0


def _bare_engine(theory, fragment, **attributes):
    engine = GeometricEngine.__new__(GeometricEngine)
    engine.__dict__.update(
        theory=theory,
        fragment=fragment,
        active_region=False,
        pbc_active=False,
        full_current_coords=[],
        iteration_count=0,
        maxiter=10,
        energy=0,
        actatoms=[],
        print_atoms_list=fragment.allatoms,
        charge=0,
        mult=1,
        conv_criteria=None,
        BOmatrix=None,
        mm_pdb_traj_write=False,
        M=SimpleNamespace(xyzs=[fragment.coords.copy()], elem=list(fragment.elems)),
    )
    engine.__dict__.update(attributes)
    return engine


def _rotation():
    angle = 0.7
    about_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    about_x = np.array([[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]])
    return about_z @ about_x


def test_pbc_step_maps_geometric_coordinates_through_the_current_cell_and_masks_the_lattice_gradient():
    fragment = _water()
    gradient = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9]])
    cell_gradient = np.arange(1.0, 10.0).reshape(3, 3)
    theory = _PeriodicTheory(LOWER_CELL, gradient=gradient, cell_gradient=cell_gradient)
    engine = _bare_engine(theory, fragment, pbc_active=True, elems_phys=list(fragment.elems), H_ref=LOWER_CELL.copy())
    engine.H_ref_inv = np.linalg.inv(LOWER_CELL)
    Path("geometric_OPTtraj.log").write_text("Step 7: Gradient = 1.0e-3\n")
    origin = np.array([0.3, -0.2, 0.1])
    requested_cell = np.array([[11.0, 0.3, 0.2], [1.0, 11.0, 0.4], [0.5, 0.5, 11.0]])
    geometric_coords = np.vstack([WATER, origin, requested_cell + origin])

    result = engine.pbc_calc(geometric_coords.copy())

    lower_cell = np.tril(requested_cell)
    np.testing.assert_allclose(theory.cells[-1], lower_cell)
    expected_physical = WATER @ np.linalg.inv(LOWER_CELL) @ lower_cell
    np.testing.assert_allclose(theory.geometries[-1], expected_physical)
    np.testing.assert_allclose(engine.full_current_coords, expected_physical)
    assert engine.iteration_count == 7
    assert result["energy"] == 0.0
    transform = np.linalg.inv(LOWER_CELL) @ lower_cell
    expected_gradient = np.vstack([gradient @ transform.T, np.zeros((1, 3)), np.tril(cell_gradient)])
    np.testing.assert_allclose(result["gradient"], expected_gradient.flatten())


def test_pbc_step_with_a_uniformly_scaled_cell_scales_the_physical_geometry():
    fragment = _water()
    theory = _PeriodicTheory(LOWER_CELL)
    engine = _bare_engine(theory, fragment, pbc_active=True, elems_phys=list(fragment.elems), H_ref=LOWER_CELL.copy())
    engine.H_ref_inv = np.linalg.inv(LOWER_CELL)
    Path("geometric_OPTtraj.log").write_text("")

    engine.pbc_calc(np.vstack([WATER, np.zeros(3), 1.1 * LOWER_CELL]))

    np.testing.assert_allclose(theory.geometries[-1], 1.1 * WATER)
    np.testing.assert_allclose(theory.cells[-1], 1.1 * LOWER_CELL)


def test_pbc_bond_orders_connect_bonded_atoms_and_the_lattice_dummies_once():
    fragment = _water()
    engine = _bare_engine(_PeriodicTheory(LOWER_CELL), fragment, pbc_active=True, elems_phys=list(fragment.elems))
    engine.M.elem += ["F"] * 4

    bond_orders = engine.calc_bondorder(None, ".")

    expected = np.zeros((7, 7), dtype=int)
    for i, j in [(0, 1), (0, 2), (3, 4), (3, 5), (3, 6)]:
        expected[i, j] = expected[j, i] = 1
    np.testing.assert_array_equal(bond_orders, expected)
    assert engine.calc_bondorder(None, ".") is bond_orders


def test_bond_orders_are_only_provided_for_periodic_engines():
    engine = _bare_engine(ZeroTheory(), _water())
    assert engine.calc_bondorder(None, ".") is None


def test_periodic_engine_aligns_cell_and_atoms_to_the_standard_orientation_and_adds_lattice_atoms():
    rotation = _rotation()
    rotated_cell = LOWER_CELL @ rotation
    rotated_coords = WATER @ rotation
    fragment = Fragment(elems=["O", "H", "H"], coords=rotated_coords, charge=0, mult=1)
    theory = _PeriodicTheory(rotated_cell)
    molecule = SimpleNamespace(xyzs=[rotated_coords.copy()], elem=["O", "H", "H"])

    engine = GeometricEngine(molecule, theory, fragment=fragment, pbc_active=True)

    np.testing.assert_allclose(engine.H_ref, LOWER_CELL, atol=1e-10)
    np.testing.assert_allclose(theory.periodic_cell_vectors, LOWER_CELL, atol=1e-10)
    np.testing.assert_allclose(cell_vectors_to_params(engine.H_ref), cell_vectors_to_params(rotated_cell))
    np.testing.assert_allclose(fragment.coords, WATER, atol=1e-10)
    assert molecule.elem == ["O", "H", "H", "F", "F", "F", "F"]
    np.testing.assert_allclose(molecule.xyzs[0][:3], WATER, atol=1e-10)
    np.testing.assert_allclose(molecule.xyzs[0][3], 0.0)
    np.testing.assert_allclose(molecule.xyzs[0][4:], LOWER_CELL, atol=1e-10)


@pytest.mark.parametrize(("option", "extension"), [("CIF", "cif"), ("XSF", "xsf"), ("POSCAR", "POSCAR")])
def test_periodic_optimization_writes_the_final_structure_in_the_requested_format(option, extension):
    theory = _PeriodicTheory(LOWER_CELL)

    result = optimize_geometry(theory=theory, fragment=_water(), pbc_format_option=option)

    assert result.energy == 0.0
    assert all(geometry.shape == (3, 3) for geometry in theory.geometries)
    np.testing.assert_allclose(theory.periodic_cell_vectors, LOWER_CELL, atol=1e-6)
    text = Path(f"Fragment-optimized.{extension}").read_text()
    if option == "XSF":
        vectors = [line.split() for line in text.splitlines()[2:5]]
        np.testing.assert_allclose(np.array(vectors, dtype=float), LOWER_CELL, atol=1e-6)
    elif option == "CIF":
        length_a = next(float(line.split()[1]) for line in text.splitlines() if line.startswith("_cell_length_a"))
        assert length_a == pytest.approx(10.0, abs=1e-6)
    else:
        assert "O" in text and "H" in text


def test_active_region_sets_hdlc_unless_the_user_forces_tric(caplog):
    with caplog.at_level(logging.WARNING, logger=GEOMETRIC_LOGGER):
        switched = GeometricOptimizer(actatoms=[0, 1])
    assert switched.active_region is True
    assert switched.coordsystem == "hdlc"
    assert "Switching to HDLC" in caplog.text

    forced = GeometricOptimizer(actatoms=[0, 1], force_coordsystem=True)
    assert forced.coordsystem == "tric"


def test_periodic_theory_activates_pbc_unless_forced_off():
    periodic = GeometricOptimizer(theory=SimpleNamespace(periodic=True), pbc_format_option="poscar")
    assert periodic.pbc_active is True
    assert periodic.coordsystem == "hdlc"
    assert periodic.pbc_format_option == "poscar"

    forced_off = GeometricOptimizer(theory=SimpleNamespace(periodic=True), force_no_pbc=True)
    assert forced_off.pbc_active is False


def _geometric_args(**overrides):
    options = {
        "coordsys": "tric",
        "maxiter": 5,
        "conv_criteria": GeometricOptimizer().conv_criteria,
        "transition": False,
        "hessian": None,
        "subfrctor": 1,
        "verbose": 0,
        "irc": False,
        "rigid": False,
        "enforce_constraints": None,
        "bothre": 0.0,
    }
    options.update(overrides)
    return GeometricArgs(SimpleNamespace(), "constraints.txt", **options)


def test_geometric_args_only_carry_rigid_and_enforce_settings_when_requested():
    plain = _geometric_args()
    assert not hasattr(plain, "conmethod")
    assert not hasattr(plain, "enforce")
    assert plain.constraints == "constraints.txt"
    assert plain.logIni.endswith("/log.ini")

    rigid = _geometric_args(rigid=True, enforce_constraints=0.1)
    assert rigid.conmethod == 1
    assert rigid.enforce == 0.1


def test_optimize_geometry_requires_theory_and_fragment():
    with pytest.raises(InputError, match="requires theory and fragment"):
        optimize_geometry(fragment=_water())


class _StoppedAtGeometricError(Exception):
    pass


def test_num_grad_option_wraps_the_theory_before_geometric_sees_it(monkeypatch):
    import geometric.optimize

    handed = []

    def record(**kwargs):
        handed.append(kwargs)
        raise _StoppedAtGeometricError

    monkeypatch.setattr(geometric.optimize, "run_optimizer", record)
    with pytest.raises(_StoppedAtGeometricError):
        optimize_geometry(theory=ZeroTheory(), fragment=_water(), num_grad=True)

    assert isinstance(handed[0]["customengine"].theory, NumGrad)


def test_a_single_atom_is_evaluated_as_a_single_point_instead_of_optimized():
    result = optimize_geometry(
        theory=ZeroTheory(), fragment=Fragment(elems=["He"], coords=[[0, 0, 0]], charge=0, mult=1)
    )

    assert result.energy == 0.0
    assert not Path("geometric_OPTtraj.log").exists()


def test_a_missing_constraints_input_file_is_reported_as_a_file_error():
    with pytest.raises(FileFormatError, match=r"absent\.txt"):
        optimize_geometry(theory=ZeroTheory(), fragment=_water(), constraintsinputfile="absent.txt")


def test_energy_logfile_writes_a_header_once_and_one_row_per_iteration():
    engine = GeometricEngine.__new__(GeometricEngine)
    engine.theory = SimpleNamespace(QMenergy=-1.5, MMenergy=0.25, QM_MM_energy=-1.25)
    engine.iteration_count = 0
    engine.write_energy_logfile()
    engine.iteration_count = 1
    engine.write_energy_logfile()

    lines = Path("optimization_energies.log").read_text().splitlines()
    assert lines[0].split() == ["Iteration", "QM-energy", "(Eh)", "MM-Energy", "(Eh)", "QM/MM-Energy", "(Eh)"]
    assert [line.split() for line in lines[1:]] == [["0", "-1.5", "0.25", "-1.25"], ["1", "-1.5", "0.25", "-1.25"]]


def test_active_region_step_moves_only_active_atoms_and_writes_every_qmmm_trajectory(caplog):
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = _bare_engine(
        qmmm,
        fragment,
        active_region=True,
        actatoms=[1],
        print_atoms_list=[1],
        charge=None,
        mult=None,
        mm_pdb_traj_write=True,
    )
    Path("geometric_OPTtraj.log").write_text("Step 2: Gradient = 1.0e-3\n")
    moved = np.array([[0.6, 0.0, 0.0]])

    with caplog.at_level(logging.DEBUG, logger=GEOMETRIC_LOGGER):
        result = engine.actregion_calc(moved.copy())

    full_bohr = np.array([[-0.5, 0, 0], [0.6, 0, 0]]) * ANG_TO_BOHR
    assert result["energy"] == pytest.approx(0.5 * 0.01 * np.sum(full_bohr**2))
    np.testing.assert_allclose(result["gradient"], 0.01 * full_bohr[1])
    np.testing.assert_allclose(engine.full_current_coords, [[-0.5, 0, 0], [0.6, 0, 0]])
    assert engine.iteration_count == 2
    assert Path("geometric_OPTtraj_Full.xyz").read_text().splitlines()[0] == "2"
    assert Path("geometric_OPTtraj_QMregion.xyz").read_text().splitlines()[0] == "2"
    assert Path("optimization_energies.log").read_text().startswith("Iteration")
    assert Path("geometric_OPTtraj-PDB.pdb").is_file()
    assert Path("Grad").is_file()
    assert Path("Grad_act").is_file()
