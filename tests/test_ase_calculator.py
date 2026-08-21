from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import CalculationFailed, CalculatorSetupError
from ase.optimize import BFGS

from openmmqmmm import Fragment, OpenMMQMMMCalculator, OpenMMTheory, QMMMTheory, ZeroTheory

TEST_DIR = Path(__file__).parent


class StubQMMMTheory(QMMMTheory):
    def __init__(self, symbols=("H", "H"), *, charge=0, mult=1):
        self.elems = list(symbols)
        self.coords = np.zeros((len(symbols), 3))
        self.fragment = SimpleNamespace(coords=self.coords.copy(), elems=list(symbols), charge=None, mult=None)
        self.qm_charge = charge
        self.qm_mult = mult
        self.calls = []
        self.QMenergy = None
        self.MMenergy = None
        self.QM_MM_energy = None

    def run(self, *, current_coords, elems, grad, label, charge, mult):
        self.calls.append(
            {
                "coords": np.asarray(current_coords).copy(),
                "elems": list(elems),
                "grad": grad,
                "label": label,
                "charge": charge,
                "mult": mult,
                "directory": Path.cwd(),
            }
        )
        self.QMenergy = 0.25
        self.MMenergy = 1.0
        self.QM_MM_energy = 1.25
        if grad:
            return self.QM_MM_energy, np.full_like(current_coords, 0.5, dtype=float)
        return self.QM_MM_energy


def test_energy_and_force_units_and_sign_are_converted(tmp_path):
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory, directory=tmp_path)

    forces = atoms.get_forces()

    assert atoms.get_potential_energy() == pytest.approx(1.25 * units.Hartree)
    assert forces == pytest.approx(np.full((2, 3), -0.5 * units.Hartree / units.Bohr))
    assert len(theory.calls) == 1, "ASE should reuse the energy produced with the force calculation"
    assert theory.calls[0]["directory"] == tmp_path
    assert theory.fragment.coords == pytest.approx(atoms.positions)


def test_energy_only_request_does_not_calculate_a_gradient():
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory)

    assert atoms.get_potential_energy() == pytest.approx(1.25 * units.Hartree)
    assert theory.calls[0]["grad"] is False


def test_calculation_reuses_fixed_element_metadata():
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.get_chemical_symbols = lambda: pytest.fail("chemical symbols were rebuilt")
    atoms.calc = OpenMMQMMMCalculator(theory)

    atoms.get_forces()

    assert theory.calls[0]["elems"] == ["H", "H"]


def test_theory_and_fragment_share_one_coordinate_snapshot():
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory)

    atoms.get_forces()

    assert theory.coords is theory.fragment.coords
    assert theory.coords == pytest.approx(atoms.positions)


def test_theory_electronic_state_is_forwarded():
    theory = StubQMMMTheory(charge=-1, mult=2)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory)

    atoms.get_potential_energy()

    assert theory.calls[0]["charge"] == -1
    assert theory.calls[0]["mult"] == 2


def test_explicit_electronic_state_is_used_when_theory_has_none():
    theory = StubQMMMTheory(charge=None, mult=None)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory, charge=1, mult=2)

    atoms.get_potential_energy()

    assert theory.calls[0]["charge"] == 1
    assert theory.calls[0]["mult"] == 2


def test_conflicting_electronic_state_is_rejected():
    theory = StubQMMMTheory(charge=-1, mult=2)

    with pytest.raises(CalculatorSetupError, match="conflicts"):
        OpenMMQMMMCalculator(theory, charge=0)


def test_changed_atom_identity_is_rejected():
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory)
    atoms.get_potential_energy()
    atoms.numbers[1] = 8

    with pytest.raises(CalculatorSetupError, match="atom count, elements, and ordering"):
        atoms.get_potential_energy()


def test_changed_cell_is_rejected():
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]], cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = OpenMMQMMMCalculator(theory)
    atoms.get_potential_energy()
    atoms.set_cell([11.0, 10.0, 10.0])

    with pytest.raises(CalculatorSetupError, match="Changing the ASE cell"):
        atoms.get_potential_energy()


def _swap_element(atoms):
    atoms.numbers[1] = 8


def _resize_cell(atoms):
    atoms.set_cell([11.0, 10.0, 10.0])


def _drop_periodicity(atoms):
    atoms.set_pbc([True, True, False])


@pytest.mark.parametrize("mutate", [_swap_element, _resize_cell, _drop_periodicity])
def test_rejected_structure_stays_rejected_when_only_positions_change(mutate):
    """ASE caches the Atoms before the calculator validates them, so a rejected system must not become the baseline."""
    theory = StubQMMMTheory()
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]], cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = OpenMMQMMMCalculator(theory)
    atoms.get_potential_energy()
    mutate(atoms)

    with pytest.raises(CalculatorSetupError):
        atoms.get_potential_energy()

    atoms.positions[1, 2] += 1e-3
    with pytest.raises(CalculatorSetupError):
        atoms.get_potential_energy()

    assert len(theory.calls) == 1, "the theory must never run against a rejected system"


@pytest.mark.parametrize(
    ("energy", "gradient"),
    [(np.nan, np.zeros((2, 3))), (0.0, np.full((2, 3), np.nan)), (0.0, np.zeros((1, 3)))],
)
def test_invalid_backend_results_are_rejected(energy, gradient):
    theory = StubQMMMTheory()
    theory.run = lambda **_kwargs: (energy, gradient)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
    atoms.calc = OpenMMQMMMCalculator(theory)

    with pytest.raises(CalculationFailed):
        atoms.get_forces()


def test_ase_optimizer_can_drive_the_calculator():
    class HarmonicQMMMTheory(StubQMMMTheory):
        def __init__(self):
            super().__init__(("H",))

        def run(self, *, current_coords, elems, grad, label, charge, mult):
            coords_bohr = np.asarray(current_coords) / units.Bohr
            energy = 0.5 * np.sum(coords_bohr**2)
            if grad:
                return energy, coords_bohr
            return energy

    theory = HarmonicQMMMTheory()
    atoms = Atoms("H", positions=[[0.4, 0.0, 0.0]])
    atoms.calc = OpenMMQMMMCalculator(theory)

    BFGS(atoms, logfile=None).run(fmax=1e-3, steps=20)

    assert np.linalg.norm(atoms.positions) < 1e-4


def test_calculator_runs_a_real_openmm_qmmm_theory():
    fragment = Fragment(xyzfile=f"{TEST_DIR}/xyzfiles/h2o_MeOH.xyz")
    fragment.write_pdbfile_openmm(filename="h2o_MeOH.pdb", skip_connectivity=True)
    mm_theory = OpenMMTheory(
        xmlfiles=[f"{TEST_DIR}/extra_files/MeOH_H2O-sigma.xml"],
        pdbfile="h2o_MeOH.pdb",
        autoconstraints=None,
        rigidwater=False,
    )
    qmmm_theory = QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory(),
        mm_theory=mm_theory,
        qmatoms=[3, 4, 5, 6, 7, 8],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
    )
    atoms = Atoms(fragment.elems, positions=fragment.coords)
    atoms.calc = OpenMMQMMMCalculator(qmmm_theory)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (fragment.numatoms, 3)
    assert np.all(np.isfinite(forces))
    assert energy == pytest.approx(qmmm_theory.QM_MM_energy * units.Hartree)
    assert forces == pytest.approx(-qmmm_theory.QM_MM_gradient * units.Hartree / units.Bohr)
