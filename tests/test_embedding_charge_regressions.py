"""Population-charge updates must preserve charge and never imply missing forces."""

import shlex
from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Fragment, OpenMMTheory, ORCATheory, QMMMTheory
from openmmqmmm.constants import ANG_TO_BOHR
from openmmqmmm.exceptions import InputError
from openmmqmmm.orca import find_orca


class _PopulationQM:
    numcores = 1

    def __init__(self, populations=(0.2, -0.2)):
        self.populations = populations
        self.charges = [9.0, 9.0]  # The explicit accessor takes precedence.
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        return 0.0

    def get_atomic_charges(self):
        return self.populations


class _LegacyPopulationQM:
    numcores = 1

    def __init__(self):
        self.charges = []

    def run(self, **kwargs):
        self.charges = [0.2, -0.2]
        return 0.0


def _qmmm(qm, *, water=False, capped=False):
    coords = [[0, 0, 0], [0.75, 0, 0], [8, 1, 0]]
    elems = ["H", "H", "He"]
    qmatoms = [0, 1]
    if water:
        coords = [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0], [8, 1, 0]]
        elems = ["O", "H", "H", "He"]
        qmatoms = [0, 1, 2]
    elif capped:
        coords = [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]]
        elems = ["C", "C", "C"]
        qmatoms = [0]
    fragment = Fragment(elems=elems, coords=coords, conncalc=False)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    if capped:
        atoms = list(mm.topology.atoms())
        mm.topology.addBond(atoms[0], atoms[1])
        mm.topology.addBond(atoms[1], atoms[2])
    mm.update_charges(list(range(len(coords))), [0.0] * (len(coords) - 1) + [0.5])
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=qmatoms,
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
        update_qm_region_charges=True,
    )
    return theory, fragment


@pytest.mark.parametrize("qm", [_PopulationQM(), _LegacyPopulationQM()])
def test_energy_only_updates_use_accessor_or_legacy_charges(qm):
    theory, fragment = _qmmm(qm)
    energy = theory.run(current_coords=fragment.coords)

    assert theory.charges == pytest.approx([0.2, -0.2, 0.5])
    assert theory.mm_theory.charges == pytest.approx(theory.charges)
    distances = np.linalg.norm(fragment.coords[:2] - fragment.coords[2], axis=1) * ANG_TO_BOHR
    expected = np.sum(np.array([0.2, -0.2]) * 0.5 / distances)
    assert energy == pytest.approx(expected, abs=1e-10)


def test_population_charge_gradients_fail_before_qm_evaluation():
    qm = _PopulationQM()
    theory, fragment = _qmmm(qm)

    with pytest.raises(InputError, match="population-charge response derivatives"):
        theory.run(current_coords=fragment.coords, grad=True)
    assert qm.calls == 0
    assert theory.runcalls == 0


def test_charge_update_backend_contract_is_checked_before_qm_evaluation():
    class UnsupportedQM:
        numcores = 1

        def run(self, **kwargs):
            pytest.fail("An unsupported population backend must not launch QM")

    theory, fragment = _qmmm(UnsupportedQM())
    with pytest.raises(InputError, match=r"get_atomic_charges\(\)"):
        theory.run(current_coords=fragment.coords)


def test_cap_populations_are_not_silently_discarded():
    qm = _PopulationQM()
    theory, fragment = _qmmm(qm, capped=True)

    with pytest.raises(InputError, match="charge-conserving mapping"):
        theory.run(current_coords=fragment.coords)
    assert qm.calls == 0


@pytest.mark.parametrize(
    "populations",
    [[], [0.2], [[0.2, -0.2]], [np.nan, 0.0], [0.0, np.inf], [0.2, -0.1], ["invalid", 0.0]],
)
def test_invalid_populations_fail_without_mutating_mm_charges(populations):
    theory, fragment = _qmmm(_PopulationQM(populations))
    initial = theory.charges.copy()

    with pytest.raises(InputError, match="QM atomic charges"):
        theory.run(current_coords=fragment.coords)
    assert theory.charges == initial
    assert theory.mm_theory.charges == initial


def test_population_net_charge_check_allows_output_rounding():
    theory, fragment = _qmmm(_PopulationQM([0.123456, -0.123455]))
    assert np.isfinite(theory.run(current_coords=fragment.coords))


def test_orca_population_updates_work_without_population_logging(fake_orca_dir, tmp_path):
    output = tmp_path / "populations.out"
    output.write_text((Path(__file__).parent / "orca_outputs" / "h2o_engrad.out").read_text())
    # Exercise ORCATheory.run and its actual subprocess/output-parser path.
    (fake_orca_dir / "orca").write_text(
        "#!/bin/sh\n"
        'if [ "$#" -eq 0 ]; then\n'
        "  echo 'This program requires the name of a parameterfile'\n"
        "  exit 2\n"
        "fi\n"
        f"cat {shlex.quote(str(output))}\n"
    )
    qm = ORCATheory(orcasimpleinput="! HF STO-3G", print_population_analysis=False)
    theory, fragment = _qmmm(qm, water=True)
    theory.run(current_coords=fragment.coords)

    assert theory.charges == pytest.approx([-0.347486, 0.173743, 0.173743, 0.5])
    assert qm.properties["Mulliken_charges"] == pytest.approx(theory.charges[:3])

    output.write_text(output.read_text().replace("-0.347486", "-0.200000").replace("0.173743", "0.100000"))
    theory.run(current_coords=fragment.coords)
    assert theory.charges == pytest.approx([-0.2, 0.1, 0.1, 0.5])
    assert qm.runcalls == 2


@pytest.mark.usefixtures("fake_orca_dir")
def test_orca_accessor_rejects_missing_population_table(tmp_path):
    qm = ORCATheory(orcasimpleinput="! HF STO-3G")
    with pytest.raises(InputError, match="No readable ORCA output"):
        qm.get_atomic_charges()
    (tmp_path / "orca.out").write_text("ORCA TERMINATED NORMALLY\n")
    with pytest.raises(InputError, match="finite Mulliken atomic charges"):
        qm.get_atomic_charges()


@pytest.mark.skipif(find_orca(required=False) is None, reason="No working ORCA installation found")
def test_real_orca_energy_only_population_update():
    qm = ORCATheory(orcasimpleinput="! HF STO-3G TightSCF", print_population_analysis=False)
    theory, fragment = _qmmm(qm)

    energy = theory.run(current_coords=fragment.coords)

    assert np.isfinite(energy)
    assert theory.charges == pytest.approx([0.0, 0.0, 0.5], abs=1e-6)
    assert qm.get_atomic_charges() == pytest.approx([0.0, 0.0], abs=1e-6)
