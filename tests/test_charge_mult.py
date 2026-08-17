from types import SimpleNamespace

import numpy as np
import pytest

from openmmqmmm import (
    Fragment,
    OpenMMTheory,
    QMMMTheory,
    ZeroTheory,
    optimize_geometry,
    single_point,
    single_point_theories,
)
from openmmqmmm.coords import _qm_region_owner, check_charge_mult
from openmmqmmm.exceptions import InputError, InternalError
from openmmqmmm.numgrad import NumGrad
from openmmqmmm.parallel import worker_par


def test_chargemult():
    fragcoords = """
    H 0.0 0.0 0.0
    F 0.0 0.0 1.0
    """
    HF_frag = Fragment(coordsstring=fragcoords)
    HF_frag2 = Fragment(coordsstring=fragcoords, charge=0, mult=1)

    assert HF_frag.charge is None, "Charge is not None"
    assert HF_frag2.charge == 0, "Charge is not 0"


def _bare_qmmm(
    *,
    qm_charge=None,
    qm_mult=None,
    qmatoms=(0,),
    natoms=3,
    fragment_charge=None,
    fragment_mult=None,
    charges=None,
    linkatoms=False,
):
    """A QMMMTheory carrying only the attributes the charge resolver and diagnostic read."""
    theory = QMMMTheory.__new__(QMMMTheory)
    theory.theorytype = "QM/MM"
    theory.qm_charge = qm_charge
    theory.qm_mult = qm_mult
    theory.qmatoms = sorted(qmatoms)
    theory.allatoms = list(range(natoms))
    theory.num_allatoms = natoms
    theory.fragment = SimpleNamespace(charge=fragment_charge, mult=fragment_mult)
    theory.charges = [0.0] * natoms if charges is None else charges
    theory.linkatoms = linkatoms
    return theory


class _RecordingQM:
    """QM stand-in recording the charge/mult it was run with."""

    def __init__(self):
        self.numcores = 1
        self.theorytype = "QM"
        self.theorynamelabel = "RecordingQM"
        self.label = "recording"
        self.calls = []

    def cleanup(self):
        pass

    def run(self, *, elems=None, qm_elems=None, grad=False, charge=None, mult=None, **_kwargs):
        self.calls.append({"charge": charge, "mult": mult})
        numatoms = len(elems if qm_elems is None else qm_elems)
        if grad:
            return 0.0, np.zeros((numatoms, 3))
        return 0.0


def _subregion_qmmm(qm_theory=None, **kwargs):
    """A real QM/MM object over two well-separated H atoms: QM region is atom 0, no link atoms."""
    fragment = Fragment(elems=["H", "H"], coords=[[1.0, 0, 0], [5.0, 0, 0]], charge=0, mult=1, conncalc=False)
    mm_theory = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    qmmm = QMMMTheory(
        fragment=fragment,
        qm_theory=ZeroTheory() if qm_theory is None else qm_theory,
        mm_theory=mm_theory,
        qmatoms=[0],
        embedding="elstat",
        dipole_correction=False,
        **kwargs,
    )
    return qmmm, fragment


# Resolver precedence


def test_qm_charge_wins_over_the_fragment_charge():
    theory = _bare_qmmm(qm_charge=-1, qm_mult=1, fragment_charge=0, fragment_mult=1)

    assert theory.resolve_qm_charge_mult() == (-1, 1)


def test_supplied_charge_is_used_when_the_theory_has_none():
    theory = _bare_qmmm(fragment_charge=0, fragment_mult=1)

    assert theory.resolve_qm_charge_mult(charge=-2, mult=3) == (-2, 3)


def test_conflicting_charge_is_rejected():
    theory = _bare_qmmm(qm_charge=-1, qm_mult=1)

    with pytest.raises(InputError, match="conflicts"):
        theory.resolve_qm_charge_mult(charge=0, mult=1)


def test_matching_charge_is_not_a_conflict():
    """A single job resolves several times on its way down to run(); repeats must not raise."""
    theory = _bare_qmmm(qm_charge=-1, qm_mult=1)

    assert theory.resolve_qm_charge_mult(charge=-1, mult=1) == (-1, 1)


def test_whole_system_qm_may_take_the_fragment_charge():
    theory = _bare_qmmm(qmatoms=(0, 1, 2), natoms=3, fragment_charge=-1, fragment_mult=2)

    assert theory.resolve_qm_charge_mult() == (-1, 2)


def test_subregion_qm_rejects_the_fragment_charge():
    """The whole-system charge is not the QM-region charge, so guessing it is not allowed."""
    theory = _bare_qmmm(qmatoms=(0,), natoms=3, fragment_charge=-1, fragment_mult=1)

    with pytest.raises(InputError, match="qm_charge"):
        theory.resolve_qm_charge_mult()


def test_undefined_charge_names_the_keyword_that_fixes_it():
    theory = _bare_qmmm()

    with pytest.raises(InputError, match="qm_charge"):
        theory.resolve_qm_charge_mult()


def test_charge_and_mult_resolve_independently():
    """qm_charge alone must not send mult off to the full-system fragment."""
    theory = _bare_qmmm(qm_charge=-1, qm_mult=None, fragment_charge=0, fragment_mult=1)

    assert theory.resolve_qm_charge_mult(mult=3) == (-1, 3)


# check_charge_mult dispatch


def test_check_charge_mult_delegates_qmmm_to_the_theory():
    theory = _bare_qmmm(qm_charge=-1, qm_mult=1)
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)

    assert check_charge_mult(None, None, theory.theorytype, fragment, "test", theory=theory) == (-1, 1)


def test_check_charge_mult_looks_through_a_numgrad_wrapper():
    """NumGrad reports theorytype 'QM', so dispatch cannot depend on that string."""
    theory = _bare_qmmm(qm_charge=-1, qm_mult=1)
    wrapper = NumGrad(theory=theory)
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)

    assert wrapper.theorytype == "QM"
    assert check_charge_mult(None, None, wrapper.theorytype, fragment, "test", theory=wrapper) == (-1, 1)


def test_check_charge_mult_rejects_a_wrapped_subregion_without_qm_charge():
    theory = _bare_qmmm(qmatoms=(0,), natoms=2, fragment_charge=0, fragment_mult=1)
    wrapper = NumGrad(theory=theory)
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)

    with pytest.raises(InputError, match="qm_charge"):
        check_charge_mult(None, None, wrapper.theorytype, fragment, "test", theory=wrapper)


def test_check_charge_mult_still_takes_a_plain_qm_charge_from_the_fragment():
    """A plain QM theory treats the whole fragment as its system, so this fallback stays."""
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)

    assert check_charge_mult(None, None, "QM", fragment, "test", theory=ZeroTheory()) == (0, 1)


def test_check_charge_mult_resets_mm_charge_to_none():
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)

    assert check_charge_mult(0, 1, "MM", fragment, "test", theory=SimpleNamespace(theorytype="MM")) == (None, None)


def test_check_charge_mult_rejects_a_qmmm_theory_without_a_resolver():
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)

    with pytest.raises(InternalError, match="resolve_qm_charge_mult"):
        check_charge_mult(None, None, "QM/MM", fragment, "test", theory=SimpleNamespace(theorytype="QM/MM"))


def test_qm_region_owner_finds_the_theory_owning_the_qm_region():
    theory = _bare_qmmm(qm_charge=0, qm_mult=1)

    assert _qm_region_owner(theory) is theory
    assert _qm_region_owner(NumGrad(theory=theory)) is theory
    assert _qm_region_owner(ZeroTheory()) is None
    assert _qm_region_owner(None) is None


# End-to-end through the job drivers


def test_subregion_qmmm_single_point_requires_qm_charge():
    """The whole-system charge must never reach the QM code as the QM-region charge."""
    qmmm, fragment = _subregion_qmmm()

    assert fragment.charge == 0, "The full-system fragment is the tempting wrong source"
    with pytest.raises(InputError, match="qm_charge"):
        single_point(theory=qmmm, fragment=fragment, result_write_to_disk=False)


def test_subregion_qmmm_single_point_accepts_an_explicit_charge():
    qmmm, fragment = _subregion_qmmm()

    result = single_point(theory=qmmm, fragment=fragment, charge=0, mult=1, result_write_to_disk=False)

    assert result.charge == 0


def test_results_charge_matches_the_charge_sent_to_the_qm_theory():
    qm_theory = _RecordingQM()
    qmmm, fragment = _subregion_qmmm(qm_theory=qm_theory, qm_charge=-1, qm_mult=1)

    result = single_point(theory=qmmm, fragment=fragment, result_write_to_disk=False)

    assert result.charge == -1
    assert qm_theory.calls[0]["charge"] == -1, "Results must record the charge the QM code actually received"


def test_charge_conflicting_with_qm_charge_is_rejected_by_single_point():
    qmmm, fragment = _subregion_qmmm(qm_charge=-1, qm_mult=1)

    with pytest.raises(InputError, match="conflicts"):
        single_point(theory=qmmm, fragment=fragment, charge=0, mult=1, result_write_to_disk=False)


def test_optimizer_does_not_stamp_the_qm_region_charge_on_the_fragment():
    qmmm, fragment = _subregion_qmmm(qm_charge=-1, qm_mult=1)

    optimize_geometry(theory=qmmm, fragment=fragment, maxiter=1)

    assert fragment.charge == 0, "The QM-region charge must not be written onto the full-system fragment"
    assert fragment.mult == 1


def test_optimizer_still_stamps_a_plain_qm_charge_on_the_fragment():
    """The stamp is what makes the optimized xyz re-readable with readchargemult=True."""
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 0.9]])

    optimize_geometry(theory=ZeroTheory(), fragment=fragment, charge=0, mult=1)

    assert (fragment.charge, fragment.mult) == (0, 1)


class _StandInQMMM:
    """Duck-typed QM/MM theory: capability dispatch must not require the real class."""

    def __init__(self, charge, mult):
        self.theorytype = "QM/MM"
        self.theorynamelabel = "StandInQMMM"
        self.numcores = 1
        self.label = "standin"
        self.charge = charge
        self.mult = mult
        self.calls = []
        self.QMenergy = 0.0
        self.MMenergy = 0.0
        self.QM_MM_energy = 0.0

    def resolve_qm_charge_mult(self, *, charge=None, mult=None):
        return self.charge, self.mult

    def cleanup(self):
        pass

    def run(self, *, elems=None, grad=False, charge=None, mult=None, **_kwargs):
        self.calls.append({"charge": charge, "mult": mult})
        if grad:
            return 0.0, np.zeros((len(elems), 3))
        return 0.0


def test_single_point_theories_does_not_carry_one_theorys_charge_into_the_next():
    """A QM/MM region charge resolved for one theory must not be reused for the next."""
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1)
    qmmm = _StandInQMMM(-1, 1)
    plain_qm = _RecordingQM()

    single_point_theories(theories=[qmmm, plain_qm], fragment=fragment)

    assert qmmm.calls[0]["charge"] == -1
    assert plain_qm.calls[0]["charge"] == 0, "The plain QM theory runs the whole fragment, so charge 0"


def test_worker_par_reads_charge_from_a_fragment_file(tmp_path):
    fragment = Fragment(elems=["H", "H"], coords=[[0, 0, 0], [0, 0, 1]], charge=0, mult=1, label="frag0")
    fragment.print_system(filename="frag0.frag")

    label, energy, _worker_dirname, _properties = worker_par(
        fragmentfile="frag0.frag", theory=ZeroTheory(), label="frag0"
    )

    assert (label, energy) == ("frag0", 0.0)


def test_worker_par_requires_a_fragment_or_a_fragmentfile():
    with pytest.raises(InputError, match="fragment"):
        worker_par(theory=ZeroTheory(), label="frag0")


# MM charge-sum diagnostic


def test_qm_charge_mismatching_the_mm_charge_sum_warns(caplog):
    theory = _bare_qmmm(qmatoms=(0, 1), natoms=3, charges=[-0.5, -0.5, 1.0])

    with caplog.at_level("WARNING"):
        theory._log_qm_charge_consistency(0)

    assert "sum to -1.0000" in caplog.text


def test_qm_charge_matching_the_mm_charge_sum_is_quiet(caplog):
    theory = _bare_qmmm(qmatoms=(0, 1), natoms=3, charges=[-0.5, -0.5, 1.0])

    with caplog.at_level("WARNING"):
        theory._log_qm_charge_consistency(-1)

    assert caplog.text == ""


def test_non_integer_mm_charge_sum_is_reported(caplog):
    theory = _bare_qmmm(qmatoms=(0,), natoms=3, charges=[-0.4, -0.6, 1.0])

    with caplog.at_level("WARNING"):
        theory._log_qm_charge_consistency(0)

    assert "not an integer" in caplog.text


def test_link_atom_region_reports_the_sum_without_a_verdict(caplog):
    """Charge shifting moves charge across a covalent boundary by design."""
    theory = _bare_qmmm(qmatoms=(0, 1), natoms=3, charges=[-0.5, -0.5, 1.0], linkatoms=True)

    with caplog.at_level("INFO"):
        theory._log_qm_charge_consistency(0)

    assert "expected" in caplog.text
    assert not [record for record in caplog.records if record.levelname == "WARNING"]
