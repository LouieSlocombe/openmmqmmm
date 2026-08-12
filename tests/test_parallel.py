"""Regression tests for the parallel-execution path.

job_parallel had no test coverage at all, which is how a renamed keyword argument
(Grad -> grad) shipped in 1.0.0 and made every worker raise TypeError. These tests use
ZeroTheory so they run everywhere, with no ORCA or OpenMM setup required.
"""

import numpy as np
import pytest

from openmmqmmm import Fragment, ZeroTheory, job_parallel
from openmmqmmm.exceptions import InputError

FRAGCOORDS = """
H 0.0 0.0 0.0
F 0.0 0.0 {bondlength}
"""


def _make_fragments(n=4):
    """A few labelled two-atom fragments, differing only in bond length."""
    return [
        Fragment(coordsstring=FRAGCOORDS.format(bondlength=0.9 + 0.1 * i), charge=0, mult=1, label=f"frag{i}")
        for i in range(n)
    ]


def test_job_parallel_energies():
    fragments = _make_fragments()

    result = job_parallel(fragments=fragments, theories=[ZeroTheory()], numcores=2)

    assert len(result.energies) == len(fragments), "One energy per fragment expected"
    assert set(result.energies_dict) == {f.label for f in fragments}
    assert all(e == 0.0 for e in result.energies_dict.values()), "ZeroTheory energies should all be 0.0"
    assert set(result.worker_dirnames) == {f.label for f in fragments}


def test_job_parallel_gradients():
    fragments = _make_fragments(3)

    result = job_parallel(fragments=fragments, theories=[ZeroTheory()], numcores=2, grad=True)

    assert len(result.energies) == len(fragments)
    assert set(result.gradients_dict) == {f.label for f in fragments}
    for fragment in fragments:
        gradient = result.gradients_dict[fragment.label]
        assert np.shape(gradient) == (fragment.numatoms, 3), "Gradient should be one row per atom"
        assert np.allclose(gradient, 0.0), "ZeroTheory gradients should all be zero"


def test_job_parallel_requires_theories():
    """Missing arguments must raise InputError, not TypeError from indexing None."""
    with pytest.raises(InputError):
        job_parallel(fragments=_make_fragments(1), theories=None, numcores=2)

    with pytest.raises(InputError):
        job_parallel(fragments=None, fragmentfiles=None, theories=[ZeroTheory()], numcores=2)
