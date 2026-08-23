from pathlib import Path

import numpy as np
import pytest

import openmmqmmm.parallel as parallel_module
from openmmqmmm import Fragment, ZeroTheory, job_parallel
from openmmqmmm.exceptions import InputError, OpenMMQMMMError
from openmmqmmm.parallel import worker_par

FRAGCOORDS = """
H 0.0 0.0 0.0
F 0.0 0.0 {bondlength}
"""


def _make_fragments(n=4):
    return [
        Fragment(coordsstring=FRAGCOORDS.format(bondlength=0.9 + 0.1 * i), charge=0, mult=1, label=f"frag{i}")
        for i in range(n)
    ]


class _FakeAsyncResult:
    def __init__(self, *, value=None, error=None):
        self.value = value
        self.error = error
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        if self.error is not None:
            raise self.error
        return self.value


def _install_fake_pool(monkeypatch, *, failing_label=None):
    pools = []

    class FakePool:
        def __init__(self, numcores):
            self.numcores = numcores
            self.async_results = []
            self.submitted_core_counts = []
            self.closed = False
            self.terminated = False
            self.join_calls = 0
            pools.append(self)

        def apply_async(self, _function, *, kwds):
            self.submitted_core_counts.append(kwds["theory"].numcores)
            label = kwds["label"]
            if label == failing_label:
                result = _FakeAsyncResult(error=InputError("worker exploded"))
            elif kwds["grad"]:
                fragment = kwds["fragment"]
                result = _FakeAsyncResult(value=(label, 0.0, np.zeros((fragment.numatoms, 3)), f"Pooljob_{label}", {}))
            else:
                result = _FakeAsyncResult(value=(label, 0.0, f"Pooljob_{label}", {}))
            self.async_results.append(result)
            return result

        def close(self):
            self.closed = True

        def terminate(self):
            self.terminated = True

        def join(self):
            self.join_calls += 1

    monkeypatch.setattr(parallel_module, "_import_pool", lambda version: FakePool)
    return pools


class _PropertyTheory(ZeroTheory):
    def get_dipole_moment(self):
        return np.array([1.0, 2.0, 3.0])

    def get_polarizability_tensor(self):
        return np.eye(3)


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


def test_async_results_are_collected_once(monkeypatch):
    pools = _install_fake_pool(monkeypatch)

    result = job_parallel(fragments=_make_fragments(3), theories=[ZeroTheory()], numcores=2)

    pool = pools[0]
    assert result.energies == [0.0, 0.0, 0.0]
    assert [async_result.get_calls for async_result in pool.async_results] == [1, 1, 1]
    assert pool.closed is True
    assert pool.terminated is False
    assert pool.join_calls == 1


def test_worker_failure_terminates_pool_chains_error_and_restores_theory_cores(monkeypatch):
    pools = _install_fake_pool(monkeypatch, failing_label="frag1")
    theory = ZeroTheory(numcores=4)

    with pytest.raises(OpenMMQMMMError, match="frag1") as exc_info:
        job_parallel(fragments=_make_fragments(3), theories=[theory], numcores=2)

    pool = pools[0]
    assert isinstance(exc_info.value.__cause__, InputError)
    assert str(exc_info.value.__cause__) == "worker exploded"
    assert pool.submitted_core_counts == [1, 1, 1]
    assert [async_result.get_calls for async_result in pool.async_results] == [1, 1, 0]
    assert pool.closed is True
    assert pool.terminated is True
    assert pool.join_calls == 1
    assert theory.numcores == 4


@pytest.mark.parametrize(
    ("allow_theory_parallelization", "submitted_core_counts"),
    [(False, [1, 1]), (True, [3, 5])],
)
def test_multiple_theories_use_unique_theory_labels_and_restore_cores(
    monkeypatch, allow_theory_parallelization, submitted_core_counts
):
    pools = _install_fake_pool(monkeypatch)
    theories = [ZeroTheory(numcores=3, label="method-a"), ZeroTheory(numcores=5, label="method-b")]

    result = job_parallel(
        fragments=_make_fragments(1),
        theories=theories,
        numcores=2,
        allow_theory_parallelization=allow_theory_parallelization,
    )

    assert set(result.energies_dict) == {"method-a", "method-b"}
    assert pools[0].submitted_core_counts == submitted_core_counts
    assert [theory.numcores for theory in theories] == [3, 5]


def test_worker_keeps_dipole_and_polarizability_properties():
    fragment = _make_fragments(1)[0]

    worker_result = worker_par(fragment=fragment, theory=_PropertyTheory(), label=fragment.label, grad=True)
    properties = worker_result[-1]
    result = parallel_module._assemble_parallel_results([worker_result], grad=True)

    assert np.allclose(properties["dipole_moment"], [1.0, 2.0, 3.0])
    assert np.allclose(properties["polarizability"], np.eye(3))
    assert fragment.label in result.displacement_dipole_dictionary
    assert fragment.label in result.displacement_polarizability_dictionary


def test_fragmentfile_worker_directory_is_a_safe_single_path_component(tmp_path):
    fragment = _make_fragments(1)[0]
    nested_dir = tmp_path / "nested" / "unsafe"
    nested_dir.mkdir(parents=True)
    fragmentfile = nested_dir / "../unsafe fragment.frag"
    fragment.print_system(filename=str(fragmentfile))

    label, _energy, worker_dirname, _properties = worker_par(
        fragmentfile=fragmentfile,
        theory=ZeroTheory(),
        label=fragmentfile,
    )

    assert label == fragmentfile
    assert Path(worker_dirname).parent == Path(".")
    assert "/" not in worker_dirname
    assert "\\" not in worker_dirname
    assert (tmp_path / worker_dirname).is_dir()


def test_fragmentfile_worker_directories_include_theory_labels():
    fragmentfile = Path("shared.frag")

    first = parallel_module._worker_directory_label("method-a", fragmentfile)
    second = parallel_module._worker_directory_label("method-b", fragmentfile)

    assert first != second
    assert first.startswith("shared_frag_method-a")
    assert second.startswith("shared_frag_method-b")


def test_equivalent_string_and_path_labels_cannot_share_a_worker_directory():
    fragmentfile = Path("shared.frag")

    from_string = parallel_module._worker_directory_label("shared.frag", fragmentfile)
    from_path = parallel_module._worker_directory_label(Path("shared.frag"), fragmentfile)

    assert from_string != from_path


def test_distinct_labels_with_the_same_rendered_text_get_distinct_directories():
    assert parallel_module._worker_directory_label((1, "2"), None) != parallel_module._worker_directory_label(
        ("1", 2), None
    )
    assert parallel_module._worker_directory_label(Path("a"), None) != parallel_module._worker_directory_label(
        "a", None
    )


@pytest.mark.parametrize("numcores", [None, 0, -1, 1.5, True])
def test_job_parallel_requires_a_positive_integer_numcores(numcores):
    with pytest.raises(InputError, match="positive integer"):
        job_parallel(fragments=_make_fragments(1), theories=[ZeroTheory()], numcores=numcores)


def test_job_parallel_rejects_empty_or_ambiguous_jobs():
    with pytest.raises(InputError, match="at least one theory"):
        job_parallel(fragments=_make_fragments(1), theories=[], numcores=1)
    with pytest.raises(InputError, match="at least one fragment"):
        job_parallel(fragments=[], theories=[ZeroTheory()], numcores=1)
    with pytest.raises(InputError, match="not both"):
        job_parallel(
            fragments=_make_fragments(1),
            fragmentfiles=["unused.frag"],
            theories=[ZeroTheory()],
            numcores=1,
        )
    with pytest.raises(InputError, match="ambiguous"):
        job_parallel(
            fragments=_make_fragments(2),
            theories=[ZeroTheory(label="a"), ZeroTheory(label="b")],
            numcores=1,
        )
    with pytest.raises(InputError, match="not one bare path"):
        job_parallel(fragmentfiles="fragment.frag", theories=[ZeroTheory()], numcores=1)
    with pytest.raises(InputError, match="not one bare path"):
        job_parallel(fragmentfiles=Path("fragment.frag"), theories=[ZeroTheory()], numcores=1)


def test_job_parallel_rejects_invalid_backend_before_starting_workers():
    with pytest.raises(InputError, match="parallel backend"):
        job_parallel(
            fragments=_make_fragments(1),
            theories=[ZeroTheory()],
            numcores=1,
            version="threads",  # type: ignore[arg-type]
        )


def test_job_parallel_rejects_labels_that_would_overwrite_result_dicts():
    duplicate_fragments = _make_fragments(2)
    duplicate_fragments[1].label = duplicate_fragments[0].label
    with pytest.raises(InputError, match="duplicate label"):
        job_parallel(fragments=duplicate_fragments, theories=[ZeroTheory()], numcores=1)

    with pytest.raises(InputError, match="duplicate label"):
        job_parallel(
            fragments=_make_fragments(1),
            theories=[ZeroTheory(label="same"), ZeroTheory(label="same")],
            numcores=1,
        )
