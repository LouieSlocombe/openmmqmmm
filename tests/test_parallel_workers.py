"""worker_par and job_parallel error paths, the OpenMPI probe and the pool plumbing of openmmqmmm.parallel."""

import logging
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from conftest import labelled_fragments, make_subregion_qmmm

import openmmqmmm.parallel as parallel_module
from openmmqmmm import ZeroTheory, job_parallel
from openmmqmmm.exceptions import ExternalProgramError, InputError, OpenMMQMMMError
from openmmqmmm.parallel import worker_par

WORKER_OPTIONS = {"mofilesdir": None, "grad": False, "copytheory": False, "optimizer": None}


def _install_mpirun(directory, version_text):
    directory.mkdir(parents=True, exist_ok=True)
    mpirun = directory / "mpirun"
    mpirun.write_text(f"#!/bin/sh\necho '{version_text}'\n")
    mpirun.chmod(mpirun.stat().st_mode | stat.S_IXUSR)
    return directory


def test_check_openmpi_reports_the_mpirun_directory_and_version(tmp_path, monkeypatch, caplog):
    bindir = _install_mpirun(tmp_path / "mpi" / "bin", "mpirun (Open MPI) 4.1.6")
    monkeypatch.setenv("PATH", str(bindir))

    with caplog.at_level(logging.INFO, logger="openmmqmmm.parallel"):
        parallel_module.check_openmpi()

    assert f"OpenMPI binary directory found: {bindir}" in caplog.text
    assert "OpenMPI version (mpirun -V): mpirun (Open MPI) 4.1.6" in caplog.text


def test_check_openmpi_without_mpirun_in_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(ExternalProgramError, match="No mpirun found in PATH"):
        parallel_module.check_openmpi()


def test_import_pool_selects_the_requested_backend():
    from multiprocessing.pool import Pool as StandardPool

    from multiprocess.pool import Pool as DillPool

    assert parallel_module._import_pool("multiprocessing") is StandardPool
    assert parallel_module._import_pool("multiprocess") is DillPool
    with pytest.raises(InputError, match="Unknown parallel backend"):
        parallel_module._import_pool("threads")


@pytest.mark.parametrize(
    ("labels", "message"),
    [([None], "needs a label"), ([[1, 2]], "must be hashable"), (["a", "a"], "must be unique")],
)
def test_job_labels_are_validated(labels, message):
    with pytest.raises(InputError, match=message):
        parallel_module._validate_unique_job_labels([{"label": label} for label in labels])


def test_labels_sharing_a_worker_directory_are_rejected(monkeypatch):
    monkeypatch.setattr(parallel_module, "_worker_directory_label", lambda label, fragmentfile: "same")

    with pytest.raises(InputError, match="resolve to the same worker directory 'same'"):
        parallel_module._validate_unique_worker_directories([{"label": "a"}, {"label": "b"}])


class _ExplodingPool:
    def terminate(self):
        raise RuntimeError("terminate failed")

    def join(self):
        raise RuntimeError("join failed")


def test_terminate_and_join_logs_both_failures(caplog):
    with caplog.at_level(logging.ERROR, logger="openmmqmmm.parallel"):
        parallel_module._terminate_and_join(_ExplodingPool())

    assert "Failed to terminate the parallel worker pool" in caplog.text
    assert "Failed to join the terminated parallel worker pool" in caplog.text


def _jobs(count):
    theory = ZeroTheory()
    return [{"theory": theory, "fragment": fragment, "label": fragment.label} for fragment in labelled_fragments(count)]


class _SyncPool:
    """Runs each job in-process when its result is collected, recording what was submitted."""

    instances: ClassVar[list] = []

    def __init__(self, numcores, *, submit_error=None, close_error=None):
        self.numcores = numcores
        self.submit_error = submit_error
        self.close_error = close_error
        self.submitted = []
        self.terminated = False
        self.joined = 0
        _SyncPool.instances.append(self)

    def apply_async(self, function, *, kwds):
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(kwds)
        return SimpleNamespace(get=lambda: function(**kwds))

    def close(self):
        if self.close_error is not None:
            raise self.close_error

    def terminate(self):
        self.terminated = True

    def join(self):
        self.joined += 1


@pytest.fixture
def sync_pool(monkeypatch):
    _SyncPool.instances.clear()
    monkeypatch.setattr(parallel_module, "_import_pool", lambda version: _SyncPool)
    return _SyncPool.instances


def test_execute_parallel_jobs_runs_workers_in_submission_order():
    completed = parallel_module._execute_parallel_jobs(
        pool_type=_SyncPool, numcores=1, jobs=_jobs(2), worker_options=WORKER_OPTIONS
    )

    assert [label for label, _energy, _dirname, _properties in completed] == ["frag0", "frag1"]
    assert [energy for _label, energy, _dirname, _properties in completed] == [0.0, 0.0]
    assert sorted(name for name in os.listdir(".") if name.startswith("Pooljob_")) == ["Pooljob_frag0", "Pooljob_frag1"]


def test_pool_creation_failure_is_reported():
    def broken_pool(numcores):
        raise RuntimeError("no forks")

    with pytest.raises(OpenMMQMMMError, match="could not create a 2-process worker pool: no forks"):
        parallel_module._execute_parallel_jobs(
            pool_type=broken_pool, numcores=2, jobs=_jobs(1), worker_options=WORKER_OPTIONS
        )


def test_keyboard_interrupt_during_submission_terminates_the_pool():
    def pool_type(numcores):
        return _SyncPool(numcores, submit_error=KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        parallel_module._execute_parallel_jobs(
            pool_type=pool_type, numcores=1, jobs=_jobs(1), worker_options=WORKER_OPTIONS
        )

    assert _SyncPool.instances[-1].terminated is True
    assert _SyncPool.instances[-1].joined == 1


def test_submission_failure_names_the_job():
    def pool_type(numcores):
        return _SyncPool(numcores, submit_error=RuntimeError("boom"))

    with pytest.raises(OpenMMQMMMError, match="worker 'frag0' failed: boom"):
        parallel_module._execute_parallel_jobs(
            pool_type=pool_type, numcores=1, jobs=_jobs(1), worker_options=WORKER_OPTIONS
        )


def test_pool_lifecycle_failure_is_distinguished_from_worker_failure():
    def pool_type(numcores):
        return _SyncPool(numcores, close_error=RuntimeError("close boom"))

    with pytest.raises(OpenMMQMMMError, match="pool lifecycle failed: close boom"):
        parallel_module._execute_parallel_jobs(
            pool_type=pool_type, numcores=1, jobs=_jobs(1), worker_options=WORKER_OPTIONS
        )

    assert _SyncPool.instances[-1].terminated is True


def test_assemble_results_rejects_duplicate_labels():
    completed = [("a", 0.0, "Pooljob_a", {}), ("a", 1.0, "Pooljob_a", {})]

    with pytest.raises(OpenMMQMMMError, match="duplicate result label 'a'"):
        parallel_module._assemble_parallel_results(completed, grad=False)


def _invalid_job(kind):
    fragments = labelled_fragments(1)
    return {
        "theories_not_a_sequence": ({"theories": 5}, "theories must be a sequence"),
        "fragments_not_a_sequence": ({"fragments": 5, "theories": [ZeroTheory()]}, "fragments must be a sequence"),
        "files_not_a_sequence": ({"fragmentfiles": 5, "theories": [ZeroTheory()]}, "fragmentfiles must be a sequence"),
        "fragments_wrong_type": ({"fragments": ["H2"], "theories": [ZeroTheory()]}, "only Fragment objects"),
        "files_wrong_type": ({"fragmentfiles": [7], "theories": [ZeroTheory()]}, "only string or PathLike"),
        "opt_with_grad": ({"fragments": fragments, "theories": [ZeroTheory()], "opt": True, "grad": True}, "grad=True"),
        "theory_without_numcores": ({"fragments": fragments, "theories": [SimpleNamespace(label="t")]}, "numcores"),
    }[kind]


@pytest.mark.parametrize(
    "kind",
    [
        "theories_not_a_sequence",
        "fragments_not_a_sequence",
        "files_not_a_sequence",
        "fragments_wrong_type",
        "files_wrong_type",
        "opt_with_grad",
        "theory_without_numcores",
    ],
)
def test_job_parallel_validates_its_inputs(kind):
    kwargs, message = _invalid_job(kind)

    with pytest.raises(InputError, match=message):
        job_parallel(numcores=1, **kwargs)


def test_job_parallel_runs_fragment_files(sync_pool):
    filenames = []
    for fragment in labelled_fragments(2):
        filenames.append(f"{fragment.label}.frag")
        fragment.print_system(filename=filenames[-1])

    result = job_parallel(fragmentfiles=filenames, theories=[ZeroTheory()], numcores=1)

    assert result.energies == [0.0, 0.0]
    assert list(result.energies_dict) == filenames
    assert [kwds["fragmentfile"] for kwds in sync_pool[0].submitted] == filenames
    assert all(Path(dirname).is_dir() for dirname in result.worker_dirnames.values())


def test_job_parallel_runs_several_theories_on_one_fragment_file(sync_pool):
    fragment = labelled_fragments(1)[0]
    fragment.print_system(filename="shared.frag")
    theories = [ZeroTheory(label="method-a"), ZeroTheory(label="method-b")]

    result = job_parallel(fragmentfiles=["shared.frag"], theories=theories, numcores=2)

    assert list(result.energies_dict) == ["method-a", "method-b"]
    assert [kwds["fragmentfile"] for kwds in sync_pool[0].submitted] == ["shared.frag", "shared.frag"]
    assert len(set(result.worker_dirnames.values())) == 2


class _StubOptimizer:
    runs: ClassVar[list] = []

    def run(self, *, theory, fragment, charge, mult):
        _StubOptimizer.runs.append((fragment.label, charge, mult))
        return SimpleNamespace(energy=-3.0)


def test_job_parallel_forwards_the_given_optimizer(sync_pool):
    _StubOptimizer.runs.clear()
    optimizer = _StubOptimizer()

    result = job_parallel(
        fragments=labelled_fragments(2), theories=[ZeroTheory()], numcores=1, opt=True, optimizer=optimizer
    )

    assert result.energies == [-3.0, -3.0]
    assert _StubOptimizer.runs == [("frag0", 0, 1), ("frag1", 0, 1)]
    assert all(kwds["optimizer"] is optimizer for kwds in sync_pool[0].submitted)


def test_job_parallel_creates_a_default_optimizer(sync_pool):
    from openmmqmmm.geometric import GeometricOptimizer

    result = job_parallel(fragments=labelled_fragments(1), theories=[ZeroTheory()], numcores=1, opt=True)

    assert isinstance(sync_pool[0].submitted[0]["optimizer"], GeometricOptimizer)
    assert result.energies == [0.0]


def test_job_parallel_warns_about_qmmm_theories(sync_pool, caplog):
    qmmm, fragment = make_subregion_qmmm(qm_charge=0, qm_mult=1)
    fragment.label = "subregion"

    with caplog.at_level(logging.WARNING, logger="openmmqmmm.parallel"):
        result = job_parallel(fragments=[fragment], theories=[qmmm], numcores=1)

    assert "experimental" in caplog.text
    assert result.energies_dict == {"subregion": 0.0}


def test_job_parallel_with_the_multiprocess_backend():
    fragments = labelled_fragments(2)

    result = job_parallel(fragments=fragments, theories=[ZeroTheory()], numcores=2, grad=True, version="multiprocess")

    assert result.energies == [0.0, 0.0]
    assert all(np.shape(result.gradients_dict[fragment.label]) == (2, 3) for fragment in fragments)


class ORCATheory:
    """Name-only stand-in: worker_par decides the mofilesdir option by the class name alone."""

    def __init__(self):
        self.numcores = 1
        self.theorytype = "QM"
        self.filename = "orca"
        self.moreadfile = None

    def run(self, **_kwargs):
        return 0.0


@pytest.mark.parametrize(
    ("label", "moreadfile"),
    [
        ((1, 2), "mos/orca_RC1_1-RC2_2.gbw"),
        ((3,), "mos/orca_RC1_3.gbw"),
        (5, "mos/orca_RC1_5.gbw"),
        (2.5, "mos/orca_RC1_2.5.gbw"),
    ],
)
def test_worker_builds_the_orbital_file_name_from_the_label(label, moreadfile):
    theory = ORCATheory()

    worker_par(fragment=labelled_fragments(1)[0], theory=theory, label=label, mofilesdir="mos")

    assert theory.moreadfile == moreadfile


@pytest.mark.parametrize("label", [(), "named"])
def test_worker_mofilesdir_needs_a_numeric_or_tuple_label(label):
    with pytest.raises(InputError, match="needs a tuple, float or int label"):
        worker_par(fragment=labelled_fragments(1)[0], theory=ORCATheory(), label=label, mofilesdir="mos")


def test_worker_mofilesdir_is_orca_only():
    with pytest.raises(InputError, match="only supported for ORCATheory, not ZeroTheory"):
        worker_par(fragment=labelled_fragments(1)[0], theory=ZeroTheory(), label=(1, 2), mofilesdir="mos")


def test_worker_requires_a_theory():
    with pytest.raises(InputError, match="requires a theory"):
        worker_par(fragment=labelled_fragments(1)[0], theory=None, label="frag0")


def test_worker_requires_a_label():
    with pytest.raises(InputError, match="No label provided"):
        worker_par(fragment=labelled_fragments(1)[0], theory=ZeroTheory(), label=None)


def test_worker_copies_the_theory_on_request():
    theory = ZeroTheory()

    worker_par(fragment=labelled_fragments(1)[0], theory=theory, label="frag0", copytheory=True)
    assert not hasattr(theory, "energy")

    worker_par(fragment=labelled_fragments(1)[0], theory=theory, label="frag0")
    assert theory.energy == 0.0


def test_worker_reuses_an_existing_directory(caplog):
    Path("Pooljob_frag0").mkdir()

    with caplog.at_level(logging.INFO, logger="openmmqmmm.parallel"):
        worker_par(fragment=labelled_fragments(1)[0], theory=ZeroTheory(), label="frag0")

    assert "Dir exists. continuing" in caplog.text


def test_worker_runs_the_optimizer_on_a_copy(caplog):
    _StubOptimizer.runs.clear()
    fragment = labelled_fragments(1)[0]

    with caplog.at_level(logging.INFO, logger="openmmqmmm.parallel"):
        label, energy, dirname, properties = worker_par(
            fragment=fragment, theory=ZeroTheory(), label="frag0", optimizer=_StubOptimizer()
        )

    assert (label, energy, dirname, properties) == ("frag0", -3.0, "Pooljob_frag0", {})
    assert fragment.energy == -3.0
    assert _StubOptimizer.runs == [("frag0", 0, 1)]
    assert "Doing optimization job" in caplog.text


def test_worker_gradient_without_property_methods_returns_no_properties():
    label, energy, gradient, _dirname, properties = worker_par(
        fragment=labelled_fragments(1)[0], theory=ZeroTheory(), label="frag0", grad=True
    )

    assert (label, energy, properties) == ("frag0", 0.0, {})
    assert np.shape(gradient) == (2, 3)


class _FailingTheory(ZeroTheory):
    def run(self, **_kwargs):
        raise RuntimeError("SCF blew up")


def test_worker_restores_the_working_directory_after_a_failure():
    before = os.getcwd()

    with pytest.raises(RuntimeError, match="SCF blew up"):
        worker_par(fragment=labelled_fragments(1)[0], theory=_FailingTheory(), label="frag0")

    assert os.getcwd() == before
