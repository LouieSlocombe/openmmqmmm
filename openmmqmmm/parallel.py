from __future__ import annotations

import copy
import hashlib
import logging
import os
import re
import shutil
import subprocess as sp
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

from openmmqmmm.coords import Fragment, check_charge_mult
from openmmqmmm.exceptions import (
    ExternalProgramError,
    InputError,
    MissingDependencyError,
    OpenMMQMMMError,
)
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.results import Results
from openmmqmmm.utils import sub_header

if TYPE_CHECKING:
    from openmmqmmm.geometric import GeometricOptimizer

logger = logging.getLogger(__name__)

StrPath: TypeAlias = str | os.PathLike[str]
Label: TypeAlias = StrPath | int | float | tuple[object, ...]
ParallelBackend: TypeAlias = Literal["multiprocessing", "multiprocess"]

_UNSAFE_WORKER_LABEL = re.compile(r"[^A-Za-z0-9_-]+")


def check_openmpi() -> None:
    try:
        openmpibindir = os.path.dirname(shutil.which("mpirun"))
    except TypeError:
        raise ExternalProgramError(
            "No mpirun found in PATH. Make sure to add OpenMPI to PATH in your environment/jobscript"
        ) from None
    logger.info("OpenMPI binary directory found: %s", openmpibindir)
    _verify_openmpi()


def _verify_openmpi() -> None:
    logger.info("Testing that mpirun is executable...")
    p = sp.Popen(["mpirun", "-V"], stdout=sp.PIPE)
    out, _err = p.communicate()
    mpiversion = out.decode()  # Now taking whole string
    logger.info("yes")
    logger.info("OpenMPI version (mpirun -V): %s", mpiversion)


def _import_pool(version: ParallelBackend = "multiprocessing") -> type[Any]:
    # NOTE: Python 3.8 and higher use spawn in MacOS (openmmqmmm import problems). Unix/Linux uses fork
    if version == "multiprocessing":
        logger.info("Using version: multiprocessing")
        from multiprocessing.pool import Pool

        logger.info("multiprocessing library successfully loaded")
    # Active fork of multiprocessing that uses dill instead of pickle etc. https://github.com/uqfoundation/multiprocess
    elif version == "multiprocess":
        logger.info("Job_parallel: Using version: multiprocess")
        try:
            from multiprocess.pool import Pool

            logger.info("multiprocess library successfully loaded")
        except ImportError:
            raise MissingDependencyError(
                "This requires the multiprocess library to be installed\nPlease install using pip: pip install "
                "multiprocess"
            ) from None
    else:
        raise InputError(f"Unknown parallel backend {version!r}; expected 'multiprocessing' or 'multiprocess'")
    return Pool


# Used for standalone SP calculations and NumFreq
# Can also be used for optimization and relaxed scans by providing Opt keyword or optimizer object

# mofilesdir. Directory containing MO-files (GBW files for ORCA). Usef for multiple fragment option
# NOTE: Experimental copytheory option
# NOTE: Can now either use built-in multiprocessing library or more reliable fork multiprocess.
# The latter uses dill serialization and should be more reliable


def _resolve_theory_parallelization(theory: Any, numcores: int, allow_theory_parallelization: bool) -> None:
    """Decide whether each parallel job may also use the theory's own cores."""
    if theory.numcores == 1:
        return
    logger.warning("Theory numcores set to: %s", theory.numcores)
    if allow_theory_parallelization is True:
        logger.warning(
            f"allow_theory_parallelization is True. Each job can use {theory.numcores} CPU cores, thus up to "
            f"{numcores * theory.numcores} CPU cores can be running simultaneously. Make sure that that's how "
            f"many slots are available."
        )
        return
    logger.warning(
        "allow_theory_parallelization is False. Now turning off theory.parallelization (setting theory "
        "numcores to 1)\nThis can be overriden by: Job_parallel(allow_theory_parallelization=True)\n"
    )
    theory.numcores = 1


def _validate_unique_job_labels(jobs: Sequence[dict[str, Any]]) -> None:
    """Reject labels that cannot safely key the result dictionaries."""
    seen = set()
    for job in jobs:
        label = job["label"]
        if label is None:
            raise InputError(
                "Every parallel job needs a label. Set Fragment.label for a one-theory job or Theory.label for a "
                "multiple-theory job."
            )
        try:
            hash(label)
        except TypeError:
            raise InputError(f"Parallel job labels must be hashable; got {label!r}") from None
        if label in seen:
            raise InputError(f"Parallel job labels must be unique; duplicate label: {label!r}")
        seen.add(label)


def _validate_unique_worker_directories(jobs: Sequence[dict[str, Any]]) -> None:
    """Reject the vanishingly rare case where two labels resolve to one directory."""
    seen: dict[str, Label] = {}
    for job in jobs:
        directory_label = _worker_directory_label(job["label"], job.get("fragmentfile"))
        if directory_label in seen:
            raise InputError(
                f"Parallel labels {seen[directory_label]!r} and {job['label']!r} resolve to the same worker "
                f"directory {directory_label!r}; choose more distinct labels"
            )
        seen[directory_label] = job["label"]


def _terminate_and_join(pool: Any) -> None:
    """Best-effort shutdown used after pool submission or worker failures."""
    try:
        pool.terminate()
    except Exception:
        logger.exception("Failed to terminate the parallel worker pool cleanly")
    try:
        pool.join()
    except Exception:
        logger.exception("Failed to join the terminated parallel worker pool cleanly")


def _execute_parallel_jobs(
    *,
    pool_type: type[Any],
    numcores: int,
    jobs: Sequence[dict[str, Any]],
    worker_options: dict[str, Any],
) -> list[Any]:
    """Submit all jobs and collect each AsyncResult exactly once in submission order."""
    try:
        pool = pool_type(numcores)
    except Exception as exc:
        raise OpenMMQMMMError(f"Job_parallel could not create a {numcores}-process worker pool: {exc}") from exc

    async_results = []
    active_label = None
    try:
        for job in jobs:
            active_label = job["label"]
            async_results.append(pool.apply_async(worker_par, kwds={**worker_options, **job}))

        active_label = None
        pool.close()
        completed = []
        for job, async_result in zip(jobs, async_results, strict=True):
            active_label = job["label"]
            completed.append(async_result.get())
        active_label = None
        pool.join()
    except (KeyboardInterrupt, SystemExit):
        _terminate_and_join(pool)
        raise
    except Exception as exc:
        _terminate_and_join(pool)
        context = "pool lifecycle" if active_label is None else f"worker {active_label!r}"
        raise OpenMMQMMMError(f"Job_parallel {context} failed: {exc}") from exc
    return completed


def _assemble_parallel_results(completed: Sequence[Any], *, grad: bool) -> Results:
    """Build the public Results object from already-collected worker values."""
    energy_dict = {}
    worker_dirnames_dict = {}
    property_dict = {}
    dipole_dict = {}
    polarizability_dict = {}
    gradient_dict = {}

    final_result = Results(label="Job_parallel", energies=[], gradients=[])
    for worker_result in completed:
        if grad:
            label, energy, gradient, worker_dirname, properties = worker_result
        else:
            label, energy, worker_dirname, properties = worker_result

        if label in energy_dict:
            raise OpenMMQMMMError(f"Parallel workers returned duplicate result label {label!r}")
        energy_dict[label] = energy
        worker_dirnames_dict[label] = worker_dirname
        final_result.energies.append(energy)

        if properties:
            property_dict[label] = properties
        if grad:
            gradient_dict[label] = gradient
            final_result.gradients.append(gradient)
            if "dipole_moment" in properties:
                dipole_dict[label] = properties["dipole_moment"]
            if "polarizability" in properties:
                polarizability_dict[label] = properties["polarizability"]

    final_result.energies_dict = energy_dict
    final_result.worker_dirnames = worker_dirnames_dict
    final_result.properties = property_dict
    if grad:
        final_result.gradients_dict = gradient_dict
        final_result.displacement_dipole_dictionary = dipole_dict
        final_result.displacement_polarizability_dictionary = polarizability_dict
    return final_result


def job_parallel(
    *,
    fragments: Sequence[Fragment] | None = None,
    fragmentfiles: Sequence[StrPath] | None = None,
    theories: Sequence[Any] | None = None,
    numcores: int | None = None,
    mofilesdir: str | None = None,
    allow_theory_parallelization: bool = False,
    grad: bool = False,
    copytheory: bool = False,
    version: ParallelBackend = "multiprocessing",
    opt: bool = False,
    optimizer: GeometricOptimizer | None = None,
) -> Results:
    """Carry out multiple single-point or optimization calculations in parallel."""
    logger.info(sub_header("Job_parallel function"))
    logger.info("copytheory: %s", copytheory)

    if theories is None:
        raise InputError("Job_parallel requires at least one theory")
    try:
        theory_jobs = list(theories)
    except TypeError:
        raise InputError("theories must be a sequence of theory objects") from None
    if not theory_jobs:
        raise InputError("Job_parallel requires at least one theory")

    try:
        fragment_jobs = [] if fragments is None else list(fragments)
    except TypeError:
        raise InputError("fragments must be a sequence of Fragment objects") from None
    if isinstance(fragmentfiles, (str, os.PathLike)):
        raise InputError("fragmentfiles must be a sequence of file paths, not one bare path")
    try:
        fragmentfile_jobs = [] if fragmentfiles is None else list(fragmentfiles)
    except TypeError:
        raise InputError("fragmentfiles must be a sequence of file paths") from None

    if any(not isinstance(fragment, Fragment) for fragment in fragment_jobs):
        raise InputError("fragments must contain only Fragment objects")
    if any(not isinstance(fragmentfile, (str, os.PathLike)) for fragmentfile in fragmentfile_jobs):
        raise InputError("fragmentfiles must contain only string or PathLike file paths")
    if fragment_jobs and fragmentfile_jobs:
        raise InputError("Provide fragments or fragmentfiles, not both")
    if not fragment_jobs and not fragmentfile_jobs:
        raise InputError("Job_parallel requires at least one fragment or fragment file")

    if isinstance(numcores, (bool, np.bool_)) or not isinstance(numcores, (int, np.integer)) or numcores <= 0:
        raise InputError(f"numcores must be a positive integer; got {numcores!r}")
    numcores = int(numcores)
    if version not in {"multiprocessing", "multiprocess"}:
        raise InputError(f"Unknown parallel backend {version!r}; expected 'multiprocessing' or 'multiprocess'")
    if opt and grad:
        raise InputError("Job_parallel does not support requesting grad=True for optimization jobs")
    if any(not hasattr(theory, "numcores") for theory in theory_jobs):
        raise InputError("Every theory passed to Job_parallel must define a numcores attribute")

    input_job_count = len(fragment_jobs) + len(fragmentfile_jobs)
    if len(theory_jobs) > 1 and input_job_count > 1:
        raise InputError("Multiple theories with multiple fragments/files is ambiguous and is not supported")

    jobs = []
    if len(theory_jobs) == 1:
        theory = theory_jobs[0]
        jobs.extend({"theory": theory, "fragment": fragment, "label": fragment.label} for fragment in fragment_jobs)
        jobs.extend(
            {"theory": theory, "fragmentfile": fragmentfile, "label": fragmentfile}
            for fragmentfile in fragmentfile_jobs
        )
    else:
        # One molecular system and multiple theories: theory labels are the only keys that
        # distinguish the results. The previous fragment-label key silently overwrote every
        # result except the last one.
        common_job = {"fragment": fragment_jobs[0]} if fragment_jobs else {"fragmentfile": fragmentfile_jobs[0]}
        jobs.extend({"theory": theory, "label": getattr(theory, "label", None), **common_job} for theory in theory_jobs)
    _validate_unique_job_labels(jobs)
    _validate_unique_worker_directories(jobs)

    if opt is True:
        logger.info("Job_parallel: Opt is True. This is an Opt_parallel job")
        if optimizer is None:
            logger.info("Job_parallel needs optimizer object which was not provided.")
            logger.debug("Creating one")
            from openmmqmmm.geometric import GeometricOptimizer

            # No options easily provided. Unclear if this is a good idea
            optimizer = GeometricOptimizer()
    else:
        logger.info("Job_parallel: No Opt. This is a Singlepoint_parallel job")
        optimizer = None

    logger.info("Number of CPU cores available:  %s", numcores)
    if any(isinstance(theory, QMMMTheory) for theory in theory_jobs):
        logger.warning(
            "Job_parallel using QMMMTheory with OpenMMTheory MM is experimental and has known issues with "
            "platform='CPU'; use 'Reference', 'OpenCL', or 'CUDA' if possible"
        )
    logger.info("Number of theories: %s", len(theory_jobs))
    logger.debug("Running single-point calculations in parallel")
    logger.info("Mofilesdir: %s", mofilesdir)
    logger.warning("Output from Job_parallel will be erratic due to simultaneous output from multiple workers")
    logger.info("Number of fragments: %s", len(fragment_jobs))
    logger.info("Number of fragmentfiles: %s", len(fragmentfile_jobs))
    logger.info("Job_parallel numcores set to: %s", numcores)
    logger.info("openmmqmmm will run %s jobs simultaneously", numcores)

    pool_type = _import_pool(version=version)
    unique_theories = []
    seen_theory_ids = set()
    for theory in theory_jobs:
        if id(theory) not in seen_theory_ids:
            unique_theories.append(theory)
            seen_theory_ids.add(id(theory))
    original_core_counts = [(theory, theory.numcores) for theory in unique_theories]

    try:
        for theory in unique_theories:
            _resolve_theory_parallelization(theory, numcores, allow_theory_parallelization)
        completed = _execute_parallel_jobs(
            pool_type=pool_type,
            numcores=numcores,
            jobs=jobs,
            worker_options={
                "mofilesdir": mofilesdir,
                "grad": grad,
                "copytheory": copytheory,
                "optimizer": optimizer,
            },
        )
    finally:
        for theory, original_numcores in original_core_counts:
            theory.numcores = original_numcores

    return _assemble_parallel_results(completed, grad=grad)


def _worker_directory_label(label: Label, fragmentfile: StrPath | None) -> str:
    """Turn a result label into one bounded, traversal-safe directory component."""
    if fragmentfile is not None:
        path_text = os.fspath(fragmentfile)
        file_label = os.path.basename(path_text) or "fragment"
        label_is_input_path = isinstance(label, (str, os.PathLike)) and os.fspath(label) == path_text
        raw_label = file_label if label_is_input_path else f"{file_label}_{label}"
        # A single fragment file can be run with several theories. Include both the
        # typed result label and the path so those jobs never share a directory.
        label_type = f"{type(label).__module__}.{type(label).__qualname__}"
        digest_source = f"{label_type}:{label!r}\0{path_text}"
    elif isinstance(label, tuple):
        raw_label = "_".join(str(part) for part in label)
        digest_source = f"{type(label).__module__}.{type(label).__qualname__}:{label!r}"
    else:
        raw_label = str(label)
        digest_source = f"{type(label).__module__}.{type(label).__qualname__}:{label!r}"

    # Dots were historically rendered as underscores. Replace all remaining path or
    # shell punctuation as well, and add a digest whenever the transformation could
    # otherwise make two distinct labels share a worker directory.
    normalized_label = raw_label.replace(".", "_")
    safe_label = _UNSAFE_WORKER_LABEL.sub("_", normalized_label).strip("_") or "job"
    # Non-string labels can render exactly like strings (1 versus "1", Path("a")
    # versus "a"), and tuple joining is not injective. Tag those forms even when
    # their text happens to be filesystem-safe.
    needs_digest = (
        fragmentfile is not None or not isinstance(label, str) or safe_label != raw_label or len(safe_label) > 80
    )
    safe_label = safe_label[:80].rstrip("_") or "job"
    if needs_digest:
        digest = hashlib.blake2s(digest_source.encode("utf-8", errors="surrogatepass"), digest_size=5).hexdigest()
        safe_label = f"{safe_label}_{digest}"
    return safe_label


# NOTE: Version intended for apply_async
def worker_par(
    *,
    fragment: Fragment | None = None,
    fragmentfile: StrPath | None = None,
    theory: Any | None = None,
    label: Label | None = None,
    mofilesdir: str | None = None,
    charge: int | None = None,
    mult: int | None = None,
    grad: bool = False,
    copytheory: bool = False,
    optimizer: GeometricOptimizer | None = None,
    version: ParallelBackend = "multiprocessing",
) -> tuple[Label, float, np.ndarray, str, dict[str, Any]] | tuple[Label, float, str, dict[str, Any]]:
    logger.info("Fragment: %s", fragment)
    logger.info("fragmentfile: %s", fragmentfile)
    logger.info("Theory: %s", theory)

    # Creating new copy of theory to avoid deactivation of certain first-run features (e.g. brokensym)
    if copytheory:
        theory = copy.deepcopy(theory)

    if fragmentfile is not None:
        logger.info("Reading fragmentfile from disk")
        fragmentfile = os.fspath(fragmentfile)
        fragment = Fragment(fragfile=fragmentfile)

    # Resolved after the load: job_parallel submits fragmentfile= without fragment=, so a resolution
    # before this point sees no fragment at all.
    if fragment is None:
        raise InputError("Worker_par requires either a fragment or a fragmentfile")
    if theory is None:
        raise InputError("Worker_par requires a theory")
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Worker_par", theory=theory)

    # Making label flexible. Can be tuple but inputfilename is converted to string below
    logger.info(f"label: {label} (type {type(label)})")
    if label is None:
        raise InputError(
            "No label provided to fragment or theory objects. This is required to distinguish between calculations"
        )
    moreadfile_path = None
    if isinstance(label, tuple):
        # RC1_0.9-RC2_170.0.xyz
        # orca_RC1_0.9RC2_170.0.gbw
        if mofilesdir is not None:
            logger.info("Mofilesdir option.")
            if len(label) == 2:
                moreadfile_path = (
                    mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label[0]) + "-" + "RC2_" + str(label[1])
                )
            elif len(label) > 0:
                moreadfile_path = mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label[0])
    elif isinstance(label, (float, int)):
        logger.info("Label is float or int")
        if mofilesdir is not None:
            logger.info("Mofilesdir option.")
            moreadfile_path = mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label)

    if mofilesdir is not None:
        if theory.__class__.__name__ != "ORCATheory":
            raise InputError(f"The mofilesdir option is only supported for ORCATheory, not {theory.__class__.__name__}")
        if moreadfile_path is None:
            raise InputError(
                f"The mofilesdir option needs a tuple, float or int label to build the MO-file name, "
                f"but the label was {label!r} (type {type(label).__name__})"
            )
        theory.moreadfile = moreadfile_path + ".gbw"
        logger.debug("Setting moreadfile to: %s", theory.moreadfile)

    worker_dirname = "Pooljob_" + _worker_directory_label(label, fragmentfile)
    try:
        os.mkdir(worker_dirname)
    except FileExistsError:
        logger.info("Dir exists. continuing")
    # Pool workers are reused for later jobs, so the cwd must be restored even if this job fails
    parent_dir = os.getcwd()
    os.chdir(worker_dirname)
    try:
        logger.info(
            f"Doing single-point Energy job on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
        )

        # Create property dict containing some results except energy and gradient
        properties = {}
        if optimizer is not None:
            optimizer_new = copy.copy(optimizer)
            result = optimizer_new.run(theory=theory, fragment=fragment, charge=charge, mult=mult)
            energy = result.energy
        elif grad:
            energy, gradient = theory.run(
                current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult, grad=grad
            )

            try:
                dm = theory.get_dipole_moment()
                properties["dipole_moment"] = dm
            except Exception:  # noqa: BLE001 - best-effort property grab
                pass
            try:
                polarizability = theory.get_polarizability_tensor()
                properties["polarizability"] = polarizability
            except Exception:  # noqa: BLE001 - best-effort property grab
                pass
        else:
            energy = theory.run(
                current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult
            )

        logger.info("Energy:  %s", energy)

        fragment.energy = energy
    finally:
        os.chdir(parent_dir)

    # Return label and energy or label, energy and gradient. Also worker_dirname
    if grad:
        return (label, energy, gradient, worker_dirname, properties)
    return (label, energy, worker_dirname, properties)
