from __future__ import annotations

import copy
import logging
import os
import shutil
import subprocess as sp
import time
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypeAlias

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
Label: TypeAlias = StrPath | float | tuple[object, ...]
ParallelBackend: TypeAlias = Literal["multiprocessing", "multiprocess"]


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


def _import_mp(version: ParallelBackend = "multiprocessing") -> tuple[ModuleType, type[Any]]:
    # NOTE: Python 3.8 and higher use spawn in MacOS (openmmqmmm import problems). Unix/Linux uses fork
    if version == "multiprocessing":
        logger.info("Using version: multiprocessing")
        import multiprocessing as mp
        from multiprocessing.pool import Pool

        logger.info("multiprocessing library successfully loaded")
    # Active fork of multiprocessing that uses dill instead of pickle etc. https://github.com/uqfoundation/multiprocess
    elif version == "multiprocess":
        logger.info("Job_parallel: Using version: multiprocess")
        try:
            import multiprocess as mp
            from multiprocess.pool import Pool

            logger.info("multiprocess library successfully loaded")
        except ImportError:
            raise MissingDependencyError(
                "This requires the multiprocess library to be installed\nPlease install using pip: pip install "
                "multiprocess"
            ) from None
    return mp, Pool


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

    # Early exits. Must come before any use of the arguments below
    if fragments is None and fragmentfiles is None:
        raise InputError("Job_parallel requires a list of fragments or a list of fragmentfilenames")
    if theories is None or numcores is None:
        raise InputError(
            f"theories: {theories}\nnumcores: {numcores}\nJob_parallel requires a theory object and a numcores value"
        )

    if isinstance(theories[0], QMMMTheory):
        logger.warning(
            "Job_parallel using QMMMTheory with OpenMMTheory MM is experimental and has known issues with "
            "platform='CPU'; use 'Reference', 'OpenCL', or 'CUDA' if possible"
        )
    logger.info("Number of theories: %s", len(theories))
    logger.debug("Running single-point calculations in parallel")
    logger.info("Mofilesdir: %s", mofilesdir)
    logger.warning("Output from Job_parallel will be erratic due to simultaneous output from multiple workers")

    if fragments is not None:
        logger.info("Number of fragments: %s", len(fragments))
    else:
        fragments = []
    if fragmentfiles is not None:
        logger.info("Number of fragmentfiles: %s", len(fragmentfiles))
    else:
        fragmentfiles = []

    mp, Pool = _import_mp(version=version)

    def terminate_pool_processes(message: BaseException) -> NoReturn:
        logger.error("Terminating Pool processes due to exception")
        logger.error("Exception message: %s", message)
        pool.terminate()
        event.set()
        raise OpenMMQMMMError(f"Terminating pool processes due to worker exception: {message}")

    pool = Pool(numcores)
    manager = mp.Manager()
    event = manager.Event()

    results = []

    def submit(**job: Any) -> None:
        results.append(
            pool.apply_async(
                worker_par,
                kwds={
                    "mofilesdir": mofilesdir,
                    "version": version,
                    "event": event,
                    "grad": grad,
                    "copytheory": copytheory,
                    "optimizer": optimizer,
                    **job,
                },
                error_callback=terminate_pool_processes,
            )
        )

    if len(theories) == 1:
        theory = theories[0]
        logger.debug("Case: Multiple fragments but one theory")
        logger.debug("\nLaunching pool.apply_async:")
        logger.info("Job_parallel numcores set to: %s", numcores)
        logger.info(f"openmmqmmm will run {numcores} jobs simultaneously")

        _resolve_theory_parallelization(theory, numcores, allow_theory_parallelization)

        if len(fragments) > 0:
            logger.info("fragments: %s", fragments)
            for fragment in fragments:
                logger.info("fragment: %s", fragment)
                submit(theory=theory, fragment=fragment, label=fragment.label)
        elif len(fragmentfiles) > 0:
            logger.debug("Launching multiprocessing and passing list of fragment files")
            for fragmentfile in fragmentfiles:
                logger.info("fragmentfile: %s", fragmentfile)
                submit(theory=theory, fragmentfile=fragmentfile, label=fragmentfile)
    elif len(fragments) == 1:
        logger.debug("Case: Multiple theories but one fragment")
        fragment = fragments[0]
        for theory in theories:
            logger.info("theory: %s", theory)
            submit(theory=theory, fragment=fragment, label=fragment.label)
    elif len(fragmentfiles) == 1:
        logger.debug("Case: Multiple theories but one fragmentfile")
        fragmentfile = fragmentfiles[0]
        for theory in theories:
            logger.info("theory: %s", theory)
            submit(theory=theory, fragmentfile=fragmentfile, label=fragmentfile)
    else:
        raise InputError("Multiple theories and multiple fragments provided.\nThis is not supported. Exiting...")

    pool.close()
    pool.join()
    event.set()

    while True:
        logger.info("Pool multiprocessing underway....")
        time.sleep(3)
        if event.is_set():
            logger.info("Event has been set! Now terminating Pool processes")
            pool.terminate()
            break

    # This prevents hanging for ApplyResult.get() if Pool did not finish correctly
    energy_dict = {}
    worker_dirnames_dict = {}
    property_dict = {}
    dipole_dict = {}
    polarizability_dict = {}

    final_result = Results(label="Job_parallel", energies=[], gradients=[])
    if grad is True:
        gradient_dict = {}
        for _i, r in enumerate(results):
            if r.ready():
                energy_dict[r.get()[0]] = r.get()[1]
                gradient_dict[r.get()[0]] = r.get()[2]
                worker_dirnames_dict[r.get()[0]] = r.get()[3]
                final_result.energies.append(r.get()[1])
                final_result.gradients.append(r.get()[2])
                if len(r.get()[4]) > 0:
                    property_dict[r.get()[0]] = r.get()[4]
                    if "dipole_moment" in r.get()[4]:
                        dipole_dict[r.get()[0]] = r.get()[4]["dipole_moment"]
                    if "polarizability" in r.get()[4]:
                        polarizability_dict[r.get()[0]] = r.get()[4]["polarizability"]

        final_result.gradients_dict = gradient_dict
        final_result.properties = property_dict
        final_result.displacement_dipole_dictionary = dipole_dict
        final_result.displacement_polarizability_dictionary = polarizability_dict
    else:
        for _i, r in enumerate(results):
            if r.ready() is True:
                energy_dict[r.get()[0]] = r.get()[1]
                worker_dirnames_dict[r.get()[0]] = r.get()[2]
                final_result.energies.append(r.get()[1])
                if len(r.get()[3]) > 0:
                    logger.info("r.get()[3]: %s", r.get()[3])
                    property_dict[r.get()[0]] = r.get()[3]
        final_result.properties = property_dict

    final_result.energies_dict = energy_dict
    final_result.worker_dirnames = worker_dirnames_dict

    # Results from jobs that died are skipped above so that a broken pool cannot hang the
    # collection loop. Silently returning the survivors would look like a successful run with
    # missing data, so report the shortfall instead.
    if len(final_result.energies) != len(results):
        raise OpenMMQMMMError(
            f"Job_parallel: only {len(final_result.energies)} of {len(results)} jobs returned a result. "
            "Check the worker output above for the underlying exception."
        )

    return final_result


# NOTE: Version intended for apply_async
def worker_par(
    *,
    fragment: Fragment | None = None,
    fragmentfile: StrPath | None = None,
    theory: Any | None = None,
    label: Label | None = None,
    mofilesdir: str | None = None,
    event: Any | None = None,
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
        fragment = Fragment(fragfile=fragmentfile)

    # Resolved after the load: job_parallel submits fragmentfile= without fragment=, so a resolution
    # before this point sees no fragment at all.
    if fragment is None:
        raise InputError("Worker_par requires either a fragment or a fragmentfile")
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Worker_par", theory=theory)

    # Making label flexible. Can be tuple but inputfilename is converted to string below
    logger.info(f"label: {label} (type {type(label)})")
    if label is None:
        raise InputError(
            "No label provided to fragment or theory objects. This is required to distinguish between calculations"
        )
    # Using label (could be tuple) to create a labelstring which is used to name worker directories
    moreadfile_path = None
    if isinstance(label, tuple):
        if len(label) == 2:
            labelstring = str(str(label[0]) + "_" + str(label[1])).replace(".", "_")
        else:
            labelstring = str(str(label[0])).replace(".", "_")
        logger.info("Labelstring: %s", labelstring)
        # RC1_0.9-RC2_170.0.xyz
        # orca_RC1_0.9RC2_170.0.gbw

        if mofilesdir is not None:
            logger.info("Mofilesdir option.")
            if len(label) == 2:
                moreadfile_path = (
                    mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label[0]) + "-" + "RC2_" + str(label[1])
                )
            else:
                moreadfile_path = mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label[0])
    elif isinstance(label, (float, int)):
        logger.info("Label is float or int")
        labelstring = str(label).replace(".", "_")
        if mofilesdir is not None:
            logger.info("Mofilesdir option.")
            moreadfile_path = mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label)
    else:
        labelstring = str(label).replace(".", "_")

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

    worker_dirname = "Pooljob_" + labelstring
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
                properties = {"dipole_moment": dm}
            except Exception:  # noqa: BLE001 - best-effort property grab
                pass
            try:
                polarizability = theory.get_polarizability_tensor()
                properties = {"polarizability": polarizability}
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
