"""Parallel execution of independent calculations with multiprocessing."""

import contextlib
import copy
import logging
import os
import shutil
import subprocess as sp
import time

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

logger = logging.getLogger(__name__)


###############################################
# CHECKS FOR OPENMPI
###############################################
def check_openmpi():
    # Find mpirun and take path
    try:
        openmpibindir = os.path.dirname(shutil.which("mpirun"))
    except TypeError:
        raise ExternalProgramError(
            "No mpirun found in PATH. Make sure to add OpenMPI to PATH in your environment/jobscript"
        ) from None
    logger.info("OpenMPI binary directory found: %s", openmpibindir)
    # Test that mpirun is executable and grab OpenMPI version number for printout
    verify_openmpi()
    return


def verify_openmpi():
    logger.info("Testing that mpirun is executable...")
    p = sp.Popen(["mpirun", "-V"], stdout=sp.PIPE)
    out, _err = p.communicate()
    mpiversion = out.decode()  # Now taking whole string
    logger.info("yes")
    logger.info("OpenMPI version (mpirun -V): %s", mpiversion)


###############################################
# MULTIPROCESS/MULTIPROCESSING handling
###############################################
def import_mp(version="multiprocessing"):
    ###############################
    # Multiprocessing Pool setup
    ###############################
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


#########################################
# Job_parallel: General PARALLEL function.
#########################################
# Used for standalone SP calculations and NumFreq
# Can also be used for optimization and relaxed scans by providing Opt keyword or optimizer object

# will run over fragments or fragmentfiles, over theories or both
# mofilesdir. Directory containing MO-files (GBW files for ORCA). Usef for multiple fragment option
# NOTE: Experimental copytheory option
# NOTE: Can now either use built-in multiprocessing library or more reliable fork multiprocess.
# The latter uses dill serialization and should be more reliable


# Used to be Singlepoint_parallel. Default behaviour is single-point
def job_parallel(
    fragments=None,
    fragmentfiles=None,
    theories=None,
    numcores=None,
    mofilesdir=None,
    allow_theory_parallelization=False,
    grad=False,
    copytheory=False,
    version="multiprocessing",
    opt=False,
    optimizer=None,
) -> "Results":
    """Carry out multiple single-point or optimization calculations in parallel.

    Runs over fragments or fragmentfiles, over theories, or over both.

    Args:
        fragments: list of Fragment objects to run.
        fragmentfiles: list of fragment filenames (strings) to read from disk instead.
        theories: list of theory objects. A single theory is applied to every fragment.
        numcores: number of jobs to run simultaneously (the worker pool size). Required.
        mofilesdir: directory holding MO files (GBW files for ORCA), used with the
            multiple-fragment option.
        allow_theory_parallelization: when False (the default) each theory's own
            numcores is forced to 1, so at most numcores cores are busy. When True each
            job may use theory.numcores, so up to numcores * theory.numcores cores run
            at once — make sure that many slots are actually available.
        grad: also compute the gradient.
        copytheory: experimental. Deep-copy the theory for each job so that first-run-only
            features (brokensym, for one) are not deactivated by a preceding run.
        version: which library to parallelize with — "multiprocessing" (standard library)
            or "multiprocess" (a fork using dill, more reliable for objects that pickle
            badly).
        opt: run geometry optimizations rather than single points.
        optimizer: optimizer object to use when opt is True. A default GeometricOptimizer
            is created if none is given.

    Returns:
        Results labelled "Job_parallel", holding one energy per job (and one gradient
        per job when grad=True).
    """
    logger.info("")
    logger.info(sub_header("Job_parallel function"))

    logger.info("copytheory: %s", copytheory)

    # OPT
    if opt is True:
        logger.info("Job_parallel: Opt is True. This is an Opt_parallel job")
        if optimizer is None:
            logger.info("Job_parallel needs optimizer object which was not provided.")
            logger.info("Creating one")
            from openmmqmmm.geometric import GeometricOptimizer

            # No options easily provided. Unclear if this is a good idea
            optimizer = GeometricOptimizer()
    # SP
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
        logger.warning("Job_parallel using QMMMTheory with OpenMMTheory MM is experimental")
        logger.info("Specifically there are issues with platform='CPU'.")
        logger.info("Try platform='Reference' instead or GPU options OpenCL or CUDA if possible")
    logger.info("Number of theories: %s", len(theories))
    logger.info("Running single-point calculations in parallel")
    logger.info("Mofilesdir: %s", mofilesdir)
    logger.warning("Output from Job_parallel will be erratic due to simultaneous output from multiple workers")

    # Fragment objects passed or name of fragmentfiles
    if fragments is not None:
        logger.info("Number of fragments: %s", len(fragments))
    else:
        fragments = []
    if fragmentfiles is not None:
        logger.info("Number of fragmentfiles: %s", len(fragmentfiles))
    else:
        fragmentfiles = []

    ###############################
    # Multiprocessing Pool setup
    ###############################

    # Import multiprocess/multiprocessing library
    mp, Pool = import_mp(version=version)

    # Function to handle exception of child processes
    def terminate_pool_processes(message):
        logger.error("Terminating Pool processes due to exception")
        logger.error("Exception message: %s", message)
        pool.terminate()
        event.set()
        raise OpenMMQMMMError(f"Terminating pool processes due to worker exception: {message}")

    pool = Pool(numcores)
    # Manager
    manager = mp.Manager()
    event = manager.Event()

    ##############################################################
    # Calling Pool for different fragment vs. theory scenarios
    ###############################################################
    # Case: 1 theory, multiple fragments
    results = []
    if len(theories) == 1:
        theory = theories[0]
        logger.info("Case: Multiple fragments but one theory")
        logger.info("")
        logger.info("Launching pool.apply_async:")
        logger.info("Job_parallel numcores set to: %s", numcores)
        logger.info(f"openmmqmmm will run {numcores} jobs simultaneously")

        # Whether to allow theory parallelization or not
        if theory.numcores != 1:
            logger.warning("Theory numcores set to: %s", theory.numcores)
            if allow_theory_parallelization is True:
                totnumcores = numcores * theory.numcores
                logger.warning("allow_theory_parallelization is True.")
                logger.warning(
                    f"Each job can use {theory.numcores} CPU cores, thus up to {totnumcores} CPU cores can be running "
                    f"simultaneously. Make sure that that's how many slots are available."
                )
            else:
                logger.warning(
                    "allow_theory_parallelization is False. Now turning off theory.parallelization (setting theory "
                    "numcores to 1)"
                )
                logger.warning("This can be overriden by: Job_parallel(allow_theory_parallelization=True)\n")
                theory.numcores = 1

        # Passing list of fragments
        if len(fragments) > 0:
            logger.info("fragments: %s", fragments)
            for fragment in fragments:
                logger.info("fragment: %s", fragment)
                results.append(
                    pool.apply_async(
                        worker_par,
                        kwds={
                            "theory": theory,
                            "fragment": fragment,
                            "label": fragment.label,
                            "mofilesdir": mofilesdir,
                            "version": version,
                            "event": event,
                            "grad": grad,
                            "copytheory": copytheory,
                            "optimizer": optimizer,
                        },
                        error_callback=terminate_pool_processes,
                    )
                )
        # Passing list of fragment files
        elif len(fragmentfiles) > 0:
            logger.info("Launching multiprocessing and passing list of fragment files")
            for fragmentfile in fragmentfiles:
                logger.info("fragmentfile: %s", fragmentfile)
                results.append(
                    pool.apply_async(
                        worker_par,
                        kwds={
                            "theory": theory,
                            "fragmentfile": fragmentfile,
                            "label": fragmentfile,
                            "mofilesdir": mofilesdir,
                            "version": version,
                            "event": event,
                            "grad": grad,
                            "copytheory": copytheory,
                            "optimizer": optimizer,
                        },
                        error_callback=terminate_pool_processes,
                    )
                )
    # Case: Multiple theories, 1 fragment
    elif len(fragments) == 1:
        logger.info("Case: Multiple theories but one fragment")
        fragment = fragments[0]
        for theory in theories:
            logger.info("theory: %s", theory)
            results.append(
                pool.apply_async(
                    worker_par,
                    kwds={
                        "theory": theory,
                        "fragment": fragment,
                        "label": fragment.label,
                        "mofilesdir": mofilesdir,
                        "version": version,
                        "event": event,
                        "grad": grad,
                        "copytheory": copytheory,
                        "optimizer": optimizer,
                    },
                    error_callback=terminate_pool_processes,
                )
            )
    # Case: Multiple theories, 1 fragmentfile
    elif len(fragmentfiles) == 1:
        logger.info("Case: Multiple theories but one fragmentfile")
        fragmentfile = fragmentfiles[0]
        for theory in theories:
            logger.info("theory: %s", theory)
            results.append(
                pool.apply_async(
                    worker_par,
                    kwds={
                        "theory": theory,
                        "fragmentfile": fragmentfile,
                        "label": fragmentfile,
                        "mofilesdir": mofilesdir,
                        "version": version,
                        "event": event,
                        "grad": grad,
                        "copytheory": copytheory,
                        "optimizer": optimizer,
                    },
                    error_callback=terminate_pool_processes,
                )
            )
    else:
        raise InputError("Multiple theories and multiple fragments provided.\nThis is not supported. Exiting...")

    pool.close()
    pool.join()
    event.set()

    # While loop that is only terminated if processes finished or exception occurred
    while True:
        logger.info("Pool multiprocessing underway....")
        time.sleep(3)
        if event.is_set():
            logger.info("Event has been set! Now terminating Pool processes")
            pool.terminate()
            break

    ##############################################################
    # END OF POOL
    ###############################################################

    ###########
    # RESULTS
    ###########
    # Going through each result-object and adding to energy_dict if ready
    # This prevents hanging for ApplyResult.get() if Pool did not finish correctly
    energy_dict = {}
    worker_dirnames_dict = {}
    property_dict = {}
    # Dipole-dict, polarizability-dict etc.
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
                    # Dipole and polarizability
                    if "dipole_moment" in r.get()[4]:
                        dipole_dict[r.get()[0]] = r.get()[4]["dipole_moment"]
                    if "polarizability" in r.get()[4]:
                        polarizability_dict[r.get()[0]] = r.get()[4]["polarizability"]

        final_result.gradients_dict = gradient_dict
        final_result.properties = property_dict
        # Dipole and polarizability
        final_result.displacement_dipole_dictionary = dipole_dict
        final_result.displacement_polarizability_dictionary = polarizability_dict

    else:
        for _i, r in enumerate(results):
            if r.ready() is True:
                energy_dict[r.get()[0]] = r.get()[1]
                worker_dirnames_dict[r.get()[0]] = r.get()[2]
                final_result.energies.append(r.get()[1])
                # Optional property dict
                if len(r.get()[3]) > 0:
                    logger.info("r.get()[3]: %s", r.get()[3])
                    property_dict[r.get()[0]] = r.get()[3]
        final_result.properties = property_dict

    # Adding energy dictionary also
    final_result.energies_dict = energy_dict
    # And dictionary with dirnames used (so we can look up stuff)
    final_result.worker_dirnames = worker_dirnames_dict

    # Results from jobs that died are skipped above so that a broken pool cannot hang the
    # collection loop. Silently returning the survivors would look like a successful run with
    # missing data, so report the shortfall instead.
    if len(final_result.energies) != len(results):
        raise OpenMMQMMMError(
            f"Job_parallel: only {len(final_result.energies)} of {len(results)} jobs returned a result. "
            "Check the worker output above for the underlying exception."
        )

    # TODO: JSON-array problem, reenable later
    return final_result


# Worker_par for both Singlepoint-type and Opt-type jobs
# NOTE: Version intended for apply_async
# TODO: This function contains 2 many QM-code specifics. Needs to be generalized (QM-specifics moved to QMtheory class)
def worker_par(
    fragment=None,
    fragmentfile=None,
    theory=None,
    label=None,
    mofilesdir=None,
    event=None,
    charge=None,
    mult=None,
    grad=False,
    copytheory=False,
    optimizer=None,
    version="multiprocessing",
):
    # Should not be necessary to import
    # Check charge/mult.
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Worker_par", theory=theory)
    # BASIC PRINTING
    logger.info("Fragment: %s", fragment)
    logger.info("fragmentfile: %s", fragmentfile)
    logger.info("Theory: %s", theory)

    # Creating new copy of theory to avoid deactivation of certain first-run features (e.g. brokensym)
    # NOTE: Alternatively add if-statement inside orca.run
    if copytheory:
        theory = copy.deepcopy(theory)
    else:
        pass

    # Optional fragment-creation from disk
    if fragmentfile is not None:
        logger.info("Reading fragmentfile from disk")
        fragment = Fragment(fragfile=fragmentfile)

    ###############################
    # Labels distinguishing jobs
    ###############################
    # Making label flexible. Can be tuple but inputfilename is converted to string below
    logger.info(f"label: {label} (type {type(label)})")
    if label is None:
        raise InputError(
            "No label provided to fragment or theory objects. This is required to distinguish between calculations"
        )
    # Using label (could be tuple) to create a labelstring which is used to name worker directories
    # Tuple-label (1 or 2 elements).
    # Otherwise normally string
    # TODO: Needs to be generalized.  Remove RC1, RC2 strings
    moreadfile_path = None
    if isinstance(label, tuple):
        if len(label) == 2:
            labelstring = str(str(label[0]) + "_" + str(label[1])).replace(".", "_")
        else:
            labelstring = str(str(label[0])).replace(".", "_")
        logger.info("Labelstring: %s", labelstring)
        # RC1_0.9-RC2_170.0.xyz
        # orca_RC1_0.9RC2_170.0.gbw
        # TODO: what if tuple is only a single number???

        if mofilesdir is not None:
            logger.info("Mofilesdir option.")
            if len(label) == 2:
                moreadfile_path = (
                    mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label[0]) + "-" + "RC2_" + str(label[1])
                )
            else:
                moreadfile_path = mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label[0])

    # Label is not a tuple
    elif isinstance(label, (float, int)):
        logger.info("Label is float or int")
        labelstring = str(label).replace(".", "_")
        # Label is float or int.
        if mofilesdir is not None:
            logger.info("Mofilesdir option.")
            moreadfile_path = mofilesdir + "/" + theory.filename + "_" + "RC1_" + str(label)
    else:
        # Label is not tuple. String or single number
        labelstring = str(label).replace(".", "_")

    ###############################
    # TODO: Need to revisit all of this, ideally remove
    if mofilesdir is not None:
        if theory.__class__.__name__ != "ORCATheory":
            raise InputError(f"The mofilesdir option is only supported for ORCATheory, not {theory.__class__.__name__}")
        if moreadfile_path is None:
            raise InputError(
                f"The mofilesdir option needs a tuple, float or int label to build the MO-file name, "
                f"but the label was {label!r} (type {type(label).__name__})"
            )
        theory.moreadfile = moreadfile_path + ".gbw"
        logger.info("Setting moreadfile to: %s", theory.moreadfile)

    ####################################
    # Handling Directory
    ####################################
    # Creating new dir and running calculation inside
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

        #####################
        # RUN WORKER JOB
        #####################
        # Create property dict containing some results except energy and gradient
        properties = {}
        # Optimizer
        if optimizer is not None:
            # Make copy of optimizer
            optimizer_new = copy.copy(optimizer)
            result = optimizer_new.run(theory=theory, fragment=fragment, charge=charge, mult=mult)
            energy = result.energy
        # Singlepoint Grad
        elif grad:
            energy, gradient = theory.run(
                current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult, grad=grad
            )

            # Dipole and polarizability
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

        # Singlepoint energy
        else:
            energy = theory.run(
                current_coords=fragment.coords, elems=fragment.elems, label=label, charge=charge, mult=mult
            )
        #####################

        logger.info("Energy:  %s", energy)

        # Now adding total energy to fragment.
        # NOTE: Add to theory also?
        fragment.energy = energy
    finally:
        # Exiting workerdir
        os.chdir(parent_dir)

    # Return label and energy or label, energy and gradient. Also worker_dirname
    if grad:
        return (label, energy, gradient, worker_dirname, properties)
    else:
        return (label, energy, worker_dirname, properties)


# Simple parallel function for cases where no file handling is needed.
# parameter_dict: dict of input keywords for jobfunction
# separate_dirs: creates and enters separate dirs per process
def simple_parallel(
    jobfunction=None,
    parameter_dict=None,
    separate_dirs=False,
    numcores=None,
    copytheory=False,
    version="multiprocessing",
) -> dict:
    """Run a list of independent function calls in parallel worker processes."""
    logger.info("")
    logger.info(sub_header("Simple_parallel function"))
    logger.info("Number of CPU cores available:  %s", numcores)

    if parameter_dict is None:
        parameter_dict = {}

    ############
    # POOL
    ###########
    # Import multiprocess/multiprocessing
    mp, Pool = import_mp(version=version)
    # Creating Pool
    logger.info(f"Pool of {numcores} created")
    pool = Pool(numcores)
    manager = mp.Manager()
    event = manager.Event()

    # Function to handle exception of child processes
    def terminate_pool_processes(message):
        logger.error("Terminating Pool processes due to exception")
        logger.error("Exception message: %s", message)
        pool.terminate()
        event.set()
        raise OpenMMQMMMError(f"Terminating pool processes due to worker exception: {message}")

    # ----------
    # START
    # ----------
    if separate_dirs is True:
        for i in range(numcores):
            workerdir = f"Pooljob_{i}"
            logger.info(f"separate_dirs option True. Creating dir {workerdir}")
            logger.info("Creating workerdir: %s", workerdir)
            with contextlib.suppress(FileExistsError):
                os.mkdir(workerdir)

        # Default 0
    # Collecting results in a list of tuples from each process
    results = []
    logger.info("Now looping")

    # Starting process loop
    for process in range(numcores):
        logger.info("Starting process: %s", process)
        # Taking copy of parameter_dict
        parameter_dict_new = copy.copy(parameter_dict)
        # Adding process_id to parameter_dict
        # NOTE: jobfunction run method must have a process_id keyword to be compatible. Add as needed?
        parameter_dict_new["process_id"] = process
        if separate_dirs is True:
            parameter_dict_new["workerdir"] = f"Pooljob_{process}"
        logger.info("parameter_dict_new: %s", parameter_dict_new)
        # Calling apply_async.
        results.append(
            (process, pool.apply_async(jobfunction, kwds=parameter_dict_new, error_callback=terminate_pool_processes))
        )

    # CLOSING POOL
    pool.close()
    pool.join()
    event.set()
    # While loop that is only terminated if processes finished or exception occurred
    while True:
        logger.info("Pool multiprocessing underway....")
        time.sleep(3)
        if event.is_set():
            logger.info("Event has been set! Now terminating Pool processes")
            pool.terminate()
            break

    ##############################################################
    # END OF POOL
    ###############################################################
    ###########
    # RESULTS
    ###########
    results_dict = {}
    # results_dict is a dictionary of a result objection from jobfunction (whatever that may be)
    # where keys are process-IDs
    for pr, res in results:
        results_dict[pr] = res.get()
    logger.info("Returning result of Simple_parallel as dict: %s", results_dict)
    return results_dict
