from __future__ import annotations

import contextlib
import copy
import fcntl
import functools
import logging
import math
import os
import shutil
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

import openmmqmmm.constants
import openmmqmmm.coords
import openmmqmmm.orca
from openmmqmmm.coords import Fragment, check_charge_mult
from openmmqmmm.exceptions import ExternalProgramError, InputError, InternalError
from openmmqmmm.numgrad import _displaced_geometries, _displacement_label
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.results import Results
from openmmqmmm.utils import (
    clean_number,
    listdiff,
    log_time_since,
    main_header,
    require_int_in_range,
    require_positive_finite,
)

logger = logging.getLogger(__name__)

type Displacement = tuple[int, int, str] | str

# A linear molecule off the Cartesian axes gets roundoff (~1e-16 relative), not 0.0, for its zero moment.
_ZERO_MOMENT_RTOL = 1e-10

_NUMFREQ_DIRECTORY = "Numfreq_dir"
_NUMFREQ_MARKER = ".openmmqmmm-managed"
_NUMFREQ_MARKER_CONTENT = "Managed numerical-frequency workspace. Its contents may be replaced.\n"
_NUMFREQ_LOCK = ".openmmqmmm-active.lock"
_NUMFREQ_PROCESS_LOCK = threading.Lock()
_NUMFREQ_LOCK_OWNER: ContextVar[str | None] = ContextVar("numfreq_lock_owner", default=None)
_NUMFREQ_LOCK_HANDLE: ContextVar[Any | None] = ContextVar("numfreq_lock_handle", default=None)


def _release_numfreq_lock(parent: Path, owner: str) -> None:
    """Release this invocation's advisory lock without disturbing another owner."""
    lock = parent / _NUMFREQ_DIRECTORY / _NUMFREQ_LOCK
    lock_file = _NUMFREQ_LOCK_HANDLE.get()
    if lock_file is None:
        return
    try:
        lock_file.seek(0)
        stored_owner = lock_file.read()
        if stored_owner != owner:
            logger.warning("Numerical-frequency lock %s changed owner while held", lock)
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.flush()
    except (OSError, UnicodeError) as error:
        logger.warning("Could not clear numerical-frequency workspace lock %s: %s", lock, error)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            lock_file.close()
        _NUMFREQ_LOCK_HANDLE.set(None)


def _restore_working_directory[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Serialize process-local runs and restore their directory and owned lock."""

    @functools.wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if not _NUMFREQ_PROCESS_LOCK.acquire(blocking=False):
            raise InputError("Another numerical-frequency calculation is already active in this process")
        try:
            original_directory = Path.cwd()
        except BaseException:
            _NUMFREQ_PROCESS_LOCK.release()
            raise

        owner = f"{os.getpid()}:{threading.get_ident()}:{uuid4().hex}"
        owner_context = _NUMFREQ_LOCK_OWNER.set(owner)
        handle_context = _NUMFREQ_LOCK_HANDLE.set(None)
        try:
            return function(*args, **kwargs)
        finally:
            try:
                os.chdir(original_directory)
            finally:
                try:
                    _release_numfreq_lock(original_directory, owner)
                finally:
                    _NUMFREQ_LOCK_HANDLE.reset(handle_context)
                    _NUMFREQ_LOCK_OWNER.reset(owner_context)
                    _NUMFREQ_PROCESS_LOCK.release()

    return wrapped


def _prepare_numfreq_directory(parent: Path, owner: str) -> Path:
    """Acquire and safely refresh the managed numerical-frequency workspace."""
    directory = parent / _NUMFREQ_DIRECTORY
    marker = directory / _NUMFREQ_MARKER
    lock = directory / _NUMFREQ_LOCK

    if not directory.exists() and not directory.is_symlink():
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        else:
            marker.write_text(_NUMFREQ_MARKER_CONTENT, encoding="utf-8")

    managed = False
    if directory.is_dir() and not directory.is_symlink() and marker.is_file() and not marker.is_symlink():
        with contextlib.suppress(OSError, UnicodeError):
            managed = marker.read_text(encoding="utf-8") == _NUMFREQ_MARKER_CONTENT
    if not managed:
        raise InputError(
            f"{directory.name} already exists and is not an openmmqmmm-managed workspace; refusing to delete "
            "or overwrite it. Move it aside and retry."
        )
    if lock.is_symlink():
        raise InputError(f"{_NUMFREQ_LOCK} must be a regular file, not a symbolic link")

    lock_file = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        existing_owner = lock_file.read()
        lock_file.close()
        owner_detail = f" (owner {existing_owner})" if existing_owner else ""
        raise InputError(
            f"{directory.name} is already in use by another numerical-frequency calculation{owner_detail}"
        ) from None
    except OSError:
        lock_file.close()
        raise

    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(owner)
        lock_file.flush()
        os.fsync(lock_file.fileno())
    except OSError:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
        raise
    _NUMFREQ_LOCK_HANDLE.set(lock_file)

    # A lock file left by a killed process is harmless: the kernel releases the
    # advisory lock, this invocation acquires it above, and the recorded owner is
    # replaced. Only a descriptor that is still locked blocks workspace cleanup.
    for entry in directory.iterdir():
        if entry in (marker, lock):
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    return directory


def _copy_orca_guess(theory: Any, source_directory: Path, scratch_directory: Path) -> None:
    """Copy an available ORCA GBW guess into the displacement workspace."""
    orca_theory = None
    if isinstance(theory, openmmqmmm.orca.ORCATheory):
        orca_theory = theory
    elif isinstance(getattr(theory, "qm_theory", None), openmmqmmm.orca.ORCATheory):
        orca_theory = theory.qm_theory

    filename = getattr(orca_theory, "filename", None)
    if filename is None:
        return

    source = source_directory / f"{filename}.gbw"
    if not source.is_file():
        return

    destination = scratch_directory / source.name
    try:
        shutil.copy2(source, destination)
    except OSError as error:
        logger.warning("Could not copy ORCA GBW guess %s into %s: %s", source, scratch_directory, error)
    else:
        logger.info("Copied ORCA GBW guess into %s", scratch_directory.name)


def _validate_thermo_options(
    temp: float,
    pressure: float,
    qrrho: bool,
    qrrho_method: str,
    qrrho_omega_0: float,
    symmetry_number: int | None,
    rotmode_threshold: float,
    scaling_factor: float,
) -> dict[str, Any]:
    """Normalize shared analysis options before any external job or workspace mutation."""
    temp = require_positive_finite(temp, "temperature must be a positive finite number")
    pressure = require_positive_finite(pressure, "pressure must be a positive finite number")
    scaling_factor = require_positive_finite(scaling_factor, "scaling_factor must be a positive finite number")
    message = "rotmode_threshold must be a non-negative finite number"
    if isinstance(rotmode_threshold, (bool, np.bool_)):
        raise InputError(message)
    try:
        rotmode_threshold = float(rotmode_threshold)
    except (OverflowError, TypeError, ValueError):
        raise InputError(message) from None
    if not math.isfinite(rotmode_threshold) or rotmode_threshold < 0:
        raise InputError(message)
    if qrrho:
        if qrrho_method not in ("Grimme", "Truhlar"):
            raise InputError("Unknown qrrho_method. Choose 'Grimme' or 'Truhlar'.")
        qrrho_omega_0 = require_positive_finite(qrrho_omega_0, "qrrho_omega_0 must be a positive finite number")
    if symmetry_number is not None:
        symmetry_number = require_int_in_range(symmetry_number, "symmetry_number must be a positive integer")
    return {
        "temp": temp,
        "pressure": pressure,
        "qrrho": qrrho,
        "qrrho_method": qrrho_method,
        "qrrho_omega_0": qrrho_omega_0,
        "symmetry_number": symmetry_number,
        "rotmode_threshold": rotmode_threshold,
        "scaling_factor": scaling_factor,
    }


def _validated_masses(masses: Sequence[float], count: int, name: str) -> list[float]:
    try:
        values = list(masses)
    except TypeError:
        raise InputError(f"{name} must be a sequence of positive finite numbers") from None
    if len(values) != count:
        raise InputError(
            f"Number of provided masses ({name} keyword) is not equal to number of Hessian-atoms. Check input masses!"
        )
    return [require_positive_finite(value, f"{name} must contain only positive finite numbers") for value in values]


def analytic_frequencies(
    *,
    fragment: Fragment | None = None,
    theory: Any | None = None,
    charge: int | None = None,
    mult: int | None = None,
    temp: float = 298.15,
    masses: Sequence[float] | None = None,
    pressure: float = 1.0,
    qrrho: bool = True,
    qrrho_method: str = "Grimme",
    qrrho_omega_0: float = 100,
    scaling_factor: float = 1.0,
    symmetry_number: int | None = None,
    rotmode_threshold: float = 1e-4,
) -> Results:
    """Compute vibrational frequencies from an analytical Hessian provided by the theory."""
    module_init_time = time.time()
    logger.info("------------ANALYTICAL FREQUENCIES-------------")

    if fragment is None or theory is None:
        raise InputError("analytic_frequencies requires a fragment and a theory object")

    thermo_options = _validate_thermo_options(
        temp, pressure, qrrho, qrrho_method, qrrho_omega_0, symmetry_number, rotmode_threshold, scaling_factor
    )
    hessatoms = list(range(fragment.numatoms))
    masses = _validated_masses(fragment.list_of_masses if masses is None else masses, fragment.numatoms, "masses")
    if not getattr(theory, "analytic_hessian", False):
        raise InputError(
            f"Analytical frequencies are not available for {theory.__class__.__name__}. "
            "Use numerical_frequencies instead."
        )
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "AnFreq", theory=theory)
    theory.run(current_coords=fragment.coords, elems=fragment.elems, charge=charge, mult=mult, hessian=True)
    # Preserve the theory's IR intensities: custom masses do not recompute this property.
    try:
        analytic_ir = theory.ir_intensities
    except (AttributeError, KeyError, TypeError):
        analytic_ir = None
    result = _analyse_hessian(
        fragment=fragment,
        hessian=theory.hessian,
        hessatoms=hessatoms,
        hessmasses=masses,
        mult=mult,
        projection=True,
        label="Anfreq",
        analytic_ir=analytic_ir,
        **thermo_options,
    )
    logger.info("------------ANALYTICAL FREQUENCIES END-------------")
    log_time_since(module_init_time, "AnFreq")
    result.write_to_disk(filename="results_anfreq.json")
    return result


# Numerical frequencies retain a 0.005 Å default; NumGrad uses 0.005 Bohr in Å.
def _build_displacements(
    *,
    coords: np.ndarray,
    elems: Sequence[str],
    hessatoms: Sequence[int],
    displacement: float,
    npoint: int,
    charge: int,
    mult: int,
) -> tuple[list[np.ndarray], list[Displacement], list[str], list[Fragment]]:
    """Return the displaced geometries, their displacement tuples, log labels and fragments."""
    geometries, displacements = _displaced_geometries(coords, hessatoms, displacement, npoint)
    if npoint == 1:
        # Forward difference needs the undisplaced gradient as its reference
        geometries.append(np.array(coords, copy=True))
        displacements.append("Originalgeo")
    logger.info("List of displacements: %s", displacements)

    axis_names = ("x", "y", "z")
    labels = []
    fragments = []
    for geometry, disp in zip(geometries, displacements, strict=True):
        if disp == "Originalgeo":
            calclabel = stringlabel = "Originalgeo"
        else:
            atom_disp, axis, direction = disp
            calclabel = f"Atom: {atom_disp} Coord: {axis_names[axis]} Direction: {direction}"
            stringlabel = _displacement_label(disp)
        frag = openmmqmmm.Fragment(coords=geometry, elems=elems, label=stringlabel, charge=charge, mult=mult)
        fragments.append(frag)
        labels.append(calclabel)
    return geometries, displacements, labels, fragments


def _run_displacements_serially(
    *,
    theory: Any,
    elems: Sequence[str],
    charge: int,
    mult: int,
    geometries: Sequence[np.ndarray],
    displacements: Sequence[Displacement],
    labels: Sequence[str],
    IR: bool,
    Raman: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Sequence[float] | np.ndarray | None],
    dict[str, np.ndarray],
]:
    """Run every displaced geometry in turn, collecting gradients and optional properties."""
    grads = {}
    dipoles = {}
    polarizabilities = {}
    logger.info(
        "Runmode: serial. Only theory parallelization is active; theory numcores is set to: %s", theory.numcores
    )
    for numdisp, (disp, label, geo) in enumerate(zip(displacements, labels, geometries, strict=True)):
        if label == "Originalgeo":
            stringlabel = "Originalgeo"
            logger.debug("Doing original geometry calc.")
        else:
            stringlabel = _displacement_label(disp)
            logger.debug("Running displacement %s / %s: %s", numdisp + 1, len(labels), label)
        _energy, gradient = theory.run(current_coords=geo, elems=elems, grad=True, charge=charge, mult=mult)
        grads[stringlabel] = gradient

        if IR is True:
            with contextlib.suppress(Exception):  # best-effort property grab
                dipoles[stringlabel] = theory.get_dipole_moment()

        if Raman is True:
            try:
                logger.debug("Getting polarizability tensor")
                displacement_pol = theory.get_polarizability_tensor()
                if not np.any(displacement_pol):
                    logger.warning("No polarizability information found")
                polarizabilities[stringlabel] = displacement_pol
            except Exception:  # noqa: BLE001 - best-effort polarizability grab
                logger.warning("Problem getting polarizability tensor from theory interface. Skipping")
    return grads, dipoles, polarizabilities


def _run_displacements_in_parallel(
    *, theory: Any, fragments: Sequence[Fragment], numcores: int
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Sequence[float] | np.ndarray],
    dict[str, np.ndarray],
]:
    """Run every displaced geometry through the job-parallel driver."""
    if isinstance(theory, openmmqmmm.QMMMTheory):
        logger.info("Numfreq in runmode='parallel' with QM/MM is quite experimental")
    logger.debug(
        "Starting Numfreq calculations in parallel mode (numcores=%s) over %s displacements",
        numcores,
        len(fragments),
    )
    result = openmmqmmm.job_parallel(
        fragments=fragments,
        theories=[theory],
        numcores=numcores,
        allow_theory_parallelization=True,
        grad=True,
        copytheory=True,
    )
    return (
        result.gradients_dict,
        result.displacement_dipole_dictionary,
        result.displacement_polarizability_dictionary,
    )


def _expected_displacement_labels(npoint: int, hessatoms: Sequence[int]) -> list[str]:
    labels = []
    directions = ("+",) if npoint == 1 else ("+", "-")
    for atomindex in hessatoms:
        for coordinate in (0, 1, 2):
            labels.extend(_displacement_label((atomindex, coordinate, direction)) for direction in directions)
    if npoint == 1:
        labels.append("Originalgeo")
    return labels


def _property_mapping_is_complete(
    values: Mapping[str, Any],
    expected_labels: Sequence[str],
    property_name: str,
    expected_shape: tuple[int, ...],
) -> bool:
    """Validate optional displacement data, disabling it only when entirely absent."""
    present = []
    missing = []
    malformed = []
    for label in expected_labels:
        if label not in values or values[label] is None:
            missing.append(label)
            continue
        try:
            value = np.asarray(values[label])
        except (TypeError, ValueError):
            present.append(label)
            malformed.append(label)
            continue
        if value.size == 0:
            missing.append(label)
            continue
        present.append(label)
        try:
            finite = bool(np.all(np.isfinite(value)))
        except TypeError:
            finite = False
        if value.shape != expected_shape or not finite:
            malformed.append(label)

    if not present:
        return False
    if missing:
        malformed_detail = f"; malformed labels: {', '.join(malformed)}" if malformed else ""
        raise InternalError(
            f"Incomplete {property_name} data for numerical frequencies; missing displacement labels: "
            f"{', '.join(missing)}{malformed_detail}"
        )
    if malformed:
        raise InternalError(
            f"Invalid {property_name} data for numerical frequencies; expected finite arrays with shape "
            f"{expected_shape}; malformed displacement labels: {', '.join(malformed)}"
        )
    return True


def _assemble_hessian(
    *,
    npoint: int,
    hessatoms: Sequence[int],
    displacement_bohr: float,
    grads: Mapping[str, np.ndarray],
    dipoles: Mapping[str, Sequence[float] | np.ndarray | None],
    polarizabilities: Mapping[str, np.ndarray],
    IR: bool,
    Raman: bool,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Finite-difference the displacement data into a symmetrised Hessian plus dipole and polarizability derivatives."""
    logger.info("Assembling the %s-point Hessian", npoint)
    hesslength = 3 * len(hessatoms)
    hessian = np.zeros((hesslength, hesslength))
    dipole_derivs = np.zeros((hesslength, 3))
    polarizability_derivs = []

    expected_labels = _expected_displacement_labels(npoint, hessatoms)
    missing_gradients = [label for label in expected_labels if label not in grads or grads[label] is None]
    if missing_gradients:
        raise InternalError(
            "Incomplete gradient data for numerical frequencies; missing displacement labels: "
            f"{', '.join(missing_gradients)}"
        )

    want_dipoles = IR is True and _property_mapping_is_complete(dipoles, expected_labels, "dipole", (3,))
    want_polarizabilities = Raman is True and _property_mapping_is_complete(
        polarizabilities, expected_labels, "polarizability", (3, 3)
    )
    # Forward difference measures against the undisplaced geometry over one step; central
    # difference measures the two displacements against each other, so over two.
    step = displacement_bohr if npoint == 1 else 2 * displacement_bohr

    hessindex = 0
    for atomindex in hessatoms:
        for crd in (0, 1, 2):
            plus = _displacement_label((atomindex, crd, "+"))
            minus = "Originalgeo" if npoint == 1 else _displacement_label((atomindex, crd, "-"))

            grad_plus = np.ravel(_get_partial_matrix(grads[plus], hessatoms))
            grad_minus = np.ravel(_get_partial_matrix(grads[minus], hessatoms))
            hessian[hessindex, :] = (grad_plus - grad_minus) / step

            if want_dipoles:
                dipole_derivs[hessindex, :] = (np.array(dipoles[plus]) - np.array(dipoles[minus])) / step
            if want_polarizabilities:
                polarizability_derivs.append(
                    (np.array(polarizabilities[plus]) - np.array(polarizabilities[minus])) / step
                )
            hessindex += 1

    return (hessian + hessian.transpose()) / 2, dipole_derivs, polarizability_derivs


@_restore_working_directory
def numerical_frequencies(
    *,
    fragment: Fragment | None = None,
    theory: Any | None = None,
    charge: int | None = None,
    mult: int | None = None,
    npoint: int = 2,
    displacement: float = 0.005,
    hessatoms: Sequence[int] | None = None,
    numcores: int = 1,
    runmode: str = "serial",
    temp: float = 298.15,
    pressure: float = 1.0,
    hessatoms_masses: Sequence[float] | None = None,
    qrrho: bool = True,
    qrrho_method: str = "Grimme",
    qrrho_omega_0: float = 100,
    IR: bool = True,
    Raman: bool = False,
    rotmode_threshold: float = 1e-4,
    scaling_factor: float = 1.0,
    symmetry_number: int | None = None,
    force_projection: bool | None = None,
) -> Results:
    """Compute vibrational frequencies from numerical differentiation of gradients."""
    module_init_time = time.time()
    logger.info("------------NUMERICAL FREQUENCIES-------------")
    if fragment is None or theory is None:
        raise InputError("numerical_frequencies requires a fragment and a theory object")
    if isinstance(theory, QMMMTheory) and theory.embedding == "elstat" and theory.truncated_pc:
        theory._validate_truncated_pc_settings(require_gradients=True)

    npoint = require_int_in_range(
        npoint, "Unknown npoint option. npoint should be 1 (forward) or 2 (central difference).", maximum=2
    )
    if runmode not in ("serial", "parallel"):
        raise InputError("Unknown runmode. Choose 'serial' or 'parallel'.")
    numcores = require_int_in_range(numcores, "numcores must be a positive integer")
    displacement = require_positive_finite(displacement, "displacement must be a positive finite number")
    thermo_options = _validate_thermo_options(
        temp, pressure, qrrho, qrrho_method, qrrho_omega_0, symmetry_number, rotmode_threshold, scaling_factor
    )
    if force_projection is not None and not isinstance(force_projection, bool):
        raise InputError("force_projection must be True, False, or None")

    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "NumFreq", theory=theory)
    coords = fragment.coords
    elems = copy.deepcopy(fragment.elems)
    numatoms = len(elems)
    allatoms = list(range(numatoms))

    if hessatoms is None:
        logger.info("No Hessatoms provided. Full Hessian assumed. Rot+trans projection is on!")
        if isinstance(theory, QMMMTheory):
            logger.info("Theory object provided is a QM/MM Theory")
            raise InputError(
                "No hessatoms option was provided. This is required for QM/MM Theories\nPlease provide a list "
                "of atom indices to the hessatoms keyword of numerical_frequencies to define the partial "
                "Hessian\nFor QM/MM "
                "numerical frequencies you want the list of hessatoms to be the same atoms used to define the "
                "\nactive-region in the optimization (or the QM-region)"
            )
        hessatoms = allatoms
    else:
        try:
            hessatoms = list(hessatoms)
        except TypeError:
            raise InputError("hessatoms must be a sequence of atom indices") from None
        if not hessatoms:
            raise InputError("hessatoms list is empty")
        invalid_types = [index for index in hessatoms if not isinstance(index, Integral) or isinstance(index, bool)]
        if invalid_types:
            raise InputError(f"hessatoms contains non-integer indices: {invalid_types}")
        invalid_indices = [int(index) for index in hessatoms if index < 0 or index >= numatoms]
        if invalid_indices:
            raise InputError(f"hessatoms contains indices outside 0..{numatoms - 1}: {invalid_indices}")
        hessatoms = [int(index) for index in hessatoms]
        duplicate_indices = sorted(index for index, count in Counter(hessatoms).items() if count > 1)
        if duplicate_indices:
            raise InputError(f"hessatoms contains duplicate indices: {duplicate_indices}")

    if len(hessatoms) == fragment.numatoms:
        logger.info("Hessian covers every fragment atom. Rot+trans projection is on!")
        projection = True
    else:
        logger.info("Hessatoms list provided, partial Hessian. Turning off rot+trans projection")
        projection = False

    if force_projection is not None:
        logger.warning("Option force_projection is in use")
        if force_projection is True:
            logger.info("force_projection set to True. Turning projection on")
            projection = True
        elif force_projection is False:
            logger.info("force_projection set to to False. Turning projection off")
            projection = False

    if hessatoms_masses is not None:
        hessatoms_masses = _validated_masses(hessatoms_masses, len(hessatoms), "hessatoms_masses")
    original_directory = Path.cwd()
    lock_owner = _NUMFREQ_LOCK_OWNER.get()
    if lock_owner is None:
        raise InternalError("Numerical-frequency workspace owner was not initialized")
    scratch_directory = _prepare_numfreq_directory(original_directory, lock_owner)
    _copy_orca_guess(theory, original_directory, scratch_directory)
    os.chdir(scratch_directory)
    logger.debug("Using managed displacement workspace: %s", scratch_directory)

    displacement_bohr = displacement * openmmqmmm.constants.ANG_TO_BOHR
    logger.info("Starting Numerical Frequencies job for fragment")
    logger.info("Hessian atoms: %s", hessatoms)
    if hessatoms != allatoms:
        logger.info("This is a partial Hessian job.")
    if npoint == 1:
        logger.info("One-point formula used (forward difference)")
    else:
        logger.info("Two-point formula used (central difference)")
    if runmode == "serial":
        logger.info("Numfreq running in serial mode")
    else:
        logger.info("Numfreq running in parallel mode")
    logger.info(f"\nDisplacement: {displacement:5.4f} Å ({displacement_bohr:5.4f} Bohr)")
    logger.debug("\nStarting geometry:")
    logger.info("Printing hessatoms geometry...")
    openmmqmmm.coords.print_coords_for_atoms(coords, elems, hessatoms)

    list_of_displaced_geos, list_of_displacements, list_of_labels, all_disp_fragments = _build_displacements(
        coords=coords,
        elems=elems,
        hessatoms=hessatoms,
        displacement=displacement,
        npoint=npoint,
        charge=charge,
        mult=mult,
    )

    if runmode == "serial":
        grads, dipoles, polarizabilities = _run_displacements_serially(
            theory=theory,
            elems=elems,
            charge=charge,
            mult=mult,
            geometries=list_of_displaced_geos,
            displacements=list_of_displacements,
            labels=list_of_labels,
            IR=IR,
            Raman=Raman,
        )
    else:
        grads, dipoles, polarizabilities = _run_displacements_in_parallel(
            theory=theory, fragments=all_disp_fragments, numcores=numcores
        )
    displacement_grad_dictionary = grads
    displacement_dipole_dictionary = dipoles
    displacement_polarizability_dictionary = polarizabilities

    logger.info("numerical_frequencies displacement calculations are done!\n")

    if len(displacement_grad_dictionary) == 0:
        raise InputError(
            "Missing gradients for displacement.\nSomething went wrong in the numerical_frequencies "
            "displacement calculations."
        )
    logger.info("Length of displacement_grad_dictionary %s", len(displacement_grad_dictionary))
    hessian, dipole_derivs, polarizability_derivs = _assemble_hessian(
        npoint=npoint,
        hessatoms=hessatoms,
        displacement_bohr=displacement_bohr,
        grads=displacement_grad_dictionary,
        dipoles=displacement_dipole_dictionary,
        polarizabilities=displacement_polarizability_dictionary,
        IR=IR,
        Raman=Raman,
    )

    if hessatoms_masses is None:
        logger.info("allatoms: %s", allatoms)
        logger.info("hessatoms: %s", hessatoms)
        logger.debug("Atomic masses: %s", fragment.list_of_masses)
        hessmasses = [fragment.list_of_masses[index] for index in hessatoms]
    else:
        hessmasses = hessatoms_masses

    logger.info("hessmasses: %s", hessmasses)
    hesselems = [elems[index] for index in hessatoms]

    hesscoords = np.take(fragment.coords, hessatoms, axis=0)
    logger.info("Elements: %s", hesselems)
    logger.info("Masses used: %s", hessmasses)

    result = _analyse_hessian(
        fragment=fragment,
        hessian=hessian,
        hessatoms=hessatoms,
        hessmasses=hessmasses,
        mult=mult,
        projection=projection,
        label="Numfreq",
        IR=IR,
        Raman=Raman,
        dipole_derivs=dipole_derivs,
        polarizability_derivs=polarizability_derivs,
        **thermo_options,
    )
    openmmqmmm.orca.write_orca_hessfile(hessian, hesscoords, hesselems, hessmasses, "orcahessfile.hess")
    logger.info("------------NUMERICAL FREQUENCIES END-------------")
    os.chdir(original_directory)
    log_time_since(module_init_time, "NumFreq")
    result.write_to_disk(filename="results_numfreq.json")
    return result


def _analyse_hessian(
    *,
    fragment: Fragment,
    hessian: np.ndarray,
    hessatoms: Sequence[int],
    hessmasses: Sequence[float],
    mult: int | None,
    projection: bool,
    label: str,
    scaling_factor: float,
    IR: bool = False,
    Raman: bool = False,
    analytic_ir: Sequence[float] | None = None,
    dipole_derivs: np.ndarray | None = None,
    polarizability_derivs: Sequence[np.ndarray] = (),
    **thermo_options: Any,
) -> Results:
    """Analyse either source of Hessians, preserving the Hessian atom ordering."""
    hesselems = [fragment.elems[index] for index in hessatoms]
    hesscoords = np.take(fragment.coords, hessatoms, axis=0)
    tr_modenum = _tr_mode_count(hesscoords, hessmasses, thermo_options["rotmode_threshold"])
    # Evectors: eigenvectors of the mass-weighted Hessian
    # Normal modes: unweighted
    frequencies, nmodes, evectors, _mode_order = _diagonalize_hessian(
        hesscoords,
        hessian,
        hessmasses,
        hesselems,
        tr_modenum=tr_modenum,
        projection=projection,
        rotmode_threshold=thermo_options["rotmode_threshold"],
    )
    logger.info("Diagonalization of frequencies complete")
    logger.info("Now scaling frequencies by scaling factor: %s", scaling_factor)
    frequencies = scaling_factor * np.array(frequencies)

    IR_intens_values = analytic_ir
    if label == "Anfreq" and analytic_ir is not None:
        if len(analytic_ir) == 0:
            IR_intens_values = None
        elif len(analytic_ir) < len(frequencies):
            IR_intens_values = [0.0] * (len(frequencies) - len(analytic_ir)) + list(analytic_ir)
    elif IR and np.any(dipole_derivs):
        IR_intens_values = _calc_ir_intensities(nmodes, dipole_derivs)

    if Raman is True:
        logger.info("Raman calculation active")
        if len(polarizability_derivs) == 0:
            logger.debug("No polarizability information found. Skipping Raman.")
            raman_activities = None
            depolarization_ratios = None
        else:
            logger.info("Polarizability derivatives are available.")
            raman_activities, depolarization_ratios = _calc_raman_activities(nmodes, polarizability_derivs)
    else:
        raman_activities = None
        depolarization_ratios = None

    _log_frequencies(
        frequencies,
        len(hessatoms),
        tr_modenum=tr_modenum,
        intensities=IR_intens_values,
        raman_activities=raman_activities,
    )

    logger.info("Normal mode composition factors by element")
    _log_frequencies_and_mode_compositions(frequencies, fragment, evectors, hessatoms=hessatoms, tr_modenum=tr_modenum)

    logger.debug("\nNow doing thermochemistry")

    thermodict = calc_thermochemistry(
        frequencies,
        hessatoms,
        fragment,
        mult,
        **thermo_options,
    )

    fragment.hessian = hessian
    write_hessian(hessian, hessfile="Hessian")
    _write_dummy_orca_file(hesselems, hesscoords, frequencies, nmodes, "orcahessfile.hess")
    return Results(
        label=label,
        hessian=hessian,
        vib_eigenvectors=evectors,
        frequencies=frequencies,
        raman_activities=raman_activities,
        depolarization_ratios=depolarization_ratios,
        ir_intensities=IR_intens_values,
        freq_atoms=hessatoms,
        freq_elems=hesselems,
        freq_coords=hesscoords,
        freq_masses=hessmasses,
        freq_tr_modenum=tr_modenum,
        freq_projection=projection,
        freq_scaling_factor=scaling_factor,
        freq_dipole_derivs=dipole_derivs,
        normal_modes=nmodes,
        thermochemistry=thermodict,
        freq_raman=Raman,
        freq_polarizability_derivs=polarizability_derivs,
    )


def _get_partial_matrix(matrix: np.ndarray, hessatoms: Sequence[int]) -> np.ndarray:
    return np.take(matrix, hessatoms, axis=0)


def _diagonalize_hessian(
    coords: np.ndarray,
    hessian: np.ndarray,
    masses: Sequence[float],
    elems: Sequence[str],
    projection: bool = True,
    tr_modenum: int | None = None,
    LargeImagFreqThreshold: float = -100,
    rotmode_threshold: float = 1e-4,
) -> tuple[np.ndarray | list[float], np.ndarray, np.ndarray, list[int]]:
    logger.info("\nDiagonalizing Hessian")
    atomlist = []
    for i, j in enumerate(elems):
        atomlist.append(str(j) + "-" + str(i))

    if projection is True:
        logger.info("Projection of out rotational and translational modes active!")
        vfreqs, evectors, nmodes, tr_modenum = _project_rot_and_trans(
            coords, masses, hessian, rotmode_threshold=rotmode_threshold
        )
        for _ in range(tr_modenum):
            vfreqs = np.insert(vfreqs, 0, 0.0)
        for _ in range(tr_modenum):
            evectors = np.insert(evectors, 0, [0.0] * evectors.shape[1], axis=0)
            nmodes = np.insert(nmodes, 0, [0.0] * nmodes.shape[1], axis=0)

        mode_order = list(range(len(nmodes)))
        return vfreqs, nmodes, evectors, mode_order
    logger.debug("No projection of rotational and translational modes will be done!")
    mwhessian, massmatrix = _mass_weight_hessian(hessian, masses)
    evalues, evectors = np.linalg.eigh(mwhessian)
    evectors = np.transpose(evectors)

    # Unweight eigenvectors to get normal modes
    nmodes = evectors * massmatrix

    vfreqs = _eigenvalues_to_wavenumbers(evalues)

    logger.info("Calculated frequencies: %s", vfreqs)
    # Unprojected, the lowest modes mix TR modes with saddle-point modes. Heuristic: modes below
    # LargeImagFreqThreshold are SP modes; the other small imaginary or low positive ones are TR
    # modes, and their frequencies are not zeroed.
    logger.info("Identifying TRmodes and SPmodes")
    TRmodes = []
    SPmodes = []
    for i, f in enumerate(vfreqs):
        if f < 0.0:
            if f < LargeImagFreqThreshold:
                logger.info("High negative freq found (< -100). Assumed to be SP-mode.")
                SPmodes.append(i)
            else:
                TRmodes.append(i)
        elif len(TRmodes) < tr_modenum:
            logger.info("Not enough TRmodes found. Adding mode to TRmodes")
            TRmodes.append(i)

    logger.info("TRmodes: %s", TRmodes)
    logger.info("SPmodes: %s", SPmodes)
    logger.info("Reordering modes so that TRmodes come first, then SP modes, then rest")
    neworder = TRmodes + SPmodes + listdiff(range(len(vfreqs)), TRmodes + SPmodes)
    vfreqs = [vfreqs[i] for i in neworder]
    evectors = evectors[neworder]
    nmodes = nmodes[neworder]

    return vfreqs, nmodes, evectors, neworder


def _calc_ir_intensities(nmodes: np.ndarray, dipole_derivs: np.ndarray) -> np.ndarray:
    de_q = nmodes @ dipole_derivs
    return openmmqmmm.constants.IR_INTENSITY_AU_TO_KM_PER_MOL * np.einsum("qt, qt -> q", de_q, de_q)


def _mass_weight_hessian(matrix: np.ndarray, masses: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    inverse_sqrt_masses = 1 / np.sqrt(np.repeat(masses, 3))
    return matrix * np.outer(inverse_sqrt_masses, inverse_sqrt_masses), inverse_sqrt_masses


def _eigenvalues_to_wavenumbers(evalues: Sequence[float]) -> np.ndarray:
    """Convert mass-weighted Hessian eigenvalues, using negative imaginary modes."""
    values = np.asarray(evalues)
    scale = np.sqrt(
        openmmqmmm.constants.HARTREE_TO_J / openmmqmmm.constants.BOHR_TO_M**2 / openmmqmmm.constants.AMU_TO_KG
    )
    scale /= 2 * np.pi * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S
    return scale * np.sqrt(np.abs(values)) * np.sign(values)


def _log_frequencies(
    vfreq: Sequence[float] | np.ndarray,
    numatoms: int,
    tr_modenum: int = 6,
    intensities: Sequence[float] | np.ndarray | None = None,
    raman_activities: Sequence[float] | np.ndarray | None = None,
) -> None:
    logger.info("%s", "-" * 40)
    logger.info("VIBRATIONAL FREQUENCY SUMMARY")
    logger.info("%s", "-" * 40)
    if intensities is None:
        logger.debug("No IR intensities were calculated. Setting values to 0.0.")
    if raman_activities is None:
        logger.debug(
            "No Raman activities were calculated (polarizabilities not available in QM-program interface). Setting "
            "values to 0.0."
        )
    logger.info("Note: imaginary modes shown as negative")
    logger.info(
        "%s", "{:>6}{:>16}  {:>16} {:>20}".format("Mode", "Freq(cm**-1)", "IR Int.(km/mol)", "Raman Act.(Å^4/amu)")
    )
    for mode in range(3 * numatoms):
        vib = vfreq[mode]
        intensity = 0.0 if intensities is None else intensities[mode]
        raman_act = 0.0 if raman_activities is None else raman_activities[mode]
        line = f"  {mode:<6d}{vib:>14.4f}{intensity:>14.4f}{raman_act:>16.4f}"
        if mode < tr_modenum:
            line = line + "            (TR mode)"
        logger.info("%s", line)


def _log_frequencies_and_mode_compositions(
    vfreq: Sequence[float] | np.ndarray,
    fragment: Fragment,
    evectors: np.ndarray,
    hessatoms: Sequence[int] | None = None,
    tr_modenum: int = 6,
    numdigits: int = 3,
) -> None:
    with open("normalmodecomposition_factors.txt", "w") as f:
        numatoms = len(hessatoms)
        logger.info("%s", "{:>6}{:>16}  {:<18}".format("Mode", "Freq(cm**-1)", "Elemental composition factors"))
        for mode in range(3 * numatoms):
            normmodecompelemsdict = _normal_mode_components_by_element(mode, fragment, evectors, hessatoms=hessatoms)
            normmodecompelemsdict_list = [f"{k}: {v:.{numdigits}f}" for k, v in normmodecompelemsdict.items()]
            normmodecompelemsdict_string = "   ".join(normmodecompelemsdict_list)
            vib = vfreq[mode]
            line = f"  {mode:<4d}{vib:>14.4f}    {normmodecompelemsdict_string}"

            if mode < tr_modenum:
                line = line + " (TR mode)"
            logger.info("%s", line)
            f.write(line + "\n")


def _rotational_temperature(moment_of_inertia_si: float) -> float:
    """Return the rotational temperature in K for one principal moment of inertia."""
    return openmmqmmm.constants.PLANCK_J_S**2 / (
        8 * math.pi**2 * openmmqmmm.constants.BOLTZMANN_J_PER_K * moment_of_inertia_si
    )


def _rotational_thermochemistry(
    *,
    fragment: Fragment,
    coords: np.ndarray,
    elems: Sequence[str],
    moltype: str,
    temp: float,
    symmetry_number: int | None,
) -> dict[str, Any]:
    """Return the rotational energy, entropy term and the quantities the report needs."""
    if moltype == "atom":
        return {
            "E_rot": 0.0,
            "TS_rot": 0.0,
            "rinertia": None,
            "rotconstants": None,
            "inertia_avg": None,
            "sigma_r": None,
        }

    logger.debug("\nDoing rotatational analysis:")
    rinertia = [float(i) for i in inertia(elems, coords, _get_center(coords, elems=elems))]
    logger.info("Moments of inertia (amu Å^2): %s", rinertia)
    inertia_si = np.array(rinertia) * openmmqmmm.constants.AMU_TO_KG * openmmqmmm.constants.ANG_TO_M**2
    inertia_avg = float(np.mean(inertia_si))
    rotconstants = _rotational_constants_from_moments(rinertia)

    if moltype == "linear":
        rot_temps = [_rotational_temperature(in_I) for in_I in _nonzero_moments(inertia_si)]
        logger.info(f"Rotational temperatures: {rot_temps} K")
        sigma_r = 1.0 if symmetry_number is None else symmetry_number
        q_r = (1 / sigma_r) * (temp / rot_temps[0])
        S_rot = openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (math.log(q_r) + 1.0)
        E_rot = openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * temp
    else:
        rot_temps = [_rotational_temperature(in_I) for in_I in inertia_si]
        logger.info(f"Rotational temperatures: {rot_temps[0]}, {rot_temps[1]}, {rot_temps[2]} K")
        if symmetry_number is None:
            logger.debug(
                "Case: nonlinear system and no user-provided symmetry_number.\n"
                "Setting symmetry number to 1.0 (appropriate for C1, Ci and Cs pointgroups)"
            )
            sigma_r = 1.0
        else:
            logger.debug("Case: nonlinear system and user-provided symmetry_number: %s", symmetry_number)
            sigma_r = symmetry_number
        q_r = (math.pi ** (1 / 2) / sigma_r) * (temp ** (3 / 2)) / (math.prod(rot_temps) ** (1 / 2))
        S_rot = openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (math.log(q_r) + 1.5)
        E_rot = 1.5 * openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * temp

    return {
        "E_rot": E_rot,
        "TS_rot": temp * S_rot,
        "rinertia": rinertia,
        "rotconstants": rotconstants,
        "inertia_avg": inertia_avg,
        "sigma_r": sigma_r,
    }


def _vibrational_thermochemistry(
    *,
    vfreq: Sequence[float | complex],
    atoms: Sequence[int],
    tr_modenum: int,
    temp: float,
    qrrho: bool,
    qrrho_method: str,
    qrrho_omega_0: float,
    inertia_avg: float | None,
    moltype: str,
) -> dict[str, Any]:
    """Return the zero-point energy, thermal vibrational energy and vibrational entropy term."""
    if moltype == "atom":
        return {"zpve": 0.0, "E_vib": 0.0, "vibenergycorr": 0.0, "TS_vib": 0.0, "freqs": []}

    logger.debug("\nDoing vibrational analysis:")
    logger.info("Vibrational frequencies (cm**-1): %s", vfreq)
    freqs = []
    vibtemps = []
    for mode in range(3 * len(atoms)):
        if mode < tr_modenum:
            logger.info("%s %s", f"skipping TR mode ({mode}) with freq:", clean_number(vfreq[mode]))
            continue
        vib = clean_number(vfreq[mode])
        if np.iscomplex(vib):
            logger.info(f"Mode {mode} with frequency {vib} is imaginary. Skipping in thermochemistry")
        elif vib <= 0:
            # A zero frequency is not a vibration (an unprojected translation or
            # rotation, or a completely flat direction): its harmonic entropy diverges
            # and its thermal-energy term is 0/0, so it is excluded like a negative one.
            logger.info(f"Mode {mode} with frequency {vib} is not positive. Skipping in thermochemistry")
        else:
            freqs.append(float(vib))
            vibtemps.append(_vibrational_temperature(vib))

    zpve = sum(i * openmmqmmm.constants.HALF_HC_HARTREE_PER_WAVENUMBER for i in freqs)

    # Thermal vibrational energy: R * sum over modes of theta*(1/2 + 1/(exp(theta/T) - 1)),
    # the harmonic-oscillator internal energy. The Bose-Einstein factor is
    # 1/(exp(x) - 1); writing it as 1/exp(x - 1) overestimates the thermal
    # correction (2.7x for water) and does not reach the classical RT limit.
    sumb = sum(v * (0.5 + (1 / (np.exp(v / temp) - 1))) for v in vibtemps)
    E_vib = sumb * openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K

    if qrrho is not True:
        TS_vib = _s_vib(freqs, temp)
    elif qrrho_method == "Grimme":
        logger.info("QRRHO is True. Doing quasi-RRHO for the vibrational entropy")
        TS_vib = s_vib_qrrho_grimme(freqs, temp, omega_0=qrrho_omega_0, i_av=inertia_avg)
    elif qrrho_method == "Truhlar":
        logger.info("QRRHO is True. Doing quasi-RRHO for the vibrational entropy")
        TS_vib = s_vib_qrrho_truhlar(freqs, temp, lowfreq_thresh=qrrho_omega_0)
    else:
        raise InputError("Unknown QRRHO_method.")

    return {"zpve": zpve, "E_vib": E_vib, "vibenergycorr": E_vib - zpve, "TS_vib": TS_vib, "freqs": freqs}


def calc_thermochemistry(
    vfreq: Sequence[float | complex],
    atoms: Sequence[int],
    fragment: Fragment,
    multiplicity: int | None,
    *,
    temp: float = 298.15,
    pressure: float = 1.0,
    qrrho: bool = True,
    qrrho_method: str = "Grimme",
    qrrho_omega_0: float = 100,
    use_full_geo_in_rotational_analysis: bool = True,
    symmetry_number: int | None = None,
    rotmode_threshold: float = 1e-4,
) -> dict[str, Any]:
    module_init_time = time.time()
    logger.info(main_header("Thermochemistry via rigid-rotor harmonic oscillator approximation"))
    if len(atoms) == 1:
        logger.info("System is an atom.")
        moltype = "atom"
        # 3 translations, no rotations and no vibrations
        tr_modenum = 3
    elif len(atoms) == 2:
        logger.info("System contains 2 atoms and thus linear.")
        moltype = "linear"
        tr_modenum = 5
    else:
        logger.info("System size > 2, checking if linear")
        linearcheck = detect_linear(fragment, threshold=rotmode_threshold)
        if linearcheck is True:
            logger.info("Structure is linear. 5 translational+rotational modes present")
            moltype = "linear"
            tr_modenum = 5
        else:
            logger.info("Structure is non-linear. 6 translational+rotational modes present")
            moltype = "nonlinear"
            tr_modenum = 6

    if use_full_geo_in_rotational_analysis:
        logger.info("Using full geometry in rotational analysis")
        coords = fragment.coords
        elems = fragment.elems
    else:
        logger.info("Using Hessian-geometry in rotational analysis")
        coords = np.take(fragment.coords, atoms, axis=0)
        elems = [fragment.elems[i] for i in atoms]

    totalmass = sum(fragment.masses)
    logger.info("Total mass of molecule: %s", totalmass)

    rotational = _rotational_thermochemistry(
        fragment=fragment, coords=coords, elems=elems, moltype=moltype, temp=temp, symmetry_number=symmetry_number
    )
    E_rot, TS_rot = rotational["E_rot"], rotational["TS_rot"]
    rinertia, rotconstants, sigma_r = rotational["rinertia"], rotational["rotconstants"], rotational["sigma_r"]

    vibrational = _vibrational_thermochemistry(
        vfreq=vfreq,
        atoms=atoms,
        tr_modenum=tr_modenum,
        temp=temp,
        qrrho=qrrho,
        qrrho_method=qrrho_method,
        qrrho_omega_0=qrrho_omega_0,
        inertia_avg=rotational["inertia_avg"],
        moltype=moltype,
    )
    zpve, E_vib = vibrational["zpve"], vibrational["E_vib"]
    vibenergycorr, TS_vib, freqs = vibrational["vibenergycorr"], vibrational["TS_vib"], vibrational["freqs"]

    E_trans = 1.5 * openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * temp

    qtrans = (openmmqmmm.constants.TRANS_PARTITION_PREFACTOR * temp**2.5 * totalmass**1.5) / pressure
    TS_trans = temp * openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (math.log(qtrans) + 2.5)

    if multiplicity is not None:
        q_el = multiplicity
        S_el = openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * math.log(q_el)
        TS_el = temp * S_el
    else:
        # E.g. OpenMMTheory
        TS_el = 0.0

    E_tot = E_vib + E_trans + E_rot
    Hcorr = E_vib + E_trans + E_rot + openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * temp
    TS_tot = TS_el + TS_trans + TS_rot + TS_vib
    Gcorr = Hcorr - TS_tot

    logger.info("\nThermochemistry")
    logger.info("--------------------")
    logger.info("Temperature: %s K", temp)
    logger.info("Pressure: %s atm", pressure)
    logger.info("Hessian atomlist: %s", atoms)
    logger.info("Total mass: %s", totalmass)

    if moltype != "atom":
        logger.info("Moments of inertia: %s", rinertia)
        logger.info("Rotational constants (cm-1): %s", rotconstants)

    logger.info("\nEnergy corrections:")
    logger.info("Zero-point vibrational energy: %s", zpve)
    logger.info("%s", "{} {} {} {} {}".format("Translational energy (", temp, "K) :", E_trans, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Rotational energy (", temp, "K) :", E_rot, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Total vibrational energy (", temp, "K) :", E_vib, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Vibrational energy correction (", temp, "K) :", vibenergycorr, "Eh"))
    logger.info("\nEntropy terms (TS):")
    logger.info("%s", "{} {} {} {} {}".format("Translational entropy (TS_trans) (", temp, "K) :", TS_trans, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Rotational entropy (TS_rot) (", temp, "K) :", TS_rot, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Vibrational entropy (TS_vib) (", temp, "K) :", TS_vib, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Electronic entropy (TS_el) (", temp, "K) :", TS_el, "Eh"))
    if moltype != "atom":
        logger.info(f"Note: symmetry number : {sigma_r} used for rotational entropy")
    logger.info("\nThermodynamic terms:")
    logger.info("%s", "{} {} {} {} {}".format("Enthalpy correction (Hcorr) (", temp, "K) :", Hcorr, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Entropy correction (TS_tot) (", temp, "K) :", TS_tot, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Gibbs free energy correction (Gcorr) (", temp, "K) :", Gcorr, "Eh"))

    thermochemcalc_dict = {}
    thermochemcalc_dict["frequencies"] = freqs
    thermochemcalc_dict["ZPVE"] = zpve
    thermochemcalc_dict["E_trans"] = E_trans
    thermochemcalc_dict["E_rot"] = E_rot
    thermochemcalc_dict["E_vib"] = E_vib
    thermochemcalc_dict["E_tot"] = E_tot
    thermochemcalc_dict["TS_trans"] = TS_trans
    thermochemcalc_dict["TS_rot"] = TS_rot
    thermochemcalc_dict["TS_vib"] = TS_vib
    thermochemcalc_dict["TS_el"] = TS_el
    thermochemcalc_dict["vibenergycorr"] = vibenergycorr
    thermochemcalc_dict["Hcorr"] = Hcorr
    thermochemcalc_dict["Gcorr"] = Gcorr
    thermochemcalc_dict["TS_tot"] = TS_tot
    log_time_since(module_init_time, "calc_thermochemistry")
    return thermochemcalc_dict


def _write_dummy_orca_file(
    elems: Sequence[str],
    coords: np.ndarray,
    vfreq: Sequence[float | complex],
    nmodes: np.ndarray,
    hessfile: str,
) -> None:
    orca_header = """                                 *****************
                                 * O   R   C   A *
                                 *****************

           --- An Ab Initio, DFT and Semiempirical electronic structure package ---

                       *****************************
                       * Geometry Optimization Run *
                       *****************************

         *************************************************************
         *                GEOMETRY OPTIMIZATION CYCLE   1            *
         *************************************************************
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------"""
    with open(hessfile + "_dummy.out", "w") as outfile:
        outfile.write(orca_header + "\n")
        for el, coord in zip(elems, coords, strict=False):
            x = coord[0]
            y = coord[1]
            z = coord[2]
            line = f"  {el:2s} {x:11.6f} {y:12.6f} {z:13.6f}"
            outfile.write(line + "\n")
        outfile.write("\n")
        outfile.write("-----------------------\n")
        outfile.write("VIBRATIONAL FREQUENCIES\n")
        outfile.write("-----------------------\n")
        outfile.write("\n")
        outfile.write(
            "Scaling factor for frequencies =  1.000000000 (Found in file - NOT applied to frequencies read from HESS "
            "file)\n"
        )
        outfile.write("\n")
        numatoms = len(elems)
        complexflag = False
        for mode in range(3 * numatoms):
            smode = str(mode) + ":"
            freq = clean_number(vfreq[mode])
            if np.iscomplex(freq):
                imagfreq = -1 * abs(freq)
                complexflag = True
            else:
                complexflag = False
            if complexflag:
                line = f"{smode:>5s}{imagfreq:13.2f} cm**-1 ***imaginary mode***"
            else:
                line = f"{smode:>5s}{freq:13.2f} cm**-1"
            outfile.write(line + "\n")

        normalmodeheader = """------------
    NORMAL MODES
    ------------

    These modes are the cartesian displacements weighted by the diagonal matrix
    M(i,i)=1/sqrt(m[i]) where m[i] is the mass of the displaced atom
    Thus, these vectors are normalized but *not* orthogonal"""

        outfile.write("\n")
        outfile.write("\n")
        outfile.write(normalmodeheader)
        outfile.write("\n")
        outfile.write("\n")

        openmmqmmm.orca._write_orca_column_blocks(
            outfile,
            np.asarray(nmodes).T,
            ncols=6,
            index_fmt=" {:>6d}",
            value_fmt=" {:>10.6f}",
            first_value_fmt=" {:>14.6f}",
            header_fmt="          {}",
            header_prefix="        ",
            header_suffix="    ",
        )

        irtable = """

    -----------
    IR SPECTRUM
    -----------

     Mode   freq       eps      Int      T**2         TX        TY        TZ
    DUMMY NUMBERS BELOW
    ----------------------------------------------------------------------------

     """
        outfile.write(irtable)
        for i in range(6, 3 * numatoms):
            d = str(i) + ":"
            outfile.write(f"{d:>4s}   1606.67   0.009763   49.34  0.001896  ( 0.000000 -0.000000 -0.043546)\n")
    logger.info("Created dummy ORCA outputfile:  %s", hessfile + "_dummy.out")


def _get_center(
    coords: np.ndarray,
    masses: Sequence[float] | None = None,
    elems: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    if masses is None:
        if elems is None:
            raise InputError("Need to provide either masses or elems")
        logger.debug("No masses provided. Using built-in atom masses.")
        masses = openmmqmmm.coords.list_of_masses(elems)
    xcom = np.sum(masses * coords[:, 0]) / np.sum(masses)
    ycom = np.sum(masses * coords[:, 1]) / np.sum(masses)
    zcom = np.sum(masses * coords[:, 2]) / np.sum(masses)
    return xcom, ycom, zcom


def _inertia_tensor(coords: np.ndarray, masses: Sequence[float], center: Sequence[float] | None = None) -> np.ndarray:
    """Return the inertia tensor in amu Å² for coordinates in Å."""
    masses = np.asarray(masses)
    if center is None:
        center = _get_center(coords, masses=masses)
    centered = np.asarray(coords) - center
    return np.eye(3) * np.sum(masses[:, None] * centered**2) - (centered * masses[:, None]).T @ centered


def _principal_moments(coords: np.ndarray, masses: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return principal moments in amu Å² and principal axes as columns."""
    return np.linalg.eigh(_inertia_tensor(coords, masses))


def _tr_mode_count(coords: np.ndarray, masses: Sequence[float], threshold: float) -> int:
    moments, _axes = _principal_moments(coords, masses)
    return 3 + int(np.count_nonzero(np.abs(moments) > threshold))


def inertia(elems: Sequence[str], coords: np.ndarray, center: Sequence[float]) -> np.ndarray:
    # Preserve this public helper's unsorted principal moments.
    return np.linalg.eigvals(_inertia_tensor(coords, openmmqmmm.coords.list_of_masses(elems), center))


def calc_rotational_constants(frag: Fragment) -> list[float]:
    """Return a fragment's rotational constants in cm**-1 (the GHz values are logged as well)."""
    coords = frag.coords
    elems = frag.elems
    center = _get_center(coords, elems=elems)
    rinertia = [float(i) for i in inertia(elems, coords, center)]

    return _rotational_constants_from_moments(rinertia)


def _nonzero_moments(moments: Sequence[float]) -> list[float]:
    threshold = _ZERO_MOMENT_RTOL * np.max(np.abs(moments))
    return [float(moment) for moment in moments if abs(moment) > threshold]


def _rotational_constants_from_moments(rinertia: Sequence[float]) -> list[float]:
    rot_constants = []
    for inertval in _nonzero_moments(rinertia):
        rot_ghz = openmmqmmm.constants.ROT_CONSTANT_GHZ_AMU_ANG2 / inertval
        rot_constants.append(rot_ghz)

    rot_constants_cm = [i * openmmqmmm.constants.GHZ_TO_WAVENUMBER for i in rot_constants]
    logger.info("Moments of inertia (amu A^2 ): %s", rinertia)
    logger.info("Rotational constants (GHz): %s", rot_constants)
    logger.info("Rotational constants (cm-1): %s", rot_constants_cm)
    logger.info("Note: If moment of inertia is zero then rotational constant is infinite and not printed ")

    return rot_constants_cm


def _calc_model_hessian_orca(
    fragment: Fragment,
    model: str = "Almloef",
    *,
    charge: int | None = None,
    mult: int | None = None,
) -> np.ndarray:
    orcasimple = "! hf"
    extraline = "!noiter opt"
    orcablocks = f"""
    %geom
    maxiter 1
    inhess {model}
    end
"""
    orcadummycalc = openmmqmmm.orca.ORCATheory(orcasimpleinput=orcasimple, orcablocks=orcablocks, extraline=extraline)
    openmmqmmm.single_point(theory=orcadummycalc, fragment=fragment, charge=charge, mult=mult)
    hesstake = False
    j = 0
    # ORCA writes the .opt Hessian in 6-column blocks; .hess files use 5 (see orca.grab_hessian)
    orcacoldim = 6
    shiftpar = 0
    lastchunk = False
    grabsize = False
    opt_path = orcadummycalc.filename + ".opt"
    with open(opt_path, "rb") as optfile:
        if b"\x00" in optfile.read(256):
            raise ExternalProgramError(
                "ORCA wrote a binary .opt model Hessian (as in ORCA 6). Reading this format is not supported; "
                "use rest_hessian='zero' or 'unit', or supply an explicitly computed Hessian. "
                "Model-Hessian support is tracked in issue #68."
            )
    with open(opt_path) as optfile:
        for line in optfile:
            if "$bmatrix" in line:
                hesstake = False
                continue
            if hesstake and len(line.split()) == 2 and grabsize:
                grabsize = False
                hessdim = int(line.split()[0])

                hessarray2d = np.zeros((hessdim, hessdim))
            if hesstake and len(line.split()) == 6:
                continue
            if hesstake and lastchunk and len(line.split()) == hessdim - shiftpar + 1:
                for i in range(hessdim - shiftpar):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
            if hesstake and len(line.split()) == 7:
                for i in range(orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if "$hessian_approx" in line:
                hesstake = True
                grabsize = True

    return np.array(hessarray2d)


def approximate_full_hessian_from_smaller(
    fragment: Fragment,
    hessian_small: np.ndarray,
    small_atomindices: Sequence[int],
    large_atomindices: Sequence[int] | None = None,
    rest_hessian: str | None = "zero",
    projection: bool = False,
    charge: int | None = None,
    mult: int | None = None,
) -> np.ndarray:
    """Embed a small computed Hessian in a larger zero, unit or ORCA model Hessian."""
    logger.info("approximate_full_hessian_from_smaller\n")
    write_hessian(hessian_small, hessfile="smallhessian")

    if large_atomindices is None or len(large_atomindices) == 0:
        hess_size = fragment.numatoms * 3
        logger.info("Hessian dimension %s", hess_size)
        correct_small_atomindices = small_atomindices
        usedfragment = fragment
    else:
        logger.info("small_atomindices: %s", small_atomindices)
        logger.info("large_atomindices: %s", large_atomindices)
        hess_size = len(large_atomindices) * 3

        if all(item in large_atomindices for item in small_atomindices) is False:
            raise InputError(
                "{}\nThis does not make sense.".format(
                    f"small_atomindices: {small_atomindices} are not all present in large_atomindices: "
                    f"{large_atomindices}"
                )
            )
        correct_small_atomindices = [large_atomindices.index(i) for i in small_atomindices]
        logger.info("correct_small_atomindices: %s", correct_small_atomindices)
        subcoords, subelems = fragment.get_coords_for_atoms(large_atomindices)
        # No charge/mult: this is a sub-region of fragment, so the fragment's whole-system values
        # do not describe it. A model Hessian over this region takes them as arguments instead.
        usedfragment = openmmqmmm.Fragment(elems=subelems, coords=subcoords)

    logger.info("Initializing full size Hessian of dimension: %s", hess_size)
    fullhessian = np.zeros((hess_size, hess_size))
    logger.info("Initial fullhessian: %s", fullhessian)
    logger.info("Number of Hessian elements: %s", fullhessian.size)
    write_hessian(fullhessian, hessfile="initialfullhessian")

    hessian_small = np.array(hessian_small)
    logger.info("hessian_small: %s", hessian_small)
    model_hessians = {name.lower(): name for name in ("Almloef", "Lindh", "Schlegel", "Swart")}
    rest_choice = "zero" if rest_hessian is None else str(rest_hessian).lower()
    if rest_choice in model_hessians:
        rest_hessian = model_hessians[rest_choice]
        logger.info("rest_hessian: %s", rest_hessian)
        if charge is None or mult is None:
            # A sub-region has no derivable net charge, so only a Hessian region spanning the whole
            # fragment may fall back to the fragment's own values.
            if usedfragment.numatoms != fragment.numatoms:
                raise InputError(
                    f"A model Hessian over a {usedfragment.numatoms}-atom region of a {fragment.numatoms}-atom "
                    f"fragment needs charge= and mult= passed explicitly; the fragment's own values describe the "
                    f"whole system"
                )
            if fragment.charge is None or fragment.mult is None:
                raise InputError("A model Hessian needs a charge and mult, and the fragment carries neither")
            charge = fragment.charge
            mult = fragment.mult
            logger.info(f"Model Hessian spans the whole fragment. Using charge={charge} mult={mult}")
        fullhessian = _calc_model_hessian_orca(usedfragment, model=rest_hessian, charge=charge, mult=mult)
    elif rest_choice == "xtb":
        raise InputError(
            "rest_hessian='xtb' is not available in this ORCA+OpenMM build. Use an ORCA model Hessian, 'unit' or "
            "'zero' instead."
        )
    elif rest_choice in {"unit", "identity"}:
        logger.info("rest_hessian is unit/identity")
        fullhessian = np.identity(hess_size)
    elif rest_choice == "zero":
        logger.info("rest_hessian is zero.")
    else:
        raise InputError(
            f"Unknown rest_hessian {rest_hessian!r}. Choose 'zero', 'unit', 'identity', 'Almloef', 'Lindh', "
            "'Schlegel' or 'Swart'."
        )
    logger.info("Intermediate fullhessian: %s", fullhessian)
    logger.info("Size: %s", fullhessian.size)
    write_hessian(fullhessian, hessfile="intermedfullhessian")
    athessindices = [3 * i + j for i in correct_small_atomindices for j in [0, 1, 2]]
    for s_i, i in enumerate(athessindices):
        for s_j, j in enumerate(athessindices):
            fullhessian[i, j] = hessian_small[s_i, s_j]
    logger.info("Final fullhessian: %s", fullhessian)
    write_hessian(fullhessian, hessfile="intermedfullhessian_after_small_update")
    tr_modenum = _tr_mode_count(usedfragment.coords, usedfragment.masses, 1e-4)

    logger.info("Now diagonalizing full Hessian")
    frequencies, _normal_modes, _evectors, _mode_order = _diagonalize_hessian(
        usedfragment.coords,
        fullhessian,
        usedfragment.masses,
        usedfragment.elems,
        tr_modenum=tr_modenum,
        projection=projection,
    )
    logger.info("Size: %s", fullhessian.size)
    logger.info("Frequencies of full Hessian: %s", frequencies)
    write_hessian(fullhessian, hessfile="Finalfullhessian")
    return fullhessian


def _normal_mode_component(evectors: np.ndarray, j: int, a: int) -> float:
    esq_j = [i**2 for i in evectors[j]]
    esq_ja = []
    esq_ja.append(esq_j[a * 3 + 0])
    esq_ja.append(esq_j[a * 3 + 1])
    esq_ja.append(esq_j[a * 3 + 2])
    return sum(esq_ja)


def _normal_mode_components_all(
    mode: int,
    fragment: Fragment,
    evectors: np.ndarray,
    hessatoms: Sequence[int] | None = None,
) -> list[float]:
    numatoms = fragment.numatoms if hessatoms is None else len(hessatoms)
    normcomplist = []
    for n in range(numatoms):
        normcomp = _normal_mode_component(evectors, mode, n)
        normcomplist.append(normcomp)

    return normcomplist


def _normal_mode_components_by_element(
    mode: int,
    fragment: Fragment,
    evectors: np.ndarray,
    hessatoms: Sequence[int] | None = None,
) -> dict[str, float]:
    normcomplist = _normal_mode_components_all(mode, fragment, evectors, hessatoms=hessatoms)
    elementnormcomplist = []

    hesselems = [fragment.elems[i] for i in hessatoms] if hessatoms is not None else fragment.elems

    uniqelems = []
    for i in hesselems:
        if i not in uniqelems:
            uniqelems.append(i)
    normmodecompelemsdict = {}
    for u in uniqelems:
        elcompsum = 0.0
        elindices = [i for i, j in enumerate(hesselems) if j == u]
        for h in elindices:
            elcompsum = float(elcompsum + float(normcomplist[h]))
        elementnormcomplist.append(elcompsum)
        normmodecompelemsdict[u] = elcompsum
    return normmodecompelemsdict


def _vibrational_temperature(frequency: float) -> float:
    return (
        frequency
        * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S
        * openmmqmmm.constants.PLANCK_HARTREE_S
        / openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K
    )


def _harmonic_ts_vib_mode(theta: float, temperature: float) -> float:
    x = theta / temperature
    return (
        temperature * openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (x / math.expm1(x) - math.log(-math.expm1(-x)))
    )


def _s_vib(freqs: Sequence[float], T: float) -> float:
    return sum(_harmonic_ts_vib_mode(_vibrational_temperature(frequency), T) for frequency in freqs)


def s_vib_qrrho_truhlar(freqs: Sequence[float], T: float, lowfreq_thresh: float = 100) -> float:
    logger.warning("Quasi-RRHO by Truhlar approximation active.")
    logger.info(
        "This means that the vibrational entropy is calculated according to Truhlar-approach of raising low-energy "
        f"vibrations to {lowfreq_thresh} cm-1"
    )
    logger.info("Cite: R. F. Ribeiro et al. J. Phys. Chem. B, 115, 14556 (2011) ")
    TS_vib_final = 0.0
    for f in freqs:
        freq_value = f
        if f < lowfreq_thresh:
            logger.warning(
                f"Frequency ({f}) is below low-freq threshold ({lowfreq_thresh}) cm-1. Setting to {lowfreq_thresh} cm-1"
            )
            freq_value = lowfreq_thresh
        vibtemp = _vibrational_temperature(freq_value)
        logger.info("vibtemp: %s", vibtemp)
        TS_vib_f = _harmonic_ts_vib_mode(vibtemp, T)
        TS_vib_final += TS_vib_f
        logger.info("TS_vib_final: %s", TS_vib_final)

    return TS_vib_final


def s_vib_qrrho_grimme(freqs: Sequence[float], T: float, omega_0: float = 100, i_av: float | None = None) -> float:
    logger.warning("Quasi-RRHO approximation by Grimme active.")
    logger.info("This means that the vibrational entropy uses the Grimme-type interpolation formula")
    logger.info("Cite: S. Grimme, Chem. Eur. J. 2012, 18, 9955-9964.")
    TS_vib_final = 0.0
    for f in freqs:
        TS_vib_f = _harmonic_ts_vib_mode(_vibrational_temperature(f), T)
        m_si = (
            openmmqmmm.constants.PLANCK_J_S
            * openmmqmmm.constants.PLANCK_J_S
            / (8 * math.pi * math.pi * f * openmmqmmm.constants.HC_J_CM)
        )
        mp_si = m_si * i_av / (m_si + i_av)
        TS_rot_f_au = (
            T
            * openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K
            * (
                0.5
                + math.log(
                    math.sqrt(
                        8
                        * math.pi
                        * math.pi
                        * math.pi
                        * mp_si
                        * openmmqmmm.constants.BOLTZMANN_J_PER_K
                        * T
                        / (openmmqmmm.constants.PLANCK_J_S * openmmqmmm.constants.PLANCK_J_S)
                    )
                )
            )
        )
        w = 1 / (1 + pow(omega_0 / f, 4))
        TS_vib_final += w * TS_vib_f + (1 - w) * TS_rot_f_au
    return TS_vib_final


def write_hessian(hessian: np.ndarray, hessfile: str | PathLike[str] = "Hessian") -> None:
    """Write a Hessian matrix to a text file."""
    np.savetxt(hessfile, hessian)
    logger.info(f"Wrote Hessian to file: {hessfile}")


def read_hessian(file: str | PathLike[str]) -> np.ndarray:
    """Read a Hessian matrix from a text file written by write_hessian."""
    logger.info(f"Reading Hessian from file: {file}")
    return np.loadtxt(file)


def detect_linear(
    fragment: Fragment | None = None,
    coords: np.ndarray | None = None,
    elems: Sequence[str] | None = None,
    threshold: float = 1e-4,
) -> bool:
    if fragment is None:
        numatoms = len(coords)
    else:
        coords = fragment.coords
        elems = fragment.elems
        numatoms = fragment.numatoms
    if numatoms == 1:
        return True
    if numatoms == 2:
        return True
    if _tr_mode_count(coords, openmmqmmm.coords.list_of_masses(elems), threshold) < 6:
        logger.info("Molecule is linear")
        return True
    logger.info("Molecule is non-linear")
    return False


def _project_rot_and_trans(
    coords: np.ndarray,
    mass: Sequence[float],
    hessian: np.ndarray,
    rotmode_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    mass = np.array(mass)
    # rotmode_threshold is consistently in amu Å², including detect_linear.
    Ivals, Ivecs = _principal_moments(coords, mass)
    Ivecs = Ivecs.T
    coords = np.array(coords) * openmmqmmm.constants.ANG_TO_BOHR
    coords = coords.copy().reshape(-1, 3)
    na = coords.shape[0]
    TotDOF = 3 * na
    wHessian, invsqrtm3 = _mass_weight_hessian(hessian, mass)

    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0) / np.sum(mass)

    xcm = coords - cxyz[np.newaxis, :]

    TR_DOF = 3 + int(np.count_nonzero(np.abs(Ivals) > rotmode_threshold))
    logger.info("TR_DOF: %s", TR_DOF)
    if TR_DOF not in (5, 6):
        logger.warning(f"Unexpected number of trans+rot DOF: {TR_DOF} not in (5, 6)")

    ic_eckart = np.zeros((6, TotDOF))
    for i in range(na):
        p_vec = np.dot(Ivecs, xcm[i])
        smass = np.sqrt(mass[i])
        ic_eckart[0, 3 * i] = smass
        ic_eckart[1, 3 * i + 1] = smass
        ic_eckart[2, 3 * i + 2] = smass
        for ix in range(3):
            ic_eckart[3, 3 * i + ix] = smass * (Ivecs[2, ix] * p_vec[1] - Ivecs[1, ix] * p_vec[2])
            ic_eckart[4, 3 * i + ix] = smass * (Ivecs[2, ix] * p_vec[0] - Ivecs[0, ix] * p_vec[2])
            ic_eckart[5, 3 * i + ix] = smass * (Ivecs[0, ix] * p_vec[1] - Ivecs[1, ix] * p_vec[0])

    # Sort the rotation ICs by their norm in descending order, then normalize them
    ic_eckart_norm = np.sqrt(np.sum(ic_eckart**2, axis=1))
    # If the norm is equal to zero, then do not scale.
    ic_eckart_norm += ic_eckart_norm == 0.0
    sortidx = np.concatenate((np.array([0, 1, 2]), 3 + np.argsort(ic_eckart_norm[3:])[::-1]))
    ic_eckart1 = ic_eckart[sortidx, :]
    ic_eckart1 /= ic_eckart_norm[sortidx, np.newaxis]
    ic_eckart = ic_eckart1.copy()

    # Using Gram-Schmidt orthogonalization, create a basis where translation
    # and rotation is projected out of Cartesian coordinates
    proj_basis = np.identity(TotDOF)
    maxIt = 100
    for iteration in range(maxIt):
        max_overlap = 0.0
        for i in range(TotDOF):
            for n in range(TR_DOF):
                proj_basis[i] -= np.dot(ic_eckart[n], proj_basis[i]) * ic_eckart[n]
            overlap = np.sum(np.dot(ic_eckart, proj_basis[i]))
            max_overlap = max(overlap, max_overlap)
        if max_overlap < 1e-12:
            break
        if iteration == maxIt - 1:
            logger.warning(f"Gram-Schmidt orthogonalization failed after {maxIt} iterations")

    # Diagonalize the overlap matrix to create (3N - TR_DOF) orthonormal basis vectors
    # constructed from translation and rotation-projected proj_basis
    proj_overlap = np.dot(proj_basis, proj_basis.T)
    proj_vals, proj_vecs = np.linalg.eigh(proj_overlap)
    proj_vecs = proj_vecs.T

    # The projection should leave exactly TR_DOF vanishing eigenvalues. Counting them
    # liberally and conservatively brackets the true number: the liberal count should be
    # at least TR_DOF and the conservative one at most TR_DOF. Outside that bracket the
    # translation/rotation projection did not separate cleanly and the frequencies below
    # are unreliable.
    n_zeros_liberal = int(np.sum(abs(proj_vals) < 1.0e-8))
    n_zeros_conservative = int(np.sum(abs(proj_vals) < 1.0e-12))
    if not (n_zeros_conservative <= TR_DOF <= n_zeros_liberal):
        logger.warning(
            "Translation/rotation projection is not clean: expected %d vanishing eigenvalues, "
            "found between %d and %d. Frequencies may be unreliable.",
            TR_DOF,
            n_zeros_conservative,
            n_zeros_liberal,
        )

    norm_vecs = proj_vecs[TR_DOF:] / np.sqrt(proj_vals[TR_DOF:, np.newaxis])

    # These are the orthonormal, TR-projected internal coordinates
    ic_basis = np.dot(norm_vecs, proj_basis)
    ic_hessian = np.linalg.multi_dot((ic_basis, wHessian, ic_basis.T))
    ichess_vals, ichess_vecs = np.linalg.eigh(ic_hessian)
    ichess_vecs = ichess_vecs.T
    normal_modes = np.dot(ichess_vecs, ic_basis)
    normal_modes_cart = normal_modes * invsqrtm3[np.newaxis, :]

    freqs_wavenumber = _eigenvalues_to_wavenumbers(ichess_vals)

    return freqs_wavenumber, normal_modes, normal_modes_cart, TR_DOF


def _calc_raman_activities(
    nmodes: np.ndarray, polarizability_derivs: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Calculating Raman activities")
    hesslength = len(nmodes)
    displacements = nmodes.T

    A_der = np.zeros((hesslength, 9))
    for i in range(hesslength):
        A_der[i, :] = polarizability_derivs[i].reshape(1, 9)

    # Transform polarizability derivatives to normal coordinates
    A_der_q_tmp = np.dot(A_der.T, displacements)
    A_der_q = []
    for i in range(hesslength):
        one_alpha_der = np.zeros((3, 3))
        jk = 0
        for j in range(3):
            for k in range(3):
                one_alpha_der[j, k] = A_der_q_tmp[jk, i]
                jk += 1
        A_der_q.append(one_alpha_der)

    # alpha, beta^2, Raman activity and depolarization ratio as in Neugebauer, J Comput Chem 2002
    alpha = np.zeros(hesslength)
    beta2 = np.zeros(hesslength)
    depol_ratio = np.zeros(hesslength)
    raman_act = np.zeros(hesslength)
    for i in range(hesslength):
        axx = A_der_q[i][0, 0]
        ayy = A_der_q[i][1, 1]
        azz = A_der_q[i][2, 2]
        axy = A_der_q[i][0, 1]
        axz = A_der_q[i][0, 2]
        ayz = A_der_q[i][1, 2]
        alpha[i] = 1 / 3 * (axx + ayy + azz)
        beta2[i] = 0.5 * ((axx - ayy) ** 2 + (axx - azz) ** 2 + (ayy - azz) ** 2 + 6 * (axy**2 + axz**2 + ayz**2))
        depol_ratio[i] = 3 * beta2[i] / ((45 * alpha[i] * alpha[i]) + 4 * beta2[i])
        raman_act[i] = 45 * alpha[i] * alpha[i] + 7 * beta2[i]

    # Converting to Angstrom^4/amu
    raman_unit = 1 / openmmqmmm.constants.BOHR_TO_ANG**4
    raman_act = raman_act / raman_unit

    logger.info("Calculated Raman activities for each normal mode: %s", raman_act)
    logger.info("Calculated Raman depolarization ratios for each normal mode: %s", depol_ratio)
    return raman_act, depol_ratio
