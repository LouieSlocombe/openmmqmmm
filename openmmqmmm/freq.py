from __future__ import annotations

import contextlib
import copy
import logging
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any, TypeAlias

import numpy as np

import openmmqmmm.constants
import openmmqmmm.coords
import openmmqmmm.orca
from openmmqmmm.coords import Fragment, check_charge_mult
from openmmqmmm.exceptions import InputError
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.results import Results
from openmmqmmm.utils import clean_number, listdiff, log_time_since, main_header

logger = logging.getLogger(__name__)

Displacement: TypeAlias = tuple[int, int, str] | str


# Analytical frequencies function. Only for theories with this option added (e.g. ORCATheory and CFourTheory)
# Checked by analytic_hessian attribute True
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
        raise InputError("AnFreq requires a fragment and a theory object")

    # Checking for linearity. Determines how many Trans+Rot modes
    if detect_linear(coords=fragment.coords, elems=fragment.elems, threshold=rotmode_threshold) is True:
        tr_modenum = 5
    else:
        tr_modenum = 6
    hessatoms = list(range(fragment.numatoms))

    if masses is None:
        masses = fragment.list_of_masses

    # Only theories that actually provide a Hessian set analytic_hessian; QMMMTheory and the
    # wrapper theories never define it at all.
    if getattr(theory, "analytic_hessian", False):
        logger.info(f"Requesting analytical Hessian calculation from {theory.theorynamelabel}\n")
        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "AnFreq", theory=theory)
        theory.run(current_coords=fragment.coords, elems=fragment.elems, charge=charge, mult=mult, hessian=True)

        logger.info("Getting analytic Hessian from theory object")
        hessian = theory.hessian
        frequencies, nmodes, evectors, _mode_order = _diagonalize_hessian(
            fragment.coords, theory.hessian, masses, fragment.elems, tr_modenum=tr_modenum, projection=True
        )
        logger.info("Now scaling frequencies by scaling factor: %s", scaling_factor)
        frequencies = scaling_factor * frequencies

        # For IR intensities it might be preferable to get dipole derivatives from theory
        # and then calculate IR intensities directly using calc_IR_Intensities function
        # Would ensure completely correct masses at least
        # For now grabbing directly from theory object
        # Tested with pyscf, ORCA
        IR_intens_values = None
        try:
            IR_intens_values = theory.ir_intensities
            if len(IR_intens_values) == 0:
                logger.info("Found no IR intensities")
                IR_intens_values = None
            elif len(IR_intens_values) < len(frequencies):
                logger.info("Found IR intensities, zero-capping needed")
                IR_intens_values = [0.0] * 6 + list(IR_intens_values)
                logger.info("Found IR intensities")
        except (AttributeError, KeyError, TypeError):
            logger.info("Found no IR intensities in theory object")
            IR_intens_values = None
        raman_activities = None

        _log_frequencies(
            frequencies,
            len(hessatoms),
            tr_modenum=tr_modenum,
            intensities=IR_intens_values,
            raman_activities=raman_activities,
        )
        logger.info("Normal mode composition factors by element")
        _log_frequencies_and_mode_compositions(
            frequencies, fragment, evectors, hessatoms=hessatoms, tr_modenum=tr_modenum
        )
        thermodict = calc_thermochemistry(
            frequencies,
            hessatoms,
            fragment,
            mult,
            temp=temp,
            pressure=pressure,
            qrrho=qrrho,
            qrrho_method=qrrho_method,
            qrrho_omega_0=qrrho_omega_0,
            symmetry_number=symmetry_number,
            rotmode_threshold=rotmode_threshold,
        )

        fragment.hessian = hessian
        write_hessian(hessian, hessfile="Hessian")

        _write_dummy_orca_file(fragment.elems, fragment.coords, frequencies, nmodes, "orcahessfile.hess")
        logger.info("Wrote dummy ORCA outputfile with frequencies and normal modes: orcahessfile.hess_dummy.out")
        logger.info("Can be used for visualization")

        logger.info("------------ANALYTICAL FREQUENCIES END-------------")
        log_time_since(module_init_time, "AnFreq")

        result = Results(
            label="Anfreq",
            hessian=hessian,
            frequencies=frequencies,
            vib_eigenvectors=evectors,
            normal_modes=nmodes,
            thermochemistry=thermodict,
        )
        result.write_to_disk(filename="results_anfreq.json")
        return result

    raise InputError(
        f"Analytical frequencies are not available for {theory.__class__.__name__}. Use numerical_frequencies instead."
    )


# ORCA uses 0.005 Bohr = 0.0026458861 Ang, CHemshell uses 0.01 Bohr = 0.00529 Ang
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
    """Return the displaced geometries, their dictionary keys, log labels and fragments."""
    current = np.array(coords)
    geometries = []
    displacements = []
    signs = [(1, "+")] if npoint == 1 else [(1, "+"), (-1, "-")]
    for atom_index in hessatoms:
        for coord_index in range(3):
            val = current[atom_index, coord_index]
            for sign, direction in signs:
                current[atom_index, coord_index] = val + sign * displacement
                geometries.append(current.copy())
                displacements.append((atom_index, coord_index, direction))
            current[atom_index, coord_index] = val
    if npoint == 1:
        # Forward difference needs the undisplaced gradient as its reference
        geometries.append(current.copy())
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
            stringlabel = f"{atom_disp}_{axis}_{direction}"
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
            stringlabel = f"{disp[0]}_{disp[1]}_{disp[2]}"
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
    """Finite-difference the displaced gradients into a symmetrised Hessian and its derivatives."""
    logger.info("Assembling the %s-point Hessian", npoint)
    hesslength = 3 * len(hessatoms)
    hessian = np.zeros((hesslength, hesslength))
    dipole_derivs = np.zeros((hesslength, 3))
    polarizability_derivs = []

    want_dipoles = IR is True and len(dipoles) > 0 and not any(value is None for value in dipoles.values())
    want_polarizabilities = Raman is True and len(polarizabilities) > 0
    # Forward difference measures against the undisplaced geometry over one step; central
    # difference measures the two displacements against each other, so over two.
    step = displacement_bohr if npoint == 1 else 2 * displacement_bohr

    hessindex = 0
    for atomindex in hessatoms:
        for crd in (0, 1, 2):
            plus = f"{atomindex}_{crd}_+"
            minus = "Originalgeo" if npoint == 1 else f"{atomindex}_{crd}_-"

            grad_plus = np.ravel(_get_partial_matrix(grads[plus], hessatoms))
            grad_minus = np.ravel(_get_partial_matrix(grads[minus], hessatoms))
            hessian[hessindex, :] = (grad_plus - grad_minus) / step

            if want_dipoles and len(dipoles[plus]) > 0:
                dipole_derivs[hessindex, :] = (np.array(dipoles[plus]) - np.array(dipoles[minus])) / step
            if want_polarizabilities:
                polarizability_derivs.append(
                    (np.array(polarizabilities[plus]) - np.array(polarizabilities[minus])) / step
                )
            hessindex += 1

    return (hessian + hessian.transpose()) / 2, dipole_derivs, polarizability_derivs


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
        raise InputError("NumFreq requires a fragment and a theory object")
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "NumFreq", theory=theory)
    coords = fragment.coords
    elems = copy.deepcopy(fragment.elems)
    numatoms = len(elems)
    allatoms = list(range(numatoms))

    # Hessatoms list is allatoms (if hessatoms list not provided). If hessatoms provided we do a partial Hessian
    if hessatoms is None:
        logger.info("No Hessatoms provided. Full Hessian assumed. Rot+trans projection is on!")
        if isinstance(theory, QMMMTheory):
            logger.info("Theory object provided is a QM/MM Theory")
            raise InputError(
                "Error: No hessatoms option was provided. This is required for QM/MM Theories\nPlease provide a list "
                "of atom indices to the hessatoms keyword of NumFreq to define the partial Hessian\nFor QM/MM "
                "numerical frequencies you want the list of hessatoms to be the same atoms used to define the "
                "\nactive-region in the optimization (or the QM-region)\nExiting now."
            )
        hessatoms = allatoms
        projection = True
    elif len(hessatoms) == fragment.numatoms:
        logger.info("Hessatoms list provided but equal to number of fragment atoms. Rot+trans projection is on!")
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

    hessatoms = sorted(set(hessatoms))

    if hessatoms_masses is not None and len(hessatoms_masses) != len(hessatoms):
        raise InputError(
            "Error: Number of provided masses (hessatoms_masses keyword) is not equal to number of "
            "Hessian-atoms.\nCheck input masses!"
        )
    # Checking for linearity. Determines how many Trans+Rot modes
    if detect_linear(coords=fragment.coords, elems=fragment.elems, threshold=rotmode_threshold) is True:
        tr_modenum = 5
    else:
        tr_modenum = 6
    # ORCA-specific: Copy old GBW file from .. dir
    try:
        if theory.theorytype == "QM":
            if isinstance(theory, openmmqmmm.orca.ORCATheory):
                logger.info("Copying GBW file into Numfreq_dir")
                shutil.copy("../" + theory.filename + ".gbw", "./" + theory.filename + ".gbw")
        elif theory.theorytype == "QM/MM" and isinstance(theory.qm_theory, openmmqmmm.orca.ORCATheory):
            logger.info("Copying GBW file into Numfreq_dir")
            shutil.copy("../" + theory.qm_theory.filename + ".gbw", "./" + theory.qm_theory.filename + ".gbw")
    except (OSError, AttributeError):
        pass

    shutil.rmtree("Numfreq_dir", ignore_errors=True)
    os.mkdir("Numfreq_dir")
    os.chdir("Numfreq_dir")
    logger.debug("Creating separate directory for displacement calculations: Numfreq_dir ")

    displacement_bohr = displacement * openmmqmmm.constants.ANG_TO_BOHR
    logger.info("Starting Numerical Frequencies job for fragment")
    logger.info("Hessian atoms: %s", hessatoms)
    if hessatoms != allatoms:
        logger.info("This is a partial Hessian job.")
        if len(hessatoms) == 0:
            raise InputError("hessatoms list is empty. Exiting.")
    if npoint == 1:
        logger.info("One-point formula used (forward difference)")
    elif npoint == 2:
        logger.info("Two-point formula used (central difference)")
    else:
        raise InputError("Unknown npoint option. npoint should be set to 1 (one-point) or 2 (two-point formula).")
    if runmode == "serial":
        logger.info("Numfreq running in serial mode")
    elif runmode == "parallel":
        logger.info("Numfreq running in parallel mode")
    logger.info(f"\nDisplacement: {displacement:5.4f} Å ({displacement_bohr:5.4f} Bohr)")
    logger.debug("\nStarting geometry:")
    logger.info("Printing hessatoms geometry...")
    openmmqmmm.coords.print_coords_for_atoms(coords, elems, hessatoms)

    # Only displacing atoms in the hessatoms list, i.e. a possible partial Hessian
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
    elif runmode == "parallel":
        grads, dipoles, polarizabilities = _run_displacements_in_parallel(
            theory=theory, fragments=all_disp_fragments, numcores=numcores
        )
    else:
        raise InputError("Unknown runmode.")
    displacement_grad_dictionary = grads
    displacement_dipole_dictionary = dipoles
    displacement_polarizability_dictionary = polarizabilities

    logger.info("NumFreq Displacement calculations are done!\n")

    if len(displacement_grad_dictionary) == 0:
        raise InputError(
            "Missing gradients for displacement.\nSomething went wrong in Numfreq displacement calculations."
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
        hessmasses = openmmqmmm.coords.get_partial_list(allatoms, hessatoms, fragment.list_of_masses)
    else:
        hessmasses = hessatoms_masses

    logger.info("hessmasses: %s", hessmasses)
    _mwhessian, _massmatrix = _mass_weight_hessian(hessian, hessmasses)
    hesselems = openmmqmmm.coords.get_partial_list(allatoms, hessatoms, elems)

    hesscoords = np.take(fragment.coords, hessatoms, axis=0)
    logger.info("Elements: %s", hesselems)
    logger.info("Masses used: %s", hessmasses)

    # Evectors: eigenvectors of the mass-weighed Hessian
    # Normal modes: unweighted
    frequencies, nmodes, evectors, mode_order = _diagonalize_hessian(
        hesscoords,
        hessian,
        hessmasses,
        hesselems,
        tr_modenum=tr_modenum,
        projection=projection,
        rotmode_threshold=rotmode_threshold,
    )
    logger.info("Diagonalization of frequencies complete")
    logger.info("Now scaling frequencies by scaling factor: %s", scaling_factor)
    frequencies = scaling_factor * np.array(frequencies)

    IR_intens_values = None
    if IR is True and np.any(dipole_derivs):
        dipole_derivs = dipole_derivs[mode_order]
        IR_intens_values = _calc_ir_intensities(hessmasses, evectors, dipole_derivs)

    if Raman is True:
        logger.info("Raman calculation active")
        if len(polarizability_derivs) == 0:
            logger.debug("No polarizability information found. Skipping Raman.")
            raman_activities = None
            depolarization_ratios = None
        else:
            logger.info("Polarizability derivatives are available.")
            polarizability_derivs = [polarizability_derivs[i] for i in mode_order]
            raman_activities, depolarization_ratios = _calc_raman_activities(
                hessmasses, evectors, polarizability_derivs
            )
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
        temp=temp,
        pressure=pressure,
        qrrho=qrrho,
        qrrho_method=qrrho_method,
        qrrho_omega_0=qrrho_omega_0,
        symmetry_number=symmetry_number,
        rotmode_threshold=rotmode_threshold,
    )

    write_hessian(hessian, hessfile="Hessian")

    openmmqmmm.orca.write_orca_hessfile(hessian, hesscoords, hesselems, hessmasses, "orcahessfile.hess")

    _write_dummy_orca_file(hesselems, hesscoords, frequencies, nmodes, "orcahessfile.hess")
    logger.info("Wrote dummy ORCA outputfile with frequencies and normal modes: orcahessfile.hess_dummy.out")
    logger.info("Can be used for visualization\n")
    logger.info("------------NUMERICAL FREQUENCIES END-------------")

    fragment.hessian = hessian  # Hessian

    os.chdir("..")
    log_time_since(module_init_time, "NumFreq")
    result = Results(
        label="Numfreq",
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
    result.write_to_disk(filename="results_numfreq.json")
    return result


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
        vfreqs, evectors, nmodes = _project_rot_and_trans(coords, masses, hessian, rotmode_threshold=rotmode_threshold)
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
    nmodes = np.dot(evectors, massmatrix)

    vfreqs = _frequencies_from_eigenvalues(evalues)

    vfreqs = _clean_frequencies(vfreqs)

    logger.info("Calculated frequencies: %s", vfreqs)
    # NOTE: Since no projection the first freqs and modes are either TRmodes or imaginary SP modes (unknown)
    # How to deal with this properly
    # For now: let's assume large imaginary freqs are proper modes and other small imag/pos modes are TRmodes.
    # TRmodes are not set to zero though
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
    # First TRmodes, then SPmodes then rest
    logger.info("Reordering modes so that TRmodes come first, then SP modes, then rest")
    neworder = TRmodes + SPmodes + listdiff(range(len(vfreqs)), TRmodes + SPmodes)
    vfreqs = [vfreqs[i] for i in neworder]
    evectors = evectors[neworder]
    nmodes = nmodes[neworder]

    return vfreqs, nmodes, evectors, neworder


def _calc_ir_intensities(hessmasses: Sequence[float], evectors: np.ndarray, dipole_derivs: np.ndarray) -> np.ndarray:
    mass_matrix = np.repeat(hessmasses, 3)
    inv_sqrt_mass_matrix = np.diag(1 / (mass_matrix**0.5))
    displacements = inv_sqrt_mass_matrix.dot(np.transpose(evectors))
    de_q = displacements.T @ dipole_derivs
    return openmmqmmm.constants.IR_INTENSITY_AU_TO_KM_PER_MOL * np.einsum("qt, qt -> q", de_q, de_q)


def _mass_weight_hessian(matrix: np.ndarray, masses: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    numatoms = len(masses)
    mass_mat = np.zeros((3 * numatoms, 3 * numatoms), dtype=float)
    molwt = [masses[int(i)] for i in range(numatoms) for j in range(3)]
    for i in range(len(molwt)):
        mass_mat[i, i] = molwt[i] ** -0.5
    mwhessian = np.dot((np.dot(mass_mat, matrix)), mass_mat)
    return mwhessian, mass_mat


def _frequencies_from_eigenvalues(evalues: Sequence[float]) -> list[complex]:
    evalues_si = [
        val * openmmqmmm.constants.HARTREE_TO_J / openmmqmmm.constants.BOHR_TO_M**2 / openmmqmmm.constants.AMU_TO_KG
        for val in evalues
    ]
    vfreq_hz = [1 / (2 * math.pi) * np.sqrt(np.complex128(val)) for val in evalues_si]
    return [val / openmmqmmm.constants.LIGHT_SPEED_CM_PER_S for val in vfreq_hz]


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


# NOTE: THIS IS NOT CORRECT
# FOR SADDLEPOINT, the SP mode will be the largest imaginary mode, hence mode 0.


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
    rotconstants = calc_rotational_constants(fragment)

    if moltype == "linear":
        rot_temps = [_rotational_temperature(in_I) for in_I in inertia_si if in_I != 0.0]
        logger.info(f"Rotational temperatures: {rot_temps} K")
        sigma_r = 1.0
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
            # rotation, or a completely flat direction) and its harmonic entropy
            # and thermal energy both diverge, so it is excluded like a negative one.
            logger.info(f"Mode {mode} with frequency {vib} is not positive. Skipping in thermochemistry")
        else:
            freqs.append(float(vib))
            freq_hz = vib * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S
            vibtemps.append(
                (openmmqmmm.constants.PLANCK_HARTREE_S * freq_hz) / openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K
            )

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
        logger.info("QRHHO is True. Doing quasi-RRHO for the vibrational entropy")
        TS_vib = s_vib_qrrho_grimme(freqs, temp, omega_0=qrrho_omega_0, i_av=inertia_avg)
    elif qrrho_method == "Truhlar":
        logger.info("QRHHO is True. Doing quasi-RRHO for the vibrational entropy")
        TS_vib = s_vib_qrrho_truhlar(freqs, temp, lowfreq_thresh=qrrho_omega_0)
    else:
        raise InputError("Unknown QRRHO_method. Exiting.")

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
    S_trans = openmmqmmm.constants.GAS_CONSTANT_KCAL_PER_MOL_K * (math.log(qtrans) + 2.5)

    TS_trans = temp * S_trans / openmmqmmm.constants.HARTREE_TO_KCAL_PER_MOL  # Energy term converted to Eh

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
    log_time_since(module_init_time, "thermochemcalc")
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

        orcahesscoldim = 6
        hessdim = 3 * numatoms
        index = 0
        line = ""
        chunkheader = ""

        chunks = hessdim // orcahesscoldim
        left = hessdim % orcahesscoldim

        if left > 0:
            chunks = chunks + 1
        for chunk in range(chunks):
            if chunk == chunks - 1:
                if left == 0:
                    left = 6
                for temp in range(index, index + left):
                    chunkheader = chunkheader + "          " + str(temp)
            else:
                for temp in range(index, index + orcahesscoldim):
                    chunkheader = chunkheader + "          " + str(temp)
            outfile.write("        " + str(chunkheader) + "    \n")
            for i in range(hessdim):
                firstcolumnindex = 6 * chunk
                j = firstcolumnindex
                # If chunk = 0 then we are dealing with TR modes in first 6 columns
                # NOTE: RB note: but TS mode should also be here. Let's not set anything to zero
                if hessdim - j == 1:
                    val1 = nmodes[j][i]
                elif hessdim - j == 2:
                    val1 = nmodes[j][j]
                    val2 = nmodes[j + 1][i]
                elif hessdim - j == 3:
                    val1 = nmodes[j][i]
                    val2 = nmodes[j + 1][i]
                    val3 = nmodes[j + 2][i]
                elif hessdim - j == 4:
                    val1 = nmodes[j][i]
                    val2 = nmodes[j + 1][i]
                    val3 = nmodes[j + 2][i]
                    val4 = nmodes[j + 3][i]
                elif hessdim - j == 5:
                    val1 = nmodes[j][i]
                    val2 = nmodes[j + 1][i]
                    val3 = nmodes[j + 2][i]
                    val4 = nmodes[j + 3][i]
                    val5 = nmodes[j + 4][i]
                elif hessdim - j >= 6:
                    val1 = nmodes[j][i]
                    val2 = nmodes[j + 1][i]
                    val3 = nmodes[j + 2][i]
                    val4 = nmodes[j + 3][i]
                    val5 = nmodes[j + 4][i]
                    val6 = nmodes[j + 5][i]
                else:
                    raise InputError(f"problem\nhessdim - j :  {hessdim - j}")

                if chunk == chunks - 1:
                    for _k in range(index, index + left):
                        if left == 6:
                            line = (
                                f"{i:>6d} {val1:>14.6f} {val2:>10.6f} {val3:>10.6f} "
                                f"{val4:>10.6f} {val5:>10.6f} {val6:>10.6f}"
                            )
                        elif left == 5:
                            line = f"{i:>6d} {val1:>14.6f} {val2:>10.6f} {val3:>10.6f} {val4:>10.6f} {val5:>10.6f}"
                        elif left == 4:
                            line = f"{i:>6d} {val1:>14.6f} {val2:>10.6f} {val3:>10.6f} {val4:>10.6f}"
                        elif left == 3:
                            line = f"{i:>6d} {val1:>14.6f} {val2:>10.6f} {val3:>10.6f}"
                        elif left == 2:
                            line = f"{i:>6d} {val1:>14.6f} {val2:>10.6f}"
                        elif left == 1:
                            line = f"{i:>6d} {val1:>14.6f}"
                else:
                    for _k in range(index, index + orcahesscoldim):
                        line = (
                            f"{i:>6d} {val1:>14.6f} {val2:>10.6f} {val3:>10.6f} "
                            f"{val4:>10.6f} {val5:>10.6f} {val6:>10.6f}"
                        )
                outfile.write(" " + str(line) + "\n")
                line = ""
                chunkheader = ""
            index += 6

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
        masses = [openmmqmmm.coords.atommasses[openmmqmmm.coords.elematomnumbers[el.lower()] - 1] for el in elems]
    xcom = np.sum(masses * coords[:, 0]) / np.sum(masses)
    ycom = np.sum(masses * coords[:, 1]) / np.sum(masses)
    zcom = np.sum(masses * coords[:, 2]) / np.sum(masses)
    return xcom, ycom, zcom


def inertia(elems: Sequence[str], coords: np.ndarray, center: Sequence[float]) -> np.ndarray:
    xcom = center[0]
    ycom = center[1]
    zcom = center[2]
    Ixx = 0.0
    Iyy = 0.0
    Izz = 0.0
    Ixy = 0.0
    Ixz = 0.0
    Iyz = 0.0

    for _index, (el, coord) in enumerate(zip(elems, coords, strict=False)):
        mass = openmmqmmm.coords.atommasses[openmmqmmm.coords.elematomnumbers[el.lower()] - 1]
        x = coord[0] - xcom
        y = coord[1] - ycom
        z = coord[2] - zcom

        Ixx += mass * (y**2.0 + z**2.0)
        Iyy += mass * (x**2.0 + z**2.0)
        Izz += mass * (x**2.0 + y**2.0)
        Ixy += mass * x * y
        Ixz += mass * x * z
        Iyz += mass * y * z

    # np.array, not np.matrix: the matrix subclass is pending deprecation in numpy
    inertia_tensor = np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])
    return np.linalg.eigvals(inertia_tensor)


def calc_rotational_constants(frag: Fragment) -> list[float]:
    """Return a fragment's rotational constants in cm**-1 (the GHz values are logged as well)."""
    coords = frag.coords
    elems = frag.elems
    center = _get_center(coords, elems=elems)
    rinertia = [float(i) for i in inertia(elems, coords, center)]

    rot_constants = []
    for inertval in rinertia:
        if inertval != 0.0:
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
    # Run ORCA dummy job to get Almloef/Lindh/Schlegel Hessian
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
    # Different from orca.hess apparently
    orcacoldim = 6
    shiftpar = 0
    lastchunk = False
    grabsize = False
    with open(orcadummycalc.filename + ".opt") as optfile:
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


# atomindices refer to what atoms in the large fragment the small partial Hessian was generated for
# NOTE: Capping atom option is now disabled. Best made into a separate function
# NOTE: Trans+rot projection off right now
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
    """Build an approximate full-system Hessian by combining a small computed Hessian with a model Hessian."""
    logger.info("approximate_full_Hessian_from_smaller\n")
    write_hessian(hessian_small, hessfile="smallhessian")

    if large_atomindices is None or len(large_atomindices) == 0:
        hess_size = fragment.numatoms * 3
        logger.info("Hessian dimension %s", hess_size)
        # If Hessian is for full fragment then we use the input atomindices directly
        correct_small_atomindices = small_atomindices
        usedfragment = fragment
    elif len(large_atomindices) > 0:
        logger.info("small_atomindices: %s", small_atomindices)
        logger.info("large_atomindices: %s", large_atomindices)
        hess_size = len(large_atomindices) * 3
        fullhessian = np.zeros((hess_size, hess_size))

        if all(item in large_atomindices for item in small_atomindices) is False:
            raise InputError(
                "{}\nThis does not make sense. Exiting".format(
                    f"small_atomindices: {small_atomindices} are not all present in large_atomindices: "
                    f"{large_atomindices}"
                )
            )
        # If large Hessian is a partial Hessian of the full system then we need to change small Hessian atomindices
        correct_small_atomindices = [large_atomindices.index(i) for i in small_atomindices]
        logger.info("correct_small_atomindices: %s", correct_small_atomindices)
        subcoords, subelems = fragment.get_coords_for_atoms(large_atomindices)
        # No charge/mult: this is a sub-region of fragment, so the fragment's whole-system values
        # do not describe it. A model Hessian over this region takes them as arguments instead.
        usedfragment = openmmqmmm.Fragment(elems=subelems, coords=subcoords)
    else:
        raise InputError(
            f"small_atomindices: {small_atomindices}\nlarge_atomindices: {large_atomindices}\nSomething went wrong"
        )

    logger.info("Initializing full size Hessian of dimension: %s", hess_size)
    fullhessian = np.zeros((hess_size, hess_size))
    logger.info("Initial fullhessian: %s", fullhessian)
    logger.info("Number of Hessian elements: %s", fullhessian.size)
    write_hessian(fullhessian, hessfile="initialfullhessian")

    hessian_small = np.array(hessian_small)
    logger.info("hessian_small: %s", hessian_small)
    if rest_hessian in {"Almloef", "Lindh", "Schlegel", "Swart"}:
        logger.info("restHessian: %s", rest_hessian)
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
    elif rest_hessian == "xtb":
        raise InputError(
            "Error: restHessian='xtb' is not available in this ORCA+OpenMM build. Use an ORCA model Hessian, 'unit' or "
            "'zero' instead."
        )
    elif rest_hessian in {"unit", "identity"}:
        logger.info("restHessian is unit/identity")
        fullhessian = np.identity(hess_size)
    elif rest_hessian is None or rest_hessian.lower() == "zero":
        logger.info("RestHessian is zero.")
    else:
        logger.info("RestHessian is zero.")
    logger.info("Intermediate fullhessian: %s", fullhessian)
    logger.info("Size: %s", fullhessian.size)
    write_hessian(fullhessian, hessfile="intermedfullhessian")
    athessindices = [3 * i + j for i in correct_small_atomindices for j in [0, 1, 2]]
    for s_i, i in enumerate(athessindices):
        for s_j, j in enumerate(athessindices):
            fullhessian[i, j] = hessian_small[s_i, s_j]
    logger.info("Final fullhessian: %s", fullhessian)
    write_hessian(fullhessian, hessfile="intermedfullhessian_after_small_update")
    # Checking for linearity. Determines how many Trans+Rot modes
    tr_modenum = 5 if detect_linear(coords=fragment.coords, elems=fragment.elems) is True else 6

    logger.info("Now diagonalizing full Hessian")
    frequencies, _normal_modes, _evectors, _mode_order = _diagonalize_hessian(
        fragment.coords,
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


def _s_vib(freqs: Sequence[float], T: float) -> float:
    vibtemps = [
        (f * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S * openmmqmmm.constants.PLANCK_HARTREE_S)
        / openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K
        for f in freqs
    ]
    entropy = 0.0
    for vibtemp in vibtemps:
        entropy += openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (vibtemp / T) / (
            math.exp(vibtemp / T) - 1
        ) - openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * math.log(1 - math.exp(-1 * vibtemp / T))
    return entropy * T


def s_vib_qrrho_truhlar(freqs: Sequence[float], T: float, lowfreq_thresh: float = 100) -> float:
    logger.warning("Quasi-RRHO by Truhlar approximation active.")
    logger.info(
        "This means that the vibrational entropy is calculated according to Truhlar-approach of raising low-energy "
        f"vibrations to {lowfreq_thresh} cm-1"
    )
    logger.info("Cite: R. F. Riberio et al. J. Phys. Chem. B, 115, 14556 (2011) ")
    TS_vib_final = 0.0
    for f in freqs:
        freq_value = f
        if f < lowfreq_thresh:
            logger.warning(
                f"Frequency ({f}) is below low-freq threshold ({lowfreq_thresh}) cm-1. Setting to {lowfreq_thresh} cm-1"
            )
            freq_value = lowfreq_thresh
        vibtemp = (
            freq_value * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S * openmmqmmm.constants.PLANCK_HARTREE_S
        ) / openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K
        logger.info("vibtemp: %s", vibtemp)
        TS_vib_f = T * (
            openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (vibtemp / T) / (math.exp(vibtemp / T) - 1)
            - openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * math.log(1 - math.exp(-1 * vibtemp / T))
        )
        TS_vib_final += TS_vib_f
        logger.info("TS_vib_final: %s", TS_vib_final)

    return TS_vib_final


def s_vib_qrrho_grimme(freqs: Sequence[float], T: float, omega_0: float = 100, i_av: float | None = None) -> float:
    logger.warning("Quasi-RRHO approximation by Grimme active.")
    logger.info("This means that the vibrational entropy uses the Grimme-type interpolation formula")
    logger.info("Cite: S. Grimme, Chem. Eur. J. 2012, 18, 9955-9964.")
    TS_vib_final = 0.0
    for f in freqs:
        vibtemp = (
            f * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S * openmmqmmm.constants.PLANCK_HARTREE_S
        ) / openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K
        TS_vib_f = T * (
            openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * (vibtemp / T) / (math.exp(vibtemp / T) - 1)
            - openmmqmmm.constants.GAS_CONSTANT_HARTREE_PER_K * math.log(1 - math.exp(-1 * vibtemp / T))
        )
        m_si = (
            openmmqmmm.constants.PLANCK_J_S
            * openmmqmmm.constants.PLANCK_J_S
            / (8 * math.pi * math.pi * f * openmmqmmm.constants.HC_J_CM)
        )
        mp_si = m_si * i_av / (m_si + i_av)
        TS_rot_f_kcal = (
            T
            * openmmqmmm.constants.GAS_CONSTANT_KCAL_PER_MOL_K
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
        TS_rot_f_au = TS_rot_f_kcal / openmmqmmm.constants.HARTREE_TO_KCAL_PER_MOL  # Converting from kcal/mol to a.u.
        w = 1 / (1 + pow(omega_0 / f, 4))  # Weighting function
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
    center = _get_center(coords, elems=elems)
    rinertia = [float(i) for i in inertia(elems, coords, center)]
    if any(abs(i) < threshold for i in rinertia) is True:
        logger.info("Molecule is linear")
        return True
    logger.info("Molecule is non-linear")
    return False


# If imaginary part is larger then we convert into negative number
# Used to report vibrational frequencies
def _get_relevant_part_of_complex(numb: complex) -> float:
    if numb.real > numb.imag:
        return numb.real
    return numb.imag * -1


def _clean_frequencies(freqs: Sequence[complex]) -> list[float]:
    clean = []
    for f in freqs:
        bla = _get_relevant_part_of_complex(f)
        clean.append(bla)
    return clean


def _project_rot_and_trans(
    coords: np.ndarray,
    mass: Sequence[float],
    hessian: np.ndarray,
    rotmode_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mass = np.array(mass)
    coords = np.array(coords) * openmmqmmm.constants.ANG_TO_BOHR
    coords = coords.copy().reshape(-1, 3)
    na = coords.shape[0]
    wavenumber_scaling = (
        1e10
        * np.sqrt(openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL / openmmqmmm.constants.BOHR_TO_NM**2)
        / (2 * np.pi * openmmqmmm.constants.LIGHT_SPEED_CM_PER_S * 0.01)
    )
    TotDOF = 3 * na

    invsqrtm3 = 1.0 / np.sqrt(np.repeat(mass, 3))
    wHessian = hessian.copy() * np.outer(invsqrtm3, invsqrtm3)

    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0) / np.sum(mass)

    xcm = coords - cxyz[np.newaxis, :]

    inertia_tensor = np.sum(
        [mass[i] * (np.eye(3) * (np.dot(xcm[i], xcm[i])) - np.outer(xcm[i], xcm[i])) for i in range(na)], axis=0
    )

    Ivals, Ivecs = np.linalg.eigh(inertia_tensor)
    # Eigenvectors are in the rows after transpose
    Ivecs = Ivecs.T

    RotDOF = 0
    for i in range(3):
        logger.info("Ivals[i]: %s", Ivals[i])
        if abs(Ivals[i]) > rotmode_threshold:
            RotDOF += 1
    TR_DOF = 3 + RotDOF
    logger.info("TR_DOF: %s", TR_DOF)
    if TR_DOF not in (5, 6):
        logger.info("Unexpected number of trans+rot DOF: {TR_DOF} not in (5, 6)")

    ic_eckart = np.zeros((6, TotDOF))
    for i in range(na):
        # The dot product of (the coordinates of the atoms with respect to the center of mass) and
        # the corresponding row of the matrix used to diagonalize the moment of inertia tensor
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
            logger.info(f"Gram-Schmidt orthogonalization failed after {maxIt} iterations")

    # Diagonalize the overlap matrix to create (3N-6) orthonormal basis vectors
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

    freqs_wavenumber = wavenumber_scaling * np.sqrt(np.abs(ichess_vals)) * np.sign(ichess_vals)

    return freqs_wavenumber, normal_modes, normal_modes_cart


def _calc_raman_activities(
    hessmasses: Sequence[float], evectors: np.ndarray, polarizability_derivs: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Calculating Raman activities")

    hesslength = 3 * len(hessmasses)

    mass_matrix = np.repeat(hessmasses, 3)
    inv_sqrt_mass_matrix = np.diag(1 / (mass_matrix**0.5))
    displacements = inv_sqrt_mass_matrix.dot(np.transpose(evectors))

    A_der = np.zeros((hesslength, 9))
    for i in range(hesslength):
        A_der[i, :] = polarizability_derivs[i].reshape(1, 9)

    # Transform polarizability derivatives to normal coordinates
    # A_der : 3*Natom x 9
    # Lx : 3*Natom x 3*Natom
    # A_der_q : 9 x 3*Natom
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

    # Now calculating alphas, betas (see Neugebauer J Comput Chem 2002)
    # and Raman activity and depolarization ratio
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
