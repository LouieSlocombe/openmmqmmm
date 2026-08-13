import copy
import logging
import math
import os
import shutil
import time

import numpy as np

import openmmqmmm.constants
import openmmqmmm.coords
import openmmqmmm.orca
from openmmqmmm.coords import check_charge_mult
from openmmqmmm.exceptions import (
    InputError,
    InternalError,
)
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.results import Results
from openmmqmmm.utils import clean_number, listdiff, log_time_since, main_header

logger = logging.getLogger(__name__)


# Analytical frequencies function. Only for theories with this option added (e.g. ORCATheory and CFourTheory)
# Checked by analytic_hessian attribute True
# TODO: IR/Raman intensities
def analytic_frequencies(
    fragment=None,
    theory=None,
    charge=None,
    mult=None,
    temp=298.15,
    masses=None,
    pressure=1.0,
    qrrho=True,
    qrrho_method="Grimme",
    qrrho_omega_0=100,
    scaling_factor=1.0,
    symmetry_number=None,
    rotmode_threshold=1e-4,
) -> "Results":
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
    # Hessian atoms
    hessatoms = list(range(fragment.numatoms))

    # Masses
    if masses is None:
        masses = fragment.list_of_masses

    if theory.analytic_hessian:
        logger.info(f"Requesting analytical Hessian calculation from {theory.theorynamelabel}")
        logger.info("")
        # Check charge/mult
        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "AnFreq", theory=theory)
        # Do single-point theory run with Hessian=True
        theory.run(current_coords=fragment.coords, elems=fragment.elems, charge=charge, mult=mult, hessian=True)

        # Grab Hessian from theory object
        logger.info("Getting analytic Hessian from theory object")
        hessian = theory.hessian
        # Diagonalize
        frequencies, nmodes, evectors, _mode_order = diagonalize_hessian(
            fragment.coords, theory.hessian, masses, fragment.elems, tr_modenum=tr_modenum, projection=True
        )
        logger.info("Now scaling frequencies by scaling factor: %s", scaling_factor)
        frequencies = scaling_factor * frequencies

        # NOTE:
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

        # Print out Freq output.
        printfreqs(
            frequencies,
            len(hessatoms),
            tr_modenum=tr_modenum,
            intensities=IR_intens_values,
            raman_activities=raman_activities,
        )
        logger.info("\n\n")
        logger.info("Normal mode composition factors by element")
        printfreqs_and_nm_elem_comps(frequencies, fragment, evectors, hessatoms=hessatoms, tr_modenum=tr_modenum)
        thermodict = thermochemcalc(
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

        # Add Hessian to fragment and write to file
        fragment.hessian = hessian
        write_hessian(hessian, hessfile="Hessian")

        # Create dummy-ORCA file with frequencies and normal modes
        print_dummy_orca_file(fragment.elems, fragment.coords, frequencies, nmodes, "orcahessfile.hess")
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

    raise InputError("Analytical frequencies not available for theory. Exiting.")


# Numerical frequencies function
# ORCA uses 0.005 Bohr = 0.0026458861 Ang, CHemshell uses 0.01 Bohr = 0.00529 Ang
def numerical_frequencies(
    fragment=None,
    theory=None,
    charge=None,
    mult=None,
    npoint=2,
    displacement=0.005,
    hessatoms=None,
    numcores=1,
    runmode="serial",
    temp=298.15,
    pressure=1.0,
    hessatoms_masses=None,
    qrrho=True,
    qrrho_method="Grimme",
    qrrho_omega_0=100,
    IR=True,
    Raman=False,
    rotmode_threshold=1e-4,
    scaling_factor=1.0,
    symmetry_number=None,
    force_projection=None,
) -> "Results":
    """Compute vibrational frequencies from numerical differentiation of gradients."""
    module_init_time = time.time()
    logger.info("------------NUMERICAL FREQUENCIES-------------")
    ################
    # Basic checks
    ################
    if fragment is None or theory is None:
        raise InputError("NumFreq requires a fragment and a theory object")
    # Check charge/mult
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "NumFreq", theory=theory)
    ################
    # SETUP
    ################
    # Setting variables
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

    # Making sure hessatoms list is sorted and only contains unique values
    hessatoms = sorted(set(hessatoms))

    # If hessatoms_masses list was provided
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
    #####################
    # Molecular orbitals
    #####################
    # ORCA-specific: Copy old GBW file from .. dir
    # NOTE: Pretty ugly. Not sure if there is a good alternative at the moment. Moreadfile option would override this
    # anyway
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

    ##########################
    # Calculation preparation
    ##########################
    # Creating directory
    shutil.rmtree("Numfreq_dir", ignore_errors=True)
    os.mkdir("Numfreq_dir")
    os.chdir("Numfreq_dir")
    logger.info("Creating separate directory for displacement calculations: Numfreq_dir ")

    displacement_bohr = displacement * openmmqmmm.constants.ang2bohr
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
    logger.info("")
    logger.info(f"Displacement: {displacement:5.4f} Å ({displacement_bohr:5.4f} Bohr)")
    logger.info("")
    logger.info("Starting geometry:")
    # Converting to numpy array just in case
    current_coords_array = np.array(coords)

    logger.info("Printing hessatoms geometry...")
    openmmqmmm.coords.print_coords_for_atoms(coords, elems, hessatoms)
    logger.info("")

    # Looping over each atom and each coordinate to create displaced geometries
    # Only displacing atom if in hessatoms list. i.e. possible partial Hessian
    list_of_displaced_geos = []
    list_of_displacements = []
    for atom_index in range(len(current_coords_array)):
        if atom_index in hessatoms:
            for coord_index in range(3):
                val = current_coords_array[atom_index, coord_index]
                # Displacing in + direction
                current_coords_array[atom_index, coord_index] = val + displacement
                y = current_coords_array.copy()
                list_of_displaced_geos.append(y)
                list_of_displacements.append((atom_index, coord_index, "+"))
                if npoint == 2:
                    # Displacing  - direction
                    current_coords_array[atom_index, coord_index] = val - displacement
                    y = current_coords_array.copy()
                    list_of_displaced_geos.append(y)
                    list_of_displacements.append((atom_index, coord_index, "-"))
                # Displacing back
                current_coords_array[atom_index, coord_index] = val

    # Original geo added here if onepoint
    if npoint == 1:
        list_of_displaced_geos.append(current_coords_array)
        list_of_displacements.append("Originalgeo")

    logger.info("List of displacements: %s", list_of_displacements)

    # Creating fragments
    # Creating displacement labels as strings and adding to fragment
    # Also calclabels, currently used by runmode serial only
    list_of_labels = []
    all_disp_fragments = []
    for dispgeo, disp in zip(list_of_displaced_geos, list_of_displacements, strict=False):
        # Original geo
        if disp == "Originalgeo":
            calclabel = "Originalgeo"
            stringlabel = "Originalgeo"
        # Displacements
        else:
            atom_disp = disp[0]
            if disp[1] == 0:
                crd = "x"
            elif disp[1] == 1:
                crd = "y"
            elif disp[1] == 2:
                crd = "z"
            drection = disp[2]
            calclabel = f"Atom: {atom_disp!s} Coord: {crd!s} Direction: {drection!s}"
            stringlabel = f"{disp[0]}_{disp[1]}_{disp[2]}"
        # Create fragment
        frag = openmmqmmm.Fragment(coords=dispgeo, elems=elems, label=stringlabel, charge=charge, mult=mult)
        all_disp_fragments.append(frag)
        list_of_labels.append(calclabel)

    if len(list_of_labels) != len(list_of_displaced_geos):
        raise InternalError("Mismatch between number of labels and displaced geometries")

    ########################
    # RUNNING displacements
    ########################
    displacement_grad_dictionary = {}
    displacement_dipole_dictionary = {}
    displacement_polarizability_dictionary = {}
    # TODO: Have serial use all_disp_fragments instead to be consistent with parallel
    if runmode == "serial":
        logger.info("Runmode: serial")
        logger.info("Only theory parallelization is active.")
        logger.info("Theory numcores attributes is set to: %s", theory.numcores)
        # Looping over geometries and running.
        #   key: AtomNCoordPDirectionm   where N=atomnumber, P=x,y,z and direction m: + or -
        for numdisp, (disp, label, geo) in enumerate(
            zip(list_of_displacements, list_of_labels, list_of_displaced_geos, strict=False)
        ):
            if label == "Originalgeo":
                calclabel = "Originalgeo"
                logger.info("Doing original geometry calc.")
                stringlabel = calclabel
            else:
                calclabel = label
                # for index,(el,coord) in enumerate(zip(elems,coords))
                logger.info(f"Running displacement: {numdisp + 1} / {len(list_of_labels)}")
                logger.info("%s", calclabel)
                # Now using string label
                stringlabel = f"{disp[0]}_{disp[1]}_{disp[2]}"
            _energy, gradient = theory.run(current_coords=geo, elems=elems, grad=True, charge=charge, mult=mult)
            displacement_grad_dictionary[stringlabel] = gradient

            # Grabbing dipole moment if available
            if IR is True:
                try:
                    displacement_dm = theory.get_dipole_moment()
                    displacement_dipole_dictionary[stringlabel] = displacement_dm
                except Exception:  # noqa: BLE001 - best-effort property grab
                    pass

            # Grabbing polarizability tensor if requested
            if Raman is True:
                try:
                    logger.info("Getting polarizability tensor")
                    displacement_pol = theory.get_polarizability_tensor()
                    # Checking if array is all zero (i.e. no polarizability information was found)
                    if not np.any(displacement_pol):
                        logger.warning("No polarizability information found")
                    displacement_polarizability_dictionary[stringlabel] = displacement_pol
                except Exception:  # noqa: BLE001 - best-effort polarizability grab
                    logger.warning("Problem getting polarizability tensor from theory interface. Skipping")

    # TODO: Dipole moment/polarizability grab for parallel mode
    elif runmode == "parallel":
        if isinstance(theory, openmmqmmm.QMMMTheory):
            logger.info("Numfreq in runmode='parallel' with QM/MM is quite experimental")

        logger.info(f"Starting Numfreq calculations in parallel mode (numcores={numcores}) using Singlepoint_parallel")
        logger.info(f"There are {len(all_disp_fragments)} displacements")
        # Launching multiple energy+gradient calculations in parallel on list of fragments: all_image_fragments
        logger.info("Looping over fragments")
        result = openmmqmmm.job_parallel(
            fragments=all_disp_fragments,
            theories=[theory],
            numcores=numcores,
            allow_theory_parallelization=True,
            grad=True,
            copytheory=True,
        )
        gradient_dict = result.gradients_dict
        # Gradient_dict is already correctly formatted
        displacement_grad_dictionary = gradient_dict

        displacement_dipole_dictionary = result.displacement_dipole_dictionary
        displacement_polarizability_dictionary = result.displacement_polarizability_dictionary
    else:
        raise InputError("Unknown runmode.")

    ############################################
    logger.info("NumFreq Displacement calculations are done!")
    logger.info("")

    if len(displacement_grad_dictionary) == 0:
        raise InputError(
            "Missing gradients for displacement.\nSomething went wrong in Numfreq displacement calculations."
        )
    logger.info("Length of displacement_grad_dictionary %s", len(displacement_grad_dictionary))
    # Initialize empty Hessian
    hesslength = 3 * len(hessatoms)
    hessian = np.zeros((hesslength, hesslength))

    # Initializing dipole derivatives
    dipole_derivs = np.zeros((hesslength, 3))
    polarizability_derivs = []  # array of 3x3 tensors

    # Onepoint-formula Hessian
    if npoint == 1:
        logger.info("Assembling the one-point Hessian")
        # First, grab original geometry gradient
        # If partial Hessian remove non-hessatoms part of gradient:
        # Get partial matrix by deleting atoms not present in list.
        original_grad = get_partial_matrix(displacement_grad_dictionary["Originalgeo"], hessatoms)
        original_grad_1d = np.ravel(original_grad)
        # IR intensities if dipoles available
        if IR is True and len(displacement_dipole_dictionary) > 0:
            original_dipole = np.array(displacement_dipole_dictionary["Originalgeo"])
        # Raman if requested
        if Raman is True and len(displacement_polarizability_dictionary) > 0:
            original_polarizability = np.array(displacement_polarizability_dictionary["Originalgeo"])
            logger.info("original_polarizability: %s", original_polarizability)
        # Starting index for Hessian array
        hessindex = 0
        # Loop over Hessian atoms and grab each gradient component. Calculate Hessian component and add to matrix
        # for atomindex in range(0,len(hessatoms)):
        for atomindex in hessatoms:
            # Iterate over x,y,z components
            for crd in [0, 1, 2]:
                # Looking up each gradient for atomindex, crd-component(x=0,y=1 or z=2) and '+'
                lookup_string_pos = f"{atomindex}_{crd}_+"
                grad_pos = displacement_grad_dictionary[lookup_string_pos]
                # Getting grad as numpy matrix and converting to 1d
                # If partial Hessian remove non-hessatoms part of gradient:
                grad_pos = get_partial_matrix(grad_pos, hessatoms)
                grad_pos_1d = np.ravel(grad_pos)
                Hessrow = (grad_pos_1d - original_grad_1d) / displacement_bohr
                hessian[hessindex, :] = Hessrow
                grad_pos_1d = 0
                # IR intensities if dipoles available
                if IR is True and len(displacement_dipole_dictionary) > 0:
                    # Make sure it's not a dict of None's
                    if any(value is None for value in displacement_dipole_dictionary.values()):
                        pass
                    elif len(displacement_dipole_dictionary[lookup_string_pos]) > 0:
                        disp_dipole = np.array(displacement_dipole_dictionary[lookup_string_pos])
                        dd_deriv = (disp_dipole - original_dipole) / displacement_bohr
                        dipole_derivs[hessindex, :] = dd_deriv
                # Raman if requested
                if Raman is True and len(displacement_polarizability_dictionary) > 0:
                    disp_polarizability = np.array(displacement_polarizability_dictionary[lookup_string_pos])
                    pz_deriv = (disp_polarizability - original_polarizability) / displacement_bohr
                    polarizability_derivs.append(pz_deriv)
                hessindex += 1

    # Twopoint-formula Hessian. pos and negative directions come in order
    elif npoint == 2:
        logger.info("Assembling the two-point Hessian")
        hessindex = 0
        # Loop over Hessian atoms and grab each gradient component. Calculate Hessian component and add to matrix
        # for atomindex in range(0,len(hessatoms)):
        for atomindex in hessatoms:
            # Iterate over x,y,z components
            for crd in [0, 1, 2]:
                # Looking up each gradient for atomindex, crd-component(x=0,y=1 or z=2) and '+'
                lookup_string_pos = f"{atomindex}_{crd}_+"
                lookup_string_neg = f"{atomindex}_{crd}_-"
                grad_pos = displacement_grad_dictionary[lookup_string_pos]
                # Looking up each gradient for atomindex, crd-component(x=0,y=1 or z=2) and '-'
                grad_neg = displacement_grad_dictionary[lookup_string_neg]
                # Getting grad as numpy matrix and converting to 1d
                # If partial Hessian remove non-hessatoms part of gradient:
                grad_pos = get_partial_matrix(grad_pos, hessatoms)
                grad_pos_1d = np.ravel(grad_pos)
                grad_neg = get_partial_matrix(grad_neg, hessatoms)
                grad_neg_1d = np.ravel(grad_neg)
                Hessrow = (grad_pos_1d - grad_neg_1d) / (2 * displacement_bohr)
                hessian[hessindex, :] = Hessrow
                grad_pos_1d = 0
                grad_neg_1d = 0

                # IR intensities if dipoles available
                if IR is True and len(displacement_dipole_dictionary) > 0:
                    # Make sure it's not a dict of None's
                    if any(value is None for value in displacement_dipole_dictionary.values()):
                        pass
                    elif len(displacement_dipole_dictionary[lookup_string_pos]) > 0:
                        disp_dipole_pos = np.array(displacement_dipole_dictionary[lookup_string_pos])
                        disp_dipole_neg = np.array(displacement_dipole_dictionary[lookup_string_neg])
                        dd_deriv = (disp_dipole_pos - disp_dipole_neg) / (2 * displacement_bohr)
                        dipole_derivs[hessindex, :] = dd_deriv
                # Raman if requested
                if Raman is True and len(displacement_polarizability_dictionary) > 0:
                    disp_polarizability_pos = np.array(displacement_polarizability_dictionary[lookup_string_pos])
                    disp_polarizability_neg = np.array(displacement_polarizability_dictionary[lookup_string_neg])
                    pz_deriv = (disp_polarizability_pos - disp_polarizability_neg) / (2 * displacement_bohr)
                    polarizability_derivs.append(pz_deriv)
                hessindex += 1
    logger.info("")

    # Symmetrize Hessian by taking average of matrix and transpose
    symm_hessian = (hessian + hessian.transpose()) / 2
    hessian = symm_hessian

    # Use input masses if given, otherwise take from frament
    if hessatoms_masses is None:
        logger.info("allatoms: %s", allatoms)
        logger.info("hessatoms: %s", hessatoms)
        logger.debug("Atomic masses: %s", fragment.list_of_masses)
        hessmasses = openmmqmmm.coords.get_partial_list(allatoms, hessatoms, fragment.list_of_masses)
    else:
        hessmasses = hessatoms_masses

    logger.info("hessmasses: %s", hessmasses)
    # Mass-weighted Hessian (in case we need it)
    _mwhessian, _massmatrix = massweight(hessian, hessmasses)
    # Get partial matrix by deleting atoms not present in list.
    hesselems = openmmqmmm.coords.get_partial_list(allatoms, hessatoms, elems)

    hesscoords = np.take(fragment.coords, hessatoms, axis=0)
    logger.info("Elements: %s", hesselems)
    logger.info("Masses used: %s", hessmasses)

    # Evectors: eigenvectors of the mass-weighed Hessian
    # Normal modes: unweighted
    frequencies, nmodes, evectors, mode_order = diagonalize_hessian(
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

    # IR intensities if dipoles available
    IR_intens_values = None
    if IR is True and np.any(dipole_derivs):
        dipole_derivs = dipole_derivs[mode_order]
        IR_intens_values = calc_ir_intensities(hessmasses, evectors, dipole_derivs)

    # Raman activities if polarizabilities available
    if Raman is True:
        logger.info("Raman calculation active")
        if len(polarizability_derivs) == 0:
            logger.info("No polarizability information found. Skipping Raman.")
            raman_activities = None
            depolarization_ratios = None
        else:
            logger.info("Polarizability derivatives are available.")
            # Reordering just in case
            polarizability_derivs = [polarizability_derivs[i] for i in mode_order]
            raman_activities, depolarization_ratios = calc_raman_activities(hessmasses, evectors, polarizability_derivs)
    else:
        raman_activities = None
        depolarization_ratios = None
    logger.info("")

    # Print out Freq output. Maybe print normal mode compositions here instead???
    printfreqs(
        frequencies,
        len(hessatoms),
        tr_modenum=tr_modenum,
        intensities=IR_intens_values,
        raman_activities=raman_activities,
    )

    logger.info("\n\n")
    logger.info("Normal mode composition factors by element")
    printfreqs_and_nm_elem_comps(frequencies, fragment, evectors, hessatoms=hessatoms, tr_modenum=tr_modenum)

    logger.info("\nNow doing thermochemistry")

    # Get and print out thermochemistry
    thermodict = thermochemcalc(
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

    # Write Hessian to file
    write_hessian(hessian, hessfile="Hessian")

    # Write ORCA-style Hessian file. Hardcoded filename here. Change?
    # Note: Passing hesscords here instead of coords. Change?
    openmmqmmm.orca.write_orca_hessfile(hessian, hesscoords, hesselems, hessmasses, "orcahessfile.hess")

    # Create dummy-ORCA file with frequencies and normal modes
    print_dummy_orca_file(hesselems, hesscoords, frequencies, nmodes, "orcahessfile.hess")
    logger.info("Wrote dummy ORCA outputfile with frequencies and normal modes: orcahessfile.hess_dummy.out")
    logger.info("Can be used for visualization\n")
    logger.info("------------NUMERICAL FREQUENCIES END-------------")

    # Add things to fragment
    fragment.hessian = hessian  # Hessian

    # Return to ..
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


# Get partial matrix properly
def get_partial_matrix(matrix, hessatoms):
    return np.take(matrix, hessatoms, axis=0)


# Diagonalize Hessian from input Hessian, masses and element-strings
def diagonalize_hessian(
    coords,
    hessian,
    masses,
    elems,
    projection=True,
    tr_modenum=None,
    LargeImagFreqThreshold=-100,
    rotmode_threshold=1e-4,
):
    logger.info("\nDiagonalizing Hessian")
    atomlist = []
    for i, j in enumerate(elems):
        atomlist.append(str(j) + "-" + str(i))

    # Projecting out translations and rotations
    if projection is True:
        logger.info("Projection of out rotational and translational modes active!")
        vfreqs, evectors, nmodes = project_rot_and_trans(coords, masses, hessian, rotmode_threshold=rotmode_threshold)
        # Adding TRmodes zeros to vfreqs list
        for _ in range(tr_modenum):
            vfreqs = np.insert(vfreqs, 0, 0.0)
        # Adding zero TSmode vectors to evectors and nmodes
        for _ in range(tr_modenum):
            evectors = np.insert(evectors, 0, [0.0] * evectors.shape[1], axis=0)
            nmodes = np.insert(nmodes, 0, [0.0] * nmodes.shape[1], axis=0)

        # Moder-order unchanged
        mode_order = list(range(len(nmodes)))
        return vfreqs, nmodes, evectors, mode_order
    logger.info("No projection of rotational and translational modes will be done!")
    # Massweight Hessian
    mwhessian, massmatrix = massweight(hessian, masses)
    # Diagonalize mass-weighted Hessian
    evalues, evectors = np.linalg.eigh(mwhessian)
    evectors = np.transpose(evectors)

    # Unweight eigenvectors to get normal modes
    nmodes = np.dot(evectors, massmatrix)

    # Calculate frequencies from eigenvalues
    vfreqs = calcfreq(evalues)

    # Clean up the complex frequencies before using further
    vfreqs = clean_frequencies(vfreqs)

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
    # Now reordering freqs, and evectors
    # First TRmodes, then SPmodes then rest
    logger.info("Reordering modes so that TRmodes come first, then SP modes, then rest")
    neworder = TRmodes + SPmodes + listdiff(range(len(vfreqs)), TRmodes + SPmodes)
    vfreqs = [vfreqs[i] for i in neworder]
    evectors = evectors[neworder]
    nmodes = nmodes[neworder]

    return vfreqs, nmodes, evectors, neworder


# Calculate IR intensities from masses, (mass-weighted) eigenvectors and dipole derivative matrix
def calc_ir_intensities(hessmasses, evectors, dipole_derivs):
    intens_factor = 974.88011184
    mass_matrix = np.repeat(hessmasses, 3)
    inv_sqrt_mass_matrix = np.diag(1 / (mass_matrix**0.5))
    displacements = inv_sqrt_mass_matrix.dot(np.transpose(evectors))
    de_q = displacements.T @ dipole_derivs
    return intens_factor * np.einsum("qt, qt -> q", de_q, de_q)


# Massweight Hessian
def massweight(matrix, masses):
    numatoms = len(masses)
    mass_mat = np.zeros((3 * numatoms, 3 * numatoms), dtype=float)
    molwt = [masses[int(i)] for i in range(numatoms) for j in range(3)]
    for i in range(len(molwt)):
        mass_mat[i, i] = molwt[i] ** -0.5
    mwhessian = np.dot((np.dot(mass_mat, matrix)), mass_mat)
    return mwhessian, mass_mat


# Calculate frequencies from eigenvalus
def calcfreq(evalues):
    hartree2j = openmmqmmm.constants.hartree2j
    bohr2m = openmmqmmm.constants.bohr2m
    amu2kg = openmmqmmm.constants.amu2kg
    c = openmmqmmm.constants.c
    pi = openmmqmmm.constants.pi
    evalues_si = [val * hartree2j / bohr2m / bohr2m / amu2kg for val in evalues]
    vfreq_hz = [1 / (2 * pi) * np.sqrt(np.complex128(val)) for val in evalues_si]
    return [val / c for val in vfreq_hz]


def printfreqs(vfreq, numatoms, tr_modenum=6, intensities=None, raman_activities=None):
    logger.info("%s", "-" * 40)
    logger.info("VIBRATIONAL FREQUENCY SUMMARY")
    logger.info("%s", "-" * 40)
    if intensities is None:
        logger.info("No IR intensities were calculated. Setting values to 0.0.")
    if raman_activities is None:
        logger.info(
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


# Function to print frequencies and also elemental normal mode composition
def printfreqs_and_nm_elem_comps(vfreq, fragment, evectors, hessatoms=None, tr_modenum=6, numdigits=3):
    with open("normalmodecomposition_factors.txt", "w") as f:
        numatoms = len(hessatoms)
        logger.info("%s", "{:>6}{:>16}  {:<18}".format("Mode", "Freq(cm**-1)", "Elemental composition factors"))
        for mode in range(3 * numatoms):
            # Get elemental normalmode comps
            normmodecompelemsdict = normalmodecomp_permode_by_elems(mode, fragment, evectors, hessatoms=hessatoms)
            normmodecompelemsdict_list = [f"{k}: {v:.{numdigits}f}" for k, v in normmodecompelemsdict.items()]
            normmodecompelemsdict_string = "   ".join(normmodecompelemsdict_list)
            vib = vfreq[mode]
            line = f"  {mode:<4d}{vib:>14.4f}    {normmodecompelemsdict_string}"

            if mode < tr_modenum:
                line = line + " (TR mode)"
            logger.info("%s", line)
            f.write(line + "\n")


# NOTE: THIS IS NOT CORRECT
# TODO: Need to identify SP mode
# FOR SADDLEPOINT, the SP mode will be the largest imaginary mode, hence mode 0.


def thermochemcalc(
    vfreq,
    atoms,
    fragment,
    multiplicity,
    temp=298.15,
    pressure=1.0,
    qrrho=True,
    qrrho_method="Grimme",
    qrrho_omega_0=100,
    use_full_geo_in_rotational_analysis=True,
    symmetry_number=None,
    rotmode_threshold=1e-4,
):
    module_init_time = time.time()
    logger.info("")
    logger.info(main_header("Thermochemistry via rigid-rotor harmonic oscillator approximation"))
    logger.info("")
    if len(atoms) == 1:
        logger.info("System is an atom.")
        moltype = "atom"
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

    # What coordinates to use for rotational analysis
    if use_full_geo_in_rotational_analysis:
        logger.info("Using full geometry in rotational analysis")
        # Using full coordinates in fragment
        coords = fragment.coords
        elems = fragment.elems
    else:
        logger.info("Using Hessian-geometry in rotational analysis")
        coords = np.take(fragment.coords, atoms, axis=0)
        elems = [fragment.elems[i] for i in atoms]

    # Masses to use for translational entropy
    totalmass = sum(fragment.masses)
    logger.info("Total mass of molecule: %s", totalmass)

    ###################
    # ROTATIONAL PART
    ###################
    if moltype != "atom":
        logger.info("\nDoing rotatational analysis:")
        # Moments of inertia (amu A^2 ), eigenvalues
        center = get_center(coords, elems=elems)
        rinertia = [float(i) for i in inertia(elems, coords, center)]

        logger.info("Moments of inertia (amu Å^2): %s", rinertia)
        # Changing units to m and kg
        inertia_si = np.array(rinertia) * openmmqmmm.constants.amu2kg * openmmqmmm.constants.ang2m**2
        inertia_avg = (inertia_si[0] + inertia_si[1] + inertia_si[2]) / 3
        # Rotational energy and entropy
        if moltype == "atom":
            q_r = 1.0
            S_rot = 0.0
            E_rot = 0.0
        elif moltype == "linear":
            # Rotational temperatures (linear case)
            rot_temps = [
                float(openmmqmmm.constants.h_planck**2 / (8 * math.pi**2 * openmmqmmm.constants.k_b_jk * in_I))
                for in_I in inertia_si
                if in_I != 0.0
            ]
            logger.info(f"Rotational temperatures: {rot_temps} K")
            rot_temps_x = rot_temps[0]
            # Symmetry number
            sigma_r = 1.0
            q_r = (1 / sigma_r) * (temp / (rot_temps_x))
            S_rot = openmmqmmm.constants.R_gasconst * (math.log(q_r) + 1.0)
            E_rot = openmmqmmm.constants.R_gasconst * temp
            # Rotational constants
            rotconstants = calc_rotational_constants(fragment)
        else:
            # Nonlinear case

            # Rotational temperatures
            rot_temps_x = openmmqmmm.constants.h_planck**2 / (
                8 * math.pi**2 * openmmqmmm.constants.k_b_jk * inertia_si[0]
            )
            rot_temps_y = openmmqmmm.constants.h_planck**2 / (
                8 * math.pi**2 * openmmqmmm.constants.k_b_jk * inertia_si[1]
            )
            rot_temps_z = openmmqmmm.constants.h_planck**2 / (
                8 * math.pi**2 * openmmqmmm.constants.k_b_jk * inertia_si[2]
            )
            logger.info(f"Rotational temperatures: {rot_temps_x}, {rot_temps_y}, {rot_temps_z} K")
            # Rotational constants
            rotconstants = calc_rotational_constants(fragment)

            if symmetry_number is None:
                logger.info("Case: nonlinear system and no user-provided symmetry_number.")
                logger.info("Setting symmetry number to 1.0 (appropriate for C1, Ci and Cs pointgroups)")
                sigma_r = 1.0
            else:
                logger.info("Case: nonlinear system and user-provided symmetry_number: %s", symmetry_number)
                sigma_r = symmetry_number

            q_r = (
                (math.pi ** (1 / 2) / sigma_r)
                * (temp ** (3 / 2))
                / ((rot_temps_x * rot_temps_y * rot_temps_z) ** (1 / 2))
            )
            S_rot = openmmqmmm.constants.R_gasconst * (math.log(q_r) + 1.5)
            E_rot = 1.5 * openmmqmmm.constants.R_gasconst * temp
        TS_rot = temp * S_rot
    else:
        E_rot = 0.0
        TS_rot = 0.0
    ###################
    # VIBRATIONAL PART
    ###################
    if moltype != "atom":
        logger.info("\nDoing vibrational analysis:")
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
                freq_Hz = vib * openmmqmmm.constants.c
                vibtemp = (openmmqmmm.constants.h_planck_hartreeseconds * freq_Hz) / openmmqmmm.constants.R_gasconst
                vibtemps.append(vibtemp)

        # Zero-point vibrational energy
        zpve = sum([i * openmmqmmm.constants.halfhcfactor for i in freqs])

        # Thermal vibrational energy: R * sum over modes of theta*(1/2 + 1/(exp(theta/T) - 1)),
        # the harmonic-oscillator internal energy. The Bose-Einstein factor is
        # 1/(exp(x) - 1); writing it as 1/exp(x - 1) overestimates the thermal
        # correction (2.7x for water) and does not reach the classical RT limit.
        sumb = 0.0
        for v in vibtemps:
            sumb = sumb + v * (0.5 + (1 / (np.exp(v / temp) - 1)))
        E_vib = sumb * openmmqmmm.constants.R_gasconst
        vibenergycorr = E_vib - zpve
        # Vibrational entropy via RRHO.
        if qrrho is True:
            logger.info("QRHHO is True. Doing quasi-RRHO for the vibrational entropy")
            if qrrho_method == "Grimme":
                TS_vib = s_vib_qrrho_grimme(freqs, temp, omega_0=qrrho_omega_0, i_av=inertia_avg)
            elif qrrho_method == "Truhlar":
                TS_vib = s_vib_qrrho_truhlar(freqs, temp, lowfreq_thresh=qrrho_omega_0)
            else:
                raise InputError("Unknown QRRHO_method. Exiting.")
        else:
            TS_vib = s_vib(freqs, temp)
    else:
        zpve = 0.0
        E_vib = 0.0
        freqs = []
        vibenergycorr = 0.0
        TS_vib = 0.0

    ###################
    # TRANSLATIONAL PART
    ###################
    E_trans = 1.5 * openmmqmmm.constants.R_gasconst * temp

    # R gas constant in kcal/molK
    R_kcalpermolK = 1.987e-3
    # Conversion factor for formula.
    # TODO: cleanup
    factor = 0.025607868
    # Translation partition function and T*S_trans. Using kcal/mol
    qtrans = (factor * temp**2.5 * totalmass**1.5) / pressure
    S_trans = R_kcalpermolK * (math.log(qtrans) + 2.5)

    TS_trans = temp * S_trans / openmmqmmm.constants.harkcal  # Energy term converted to Eh

    #######################
    # Electronic entropy
    #######################
    if multiplicity is not None:
        q_el = multiplicity
        S_el = openmmqmmm.constants.R_gasconst * math.log(q_el)
        TS_el = temp * S_el
    else:
        # E.g. OpenMMTheory
        TS_el = 0.0

    #######################
    # Thermodynamic corrections
    #######################
    E_tot = E_vib + E_trans + E_rot
    Hcorr = E_vib + E_trans + E_rot + openmmqmmm.constants.R_gasconst * temp
    TS_tot = TS_el + TS_trans + TS_rot + TS_vib
    Gcorr = Hcorr - TS_tot

    #######################
    # PRINTING
    #######################
    logger.info("")
    logger.info("Thermochemistry")
    logger.info("--------------------")
    logger.info("Temperature: %s K", temp)
    logger.info("Pressure: %s atm", pressure)
    logger.info("Hessian atomlist: %s", atoms)
    logger.info("Total mass: %s", totalmass)
    logger.info("")

    if moltype != "atom":
        logger.info("Moments of inertia: %s", rinertia)
        logger.info("Rotational constants (cm-1): %s", rotconstants)

    logger.info("")
    # Thermal corrections
    logger.info("Energy corrections:")
    logger.info("Zero-point vibrational energy: %s", zpve)
    logger.info("%s", "{} {} {} {} {}".format("Translational energy (", temp, "K) :", E_trans, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Rotational energy (", temp, "K) :", E_rot, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Total vibrational energy (", temp, "K) :", E_vib, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Vibrational energy correction (", temp, "K) :", vibenergycorr, "Eh"))
    logger.info("")
    logger.info("Entropy terms (TS):")
    logger.info("%s", "{} {} {} {} {}".format("Translational entropy (TS_trans) (", temp, "K) :", TS_trans, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Rotational entropy (TS_rot) (", temp, "K) :", TS_rot, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Vibrational entropy (TS_vib) (", temp, "K) :", TS_vib, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Electronic entropy (TS_el) (", temp, "K) :", TS_el, "Eh"))
    logger.info("")
    if moltype != "atom":
        logger.info(f"Note: symmetry number : {sigma_r} used for rotational entropy")
        logger.info("")
    logger.info("Thermodynamic terms:")
    logger.info("%s", "{} {} {} {} {}".format("Enthalpy correction (Hcorr) (", temp, "K) :", Hcorr, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Entropy correction (TS_tot) (", temp, "K) :", TS_tot, "Eh"))
    logger.info("%s", "{} {} {} {} {}".format("Gibbs free energy correction (Gcorr) (", temp, "K) :", Gcorr, "Eh"))
    logger.info("")

    # Dict with properties
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


# From Hess-tool.py: Copied 13 May 2020
# Print dummy ORCA outputfile using coordinates and normal modes. Used for visualization of modes in Chemcraft
def print_dummy_orca_file(elems, coords, vfreq, nmodes, hessfile):
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
            # if mode < TRmodenum:
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
                # If last chunk and cleft is exactly 0 then all 5 columns should be done
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
                # Disabling zero-val setting below
                # if chunk == 0:
                # TODO: Here defning values to print based on values in nmodes matrix. TO be confiremd that this is
                # correct. TODO.
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


# Center of mass
def get_center(coords, masses=None, elems=None):
    if masses is None:
        if elems is None:
            raise InputError("Need to provide either masses or elems")
        logger.info("No masses provided. Using built-in atom masses.")
        masses = [openmmqmmm.coords.atommasses[openmmqmmm.coords.elematomnumbers[el.lower()] - 1] for el in elems]
    xcom = np.sum(masses * coords[:, 0]) / np.sum(masses)
    ycom = np.sum(masses * coords[:, 1]) / np.sum(masses)
    zcom = np.sum(masses * coords[:, 2]) / np.sum(masses)
    return xcom, ycom, zcom


def inertia(elems, coords, center):
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


def calc_rotational_constants(frag) -> list[float]:
    """Calculate rotational constants (GHz and cm**-1) for a fragment."""
    coords = frag.coords
    elems = frag.elems
    center = get_center(coords, elems=elems)
    rinertia = [float(i) for i in inertia(elems, coords, center)]

    # Converting from moments of inertia in amu A^2 to rotational constants in Ghz.
    # COnversion factor from http://openmopac.net/manual/thermochemistry.html
    rot_constants = []
    for inertval in rinertia:
        # Only calculating constant if moment of inertia value not zero
        if inertval != 0.0:
            rot_ghz = 5.053791e5 / (inertval * 1000)
            rot_constants.append(rot_ghz)

    rot_constants_cm = [i * openmmqmmm.constants.GHztocm for i in rot_constants]
    logger.info("Moments of inertia (amu A^2 ): %s", rinertia)
    logger.info("Rotational constants (GHz): %s", rot_constants)
    logger.info("Rotational constants (cm-1): %s", rot_constants_cm)
    logger.info("Note: If moment of inertia is zero then rotational constant is infinite and not printed ")

    return rot_constants_cm


def calc_model_hessian_orca(fragment, model="Almloef"):
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
    openmmqmmm.single_point(theory=orcadummycalc, fragment=fragment, charge=fragment.charge, mult=fragment.mult)
    # Read orca-input.opt containing Hessian under hessian_approx
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
                # Headerline
            if hesstake and lastchunk and len(line.split()) == hessdim - shiftpar + 1:
                for i in range(hessdim - shiftpar):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
            if hesstake and len(line.split()) == 7:
                # Hessianline
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


# Function to approximate large Hessian from smaller subsystem Hessian
# fragment is the large fragment
# atomindices refer to what atoms in the large fragment the small partial Hessian was generated for
# NOTE: Capping atom option is now disabled. Best made into a separate function
# Capping atom Hessian indices are skipped
# if capping_atoms != None:
# NOTE: Trans+rot projection off right now
def approximate_full_hessian_from_smaller(
    fragment,
    hessian_small,
    small_atomindices,
    large_atomindices=None,
    rest_hessian="zero",
    projection=False,
    charge=None,
    mult=None,
) -> np.ndarray:
    """Build an approximate full-system Hessian by combining a small computed Hessian with a model Hessian."""
    logger.info("approximate_full_Hessian_from_smaller")
    logger.info("")
    write_hessian(hessian_small, hessfile="smallhessian")

    # large_atomindices not provided
    if large_atomindices is None or len(large_atomindices) == 0:
        # Size of Hessian as big as fragment
        hess_size = fragment.numatoms * 3
        logger.info("Hessian dimension %s", hess_size)
        # If Hessian is for full fragment then we use the input atomindices directly
        correct_small_atomindices = small_atomindices
        usedfragment = fragment
    elif len(large_atomindices) > 0:
        logger.info("small_atomindices: %s", small_atomindices)
        logger.info("large_atomindices: %s", large_atomindices)
        hess_size = len(large_atomindices) * 3
        # Initializing full Hessian using hessatoms size
        fullhessian = np.zeros((hess_size, hess_size))

        # Check that atomindices (for small) are all part of hessatom
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
        # Create new fragment from large_atomindices
        subcoords, subelems = fragment.get_coords_for_atoms(large_atomindices)
        usedfragment = openmmqmmm.Fragment(elems=subelems, coords=subcoords, charge=fragment.charge, mult=fragment.mult)
    else:
        raise InputError(
            f"small_atomindices: {small_atomindices}\nlarge_atomindices: {large_atomindices}\nSomething went wrong"
        )

    logger.info("Initializing full size Hessian of dimension: %s", hess_size)
    fullhessian = np.zeros((hess_size, hess_size))
    logger.info("Initial fullhessian: %s", fullhessian)
    logger.info("Number of Hessian elements: %s", fullhessian.size)
    write_hessian(fullhessian, hessfile="initialfullhessian")

    # Making sure hessian_small is np array
    hessian_small = np.array(hessian_small)
    logger.info("hessian_small: %s", hessian_small)
    # Fill up hessian_large with model approximation from ORCA
    if rest_hessian in {"Almloef", "Lindh", "Schlegel", "Swart"}:
        logger.info("restHessian: %s", rest_hessian)
        if charge is None or mult is None:
            raise InputError(
                "Error: For this restHessian option we require charge and multiplicity information to be provided"
            )
        usedfragment.charge = charge
        usedfragment.mult = mult
        fullhessian = calc_model_hessian_orca(usedfragment, model=rest_hessian)
    elif rest_hessian == "xtb":
        raise InputError(
            "Error: restHessian='xtb' is not available in this ORCA+OpenMM build. Use an ORCA model Hessian, 'unit' or "
            "'zero' instead."
        )
    # Or with unit matrix
    elif rest_hessian in {"unit", "identity"}:
        logger.info("restHessian is unit/identity")
        fullhessian = np.identity(hess_size)
    # Keep matrix at zero
    elif rest_hessian is None or rest_hessian.lower() == "zero":
        logger.info("RestHessian is zero.")
    else:
        logger.info("RestHessian is zero.")
    logger.info("Intermediate fullhessian: %s", fullhessian)
    logger.info("Size: %s", fullhessian.size)
    write_hessian(fullhessian, hessfile="intermedfullhessian")
    # Large Hessian indices
    athessindices = [3 * i + j for i in correct_small_atomindices for j in [0, 1, 2]]
    # Looping over and assigning small Hessian values to large
    for s_i, i in enumerate(athessindices):
        for s_j, j in enumerate(athessindices):
            fullhessian[i, j] = hessian_small[s_i, s_j]
    logger.info("Final fullhessian: %s", fullhessian)
    write_hessian(fullhessian, hessfile="intermedfullhessian_after_small_update")
    # NOTE: Diagonalizing full Hessian just to see
    # Checking for linearity. Determines how many Trans+Rot modes
    tr_modenum = 5 if detect_linear(coords=fragment.coords, elems=fragment.elems) is True else 6

    logger.info("Now diagonalizing full Hessian")
    frequencies, _normal_modes, _evectors, _mode_order = diagonalize_hessian(
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


# Change isotopes of Hessian. Read-in hessian array or hessfile
# TODO: generalize. Input isotope-pair: 'H': 1.0, 'D' : '2.0' or something
# NOTE: Projection is off by default since coordinates are required for projection.
# NOTE: We could change this after testing

# What else?


#####################################
# NORMALMODE COMPOSITION ANALYSIS
#####################################


# Get normal mode composition factors for mode j and atom a
def normalmodecomp(evectors, j, a):
    # square elements of mode j
    esq_j = [i**2 for i in evectors[j]]
    # Squared elements of atom a in mode j
    esq_ja = []
    esq_ja.append(esq_j[a * 3 + 0])
    esq_ja.append(esq_j[a * 3 + 1])
    esq_ja.append(esq_j[a * 3 + 2])
    return sum(esq_ja)


# Get normal mode composition factors for all atoms for a specific mode only
def normalmodecomp_all(mode, fragment, evectors, hessatoms=None):
    numatoms = fragment.numatoms if hessatoms is None else len(hessatoms)
    normcomplist = []
    for n in range(numatoms):
        normcomp = normalmodecomp(evectors, mode, n)
        normcomplist.append(normcomp)

    # Returning normcomplist, a list of atomic contributions for each atom
    return normcomplist


def normalmodecomp_permode_by_elems(mode, fragment, evectors, hessatoms=None):
    normcomplist = normalmodecomp_all(mode, fragment, evectors, hessatoms=hessatoms)
    elementnormcomplist = []

    # Sum components together
    hesselems = [fragment.elems[i] for i in hessatoms] if hessatoms is not None else fragment.elems

    uniqelems = []
    for i in hesselems:
        if i not in uniqelems:
            uniqelems.append(i)
    # Dict to store results
    normmodecompelemsdict = {}
    for u in uniqelems:
        elcompsum = 0.0
        elindices = [i for i, j in enumerate(hesselems) if j == u]
        for h in elindices:
            elcompsum = float(elcompsum + float(normcomplist[h]))
        elementnormcomplist.append(elcompsum)
        normmodecompelemsdict[u] = elcompsum
    return normmodecompelemsdict


# Vibrational entropy by plain harmonic approximation
def s_vib(freqs, T):
    vibtemps = [
        (f * openmmqmmm.constants.c * openmmqmmm.constants.h_planck_hartreeseconds) / openmmqmmm.constants.R_gasconst
        for f in freqs
    ]
    # Vibrational entropy via RRHO.
    s_vib = 0.0
    for vibtemp in vibtemps:
        s_vib += openmmqmmm.constants.R_gasconst * (vibtemp / T) / (
            math.exp(vibtemp / T) - 1
        ) - openmmqmmm.constants.R_gasconst * math.log(1 - math.exp(-1 * vibtemp / T))
        TS_vib_final = s_vib * T
    return TS_vib_final


def s_vib_qrrho_truhlar(freqs, T, lowfreq_thresh=100):
    logger.warning("Quasi-RRHO by Truhlar approximation active.")
    logger.info(
        "This means that the vibrational entropy is calculated according to Truhlar-approach of raising low-energy "
        f"vibrations to {lowfreq_thresh} cm-1"
    )
    logger.info("Cite: R. F. Riberio et al. J. Phys. Chem. B, 115, 14556 (2011) ")
    # Vibrational entropy via quasi-RRHO
    TS_vib_final = 0.0
    # Looping over frequencies
    for f in freqs:
        freq_value = f
        if f < lowfreq_thresh:
            logger.warning(
                f"Warning: Frequency ({f}) is below low-freq threshold ({lowfreq_thresh}) cm-1. "
                f"Setting to {lowfreq_thresh} cm-1"
            )
            freq_value = lowfreq_thresh
        # Vib. temp and TS_vib for freq f
        vibtemp = (
            freq_value * openmmqmmm.constants.c * openmmqmmm.constants.h_planck_hartreeseconds
        ) / openmmqmmm.constants.R_gasconst
        logger.info("vibtemp: %s", vibtemp)
        TS_vib_f = T * (
            openmmqmmm.constants.R_gasconst * (vibtemp / T) / (math.exp(vibtemp / T) - 1)
            - openmmqmmm.constants.R_gasconst * math.log(1 - math.exp(-1 * vibtemp / T))
        )
        TS_vib_final += TS_vib_f
        logger.info("TS_vib_final: %s", TS_vib_final)

    return TS_vib_final


# Vibrational entropy by quasi-RRHO (Grimme)
def s_vib_qrrho_grimme(freqs, T, omega_0=100, i_av=None):
    logger.warning("Quasi-RRHO approximation by Grimme active.")
    logger.info("This means that the vibrational entropy uses the Grimme-type interpolation formula")
    logger.info("Cite: S. Grimme, Chem. Eur. J. 2012, 18, 9955-9964.")
    # Vibrational entropy via quasi-RRHO
    TS_vib_final = 0.0
    # Looping over frequencies
    for f in freqs:
        # Vib. temp and TS_vib for freq f
        vibtemp = (
            f * openmmqmmm.constants.c * openmmqmmm.constants.h_planck_hartreeseconds
        ) / openmmqmmm.constants.R_gasconst
        TS_vib_f = T * (
            openmmqmmm.constants.R_gasconst * (vibtemp / T) / (math.exp(vibtemp / T) - 1)
            - openmmqmmm.constants.R_gasconst * math.log(1 - math.exp(-1 * vibtemp / T))
        )
        # Rotational contribution with same freq f
        m_si = (
            openmmqmmm.constants.h_planck
            * openmmqmmm.constants.h_planck
            / (8 * math.pi * math.pi * f * openmmqmmm.constants.hc)
        )
        mp_si = m_si * i_av / (m_si + i_av)
        TS_rot_f_kcal = (
            T
            * openmmqmmm.constants.R_gasconst_kcalK
            * (
                0.5
                + math.log(
                    math.sqrt(
                        8
                        * math.pi
                        * math.pi
                        * math.pi
                        * mp_si
                        * openmmqmmm.constants.BOLTZMANN
                        * T
                        / (openmmqmmm.constants.h_planck * openmmqmmm.constants.h_planck)
                    )
                )
            )
        )
        TS_rot_f_au = TS_rot_f_kcal / openmmqmmm.constants.hartokcal  # Converting from kcal/mol to a.u.
        w = 1 / (1 + pow(omega_0 / f, 4))  # Weighting function
        # Regular RRHO: TS_vib_final+=TS_vib_f
        TS_vib_final += w * TS_vib_f + (1 - w) * TS_rot_f_au
    return TS_vib_final


def write_hessian(hessian, hessfile="Hessian") -> None:
    """Write a Hessian matrix to a text file."""
    np.savetxt(hessfile, hessian)
    logger.info(f"Wrote Hessian to file: {hessfile}")


# Read Hessian from file
def read_hessian(file) -> np.ndarray:
    """Read a Hessian matrix from a text file written by write_hessian."""
    logger.info(f"Reading Hessian from file: {file}")
    return np.loadtxt(file)


# Detect if geometry is linear, either via fragment or coords array
def detect_linear(fragment=None, coords=None, elems=None, threshold=1e-4):
    if fragment is None:
        numatoms = len(coords)
    else:
        coords = fragment.coords
        elems = fragment.elems
        numatoms = fragment.numatoms
    # Returning True if atom
    if numatoms == 1:
        return True
    # Returning True if diatomic
    if numatoms == 2:
        return True
    # Linear check via moments of inertia
    center = get_center(coords, elems=elems)
    rinertia = [float(i) for i in inertia(elems, coords, center)]
    # Checking if rinertia contains an almost zero-value
    if any(abs(i) < threshold for i in rinertia) is True:
        logger.info("Molecule is linear")
        return True
    logger.info("Molecule is non-linear")
    return False


# Simple function to get the relevant part (real or imaginary) part of a complex number
# If imaginary part is larger then we convert into negative number
# Used to report vibrational frequencies
def get_relevant_part_of_complex(numb):
    if numb.real > numb.imag:
        return numb.real
    return numb.imag * -1


def clean_frequencies(freqs):
    clean = []
    for f in freqs:
        bla = get_relevant_part_of_complex(f)
        clean.append(bla)
    return clean


def project_rot_and_trans(coords, mass, hessian, rotmode_threshold=1e-4):
    mass = np.array(mass)
    coords = np.array(coords) * openmmqmmm.constants.ang2bohr
    coords = coords.copy().reshape(-1, 3)
    na = coords.shape[0]
    wavenumber_scaling = (
        1e10
        * np.sqrt(openmmqmmm.constants.hartokj / openmmqmmm.constants.bohr2nm**2)
        / (2 * np.pi * openmmqmmm.constants.c * 0.01)
    )
    TotDOF = 3 * na

    # mass weighted Hessian matrix
    invsqrtm3 = 1.0 / np.sqrt(np.repeat(mass, 3))
    wHessian = hessian.copy() * np.outer(invsqrtm3, invsqrtm3)

    # Compute the center of mass
    cxyz = np.sum(coords * mass[:, np.newaxis], axis=0) / np.sum(mass)

    # Coordinates in the center-of-mass frame
    xcm = coords - cxyz[np.newaxis, :]

    # Moment of inertia tensor
    inertia_tensor = np.sum(
        [mass[i] * (np.eye(3) * (np.dot(xcm[i], xcm[i])) - np.outer(xcm[i], xcm[i])) for i in range(na)], axis=0
    )

    # Principal moments
    Ivals, Ivecs = np.linalg.eigh(inertia_tensor)
    # Eigenvectors are in the rows after transpose
    Ivecs = Ivecs.T

    # Obtain the number of rotational degrees of freedom
    RotDOF = 0
    for i in range(3):
        logger.info("Ivals[i]: %s", Ivals[i])
        if abs(Ivals[i]) > rotmode_threshold:
            RotDOF += 1
    TR_DOF = 3 + RotDOF
    logger.info("TR_DOF: %s", TR_DOF)
    if TR_DOF not in (5, 6):
        logger.info("Unexpected number of trans+rot DOF: {TR_DOF} not in (5, 6)")

    # Internal coordinates of the Eckart frame
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

    # Construct eigenvectors of unit length in the space of Cartesian displacements
    norm_vecs = proj_vecs[TR_DOF:] / np.sqrt(proj_vals[TR_DOF:, np.newaxis])

    # These are the orthonormal, TR-projected internal coordinates
    ic_basis = np.dot(norm_vecs, proj_basis)
    # Calculate the internal coordinate Hessian and diagonalize
    ic_hessian = np.linalg.multi_dot((ic_basis, wHessian, ic_basis.T))
    ichess_vals, ichess_vecs = np.linalg.eigh(ic_hessian)
    ichess_vecs = ichess_vecs.T
    normal_modes = np.dot(ichess_vecs, ic_basis)
    # mass unweighting
    normal_modes_cart = normal_modes * invsqrtm3[np.newaxis, :]

    # Convert to wavenumbers
    freqs_wavenumber = wavenumber_scaling * np.sqrt(np.abs(ichess_vals)) * np.sign(ichess_vals)

    return freqs_wavenumber, normal_modes, normal_modes_cart


# Calculate Raman activiities from masses, (mass-weighted) eigenvectors and polarizability derivative matrix
def calc_raman_activities(hessmasses, evectors, polarizability_derivs):
    logger.info("Calculating Raman activities")

    # Length of Hessian (and normal modes)
    hesslength = 3 * len(hessmasses)

    # Getting displacement vectors
    mass_matrix = np.repeat(hessmasses, 3)
    inv_sqrt_mass_matrix = np.diag(1 / (mass_matrix**0.5))
    displacements = inv_sqrt_mass_matrix.dot(np.transpose(evectors))

    # Finalizing polarizability derivative
    A_der = np.zeros((hesslength, 9))
    for i in range(hesslength):
        A_der[i, :] = polarizability_derivs[i].reshape(1, 9)

    # Transform polarizability derivatives to normal coordinates
    # A_der : 3*Natom x 9
    # Lx : 3*Natom x 3*Natom
    # A_der_q : 9 x 3*Natom
    A_der_q_tmp = np.dot(A_der.T, displacements)
    # Reorganize list of 3x3 polarizability derivatives for each Cartesian coordinate
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
    raman_unit = 1 / openmmqmmm.constants.bohr2ang**4
    raman_act = raman_act / raman_unit

    logger.info("Calculated Raman activities for each normal mode: %s", raman_act)
    logger.info("Calculated Raman depolarization ratios for each normal mode: %s", depol_ratio)
    return raman_act, depol_ratio


# Convert coordinates to center of mass using inputmasses
