"""
Singlepoint module:

Function Singlepoint

class ZeroTheory
"""

import contextlib
import logging
import shutil
import time

import numpy as np

import openmmqmmm
from openmmqmmm.exceptions import (
    InputError,
)
from openmmqmmm.utils import log_time_since, main_header
from openmmqmmm.coords import check_charge_mult
from openmmqmmm.results import ASH_Results

logger = logging.getLogger(__name__)


# Single-point energy function
def Singlepoint(fragment=None, theory=None, Grad=False, charge=None, mult=None, result_write_to_disk=True):
    """Singlepoint function: runs a single-point energy calculation using ASH theory and ASH fragment.

    Args:
        fragment (ASH fragment, optional): An ASH fragment. Defaults to None.
        theory (ASH theory, optional): Any valid ASH theory. Defaults to None.
        Grad (bool, optional): Do gradient or not Defaults to False.
        charge (int, optional): Specify charge of system. Overrides fragment charge information.
        mult (int, optional): Specify mult of system. Overrides fragment charge information.

    Returns:
        float: Energy
        or
        float,np.array : Energy and gradient array
    """
    logger.info(main_header("Singlepoint function"))
    module_init_time = time.time()
    if fragment is None or theory is None:
        raise InputError("Singlepoint requires a fragment and a theory object")
    coords = fragment.coords
    elems = fragment.elems

    # Check charge/mult
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint", theory=theory)

    # Run a single-point energy job with gradient
    if Grad:
        logger.info("")
        logger.warning(
            f"Doing single-point Energy+Gradient job on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
        )
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy, gradient = theory.run(current_coords=coords, elems=elems, Grad=True, charge=charge, mult=mult)
        logger.info("Energy:  %s", energy)
        log_time_since(module_init_time, "Singlepoint")
        result = ASH_Results(label="Singlepoint", energy=energy, gradient=gradient, charge=charge, mult=mult)
        if theory.theorytype == "QM/MM":
            result.qmmm_energy = theory.QM_MM_energy
            result.mm_energy = theory.MMenergy
            result.qm_energy = theory.QMenergy
        if result_write_to_disk:
            result.write_to_disk(filename="ASH_SP.result")
        return result
    # Run a single-point energy job without gradient (default)
    else:
        logger.info("")
        logger.info(
            f"Doing single-point Energy job on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
        )
        logger.info(f"Charge: {charge} Mult: {mult}")
        # Run
        energy = theory.run(current_coords=coords, elems=elems, charge=charge, mult=mult)

        logger.info("Energy:  %s", energy)
        # Now adding total energy to fragment
        fragment.set_energy(energy)
        log_time_since(module_init_time, "Singlepoint")
        result = ASH_Results(label="Singlepoint", energy=energy, charge=charge, mult=mult)
        if theory.theorytype == "QM/MM":
            result.qmmm_energy = theory.QM_MM_energy
            result.mm_energy = theory.MMenergy
            result.qm_energy = theory.QMenergy
        if result_write_to_disk:
            result.write_to_disk(filename="ASH_SP.result")
        return result


# Single-point energy function that runs calculations on 1 fragment using multiple theories. Returns a list of energies.
# TODO: allow Grad option?
def Singlepoint_theories(theories=None, fragment=None, charge=None, mult=None):
    logger.info(main_header("Singlepoint_theories function"))
    module_init_time = time.time()
    logger.info("Will run single-point calculation on the fragment with multiple theories")

    energies = []

    # Looping through fragmengs
    for theory in theories:
        # Check charge/mult
        charge, mult = check_charge_mult(
            charge, mult, theory.theorytype, fragment, "Singlepoint_theories", theory=theory
        )

        # Running single-point.
        result = Singlepoint(theory=theory, fragment=fragment, charge=charge, mult=mult)

        # Preserve outputfile
        calc_label = "Frag_" + theory.__class__.__name__ + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")

        logger.info(f"Theory Label: {theory.label} Energy: {result.energy} Eh")
        theory.cleanup()
        energies.append(result.energy)

    # Printing final table
    print_theories_table(theories, energies, fragment)
    result = ASH_Results(label="Singlepoint_theories", energies=energies, charge=charge, mult=mult)
    result.write_to_disk(filename="ASH_SP_theories.result")
    log_time_since(module_init_time, "Singlepoint_theories")
    return result


# Pretty table of fragments and theories
def print_theories_table(theories, energies, fragment):
    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("Singlepoint_theories: Table of energies of each theory:")
    logger.info("%s", "=" * 70)

    logger.info(
        "%s", "\n{:15} {:15} {:>7} {:>7} {:>20}".format("Theory class", "Theory Label", "Charge", "Mult", "Energy(Eh)")
    )
    logger.info("%s", "-" * 70)
    for t, e in zip(theories, energies, strict=False):
        logger.info(f"{t.__class__.__name__:15} {t.label!s:15} {fragment.charge:>7} {fragment.mult:>7} {e:>20.10f}")
    logger.info("")


# Pretty table of fragments and energies
def print_fragments_table(fragments, energies, tabletitle="Singlepoint_fragments: ", unit="Eh"):
    logger.info("")
    logger.info("%s", "=" * 100)
    logger.info(f"{tabletitle}Table of energies of each fragment:")
    logger.info("%s", "=" * 100)
    logger.info("%s", "{:15} {:<25} {:>7} {:>7} {:>30}".format("Formula", "Label", "Charge", "Mult", f"Energy({unit})"))
    logger.info("%s", "-" * 100)
    for frag, e in zip(fragments, energies, strict=False):
        label = "None" if frag.label is None else frag.label
        logger.info(f"{frag.formula:15} {label:<25} {frag.charge:>7} {frag.mult:>7} {e:>30.10f}")
    logger.info("")


# Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
# Assuming fragments have charge,mult info defined.
# If stoichiometry provided then print reaction energy
def Singlepoint_fragments(
    theory=None, fragments=None, stoichiometry=None, relative_energies=False, unit="kcal/mol", moreadfiles=None
):
    logger.info(main_header("Singlepoint_fragments function"))
    module_init_time = time.time()
    logger.info("Will run single-point calculation on each fragment")
    logger.info("Theory: %s", theory.__class__.__name__)

    energies = []

    # Looping through fragments
    for i, frag in enumerate(fragments):
        if frag.charge is None or frag.mult is None:
            raise InputError(
                "Error: Singlepoint_fragments requires charge/mult information to be associated with each fragment."
            )
        # Setting charge/mult  from fragment
        charge = frag.charge
        mult = frag.mult

        # Setting orbital file for ORCATheory or any other theory using moreadfile
        with contextlib.suppress(IndexError, TypeError):
            theory.moreadfile = moreadfiles[i]

        # Running single-point
        result = Singlepoint(theory=theory, fragment=frag, charge=charge, mult=mult)

        logger.info(f"Fragment {frag.formula} . Label: {frag.label} Energy: {result.energy} Eh")

        # Preserve outputfile
        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")

        theory.cleanup()
        energies.append(result.energy)
        # Adding energy as the fragment attribute
        frag.set_energy(result.energy)
        logger.info("")

    # Create Results object
    result = ASH_Results(label="Singlepoint_fragments", energies=energies, charge=charge, mult=mult)

    # Print table
    print_fragments_table(fragments, energies)

    # Print table
    if relative_energies is True:
        logger.info("")
        logger.info("relative_energies option is True!")
        conversionfactor = {
            "kcal/mol": 627.50946900,
            "kcalpermol": 627.50946900,
            "kJ/mol": 2625.499638,
            "kJpermol": 2625.499638,
            "eV": 27.211386245988,
            "cm-1": 219474.6313702,
            "Eh": 1.0,
            "mEh": 1000,
            "meV": 27211.386245988,
        }
        convfactor = conversionfactor[unit]
        relenergies = [(i - min(energies)) * convfactor for i in energies]
        print_fragments_table(fragments, relenergies, unit=unit)
        result.relative_energies = relenergies
        result.labels = [f.label for f in fragments]

    # Printing reaction energy if stoichiometry was provided
    if stoichiometry is not None:
        logger.info("Stoichiometry provided: %s", stoichiometry)
        r = ReactionEnergy(
            list_of_energies=energies, stoichiometry=stoichiometry, list_of_fragments=fragments, unit=unit, label="ΔE"
        )
        result.reaction_energy = r[0]
    result.write_to_disk(filename="ASH_SP_fragments.result")
    log_time_since(module_init_time, "Singlepoint_fragments")
    return result


# Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
# Assuming fragments have charge,mult info defined.
def Singlepoint_fragments_and_theories(theories=None, fragments=None, stoichiometry=None):
    logger.info(main_header("Singlepoint_fragments_and_theories"))
    module_init_time = time.time()
    # List of lists
    all_energies = []

    # Looping over theories and getting energies for list of fragments
    for theory in theories:
        result = Singlepoint_fragments(theory=theory, fragments=fragments, stoichiometry=stoichiometry)
        all_energies.append(result.energies)

    logger.info("\n")
    logger.info("SINGLEPOINT_FRAGMENTS_AND_THEORIES ALL DONE")
    logger.info("\n")
    logger.info("%s", "=" * 60)
    logger.info("Singlepoint_fragments_and_theories: FINAL RESULTS")
    logger.info("%s", "=" * 60)
    # Table
    for t, elist in zip(theories, all_energies, strict=False):
        logger.info("\nTheory: %s", t.__class__.__name__)
        logger.info("Label: %s", t.label)
        print_fragments_table(fragments, elist, tabletitle="")
        # Reaction energy if stoichiometry provided
        if stoichiometry is not None:
            logger.info("Stoichiometry provided: %s", stoichiometry)
            ReactionEnergy(
                list_of_energies=elist,
                stoichiometry=stoichiometry,
                list_of_fragments=fragments,
                unit="kcal/mol",
                label=f"{t.label}",
            )

            logger.info("%s", "_" * 60)
    logger.info("\nFinal list of lists of total energies: %s", all_energies)

    result = ASH_Results(label="Singlepoint_fragments_and_theories", energies=all_energies)
    if stoichiometry is not None:
        logger.info("Final reaction energies:")
        for elist, t in zip(all_energies, theories, strict=False):
            r = ReactionEnergy(
                list_of_energies=elist,
                stoichiometry=stoichiometry,
                list_of_fragments=fragments,
                unit="kcal/mol",
                label=f"{t.label}",
            )
            result.reaction_energies.append(r[0])
    logger.info("")
    result.write_to_disk(filename="ASH_SP_fragments_theories.result")
    log_time_since(module_init_time, "Singlepoint_fragments_and_theories")
    return result


# Single-point energy function that runs calculations on an ASH reaction object
# Assuming fragments have charge,mult info defined.
def Singlepoint_reaction(theory=None, reaction=None, moreadfiles=None):
    logger.info(main_header("Singlepoint_reaction function"))
    module_init_time = time.time()

    logger.info("Will run single-point calculation on each fragment defined in reaction")
    logger.info("Theory: %s", theory.__class__.__name__)
    logger.info("Resetting energies in reaction object")
    reaction.energies = []
    reaction.reset_energies()

    # Looping through fragments defined in Reaction object
    for i, frag in enumerate(reaction.fragments):
        # Orbital file for ORCATheory or any other theory using moreadfile
        try:
            theory.moreadfile = reaction.orbital_dictionary[moreadfiles][i]
            logger.info("Found orbital dictionary in reaction object")
            logger.info("Using orbital file: %s", theory.moreadfile)
        except (AttributeError, KeyError, IndexError, TypeError):
            with contextlib.suppress(IndexError, TypeError):
                theory.moreadfile = moreadfiles[i]
        # Running single-point
        result = Singlepoint(theory=theory, fragment=frag, charge=frag.charge, mult=frag.mult)
        energy = result.energy
        logger.info(f"Fragment {frag.formula} . Label: {frag.label} Energy: {energy} Eh")
        # Preserve outputfile
        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")
        theory.cleanup()
        reaction.energies.append(energy)

        # TODO: Change this so that instead we just grab whatever each Theory level deemed important
        # theory.properties feature?
        # Check if ORCATheory object contains ICE-CI info
        if isinstance(theory, openmmqmmm.ORCATheory):
            logger.info("theory.properties: %s", theory.properties)
            # Add selected properties to Reaction object
            try:
                reaction.properties["E_var"].append(theory.properties["E_var"])
                reaction.properties["E_PT2_rest"].append(theory.properties["E_PT2_rest"])
                reaction.properties["num_genCFGs"].append(theory.properties["num_genCFGs"])
                reaction.properties["num_selected_CFGs"].append(theory.properties["num_selected_CFGs"])
                reaction.properties["num_after_SD_CFGs"].append(theory.properties["num_after_SD_CFGs"])
            except KeyError:
                pass
        # Adding energy as the fragment attribute
        frag.set_energy(energy)
        logger.info("")

    # Print table
    print_fragments_table(reaction.fragments, reaction.energies, tabletitle="Singlepoint_reaction: ")

    # Setting unit of reaction if given (will override reaction.unit definition)
    # NOTE: Needed?
    # NOTE: Now just setting unit equal to reaction.unit. Used for components below

    reaction.calculate_reaction_energy()

    result = ASH_Results(
        label="Singlepoint_reaction", energies=reaction.energies, reaction_energy=reaction.reaction_energy
    )

    log_time_since(module_init_time, "Singlepoint_reaction")
    result.write_to_disk(filename="ASH_SP_reaction.result")
    return result


# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, numcores=1, label=None):
        """Class Zerotheory: Simple dummy theory that gives zero energy and a zero-valued gradient array
            Note: includes unnecessary attributes for consistency.

        Args:
            fragment (ASH fragment, optional): A valid ASH fragment. Defaults to None.
            numcores (int, optional): Number of cores. Defaults to 1.
            label (str, optional): String label. Defaults to None.
        """
        self.numcores = numcores
        self.label = label
        self.fragment = fragment
        self.filename = "zerotheory"
        self.theorynamelabel = "ZeroTheory"
        # Indicate that this is a QMtheory
        self.theorytype = "QM"

    def run(
        self,
        current_coords=None,
        elems=None,
        Grad=False,
        PC=False,
        numcores=None,
        charge=None,
        mult=None,
        label=None,
        current_MM_coords=None,
        MMcharges=None,
        qm_elems=None,
    ):
        self.energy = 0.0
        # Gradient as np array
        self.gradient = np.zeros((len(elems), 3))
        if not Grad:
            return self.energy
        else:
            return self.energy, self.gradient


# Simple way to create interfaces to programs


def ReactionEnergy(
    list_of_energies=None,
    stoichiometry=None,
    list_of_fragments=None,
    unit="kcal/mol",
    label=None,
    reference=None,
    silent=False,
    correction=0.0,
):
    """Calculate reaction energy from list of energies (or energies from list of fragments) and stoichiometry

    Args:
        list_of_energies ([type], optional): A list of total energies in hartrees. Defaults to None.
        stoichiometry (list, optional): A list of integers, e.g. [-1,-1,1,1]. Defaults to None.
        list_of_fragments (list, optional): A list of ASH fragments . Defaults to None.
        unit (str, optional): Unit for relative energy. Defaults to 'kcal/mol'.
        label (string, optional): Optional label for energy. Defaults to None.
        reference (float, optional): Optional shift-parameter of energy Defaults to None.

    Returns:
        tuple : energy and error in chosen unit
    """
    conversionfactor = {
        "kcal/mol": 627.50946900,
        "kcalpermol": 627.50946900,
        "kJ/mol": 2625.499638,
        "kJpermol": 2625.499638,
        "eV": 27.211386245988,
        "cm-1": 219474.6313702,
        "Eh": 1.0,
        "mEh": 1000,
        "meV": 27211.386245988,
    }
    if label is None:
        label = ""
    reactant_energy = 0.0  # hartree
    product_energy = 0.0  # hartree
    if stoichiometry is None:
        raise InputError("stoichiometry list is required")

    if correction != 0.0:
        logger.info("User-correction was added. ")
        logger.info(f"Correction to reaction energy in {correction} Eh ")
        correction_in_unit = correction * conversionfactor[unit]
        logger.info(f"correction_in_unit in {correction_in_unit} {unit}")
    else:
        correction_in_unit = 0.0

    # List of energies option
    if list_of_energies is not None:
        if len(list_of_energies) != len(stoichiometry):
            raise InputError("Number of energies not equal to number of stoichiometry values\nExiting.")

        for i, stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy = reactant_energy + list_of_energies[i] * abs(stoich)
            if stoich > 0:
                product_energy = product_energy + list_of_energies[i] * abs(stoich)
        reaction_energy = (product_energy - reactant_energy) * conversionfactor[unit] + correction_in_unit
        if reference is None:
            error = None
            if silent is False:
                logger.info(f"Reaction_energy({label}):  {reaction_energy} {unit}")
        else:
            error = reaction_energy - reference
            if silent is False:
                logger.info(f"Reaction_energy({label}):  {reaction_energy} {unit} (Error: {error})")
    else:
        logger.info("\nNo list of total energies provided. Using internal energy of each fragment instead.")
        logger.info("")
        for i, stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy = reactant_energy + list_of_fragments[i].energy * abs(stoich)
            if stoich > 0:
                product_energy = product_energy + list_of_fragments[i].energy * abs(stoich)
        reaction_energy = (product_energy - reactant_energy) * conversionfactor[unit] + correction_in_unit
        if reference is None:
            error = None
            if silent is False:
                logger.info(f"Reaction_energy({label}):  {reaction_energy} {unit}")
        else:
            error = reaction_energy - reference
            if silent is False:
                logger.info(f"Reaction_energy({label}):  {reaction_energy} {unit} (Error: {error})")
    return reaction_energy, error
