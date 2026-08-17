import contextlib
import logging
import shutil
import time

import numpy as np

import openmmqmmm
import openmmqmmm.constants
from openmmqmmm.coords import check_charge_mult
from openmmqmmm.exceptions import (
    InputError,
)
from openmmqmmm.results import Results
from openmmqmmm.utils import log_time_since, main_header

logger = logging.getLogger(__name__)


def _cleanup_theory(theory):
    cleanup = getattr(theory, "cleanup", None)
    if callable(cleanup):
        cleanup()


def single_point(
    fragment=None,
    theory=None,
    grad: bool = False,
    charge: int | None = None,
    mult: int | None = None,
    result_write_to_disk: bool = True,
) -> "Results":
    """Run a single-point energy (and optionally gradient) calculation."""
    logger.info(main_header("Singlepoint function"))
    module_init_time = time.time()
    if fragment is None or theory is None:
        raise InputError("Singlepoint requires a fragment and a theory object")
    coords = fragment.coords
    elems = fragment.elems

    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint", theory=theory)

    if grad:
        logger.info("")
        logger.info(
            f"Doing single-point Energy+Gradient job on fragment. Formula: {fragment.prettyformula} Label: "
            f"{fragment.label} "
        )
        energy, gradient = theory.run(current_coords=coords, elems=elems, grad=True, charge=charge, mult=mult)
        logger.info("Energy:  %s", energy)
        log_time_since(module_init_time, "Singlepoint")
        result = Results(label="Singlepoint", energy=energy, gradient=gradient, charge=charge, mult=mult)
        if theory.theorytype == "QM/MM":
            result.qmmm_energy = theory.QM_MM_energy
            result.mm_energy = theory.MMenergy
            result.qm_energy = theory.QMenergy
        if result_write_to_disk:
            result.write_to_disk(filename="results_singlepoint.json")
        return result
    logger.info("")
    logger.info(
        f"Doing single-point Energy job on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
    )
    logger.info(f"Charge: {charge} Mult: {mult}")
    energy = theory.run(current_coords=coords, elems=elems, charge=charge, mult=mult)

    logger.info("Energy:  %s", energy)
    fragment.set_energy(energy)
    log_time_since(module_init_time, "Singlepoint")
    result = Results(label="Singlepoint", energy=energy, charge=charge, mult=mult)
    if theory.theorytype == "QM/MM":
        result.qmmm_energy = theory.QM_MM_energy
        result.mm_energy = theory.MMenergy
        result.qm_energy = theory.QMenergy
    if result_write_to_disk:
        result.write_to_disk(filename="results_singlepoint.json")
    return result


def single_point_theories(theories=None, fragment=None, charge=None, mult=None) -> "Results":
    """Run single-point calculations of one fragment with multiple theories."""
    logger.info(main_header("Singlepoint_theories function"))
    module_init_time = time.time()
    logger.info("Will run single-point calculation on the fragment with multiple theories")

    energies = []

    for theory in theories:
        # Resolved per theory from the original arguments: rebinding charge here would carry one
        # theory's resolved value (e.g. a QM/MM region charge) into the next theory in the list.
        theory_charge, theory_mult = check_charge_mult(
            charge, mult, theory.theorytype, fragment, "Singlepoint_theories", theory=theory
        )

        result = single_point(theory=theory, fragment=fragment, charge=theory_charge, mult=theory_mult)

        calc_label = "Frag_" + theory.__class__.__name__ + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")

        logger.info(f"Theory Label: {theory.label} Energy: {result.energy} Eh")
        _cleanup_theory(theory)
        energies.append(result.energy)

    print_theories_table(theories, energies, fragment, charge=charge, mult=mult)
    result = Results(label="Singlepoint_theories", energies=energies, charge=charge, mult=mult)
    result.write_to_disk(filename="results_singlepoint_theories.json")
    log_time_since(module_init_time, "Singlepoint_theories")
    return result


def print_theories_table(theories, energies, fragment, charge=None, mult=None):
    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("Singlepoint_theories: Table of energies of each theory:")
    logger.info("%s", "=" * 70)

    # Charge/mult may have been passed to the job rather than stored on the fragment, and an
    # MM theory resolves both to None. Format via str so the table never raises on None.
    charge = fragment.charge if charge is None else charge
    mult = fragment.mult if mult is None else mult

    logger.info(
        "%s", "\n{:15} {:15} {:>7} {:>7} {:>20}".format("Theory class", "Theory Label", "Charge", "Mult", "Energy(Eh)")
    )
    logger.info("%s", "-" * 70)
    for t, e in zip(theories, energies, strict=False):
        logger.info(f"{t.__class__.__name__:15} {t.label!s:15} {charge!s:>7} {mult!s:>7} {e:>20.10f}")
    logger.info("")


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


# Assuming fragments have charge,mult info defined.
# If stoichiometry provided then print reaction energy
def single_point_fragments(
    theory=None, fragments=None, stoichiometry=None, relative_energies=False, unit="kcal/mol", moreadfiles=None
) -> "Results":
    """Run single-point calculations of one theory over multiple fragments."""
    logger.info(main_header("Singlepoint_fragments function"))
    module_init_time = time.time()
    logger.info("Will run single-point calculation on each fragment")
    logger.info("Theory: %s", theory.__class__.__name__)

    energies = []

    for i, frag in enumerate(fragments):
        if frag.charge is None or frag.mult is None:
            raise InputError(
                "Error: Singlepoint_fragments requires charge/mult information to be associated with each fragment."
            )
        charge = frag.charge
        mult = frag.mult

        # Setting orbital file for ORCATheory or any other theory using moreadfile
        with contextlib.suppress(IndexError, TypeError):
            theory.moreadfile = moreadfiles[i]

        result = single_point(theory=theory, fragment=frag, charge=charge, mult=mult)

        logger.info(f"Fragment {frag.formula} . Label: {frag.label} Energy: {result.energy} Eh")

        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")

        _cleanup_theory(theory)
        energies.append(result.energy)
        frag.set_energy(result.energy)
        logger.info("")

    result = Results(label="Singlepoint_fragments", energies=energies, charge=charge, mult=mult)

    print_fragments_table(fragments, energies)

    if relative_energies is True:
        logger.info("")
        logger.info("relative_energies option is True!")
        convfactor = openmmqmmm.constants.ENERGY_UNIT_FROM_HARTREE[unit]
        relenergies = [(i - min(energies)) * convfactor for i in energies]
        print_fragments_table(fragments, relenergies, unit=unit)
        result.relative_energies = relenergies
        result.labels = [f.label for f in fragments]

    if stoichiometry is not None:
        logger.info("Stoichiometry provided: %s", stoichiometry)
        r = reaction_energy(
            list_of_energies=energies, stoichiometry=stoichiometry, list_of_fragments=fragments, unit=unit, label="ΔE"
        )
        result.reaction_energy = r[0]
    result.write_to_disk(filename="results_singlepoint_fragments.json")
    log_time_since(module_init_time, "Singlepoint_fragments")
    return result


# Assuming fragments have charge,mult info defined.
def single_point_fragments_and_theories(theories=None, fragments=None, stoichiometry=None) -> "Results":
    """Run single-point calculations for every fragment with every theory."""
    logger.info(main_header("Singlepoint_fragments_and_theories"))
    module_init_time = time.time()
    all_energies = []

    for theory in theories:
        result = single_point_fragments(theory=theory, fragments=fragments, stoichiometry=stoichiometry)
        all_energies.append(result.energies)

    logger.info("\n")
    logger.info("SINGLEPOINT_FRAGMENTS_AND_THEORIES ALL DONE")
    logger.info("\n")
    logger.info("%s", "=" * 60)
    logger.info("Singlepoint_fragments_and_theories: FINAL RESULTS")
    logger.info("%s", "=" * 60)
    for t, elist in zip(theories, all_energies, strict=False):
        logger.info("\nTheory: %s", t.__class__.__name__)
        logger.info("Label: %s", t.label)
        print_fragments_table(fragments, elist, tabletitle="")
        if stoichiometry is not None:
            logger.info("Stoichiometry provided: %s", stoichiometry)
            reaction_energy(
                list_of_energies=elist,
                stoichiometry=stoichiometry,
                list_of_fragments=fragments,
                unit="kcal/mol",
                label=f"{t.label}",
            )

            logger.info("%s", "_" * 60)
    logger.info("\nFinal list of lists of total energies: %s", all_energies)

    result = Results(label="Singlepoint_fragments_and_theories", energies=all_energies)
    if stoichiometry is not None:
        logger.info("Final reaction energies:")
        for elist, t in zip(all_energies, theories, strict=False):
            r = reaction_energy(
                list_of_energies=elist,
                stoichiometry=stoichiometry,
                list_of_fragments=fragments,
                unit="kcal/mol",
                label=f"{t.label}",
            )
            result.reaction_energies.append(r[0])
    logger.info("")
    result.write_to_disk(filename="results_singlepoint_fragments_theories.json")
    log_time_since(module_init_time, "Singlepoint_fragments_and_theories")
    return result


# Assuming fragments have charge,mult info defined.
def single_point_reaction(theory=None, reaction=None, moreadfiles=None) -> "Results":
    """Run single-point calculations for all species of a Reaction and compute the reaction energy."""
    logger.info(main_header("Singlepoint_reaction function"))
    module_init_time = time.time()

    logger.info("Will run single-point calculation on each fragment defined in reaction")
    logger.info("Theory: %s", theory.__class__.__name__)
    logger.info("Resetting energies in reaction object")
    reaction.energies = []
    reaction.reset_energies()

    for i, frag in enumerate(reaction.fragments):
        try:
            theory.moreadfile = reaction.orbital_dictionary[moreadfiles][i]
            logger.info("Found orbital dictionary in reaction object")
            logger.info("Using orbital file: %s", theory.moreadfile)
        except (AttributeError, KeyError, IndexError, TypeError):
            with contextlib.suppress(IndexError, TypeError):
                theory.moreadfile = moreadfiles[i]
        result = single_point(theory=theory, fragment=frag, charge=frag.charge, mult=frag.mult)
        energy = result.energy
        logger.info(f"Fragment {frag.formula} . Label: {frag.label} Energy: {energy} Eh")
        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")
        _cleanup_theory(theory)
        reaction.energies.append(energy)

        if isinstance(theory, openmmqmmm.ORCATheory):
            logger.debug("Theory properties: %s", theory.properties)
            try:
                reaction.properties["E_var"].append(theory.properties["E_var"])
                reaction.properties["E_PT2_rest"].append(theory.properties["E_PT2_rest"])
                reaction.properties["num_genCFGs"].append(theory.properties["num_genCFGs"])
                reaction.properties["num_selected_CFGs"].append(theory.properties["num_selected_CFGs"])
                reaction.properties["num_after_SD_CFGs"].append(theory.properties["num_after_SD_CFGs"])
            except KeyError:
                pass
        frag.set_energy(energy)
        logger.info("")

    print_fragments_table(reaction.fragments, reaction.energies, tabletitle="Singlepoint_reaction: ")

    reaction.calculate_reaction_energy()

    result = Results(label="Singlepoint_reaction", energies=reaction.energies, reaction_energy=reaction.reaction_energy)

    log_time_since(module_init_time, "Singlepoint_reaction")
    result.write_to_disk(filename="results_singlepoint_reaction.json")
    return result


# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    """Dummy theory returning zero energy and a zero gradient (useful for testing workflows)."""

    def __init__(self, fragment=None, numcores: int = 1, label: str | None = None):
        self.numcores = numcores
        self.label = label
        self.fragment = fragment
        self.filename = "zerotheory"
        self.theorynamelabel = "ZeroTheory"
        self.theorytype = "QM"

    def cleanup(self):
        """No files to clean up; present so ZeroTheory satisfies the theory contract."""

    def run(
        self,
        *,
        current_coords=None,
        elems=None,
        grad=False,
        pc=False,
        numcores=None,
        charge=None,
        mult=None,
        label=None,
        current_mm_coords=None,
        mm_charges=None,
        qm_elems=None,
    ):
        """Return zero energy and, if requested, a zero gradient."""
        self.energy = 0.0
        self.gradient = np.zeros((len(elems), 3))
        if not grad:
            return self.energy
        return self.energy, self.gradient


def reaction_energy(
    list_of_energies=None,
    stoichiometry=None,
    list_of_fragments=None,
    unit="kcal/mol",
    label=None,
    reference=None,
    silent=False,
    correction=0.0,
) -> tuple[float, float | None]:
    """Calculate a reaction energy from energies (or fragments with energies) and stoichiometry."""
    if label is None:
        label = ""
    convfactor = openmmqmmm.constants.ENERGY_UNIT_FROM_HARTREE[unit]
    reactant_energy = 0.0  # hartree
    product_energy = 0.0  # hartree
    if stoichiometry is None:
        raise InputError("stoichiometry list is required")

    if correction != 0.0:
        logger.info("User-correction was added. ")
        logger.info(f"Correction to reaction energy in {correction} Eh ")
        correction_in_unit = correction * convfactor
        logger.info(f"correction_in_unit in {correction_in_unit} {unit}")
    else:
        correction_in_unit = 0.0

    if list_of_energies is not None:
        if len(list_of_energies) != len(stoichiometry):
            raise InputError("Number of energies not equal to number of stoichiometry values\nExiting.")

        for i, stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy = reactant_energy + list_of_energies[i] * abs(stoich)
            if stoich > 0:
                product_energy = product_energy + list_of_energies[i] * abs(stoich)
        reaction_energy = (product_energy - reactant_energy) * convfactor + correction_in_unit
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
        reaction_energy = (product_energy - reactant_energy) * convfactor + correction_in_unit
        if reference is None:
            error = None
            if silent is False:
                logger.info(f"Reaction_energy({label}):  {reaction_energy} {unit}")
        else:
            error = reaction_energy - reference
            if silent is False:
                logger.info(f"Reaction_energy({label}):  {reaction_energy} {unit} (Error: {error})")
    return reaction_energy, error
