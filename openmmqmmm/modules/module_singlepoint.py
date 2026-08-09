"""
    Singlepoint module:

    Function Singlepoint

    class ZeroTheory
    """
import numpy as np
import shutil
import subprocess as sp
import time

import openmmqmmm
from openmmqmmm.functions.functions_general import ashexit, BC, print_time_rel, print_line_with_mainheader
from openmmqmmm.modules.module_coords import check_charge_mult
from openmmqmmm.modules.module_results import ASH_Results


# Single-point energy function


# Single-point energy function
def Singlepoint(fragment=None, theory=None, Grad=False, charge=None, mult=None, printlevel=2,
                result_write_to_disk=True):
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
    if printlevel >= 1:
        print_line_with_mainheader("Singlepoint function")
    module_init_time = time.time()
    if fragment is None or theory is None:
        print(BC.FAIL, "Singlepoint requires a fragment and a theory object", BC.END)
        ashexit()
    coords = fragment.coords
    elems = fragment.elems

    # Check charge/mult
    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint", theory=theory,
                                     printlevel=printlevel)

    # Run a single-point energy job with gradient
    if Grad:
        if printlevel >= 1:
            print()
            print(BC.WARNING, "Doing single-point Energy+Gradient job on fragment. Formula: {} Label: {} ".format(
                fragment.prettyformula, fragment.label), BC.END)
        # An Energy+Gradient calculation where we change the number of cores to 12
        energy, gradient = theory.run(current_coords=coords, elems=elems, Grad=True, charge=charge, mult=mult)
        if printlevel >= 1:
            print("Energy: ", energy)
            print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
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
        if printlevel >= 1:
            print()
            print("Doing single-point Energy job on fragment. Formula: {} Label: {} ".format(fragment.prettyformula,
                                                                                             fragment.label))
            print(f"Charge: {charge} Mult: {mult}")  # Charge/mult should have been defined so we print
        # Run
        energy = theory.run(current_coords=coords, elems=elems, charge=charge, mult=mult)

        if printlevel >= 1:
            print("Energy: ", energy)
        # Now adding total energy to fragment
        fragment.set_energy(energy)
        if printlevel >= 1:
            print_time_rel(module_init_time, modulename='Singlepoint', moduleindex=1)
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
def Singlepoint_theories(theories=None, fragment=None, charge=None, mult=None, printlevel=2):
    print_line_with_mainheader("Singlepoint_theories function")
    module_init_time = time.time()
    print("Will run single-point calculation on the fragment with multiple theories")

    energies = []

    # Looping through fragmengs
    for theory in theories:
        # Check charge/mult
        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint_theories",
                                         theory=theory,
                                         printlevel=printlevel)

        # Running single-point.
        result = Singlepoint(theory=theory, fragment=fragment, charge=charge, mult=mult)

        # Preserve outputfile
        calc_label = "Frag_" + theory.__class__.__name__ + "_"
        try:
            shutil.copyfile(theory.filename + '.out', f'./{calc_label}.out')
        except:
            pass

        print("Theory Label: {} Energy: {} Eh".format(theory.label, result.energy))
        theory.cleanup()
        energies.append(result.energy)

    # Printing final table
    print_theories_table(theories, energies, fragment)
    result = ASH_Results(label="Singlepoint_theories", energies=energies, charge=charge, mult=mult)
    result.write_to_disk(filename="ASH_SP_theories.result")
    print_time_rel(module_init_time, modulename='Singlepoint_theories', moduleindex=1)
    return result


# Pretty table of fragments and theories
def print_theories_table(theories, energies, fragment):
    print()
    print("=" * 70)
    print("Singlepoint_theories: Table of energies of each theory:")
    print("=" * 70)

    print("\n{:15} {:15} {:>7} {:>7} {:>20}".format("Theory class", "Theory Label", "Charge", "Mult", "Energy(Eh)"))
    print("-" * 70)
    for t, e in zip(theories, energies):
        print("{:15} {:15} {:>7} {:>7} {:>20.10f}".format(t.__class__.__name__, str(t.label), fragment.charge,
                                                          fragment.mult, e))
    print()


# Pretty table of fragments and energies
def print_fragments_table(fragments, energies, tabletitle="Singlepoint_fragments: ", unit='Eh'):
    print()
    print("=" * 100)
    print("{}Table of energies of each fragment:".format(tabletitle))
    print("=" * 100)
    print("{:15} {:<25} {:>7} {:>7} {:>30}".format("Formula", "Label", "Charge", "Mult", f"Energy({unit})"))
    print("-" * 100)
    for frag, e in zip(fragments, energies):
        if frag.label == None:
            label = "None"
        else:
            label = frag.label
        print("{:15} {:<25} {:>7} {:>7} {:>30.10f}".format(frag.formula, label, frag.charge, frag.mult, e))
    print()


# Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
# Assuming fragments have charge,mult info defined.
# If stoichiometry provided then print reaction energy
def Singlepoint_fragments(theory=None, fragments=None, stoichiometry=None, relative_energies=False, unit='kcal/mol',
                          moreadfiles=None):
    print_line_with_mainheader("Singlepoint_fragments function")
    module_init_time = time.time()
    print("Will run single-point calculation on each fragment")
    print("Theory:", theory.__class__.__name__)

    energies = [];
    filenames = []

    # Looping through fragments
    for i, frag in enumerate(fragments):

        if frag.charge == None or frag.mult == None:
            print(BC.FAIL,
                  "Error: Singlepoint_fragments requires charge/mult information to be associated with each fragment.",
                  BC.END)
            ashexit()
        # Setting charge/mult  from fragment
        charge = frag.charge;
        mult = frag.mult

        # Setting orbital file for ORCATheory or any other theory using moreadfile
        try:
            theory.moreadfile = moreadfiles[i]
        except:
            pass

        # Running single-point
        result = Singlepoint(theory=theory, fragment=frag, charge=charge, mult=mult)

        print("Fragment {} . Label: {} Energy: {} Eh".format(frag.formula, frag.label, result.energy))

        # Preserve outputfile
        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        try:
            shutil.copyfile(theory.filename + '.out', f'./{calc_label}.out')
        except:
            pass

        theory.cleanup()
        energies.append(result.energy)
        # Adding energy as the fragment attribute
        frag.set_energy(result.energy)
        print("")

    # Create Results object
    result = ASH_Results(label="Singlepoint_fragments", energies=energies, charge=charge, mult=mult)

    # Print table
    print_fragments_table(fragments, energies)

    # Print table
    if relative_energies is True:
        print()
        print("relative_energies option is True!")
        conversionfactor = {'kcal/mol': 627.50946900, 'kcalpermol': 627.50946900, 'kJ/mol': 2625.499638,
                            'kJpermol': 2625.499638,
                            'eV': 27.211386245988, 'cm-1': 219474.6313702, 'Eh': 1.0, 'mEh': 1000,
                            'meV': 27211.386245988}
        convfactor = conversionfactor[unit]
        relenergies = [(i - min(energies)) * convfactor for i in energies]
        print_fragments_table(fragments, relenergies, unit=unit)
        result.relative_energies = relenergies
        result.labels = [f.label for f in fragments]

    # Printing reaction energy if stoichiometry was provided
    if stoichiometry != None:
        print("Stoichiometry provided:", stoichiometry)
        r = ReactionEnergy(list_of_energies=energies, stoichiometry=stoichiometry, list_of_fragments=fragments,
                           unit=unit, label='ΔE')
        result.reaction_energy = r[0]
    result.write_to_disk(filename="ASH_SP_fragments.result")
    print_time_rel(module_init_time, modulename='Singlepoint_fragments', moduleindex=1)
    return result


# Single-point energy function that runs calculations on multiple fragments. Returns a list of energies.
# Assuming fragments have charge,mult info defined.
def Singlepoint_fragments_and_theories(theories=None, fragments=None, stoichiometry=None):
    print_line_with_mainheader("Singlepoint_fragments_and_theories")
    module_init_time = time.time()
    # List of lists
    all_energies = []

    # Looping over theories and getting energies for list of fragments
    for theory in theories:
        result = Singlepoint_fragments(theory=theory, fragments=fragments, stoichiometry=stoichiometry)
        all_energies.append(result.energies)

    print("\n")
    print("SINGLEPOINT_FRAGMENTS_AND_THEORIES ALL DONE")
    print("\n")
    print("=" * 60)
    print("Singlepoint_fragments_and_theories: FINAL RESULTS")
    print("=" * 60)
    # Table
    for t, elist in zip(theories, all_energies):
        print("\nTheory:", t.__class__.__name__)
        print("Label:", t.label)
        print_fragments_table(fragments, elist, tabletitle="")
        # Reaction energy if stoichiometry provided
        if stoichiometry != None:
            print("Stoichiometry provided:", stoichiometry)
            ReactionEnergy(list_of_energies=elist, stoichiometry=stoichiometry, list_of_fragments=fragments,
                           unit='kcal/mol', label='{}'.format(t.label))

            print("_" * 60)
    print("\nFinal list of lists of total energies:", all_energies)

    result = ASH_Results(label="Singlepoint_fragments_and_theories", energies=all_energies)
    if stoichiometry != None:
        print("Final reaction energies:")
        for elist, t in zip(all_energies, theories):
            r = ReactionEnergy(list_of_energies=elist, stoichiometry=stoichiometry, list_of_fragments=fragments,
                               unit='kcal/mol', label='{}'.format(t.label))
            result.reaction_energies.append(r[0])
    print()
    result.write_to_disk(filename="ASH_SP_fragments_theories.result")
    # return all_energies
    print_time_rel(module_init_time, modulename='Singlepoint_fragments_and_theories', moduleindex=1)
    return result


# Single-point energy function that runs calculations on an ASH reaction object
# Assuming fragments have charge,mult info defined.
def Singlepoint_reaction(theory=None, reaction=None, moreadfiles=None):
    print_line_with_mainheader("Singlepoint_reaction function")
    module_init_time = time.time()

    print("Will run single-point calculation on each fragment defined in reaction")
    print("Theory:", theory.__class__.__name__)
    print("Resetting energies in reaction object")
    reaction.energies = []
    reaction.reset_energies()

    # Looping through fragments defined in Reaction object
    for i, frag in enumerate(reaction.fragments):
        # Orbital file for ORCATheory or any other theory using moreadfile
        try:
            theory.moreadfile = reaction.orbital_dictionary[moreadfiles][i]
            print("Found orbital dictionary in reaction object")
            print("Using orbital file:", theory.moreadfile)
        except:
            try:
                theory.moreadfile = moreadfiles[i]
            except:
                pass
        # Running single-point
        result = Singlepoint(theory=theory, fragment=frag, charge=frag.charge, mult=frag.mult)
        energy = result.energy
        print("Fragment {} . Label: {} Energy: {} Eh".format(frag.formula, frag.label, energy))
        # Preserve outputfile
        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        try:
            shutil.copyfile(theory.filename + '.out', f'./{calc_label}.out')
        except:
            pass
        theory.cleanup()
        reaction.energies.append(energy)

        # TODO: Change this so that instead we just grab whatever each Theory level deemed important
        # theory.properties feature?
        # Check if ORCATheory object contains ICE-CI info
        if isinstance(theory, openmmqmmm.ORCATheory):
            print("theory.properties:", theory.properties)
            # Add selected properties to Reaction object
            try:
                reaction.properties["E_var"].append(theory.properties["E_var"])
                reaction.properties["E_PT2_rest"].append(theory.properties["E_PT2_rest"])
                reaction.properties["num_genCFGs"].append(theory.properties["num_genCFGs"])
                reaction.properties["num_selected_CFGs"].append(theory.properties["num_selected_CFGs"])
                reaction.properties["num_after_SD_CFGs"].append(theory.properties["num_after_SD_CFGs"])
            except:
                pass
        # Adding energy as the fragment attribute
        frag.set_energy(energy)
        print("")

    # Print table
    print_fragments_table(reaction.fragments, reaction.energies, tabletitle="Singlepoint_reaction: ")

    # Setting unit of reaction if given (will override reaction.unit definition)
    # NOTE: Needed?
    # NOTE: Now just setting unit equal to reaction.unit. Used for components below
    unit = reaction.unit

    reaction.calculate_reaction_energy()

    result = ASH_Results(label="Singlepoint_reaction", energies=reaction.energies,
                         reaction_energy=reaction.reaction_energy)

    print_time_rel(module_init_time, modulename='Singlepoint_reaction', moduleindex=1)
    result.write_to_disk(filename="ASH_SP_reaction.result")
    return result
    # return reaction.reaction_energy


# Single-point energy function that communicates via fragment
# NOTE: NOT SURE IF WE WANT TO GO THIS ROUTE


# Theory object that always gives zero energy and zero gradient. Useful for setting constraints
class ZeroTheory:
    def __init__(self, fragment=None, printlevel=None, numcores=1, label=None):
        """Class Zerotheory: Simple dummy theory that gives zero energy and a zero-valued gradient array
            Note: includes unnecessary attributes for consistency.

        Args:
            fragment (ASH fragment, optional): A valid ASH fragment. Defaults to None.
            printlevel (int, optional): Printlevel:0,1,2 or 3. Defaults to None.
            numcores (int, optional): Number of cores. Defaults to 1.
            label (str, optional): String label. Defaults to None.
        """
        self.numcores = numcores
        self.printlevel = printlevel
        self.label = label
        self.fragment = fragment
        self.filename = "zerotheory"
        self.theorynamelabel = "ZeroTheory"
        # Indicate that this is a QMtheory
        self.theorytype = "QM"

    def run(self, current_coords=None, elems=None, Grad=False, PC=False, numcores=None, charge=None, mult=None,
            label=None,
            current_MM_coords=None, MMcharges=None, qm_elems=None):
        self.energy = 0.0
        # Gradient as np array
        self.gradient = np.zeros((len(elems), 3))
        if Grad == False:
            return self.energy
        else:
            return self.energy, self.gradient


# Theory object that executes a script present in dir and then grabs energy and gradient from files created
# Simple way to create interfaces to programs


def ReactionEnergy(list_of_energies=None, stoichiometry=None, list_of_fragments=None, unit='kcal/mol', label=None,
                   reference=None, silent=False,
                   correction=0.0):
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
    conversionfactor = {'kcal/mol': 627.50946900, 'kcalpermol': 627.50946900, 'kJ/mol': 2625.499638,
                        'kJpermol': 2625.499638,
                        'eV': 27.211386245988, 'cm-1': 219474.6313702, 'Eh': 1.0, 'mEh': 1000, 'meV': 27211.386245988}
    if label is None:
        label = ''
    reactant_energy = 0.0  # hartree
    product_energy = 0.0  # hartree
    if stoichiometry is None:
        print("stoichiometry list is required")
        ashexit()

    if correction != 0.0:
        print("User-correction was added. ")
        print(f"Correction to reaction energy in {correction} Eh ")
        correction_in_unit = correction * conversionfactor[unit]
        print(f"correction_in_unit in {correction_in_unit} {unit}")
    else:
        correction_in_unit = 0.0

    # List of energies option
    if list_of_energies is not None:

        if len(list_of_energies) != len(stoichiometry):
            print("Number of energies not equal to number of stoichiometry values")
            print("Exiting.")
            ashexit()

        for i, stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy = reactant_energy + list_of_energies[i] * abs(stoich)
            if stoich > 0:
                product_energy = product_energy + list_of_energies[i] * abs(stoich)
        reaction_energy = (product_energy - reactant_energy) * conversionfactor[unit] + correction_in_unit
        if reference is None:
            error = None
            if silent is False:
                print(BC.BOLD, "Reaction_energy({}): {} {} {}".format(label, BC.OKGREEN, reaction_energy, unit), BC.END)
        else:
            error = reaction_energy - reference
            if silent is False:
                print(BC.BOLD,
                      "Reaction_energy({}): {} {} {} (Error: {})".format(label, BC.OKGREEN, reaction_energy, unit,
                                                                         error), BC.END)
    else:
        print("\nNo list of total energies provided. Using internal energy of each fragment instead.")
        print("")
        for i, stoich in enumerate(stoichiometry):
            if stoich < 0:
                reactant_energy = reactant_energy + list_of_fragments[i].energy * abs(stoich)
            if stoich > 0:
                product_energy = product_energy + list_of_fragments[i].energy * abs(stoich)
        reaction_energy = (product_energy - reactant_energy) * conversionfactor[unit] + correction_in_unit
        if reference is None:
            error = None
            if silent is False:
                print(BC.BOLD, "Reaction_energy({}): {} {} {}".format(label, BC.OKGREEN, reaction_energy, unit), BC.END)
        else:
            error = reaction_energy - reference
            if silent is False:
                print(BC.BOLD,
                      "Reaction_energy({}): {} {} {} (Error: {})".format(label, BC.OKGREEN, reaction_energy, unit,
                                                                         error), BC.END)
    return reaction_energy, error
