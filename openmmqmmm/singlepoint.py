from __future__ import annotations

import contextlib
import logging
import math
import shutil
import time
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

import openmmqmmm
import openmmqmmm.constants
from openmmqmmm.coords import Fragment, Reaction, check_charge_mult
from openmmqmmm.exceptions import (
    InputError,
)
from openmmqmmm.results import Results
from openmmqmmm.utils import log_time_since, main_header

logger = logging.getLogger(__name__)


def _cleanup_theory(theory: Any) -> None:
    cleanup = getattr(theory, "cleanup", None)
    if callable(cleanup):
        cleanup()


def _energy_conversion_factor(unit: str) -> float:
    try:
        return openmmqmmm.constants.ENERGY_UNIT_FROM_HARTREE[unit]
    except KeyError:
        choices = ", ".join(openmmqmmm.constants.ENERGY_UNIT_FROM_HARTREE)
        raise InputError(f"Unknown energy unit {unit!r}; choose one of: {choices}") from None


def _validate_stoichiometry(stoichiometry: Sequence[float], expected_size: int) -> list[float]:
    """Return finite numeric coefficients after checking their count."""
    try:
        coefficients = list(stoichiometry)
    except TypeError:
        raise InputError("stoichiometry must be a sequence of signed numeric coefficients") from None
    if len(coefficients) != expected_size:
        raise InputError(
            f"Number of stoichiometry values ({len(coefficients)}) does not match the number of species "
            f"({expected_size})"
        )
    invalid = [
        coefficient
        for coefficient in coefficients
        if isinstance(coefficient, bool) or not isinstance(coefficient, Real) or not math.isfinite(float(coefficient))
    ]
    if invalid:
        raise InputError(f"stoichiometry must contain only finite numeric coefficients; got {invalid!r}")
    return [float(coefficient) for coefficient in coefficients]


def single_point(
    fragment: Fragment | None = None,
    theory: Any | None = None,
    grad: bool = False,
    charge: int | None = None,
    mult: int | None = None,
    result_write_to_disk: bool = True,
) -> Results:
    """Run a single-point energy (and optionally gradient) calculation."""
    logger.info(main_header("Singlepoint function"))
    module_init_time = time.time()
    if fragment is None or theory is None:
        raise InputError("Singlepoint requires a fragment and a theory object")
    coords = fragment.coords
    elems = fragment.elems

    charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "Singlepoint", theory=theory)

    if grad:
        logger.info(
            f"Doing single-point Energy+Gradient job on fragment. Formula: {fragment.prettyformula} Label: "
            f"{fragment.label} "
        )
        energy, gradient = theory.run(current_coords=coords, elems=elems, grad=True, charge=charge, mult=mult)
    else:
        logger.info(
            f"Doing single-point Energy job on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
        )
        logger.info(f"Charge: {charge} Mult: {mult}")
        energy = theory.run(current_coords=coords, elems=elems, charge=charge, mult=mult)
        gradient = None

    logger.info("Energy:  %s", energy)
    fragment.set_energy(energy)
    log_time_since(module_init_time, "Singlepoint")
    result = Results(label="Singlepoint", energy=energy, gradient=gradient, charge=charge, mult=mult)
    if theory.theorytype == "QM/MM":
        result.qmmm_energy = theory.QM_MM_energy
        result.mm_energy = theory.MMenergy
        result.qm_energy = theory.QMenergy
    if result_write_to_disk:
        result.write_to_disk(filename="results_singlepoint.json")
    return result


def single_point_theories(
    theories: Sequence[Any] | None = None,
    fragment: Fragment | None = None,
    charge: int | None = None,
    mult: int | None = None,
) -> Results:
    """Run single-point calculations of one fragment with multiple theories."""
    logger.info(main_header("Singlepoint_theories function"))
    module_init_time = time.time()
    logger.debug("Will run single-point calculation on the fragment with multiple theories")

    if fragment is None:
        raise InputError("single_point_theories requires a fragment")
    if not theories:
        raise InputError("single_point_theories requires at least one theory")

    energies = []
    resolved_states = []

    for theory in theories:
        # Resolved per theory from the original arguments: rebinding charge here would carry one
        # theory's resolved value (e.g. a QM/MM region charge) into the next theory in the list.
        theory_charge, theory_mult = check_charge_mult(
            charge, mult, theory.theorytype, fragment, "Singlepoint_theories", theory=theory
        )
        resolved_states.append((theory_charge, theory_mult))

        result = single_point(
            theory=theory,
            fragment=fragment,
            charge=theory_charge,
            mult=theory_mult,
            result_write_to_disk=False,
        )

        calc_label = "Frag_" + theory.__class__.__name__ + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")

        logger.info(f"Theory Label: {theory.label} Energy: {result.energy} Eh")
        _cleanup_theory(theory)
        energies.append(result.energy)

    _log_theories_table(theories, energies, resolved_states)
    common_state = resolved_states[0] if all(state == resolved_states[0] for state in resolved_states) else (None, None)
    result = Results(
        label="Singlepoint_theories",
        energies=energies,
        charge=common_state[0],
        mult=common_state[1],
    )
    result.write_to_disk(filename="results_singlepoint_theories.json")
    log_time_since(module_init_time, "Singlepoint_theories")
    return result


def _log_theories_table(
    theories: Sequence[Any],
    energies: Sequence[float],
    resolved_states: Sequence[tuple[int | None, int | None]],
) -> None:
    logger.info("%s", "=" * 70)
    logger.info("Singlepoint_theories: Table of energies of each theory:")
    logger.info("%s", "=" * 70)

    logger.info(
        "%s", "\n{:15} {:15} {:>7} {:>7} {:>20}".format("Theory class", "Theory Label", "Charge", "Mult", "Energy(Eh)")
    )
    logger.info("%s", "-" * 70)
    for t, e, (charge, mult) in zip(theories, energies, resolved_states, strict=True):
        logger.info(f"{t.__class__.__name__:15} {t.label!s:15} {charge!s:>7} {mult!s:>7} {e:>20.10f}\n")


def _log_fragments_table(
    fragments: Sequence[Fragment],
    energies: Sequence[float],
    tabletitle: str = "Singlepoint_fragments: ",
    unit: str = "Eh",
) -> None:
    logger.info("%s", "=" * 100)
    logger.info(f"{tabletitle}Table of energies of each fragment:")
    logger.info("%s", "=" * 100)
    logger.info("%s", "{:15} {:<25} {:>7} {:>7} {:>30}".format("Formula", "Label", "Charge", "Mult", f"Energy({unit})"))
    logger.info("%s", "-" * 100)
    for frag, e in zip(fragments, energies, strict=True):
        label = "None" if frag.label is None else str(frag.label)
        logger.info(f"{frag.formula:15} {label:<25} {frag.charge:>7} {frag.mult:>7} {e:>30.10f}\n")


# Assuming fragments have charge,mult info defined.
# If stoichiometry provided then print reaction energy
def single_point_fragments(
    theory: Any | None = None,
    fragments: Sequence[Fragment] | None = None,
    stoichiometry: Sequence[float] | None = None,
    relative_energies: bool = False,
    unit: str = "kcal/mol",
    moreadfiles: Sequence[str] | None = None,
    result_write_to_disk: bool = True,
) -> Results:
    """Run single-point calculations of one theory over multiple fragments."""
    logger.info(main_header("Singlepoint_fragments function"))
    module_init_time = time.time()
    logger.debug("Will run single-point calculation on each fragment")
    if theory is None:
        raise InputError("single_point_fragments requires a theory")
    if not fragments:
        raise InputError("single_point_fragments requires at least one fragment")
    if isinstance(moreadfiles, str):
        raise InputError("moreadfiles must be a sequence with one orbital file per fragment, not a string")
    if moreadfiles is not None and len(moreadfiles) != len(fragments):
        raise InputError(
            f"moreadfiles contains {len(moreadfiles)} files for {len(fragments)} fragments; provide one per fragment"
        )
    missing_states = [index for index, frag in enumerate(fragments) if frag.charge is None or frag.mult is None]
    if missing_states:
        raise InputError(f"Fragments at indices {missing_states} are missing charge or multiplicity")
    if stoichiometry is not None:
        stoichiometry = _validate_stoichiometry(stoichiometry, len(fragments))
    conversion_factor = _energy_conversion_factor(unit) if relative_energies or stoichiometry is not None else 1.0
    resolved_states = [(fragment.charge, fragment.mult) for fragment in fragments]
    logger.info("Theory: %s", theory.__class__.__name__)

    energies = []

    for i, frag in enumerate(fragments):
        charge = frag.charge
        mult = frag.mult

        if moreadfiles is not None:
            theory.moreadfile = moreadfiles[i]

        result = single_point(theory=theory, fragment=frag, charge=charge, mult=mult, result_write_to_disk=False)

        logger.info(f"Fragment {frag.formula} . Label: {frag.label} Energy: {result.energy} Eh")

        calc_label = "Frag_" + str(frag.formula) + "_" + str(frag.charge) + "_" + str(frag.mult) + "_"
        with contextlib.suppress(OSError, AttributeError):
            shutil.copyfile(theory.filename + ".out", f"./{calc_label}.out")

        _cleanup_theory(theory)
        energies.append(result.energy)

    common_state = resolved_states[0] if all(state == resolved_states[0] for state in resolved_states) else (None, None)
    result = Results(
        label="Singlepoint_fragments",
        energies=energies,
        charge=common_state[0],
        mult=common_state[1],
    )

    _log_fragments_table(fragments, energies)

    if relative_energies is True:
        logger.info("\nrelative_energies option is True!")
        relenergies = [(energy - min(energies)) * conversion_factor for energy in energies]
        _log_fragments_table(fragments, relenergies, unit=unit)
        result.relative_energies = relenergies
        result.labels = [f.label for f in fragments]

    if stoichiometry is not None:
        logger.info("Stoichiometry provided: %s", stoichiometry)
        r = reaction_energy(
            list_of_energies=energies, stoichiometry=stoichiometry, list_of_fragments=fragments, unit=unit, label="ΔE"
        )
        result.reaction_energy = r[0]
    if result_write_to_disk:
        result.write_to_disk(filename="results_singlepoint_fragments.json")
    log_time_since(module_init_time, "Singlepoint_fragments")
    return result


# Assuming fragments have charge,mult info defined.
def single_point_fragments_and_theories(
    theories: Sequence[Any] | None = None,
    fragments: Sequence[Fragment] | None = None,
    stoichiometry: Sequence[float] | None = None,
) -> Results:
    """Run single-point calculations for every fragment with every theory."""
    logger.info(main_header("Singlepoint_fragments_and_theories"))
    module_init_time = time.time()
    if not theories:
        raise InputError("single_point_fragments_and_theories requires at least one theory")
    if not fragments:
        raise InputError("single_point_fragments_and_theories requires at least one fragment")
    all_energies = []
    reaction_energies = []

    for theory in theories:
        result = single_point_fragments(
            theory=theory,
            fragments=fragments,
            stoichiometry=stoichiometry,
            result_write_to_disk=False,
        )
        all_energies.append(result.energies)
        if stoichiometry is not None:
            reaction_energies.append(result.reaction_energy)

    logger.info("SINGLEPOINT_FRAGMENTS_AND_THEORIES ALL DONE")
    logger.info("%s", "=" * 60)
    logger.info("Singlepoint_fragments_and_theories: FINAL RESULTS")
    logger.info("%s", "=" * 60)
    for index, (t, elist) in enumerate(zip(theories, all_energies, strict=True)):
        logger.info("\nTheory: %s", t.__class__.__name__)
        logger.info("Label: %s", t.label)
        _log_fragments_table(fragments, elist, tabletitle="")
        if stoichiometry is not None:
            logger.info("Reaction energy (%s): %s kcal/mol", t.label, reaction_energies[index])
        logger.info("%s", "_" * 60)
    logger.info("\nFinal list of lists of total energies: %s", all_energies)

    result = Results(
        label="Singlepoint_fragments_and_theories",
        energies=all_energies,
        reaction_energies=reaction_energies if stoichiometry is not None else None,
    )
    result.write_to_disk(filename="results_singlepoint_fragments_theories.json")
    log_time_since(module_init_time, "Singlepoint_fragments_and_theories")
    return result


# Assuming fragments have charge,mult info defined.
def single_point_reaction(
    theory: Any | None = None,
    reaction: Reaction | None = None,
    moreadfiles: str | Sequence[str] | None = None,
) -> Results:
    """Run single-point calculations for all species of a Reaction and compute the reaction energy."""
    logger.info(main_header("Singlepoint_reaction function"))
    module_init_time = time.time()

    if theory is None or reaction is None:
        raise InputError("single_point_reaction requires a theory and a reaction")
    reaction.check_fragments()
    _energy_conversion_factor(reaction.unit)

    orbital_files: Sequence[str] | None
    if isinstance(moreadfiles, str):
        orbital_files = reaction.orbital_dictionary.get(moreadfiles)
        if not orbital_files:
            raise InputError(f"Reaction has no orbital-file set named {moreadfiles!r}")
    else:
        orbital_files = moreadfiles
    if orbital_files is not None and len(orbital_files) != len(reaction.fragments):
        raise InputError(
            f"moreadfiles contains {len(orbital_files)} files for {len(reaction.fragments)} reaction fragments"
        )

    logger.debug("Will run single-point calculation on each fragment defined in reaction")
    logger.info("Theory: %s", theory.__class__.__name__)
    logger.info("Resetting energies in reaction object")
    reaction.reset_energies()

    for i, frag in enumerate(reaction.fragments):
        if orbital_files is not None:
            theory.moreadfile = orbital_files[i]
            logger.info("Using orbital file: %s", theory.moreadfile)
        result = single_point(
            theory=theory,
            fragment=frag,
            charge=frag.charge,
            mult=frag.mult,
            result_write_to_disk=False,
        )
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

    _log_fragments_table(reaction.fragments, reaction.energies, tabletitle="Singlepoint_reaction: ")

    reaction.calculate_reaction_energy()

    result = Results(label="Singlepoint_reaction", energies=reaction.energies, reaction_energy=reaction.reaction_energy)

    log_time_since(module_init_time, "Singlepoint_reaction")
    result.write_to_disk(filename="results_singlepoint_reaction.json")
    return result


class ZeroTheory:
    """Dummy theory returning zero energy and a zero gradient (useful for testing workflows)."""

    def __init__(self, fragment: Fragment | None = None, numcores: int = 1, label: str | None = None) -> None:
        self.numcores = numcores
        self.label = label
        self.fragment = fragment
        self.filename = "zerotheory"
        self.theorynamelabel = "ZeroTheory"
        self.theorytype = "QM"

    def cleanup(self) -> None:
        """No files to clean up; present so ZeroTheory satisfies the theory contract."""

    def run(
        self,
        *,
        current_coords: np.ndarray | None = None,
        elems: Sequence[str] | None = None,
        grad: bool = False,
        pc: bool = False,
        numcores: int | None = None,
        charge: int | None = None,
        mult: int | None = None,
        label: str | None = None,
        current_mm_coords: np.ndarray | None = None,
        mm_charges: Sequence[float] | None = None,
        qm_elems: Sequence[str] | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """Return zero energy and, if requested, a zero gradient."""
        self.energy = 0.0
        self.gradient = np.zeros((len(elems), 3))
        if not grad:
            return self.energy
        return self.energy, self.gradient


def reaction_energy(
    list_of_energies: Sequence[float] | None = None,
    stoichiometry: Sequence[float] | None = None,
    list_of_fragments: Sequence[Fragment] | None = None,
    unit: str = "kcal/mol",
    label: str | None = None,
    reference: float | None = None,
    silent: bool = False,
    correction: float = 0.0,
) -> tuple[float, float | None]:
    """Calculate a reaction energy from energies (or fragments with energies) and stoichiometry."""
    if stoichiometry is None:
        raise InputError("stoichiometry list is required")
    convfactor = _energy_conversion_factor(unit)

    if list_of_energies is None:
        if list_of_fragments is None:
            raise InputError("Provide either list_of_energies or list_of_fragments")
        missing = [index for index, fragment in enumerate(list_of_fragments) if fragment.energy is None]
        if missing:
            raise InputError(f"Fragments at indices {missing} do not have stored energies")
        list_of_energies = [fragment.energy for fragment in list_of_fragments]
    coefficients = _validate_stoichiometry(stoichiometry, len(list_of_energies))

    if correction != 0.0:
        logger.info("User-correction was added. ")
        logger.info(f"Correction to reaction energy in {correction} Eh ")
        correction_in_unit = correction * convfactor
        logger.info(f"correction_in_unit in {correction_in_unit} {unit}")
    else:
        correction_in_unit = 0.0

    delta_energy = sum(energy * coefficient for energy, coefficient in zip(list_of_energies, coefficients, strict=True))
    converted_energy = delta_energy * convfactor + correction_in_unit
    error = None if reference is None else converted_energy - reference
    if not silent:
        error_suffix = "" if error is None else f" (Error: {error})"
        logger.info(f"Reaction_energy({label or ''}):  {converted_energy} {unit}{error_suffix}")
    return converted_energy, error
