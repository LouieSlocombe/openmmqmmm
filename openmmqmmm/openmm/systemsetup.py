from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import openmm
import openmm.app
import openmm.app as openmm_app
import openmm.unit
import openmm.unit as openmm_unit
from packaging import version

import openmmqmmm.constants
import openmmqmmm.parallel
from openmmqmmm.coords import (
    Fragment,
    check_charge_mult,
    check_gradient_for_bad_atoms,
    write_pdbfile,
    write_xyzfile,
)
from openmmqmmm.exceptions import (
    InputError,
    MissingDependencyError,
)
from openmmqmmm.openbabel import xyz_to_pdb_with_connectivity
from openmmqmmm.openmm.theory import OpenMMTheory
from openmmqmmm.singlepoint import single_point
from openmmqmmm.utils import (
    log_time_since,
    main_header,
    pygrep,
    write_list_to_file,
)

logger = logging.getLogger(__name__)

# Forcefield shorthands accepted by openmm_modeller, mapped to the XML file OpenMM ships.
FORCEFIELD_XMLFILES = {
    "Amber99": "amber99sb.xml",
    "Amber99sb": "amber99sb.xml",
    "Amber99sb-ildn": "amber99sbildn.xml",
    "Amber96": "amber96.xml",
    "Amber03": "amber03.xml",
    "Amber10": "amber10.xml",
    "Amber14": "amber14-all.xml",
    "CHARMM36": "charmm36.xml",
    "CHARMM2013": "charmm_polar_2013.xml",
    "Amoeba2013": "amoeba2013.xml",
    "Amoeba2009": "amoeba2009.xml",
}


def _normalise_modeller_solvent_name(watermodel: str | None) -> str:
    """Return the water-box name expected by OpenMM's Modeller."""
    if watermodel is None:
        return "tip3p"

    model = watermodel.lower()
    # TIP3P-FB has different parameters but uses the standard three-site
    # TIP3P box geometry supplied by Modeller.
    if model in {"tip3pfb", "tip3p-fb"}:
        return "tip3p"
    return model


def print_systemsize(modeller: openmm.app.Modeller) -> None:
    logger.info(f"System size: {len(modeller.getPositions())} atoms\n")


def openmm_minimize(
    fragment: Fragment | None = None,
    theory: OpenMMTheory | None = None,
    maxiter: int = 1000,
    tolerance: float = 1,
    enforce_periodic_box: bool = True,
    traj_frequency: int = 100,
    use_reporter: bool = True,
) -> Fragment:
    """Minimize the MM energy of a fragment with OpenMM's L-BFGS minimizer."""
    module_init_time = time.time()
    logger.info(main_header("OpenMM Optimization"))

    if fragment is None:
        raise InputError("No fragment object. Exiting.")

    if isinstance(theory, OpenMMTheory):
        openmmobject = theory
    else:
        raise InputError("Only OpenMMTheory allowed in OpenMM_Opt. Exiting.")

    logger.info("Number of atoms: %s", fragment.numatoms)
    logger.info("Max iterations: %s", maxiter)
    logger.info(f"Tolerance: {tolerance} kj/mol/nm:")
    if version.parse(openmm.__version__) >= version.parse("8.1"):
        logger.info(f"Will write to trajectory every {traj_frequency} iterations")
    logger.info("OpenMM autoconstraints: %s", openmmobject.autoconstraints)
    logger.info("OpenMM hydrogenmass: %s", openmmobject.hydrogenmass)
    logger.info("OpenMM rigidwater constraints: %s", openmmobject.rigidwater)

    if openmmobject.user_constraints:
        logger.info(f"User constraints: {openmmobject.user_constraints}")
    else:
        logger.info("User constraints: None")

    if openmmobject.user_restraints:
        logger.info(f"User restraints: {openmmobject.user_restraints}")
    else:
        logger.info("User restraints: None")
    logger.info(f"Number of frozen atoms: {len(openmmobject.user_frozen_atoms)}")

    if openmmobject.autoconstraints is None:
        logger.warning(
            "Autoconstraints have not been set in OpenMMTheory; no bonds are constrained in the optimization"
        )
    if (openmmobject.rigidwater is True and len(openmmobject.user_frozen_atoms) != 0) or (
        openmmobject.autoconstraints is not None and len(openmmobject.user_frozen_atoms) != 0
    ):
        logger.warning(
            "Frozen_atoms options selected but there are general constraints defined in "
            "the OpenMM object (either rigidwater=True or autoconstraints is not None)\n"
            "OpenMM will crash if constraints and frozen atoms involve the same atoms"
        )

    openmmobject.set_simulation_parameters(timestep=0.001, temperature=1, integrator="VerletIntegrator")

    simulation = openmmobject.create_simulation()

    logger.info("Simulation created.")

    # New in OpenMM 8.1: reporters for minimizer
    if version.parse(openmm.__version__) >= version.parse("8.1") and use_reporter is True:

        class Reporter(openmm.openmm.MinimizationReporter):
            """Log minimizer progress; the OpenMM reporter hook exists only from 8.1 on."""

            def report(
                self,
                iteration: int,
                x: Sequence[float],
                grad: Sequence[float],
                args: Mapping[str, float],
            ) -> bool:
                if not hasattr(self, "totaliter"):
                    self.totaliter = -1

                self.totaliter += 1

                self.get_forces(grad)
                short = False

                if short is True:
                    logger.info(f"Iteration {iteration} ")
                else:
                    logger.info("TOTAL iteration: %s", self.totaliter)
                    logger.info(f"Micro Iteration {iteration}")
                    self.print_energy(args)
                    self.print_forces()
                    self.write_traj(x)
                if iteration == maxiter - 1:
                    logger.info("Max iterations reached. Now modifying restraints and restarting")
                    return True

                return False

            def write_traj(self, x: Sequence[float]) -> None:
                if self.totaliter % traj_frequency == 0:
                    logger.info("%s", "-" * 40)
                    logger.info("Now writing to trajectory file")
                    logger.info("%s", "-" * 40)
                    pos = 10 * np.array(x).reshape(-1, 3)
                    write_xyzfile(fragment.elems, pos, "OpenMMOpt_traj", writemode="a")

            def print_energy(self, args: Mapping[str, float]) -> None:
                system_energy = args["system energy"] / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
                restraint_energy = args["restraint energy"] / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
                logger.info("System energy: %s", system_energy)
                logger.info("Restraint energy: %s", restraint_energy)
                logger.info("Restraint strength: %s", args["restraint strength"])
                logger.info("Max constraint error: %s", args["max constraint error"])

            def get_forces(self, grad: Sequence[float]) -> None:
                g = np.array(grad).reshape(-1, 3)  # To confirm
                kjmolnm_to_atomic_factor = -openmmqmmm.constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
                self.forces_init = g / kjmolnm_to_atomic_factor
                self.rms_force = np.sqrt(sum(n * n for n in self.forces_init.flatten()) / len(forces_init.flatten()))
                self.max_force = self.forces_init.max()

            def print_forces(self) -> None:
                logger.info(f"RMS force (w restraints): {self.rms_force} Eh/Bohr")
                logger.info(f"Max force (w restraints): {self.max_force} Eh/Bohr\n")

        reporter = Reporter()

    logger.debug("Now adding coordinates")
    openmmobject.set_positions(fragment.coords, simulation)

    state = simulation.context.getState(getEnergy=True, getForces=True, enforcePeriodicBox=enforce_periodic_box)
    potE_init = (
        state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system)
        / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
    )
    logger.info(f"Initial potential energy is: {potE_init} Eh")
    kjmolnm_to_atomic_factor = -openmmqmmm.constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    forces_init = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_init.flatten()) / len(forces_init.flatten()))
    logger.info(f"Initial RMS force: {rms_force} Eh/Bohr (w/o restraints)")
    logger.info(f"Initial Max force: {forces_init.max()} Eh/Bohr (w/o restraints)")
    logger.debug("\nStarting minimization.")
    if version.parse(openmm.__version__) >= version.parse("8.1") and use_reporter is True:
        logger.info("OpenMM versions >= 8.1. Will use a reporter to output progress")
        logger.info("OpenMM_Opt trajectory will be written to: OpenMMOpt_traj.xyz")
        with contextlib.suppress(OSError):
            os.remove("OpenMMOpt_traj.xyz")
        simulation.minimizeEnergy(maxIterations=maxiter, tolerance=tolerance, reporter=reporter)
        logger.info("Minimization done.")
        logger.info("OpenMM_Opt trajectory was written to: OpenMMOpt_traj.xyz")
    else:
        simulation.minimizeEnergy(maxIterations=maxiter, tolerance=tolerance)
        logger.info("Minimization done.")

    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=enforce_periodic_box
    )
    final_energy = (
        state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system)
        / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
    )
    logger.info("Final Potential energy is: %s Eh", final_energy)
    forces_final = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_final.flatten()) / len(forces_final.flatten()))
    logger.info(f"Final RMS force: {rms_force} Eh/Bohr (w/o restraints)")
    logger.info(f"Final Max force: {forces_final.max()} Eh/Bohr (w/o restraints)")

    # Writing final PDB-file. If system is non-periodic (according to OpenMMTheory settings) then we set
    # enforcePeriodicBox to False
    # to avoid some strange geometry translation
    if openmmobject.periodic is True:
        logger.info(f"Writing final PDB file (enforcePeriodicBox={enforce_periodic_box})")
        positions = simulation.context.getState(
            getPositions=True, enforcePeriodicBox=enforce_periodic_box
        ).getPositions()
    else:
        logger.info("Writing final PDB file (enforcePeriodicBox=False)")
        positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions()
    write_pdbfile_openmm_topology(openmmobject.topology, positions, "frag-minimized.pdb")

    newcoords = (
        simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        .getPositions(asNumpy=True)
        .value_in_unit(openmm.unit.angstrom)
    )
    logger.debug("\nUpdating coordinates in fragment.")
    fragment.coords = newcoords

    logger.info("All Done!")
    log_time_since(module_init_time, "OpenMM_Opt")

    return fragment


def _resolve_named_forcefield(
    forcefield: str, watermodel: str | None, waterxmlfile: str | None
) -> tuple[str, str | None, str | None]:
    """Map a forcefield name onto its XML file and the water model that pairs with it."""
    logger.info("Forcefield: %s", forcefield)
    try:
        xmlfile = FORCEFIELD_XMLFILES[forcefield]
    except (KeyError, TypeError):
        raise InputError("Unknown forcefield") from None

    if "CHARMM" in forcefield:
        if watermodel is None and waterxmlfile is None:
            logger.debug("No watermodel or waterxmlfile selected. Using recommended CHARMM-style TIP3P")
            watermodel = "tip3p"
        # CHARMM36 ships its own TIP3P parameters
        if watermodel is not None and watermodel.lower() == "tip3p":
            waterxmlfile = "charmm36/water.xml"
    elif "Amber" in forcefield:
        if watermodel is None and waterxmlfile is None:
            logger.debug("No watermodel or waterxmlfile selected. Using TIP3P-FB, a reparameterized TIP3P")
            watermodel = "tip3pfb"
        model = watermodel.lower() if watermodel is not None else None
        if model in {"tip3pfb", "tip3p-fb"}:
            waterxmlfile = "amber14/tip3pfb.xml"
        elif model == "tip3p":
            waterxmlfile = "amber14/tip3p.xml" if forcefield == "Amber14" else "tip3p.xml"

    logger.info("watermodel: %s. Waterxmlfile selected: %s", watermodel, waterxmlfile)
    return xmlfile, watermodel, waterxmlfile


def _build_forcefield_object(
    *,
    xmlfile: str | None,
    forcefield_object: openmm.app.ForceField | None,
    extraxmlfile: str | None,
    waterxmlfile: str | None,
    watermodel: str | None,
) -> openmm.app.ForceField:
    """Return the OpenMM ForceField, built from XML files or taken from the caller."""
    if xmlfile is None:
        if forcefield_object is None:
            raise InputError("You must provide a forcefield name, forcefieldobject or xmlfile keywords!")
        logger.info("Using forcefield object provided")
        if waterxmlfile is not None:
            logger.warning("Ignoring waterxmlfile: a forcefield_object was supplied")
        if watermodel is not None:
            logger.info("Water model selects the solvent box geometry used by Modeller: %s", watermodel)
        return forcefield_object

    logger.info("XMLfile: %s. Water model: %s. Water xmlfile: %s", xmlfile, watermodel, waterxmlfile)
    if extraxmlfile is not None:
        logger.info("Using extra XML file: %s", extraxmlfile)
        if os.path.isfile(extraxmlfile) is not True:
            raise InputError(f"File {extraxmlfile} can not be found. Exiting.")
    logger.debug("Now creating forcefield object")
    files = [f for f in (xmlfile, extraxmlfile, waterxmlfile) if f is not None]
    return openmm_app.forcefield.ForceField(*files)


def _run_pdbfixer(pdbfile: str | os.PathLike[str]) -> str:
    """Add the missing residues and atoms PDBFixer can find, and return the repaired PDB path."""
    try:
        import pdbfixer
    except ImportError:
        raise MissingDependencyError(
            "Problem importing pdbfixer. Install first via conda:\nconda install -c conda-forge pdbfixer"
        ) from None

    logger.debug("\nRunning PDBFixer")
    fixer = pdbfixer.PDBFixer(pdbfile)
    fixer.findMissingResidues()
    logger.info("Found missing residues: %s", fixer.missingResidues)
    fixer.findNonstandardResidues()
    logger.info("Found non-standard residues: %s", fixer.nonstandardResidues)
    fixer.findMissingAtoms()
    logger.info("Found missing atoms: %s", fixer.missingAtoms)
    logger.info("Found missing terminals: %s", fixer.missingTerminals)
    fixer.addMissingAtoms()
    logger.info("Added missing atoms.")

    with open("system_afterfixes.pdb", "w") as pdbfh:
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, pdbfh)
    logger.warning(
        "PDBFixer can create unreasonable orientations of residues if residues are missing or "
        "multiple occupancies are present.\n         You should inspect the created PDB-file to be sure."
    )
    logger.info("PDBFixer done. Wrote PDBfile: system_afterfixes.pdb")
    return "system_afterfixes.pdb"


def openmm_modeller(
    *,
    pdbfile: str | os.PathLike[str] | None = None,
    forcefield_object: openmm.app.ForceField | None = None,
    forcefield: str | None = None,
    xmlfile: str | None = None,
    waterxmlfile: str | None = None,
    watermodel: str | None = None,
    ph: float = 7.0,
    solvent_padding: float = 10.0,
    solvent_boxdims: Sequence[float] | None = None,
    extraxmlfile: str | None = None,
    residue_variants: Mapping[str, Mapping[int, str]] | None = None,
    ionicstrength: float = 0.1,
    pos_iontype: str = "Na+",
    neg_iontype: str = "Cl-",
    use_higher_occupancy: bool = False,
    platform: str = "CPU",
    use_pdbfixer: bool = True,
    implicit: bool = False,
    implicit_solvent_xmlfile: str | None = None,
    membrane: bool = False,
    membrane_lipidtype: str = "POPC",
    membrane_padding: float = 10.0,
    membrane_center_z: float = 0.0,
    residuetemplate_choice: Mapping[str, str] | None = None,
    parameterize_nonstandard: bool = False,
    ligand_files: Mapping[str, str | os.PathLike[str]] | None = None,
    net_charges: Mapping[str, int] | None = None,
    ligand_backend: str = "gaff",
) -> tuple[OpenMMTheory, Fragment]:
    """Prepare a protein system from a raw PDB file using pdbfixer."""
    module_init_time = time.time()
    logger.info(main_header("OpenMM Modeller"))
    try:
        logger.info("Imported OpenMM library version: %s", openmm.__version__)
    except ImportError:
        raise ImportError(
            "OpenMM requires installing the OpenMM package. Try: 'conda install -c conda-forge openmm'  \
            Also see http://docs.openmm.org/latest/userguide/application.html"
        ) from None
    if pdbfile is None:
        raise InputError("You must provide a pdbfile keyword argument")

    if residue_variants is None:
        residue_variants = {}

    if forcefield is not None:
        xmlfile, watermodel, waterxmlfile = _resolve_named_forcefield(forcefield, watermodel, waterxmlfile)

    # OpenMM defaults addSolvent() to TIP3P when no model is specified. Resolve
    # that default explicitly so the xmlfile= and forcefield_object= routes, as
    # well as non-TIP3P models, always pass a defined value to the helper below.
    modeller_solvent_name = _normalise_modeller_solvent_name(watermodel)

    forcefield_obj = _build_forcefield_object(
        xmlfile=xmlfile,
        forcefield_object=forcefield_object,
        extraxmlfile=extraxmlfile,
        waterxmlfile=waterxmlfile,
        watermodel=watermodel,
    )

    if parameterize_nonstandard is True and forcefield_object is not None:
        raise InputError(
            "parameterize_nonstandard=True requires forcefield XML names (forcefield= or xmlfile=), not a "
            "forcefield_object: forcefill needs the XML file list to decide which residues are non-standard.\n"
            "Run forcefill.build_forcefield_xml yourself and load its XML into your ForceField object instead."
        )

    logger.info("PDBfile: %s", pdbfile)
    logger.info("pH: %s", ph)
    logger.info("User-provided dictionary of residue_variants: %s", residue_variants)
    logger.debug("\nNow checking PDB-file for alternate locations, i.e. multiple occupancies:\n")

    # Check PDB-file whether it contains alternate locations of residue atoms (multiple occupations)
    # Default behaviour:
    # - if no multiple occupancies return input PDBfile and go on
    # - if multiple occupancies, print list of residues and tell user to fix them. Exiting
    # - if use_higher_occupancy is set to True, user higher occupancy location, write new PDB_file and use
    pdbfile = find_alternate_locations_residues(pdbfile, use_higher_occupancy=use_higher_occupancy)

    logger.info("Using PDB-file %s", pdbfile)

    # Fix basic mistakes in PDB by PDBFixer
    # This will e.g. fix bad terminii
    pdbfile_for_modeller = _run_pdbfixer(pdbfile) if use_pdbfixer is True else pdbfile

    nonstandard_xmlfile = None
    if parameterize_nonstandard is True:
        nonstandard_xmlfile = _parameterize_nonstandard_residues(
            pdbfile_for_modeller,
            forcefield_obj=forcefield_obj,
            xmlfile=xmlfile,
            waterxmlfile=waterxmlfile,
            extraxmlfile=extraxmlfile,
            ligand_files=ligand_files,
            net_charges=net_charges,
            ligand_backend=ligand_backend,
        )

    pdb = openmm_app.PDBFile(pdbfile_for_modeller)
    logger.debug("\n\nNow loading Modeller.")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    modeller_numatoms = modeller.topology.getNumAtoms()
    numresidues = modeller.topology.getNumResidues()
    numchains = modeller.topology.getNumChains()
    list(modeller.topology.atoms())
    list(modeller.topology.bonds())
    modeller_chains = list(modeller.topology.chains())
    modeller_residues = list(modeller.topology.residues())
    logger.info(f"Modeller topology has {numresidues} residues.")
    logger.info(f"Modeller topology has {numchains} chains.")
    logger.info(f"Modeller topology has {modeller_numatoms} atoms.")
    logger.info("Chains: %s", modeller_chains)
    for chain_x in modeller_chains:
        logger.debug(
            f"This is chain {chain_x.index}, it has {len(chain_x._residues)} residues and they are: "
            f"{chain_x._residues}\n"
        )
    logger.info("User defined residue variants per chain:")
    for rv_key, rv_vals in residue_variants.items():
        logger.info(f"Chain {rv_key} : {rv_vals}")
    logger.info("\nMODELLER TOPOLOGY - RESIDUES TABLE\n")
    logger.info(
        "%s",
        "  {:<12}{:<13}{:<13}{:<13}{:<13}       {}".format(
            "Resid", "Resname", "Chain-index", "Chain-name", "ResID-in-chain", "User-modification"
        ),
    )
    logger.info("%s", "-" * 100)
    residue_states = _log_residue_table(modeller_residues, residue_variants)

    with open("system_afterfixes2.pdb", "w") as pdbfh:
        openmm_app.PDBFile.writeFile(modeller.topology, modeller.positions, pdbfh)

    if len(residue_states) != numresidues:
        raise InputError("residue_states != numresidues. Something went wrong")

    # This is were missing residue/atom errors will come
    logger.debug("Adding hydrogens for pH: %s", ph)
    logger.warning("OpenMM Modeller will fail in this step if residue information is missing")
    logger.info("residue_states: %s", residue_states)

    residueTemplates = {}  # initisal
    if residuetemplate_choice is not None:
        logger.info("Found user-specified residuetemplate_choice")
        logger.debug("Will generate residueTemplates based on residuetemplate_choice: %s", residuetemplate_choice)
        logger.info("Note: residuetemplate_choice should be a dict like this: residuetemplate_choice={'FER':'FE2'}   ")
        residueTemplates = {}
        for resname, choice in residuetemplate_choice.items():
            residueTemplates = {res: choice for res in modeller.topology.residues() if res.name == resname}
    logger.info("residueTemplates: %s", residueTemplates)

    logger.debug("\nNow checking if we have problems with unmatched residues")
    # NOTE: We would get exception in addHydrogens anyway
    try:
        forcefield_obj.getUnmatchedResidues(modeller.topology, residueTemplates=residueTemplates)
    except Exception as e:
        logger.info("Exception found during forcefield_obj.getUnmatchedResidues.")
        logger.info("Exception: %s", e)
        logger.info(
            "\nInterpretation: you probably have multiple matching templates in the forcefield XML-file for a residue"
        )
        raise InputError(
            "This occurs e.g. for the case of Fe2+ vs Fe3+ ion in the Amber FF.\nTo deal with this problem, you have "
            "to provide a residuetemplate_choice dictionary to this interface\nExample: residuetemplate_choice should "
            "be a dict like this: residuetemplate_choice={'FER':'FE2'}   \n   where FER is here the name of the "
            "residue (in PDB-file) and FE2 is the name of the desired template in the forcefield XML-file"
        ) from e
    logger.debug("No problem with unmatched residues found. Continuing")

    try:
        logger.info("residueTemplates: %s", residueTemplates)
        modeller.addHydrogens(forcefield_obj, pH=ph, variants=residue_states, residueTemplates=residueTemplates)
    except ValueError as errormessage:
        logger.error("OpenMM modeller.addHydrogens signalled a ValueError")
        logger.debug(
            "This is a common error and suggests a problem in PDB-file or missing residue information in the "
            "forcefield."
        )
        logger.info(
            "Non-standard inorganic/organic residues require providing an additional XML-file via extraxmlfile=, "
            "or can be parameterized automatically with parameterize_nonstandard=True (requires the forcefill package)"
        )
        logger.info("Note that C-terminii require the dangling O-atom to be named OXT ")
        raise InputError(
            f"Read the OpenMM documentation on dealing with this problem.\n\nFull error message from "
            f"OpenMM:\n{errormessage}"
        ) from errormessage

    write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "system_afterH.pdb")
    print_systemsize(modeller)

    # If using Residuetemplates then we have to remade after systemchange (addHydrogens)
    if residuetemplate_choice is not None:
        for resname, choice in residuetemplate_choice.items():
            residueTemplates = {res: choice for res in modeller.topology.residues() if res.name == resname}

    periodic, fragment, waterxmlfile = _add_solvent_or_membrane(
        modeller,
        forcefield_obj=forcefield_obj,
        implicit=implicit,
        implicit_solvent_xmlfile=implicit_solvent_xmlfile,
        membrane=membrane,
        membrane_lipidtype=membrane_lipidtype,
        membrane_padding=membrane_padding,
        membrane_center_z=membrane_center_z,
        modeller_solvent_name=modeller_solvent_name,
        watermodel=watermodel,
        waterxmlfile=waterxmlfile,
        solvent_boxdims=solvent_boxdims,
        solvent_padding=solvent_padding,
        ionicstrength=ionicstrength,
        pos_iontype=pos_iontype,
        neg_iontype=neg_iontype,
        residue_templates=residueTemplates,
    )

    write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "finalsystem.pdb")
    write_pdbxfile_openmm_topology(modeller.topology, modeller.positions, "finalsystem.cif")
    fragment.print_system(filename="finalsystem.frag")
    fragment.write_xyzfile(xyzfilename="finalsystem.xyz")

    logger.info("\nOpenMM_Modeller used the following XML-files to define system:")
    logger.info("General forcefield XML file: %s", xmlfile)
    logger.info("Solvent forcefield XML file: %s", waterxmlfile)
    logger.info("Extra forcefield XML file: %s", extraxmlfile)

    # Creating new OpenMM object from forcefield so that we can write out system XMLfile
    logger.debug("Creating OpenMMTheory object")
    openmmobject = OpenMMTheory(
        platform=platform,
        forcefield=forcefield_obj,
        topoforce=True,
        topology=modeller.topology,
        pdbfile=None,
        periodic=periodic,
        autoconstraints="HBonds",
        rigidwater=True,
        residuetemplate_choice=residuetemplate_choice,
    )
    systemxmlfile = "system_full.xml"

    serialized_system = openmm.XmlSerializer.serialize(openmmobject.system)
    with open(systemxmlfile, "w") as f:
        f.write(serialized_system)

    _log_output_files_and_usage(
        systemxmlfile=systemxmlfile,
        xmlfile=xmlfile,
        waterxmlfile=waterxmlfile,
        extraxmlfile=extraxmlfile,
        nonstandard_xmlfile=nonstandard_xmlfile,
        residuetemplate_choice=residuetemplate_choice,
        periodic=periodic,
    )
    logger.debug("\nNow running single-point MM job to check for bad contacts")
    # Setting sensible periodic cutoff to avoid error
    omm = OpenMMTheory(
        platform=platform,
        forcefield=forcefield_obj,
        topoforce=True,
        topology=modeller.topology,
        pdbfile=None,
        periodic=periodic,
        autoconstraints=None,
        rigidwater=False,
        residuetemplate_choice=residuetemplate_choice,
    )
    SP_result = single_point(theory=omm, fragment=fragment, grad=True)
    check_gradient_for_bad_atoms(fragment=fragment, gradient=SP_result.gradient, threshold=45000)

    log_time_since(module_init_time, "OpenMM_Modeller")

    return openmmobject, fragment


def write_pdbfile_openmm_topology(
    topology: openmm.app.Topology,
    positions: openmm.unit.Quantity | Sequence[openmm.Vec3],
    filename: str | os.PathLike[str],
    connectivity_dict: Mapping[int, Sequence[int]] | None = None,
) -> None:
    if connectivity_dict is not None:
        logger.info("Connectivity passed to write_pdbfile_openMM")
        openmm_add_bonds_to_topology(topology, connectivity_dict)

    with open(filename, "w") as pdbfh:
        openmm.app.PDBFile.writeFile(topology, positions, file=pdbfh)
    logger.info("Wrote PDB-file: %s", filename)


def write_pdbxfile_openmm_topology(
    topology: openmm.app.Topology,
    positions: openmm.unit.Quantity | Sequence[openmm.Vec3],
    filename: str | os.PathLike[str],
    connectivity_dict: Mapping[int, Sequence[int]] | None = None,
) -> None:
    if connectivity_dict is not None:
        logger.info("Connectivity passed to write_pdbxfile_openMM")
        openmm_add_bonds_to_topology(topology, connectivity_dict)

    with open(filename, "w") as pdbfh:
        openmm.app.PDBxFile.writeFile(topology, positions, file=pdbfh)
    logger.info("Wrote PDBx-file: %s", filename)


def openmm_add_bonds_to_topology(topology: openmm.app.Topology, connectivity: Mapping[int, Sequence[int]]) -> None:
    atoms = list(topology.atoms())
    for conatom, conlist in connectivity.items():
        for conl in conlist:
            topology.addBond(atoms[conatom], atoms[conl])


def solvate_small_molecule(
    fragment: Fragment | None = None,
    charge: int | None = None,
    mult: int | None = None,
    watermodel: str | None = None,
    solvent_boxdims: Sequence[float] | None = None,
    xmlfile: str | os.PathLike[str] | None = None,
    lj_treatment: str | None = None,
    skip_xmlfile: bool = False,
) -> tuple[openmm.app.ForceField, openmm.app.Topology, Fragment]:
    """Solvate a small molecule in a water box (Amber- or CHARMM-style forcefield XML)."""
    if solvent_boxdims is None:
        solvent_boxdims = [70.0, 70.0, 70.0]
    logger.info(main_header("SmallMolecule Solvator"))
    try:
        logger.info("Imported OpenMM library version: %s", openmm.__version__)
    except ImportError:
        raise ImportError(
            "OpenMM requires installing the OpenMM package. Try: conda install -c conda-forge openmm  \
            Also see http://docs.openmm.org/latest/userguide/application.html"
        ) from None

    if fragment is None:
        raise InputError("No fragment object provided. Exiting.")

    charge, mult = check_charge_mult(charge, mult, "QM", fragment, "solvate_small_molecule")

    if xmlfile is None and skip_xmlfile is False:
        raise InputError(
            "No xmlfile was provided. You must provide one.\nIf you need a forcefield XML for the solute, build one "
            'with forcefill:\n    from forcefill import build_ligand_xml\n    build_ligand_xml({"LIG": "solute.sdf"}, '
            '"lig_ff.xml")'
        )

    # Read XML-file and check for LJ treatment
    if skip_xmlfile is False:
        logger.debug("Checking xmlfile for LJ treatment")
        if pygrep('coulomb14scale="0.83333', xmlfile):
            logger.info("Found Amber-style scaling parameter.")
            lj_treatment = "amber"
        elif pygrep("LennardJonesForce", xmlfile):
            logger.info("Found CHARMM-style format.")
            lj_treatment = "charmm"
        else:
            raise InputError(
                "Unknown LJ14 scaling type in XML-file: neither CHARMM nor Amber format was recognized\nSolvation "
                "requires an Amber- or CHARMM-style forcefield XML-file"
            )

        logger.info("LJ_treatment: %s", lj_treatment)

    if watermodel in {"tip3p", "TIP3P"}:
        logger.info("Using watermodel=TIP3P")
        if lj_treatment == "amber":
            waterxmlfile = "amber14/tip3p.xml"
        elif lj_treatment == "charmm":
            waterxmlfile = "charmm36/water.xml"
        else:
            raise InputError(f"Unsupported LJ_treatment ({lj_treatment}): must be 'amber' or 'charmm'")
    else:
        raise InputError("Only TIP3P water supported for now")

    if skip_xmlfile is True:
        logger.debug("Creating forcefield using XML-files: %s", waterxmlfile)
        forcefield = openmm_app.forcefield.ForceField(*[waterxmlfile])
    else:
        logger.debug("Creating forcefield using XML-files: %s %s", xmlfile, waterxmlfile)
        forcefield = openmm_app.forcefield.ForceField(*[xmlfile, waterxmlfile])

    if skip_xmlfile is True:
        atomnames = [el + "Y" + str(i) for i, el in enumerate(fragment.elems)]
        pdbfile = write_pdbfile(fragment, outputname="smallmol", dummyname="LIG", atomnames=atomnames)
    elif pygrep("<Bond", xmlfile):
        logger.info("XML-file contains bonded parameters. Writing PDB-file with connectivity.")
        xyzfile = Fragment.write_xyzfile(fragment, xyzfilename="smallmol.xyz")
        pdbfile = xyz_to_pdb_with_connectivity(xyzfile)
    else:
        atomnames = [el + "Y" + str(i) for i, el in enumerate(fragment.elems)]
        pdbfile = write_pdbfile(fragment, outputname="smallmol", dummyname="LIG", atomnames=atomnames)

    pdb = openmm_app.PDBFile(pdbfile)
    logger.debug("Loading Modeller.")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    logger.info(f"Modeller topology has {modeller.topology.getNumResidues()} residues.")

    logger.debug("Adding solvent, watermodel: %s", watermodel)

    # NOTE: modeller.addsolvent will automatically add ions to neutralize any excess charge
    logger.warning("Modeller will automatically neutralize system with ions if system is charged")
    if solvent_boxdims is not None:
        logger.info(f"Solvent boxdimension provided: {solvent_boxdims} Å")
        modeller.addSolvent(
            forcefield,
            boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1], solvent_boxdims[2]) * openmm_unit.angstrom,
        )

    logger.info("Creating PDB-file: system_aftersolvent.pdb")
    write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "system_aftersolvent.pdb")
    print_systemsize(modeller)

    newfragment = Fragment(pdbfile="system_aftersolvent.pdb")
    newfragment.write_xyzfile(xyzfilename="system_aftersolvent.xyz")
    logger.info("Creating XYZ-file: system_aftersolvent.xyz")
    logger.info("\n\nTo use this system setup to define a future OpenMMTheory object you can  do:\n")

    logger.info(
        f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="system_aftersolvent.pdb", '
        f"periodic=True, rigidwater=True)"
    )

    return forcefield, modeller.topology, newfragment


def find_alternate_locations_residues(pdbfile: str | os.PathLike[str], use_higher_occupancy: bool = False) -> str:
    if use_higher_occupancy is True:
        logger.debug("Will keep higher occupancy atoms for alternate locations")

    pdb_atomlines = []
    bad_resids_dict = {}

    altloc_dict = {}

    with open(pdbfile) as pfile:
        for line in pfile:
            if line.startswith(("ATOM", "HETATM")):
                altloc = line[16]
                if altloc != " ":
                    chain = line[21:22]
                    if chain not in bad_resids_dict:
                        bad_resids_dict[chain] = []
                    resid = int(line[22:26].replace(" ", ""))
                    resname = line[17:20].replace(" ", "")
                    residue = resname + str(resid)
                    atomname = line[12:16].replace(" ", "")
                    occupancy = float(line[54:60])
                    # Atomstring contains only the atom-information (not alt-location label)
                    atomstring = chain + "_" + resname + "_" + str(resid) + "_" + atomname
                    if residue not in bad_resids_dict[chain]:
                        bad_resids_dict[chain].append(residue)
                    altloc_dict[(atomstring, altloc)] = [altloc, occupancy, line]
                    # Adding atomstring to list as a marker
                    if ["REPLACE_", atomstring] not in pdb_atomlines:
                        pdb_atomlines.append(["REPLACE_", atomstring])
                else:
                    pdb_atomlines.append(line)
            else:
                pdb_atomlines.append(line)

    def find_index_of_sublist_with_max_col(rows: Sequence[Sequence[Any]], index: int) -> int | None:
        max_val = 0
        result = None
        for i, s in enumerate(rows):
            if s[index] > max_val:
                max_val = s[index]
                result = i
        return result

    finalpdblines = []
    for pdbline in pdb_atomlines:
        if pdbline[0] == "REPLACE_":
            logger.info("Alternate locations for atom: %s", pdbline[1])
            options = []
            for i, j in altloc_dict.items():
                if i[0] == pdbline[1]:
                    options.append([j[0], j[1], j[2]])
            for option_row in options:
                pdblinestring = "".join(map(str, option_row[2:]))
                logger.info("%s", pdblinestring)
            ind = find_index_of_sublist_with_max_col(options, 1)
            fline = options[ind][2][:16] + " " + options[ind][2][16 + 1 :]
            logger.info(f"Choosing line with occupancy {options[ind][1]}.")
            logger.info("%s", "-" * 90)
            if fline not in finalpdblines:
                finalpdblines.append(fline)
        else:
            finalpdblines.append(pdbline)

    if len(bad_resids_dict) > 0:
        logger.warning("Found residues in the PDB file with alternate-location labels (multiple occupancies):")
        for chain, residues in bad_resids_dict.items():
            logger.info(f"\nChain {chain}:")
            for res in residues:
                logger.info("%s", res)
        logger.warning("These residues should be inspected and fixed in the PDB file before continuing")
        if use_higher_occupancy is True:
            logger.warning("The higher-occupancy location option was selected, so continuing")
            write_list_to_file(finalpdblines, "system_afteratlocfixes.pdb", separator="")
            return "system_afteratlocfixes.pdb"
        raise InputError(
            "You should delete either the labelled A or B location of the residue-atom/atoms and then remove the "
            "A/B label from column 17 in the file\nAlternatively, you can choose use_higher_occupancy=True keyword "
            "in OpenMM_Modeller and openmmqmmm will keep the higher occupied form and go on \nMake sure that there "
            "is always an A or B form present.\nExiting."
        )

    return pdbfile


def merge_pdb_files(
    pdbfile_1: str | os.PathLike[str],
    pdbfile_2: str | os.PathLike[str],
    outputname: str | os.PathLike[str] = "merged.pdb",
) -> str | os.PathLike[str]:
    """Merge two PDB files into one (e.g. protein plus ligand)."""
    pdb1 = openmm.app.PDBFile(pdbfile_1)
    pdb2 = openmm.app.PDBFile(pdbfile_2)

    modeller = openmm.app.Modeller(pdb1.topology, pdb1.positions)  # Add pdbfile1
    modeller.add(pdb2.topology, pdb2.positions)  # Add pdbfile2
    mergedPositions = modeller.positions  # merging positions

    write_pdbfile_openmm_topology(modeller.topology, mergedPositions, outputname)
    logger.info("Wrote merged PDB file: %s", outputname)

    return outputname


def _parameterize_nonstandard_residues(
    pdbfile_for_modeller: str | os.PathLike[str],
    *,
    forcefield_obj: openmm.app.ForceField,
    xmlfile: str | None,
    waterxmlfile: str | None,
    extraxmlfile: str | None,
    ligand_files: Mapping[str, str | os.PathLike[str]] | None,
    net_charges: Mapping[str, int] | None,
    ligand_backend: str,
) -> str | None:
    """Generate a forcefill XML for unmatched residues and load it into the forcefield object."""
    try:
        from forcefill import build_forcefield_xml
    except ImportError:
        raise MissingDependencyError(
            "parameterize_nonstandard=True requires the forcefill package, which is not on PyPI.\n"
            "The installers in build_tools/ clone it next to this repository and install it editable; by hand:\n"
            "  git clone https://github.com/LouieSlocombe/forcefill.git && pip install --no-deps -e forcefill"
        ) from None

    base_forcefield = [x for x in (xmlfile, extraxmlfile, waterxmlfile) if x is not None]
    logger.debug("\nNow parameterizing non-standard residues with forcefill")
    logger.info("Base forcefield XML files defining the standard residues: %s", base_forcefield)
    logger.info("NOTE: non-standard residues must carry explicit hydrogens and CONECT records in the PDB-file")
    try:
        result = build_forcefield_xml(
            pdbfile_for_modeller,
            "nonstandard_ff.xml",
            base_forcefield=base_forcefield,
            residue_files=ligand_files,
            net_charges=net_charges,
            backend=ligand_backend,
            workdir="forcefill_files",
        )
    except (ValueError, RuntimeError) as e:
        raise InputError(f"forcefill could not parameterize the non-standard residues:\n{e}") from e
    # forcefill logs on its own logger, which configure_logging() does not wire up:
    # repeat the outcome here so it lands in the calculation record.
    for name, reason in result.skipped.items():
        logger.warning("forcefill skipped %s: %s", name, reason)
    if result.forcefield_xml is None:
        logger.info("All residues matched the forcefield; nothing to parameterize.")
        return None
    logger.info("forcefill parameterized %s -> %s", result.parameterized, result.forcefield_xml)
    forcefield_obj.loadFile(result.forcefield_xml)
    return result.forcefield_xml


def _log_residue_table(
    modeller_residues: Sequence[openmm.app.topology.Residue],
    residue_variants: Mapping[str, Mapping[int, str]],
) -> list[str | None]:
    current_chainindex = 0
    # Also using loop to get residue_states list that we pass on to modeller.addHydrogens
    residue_states = []
    for each_residue in modeller_residues:
        if each_residue.chain.index != current_chainindex:
            logger.info("%s", "--" * 30)
        resid = each_residue.index
        resid_in_chain = int(each_residue.id)
        resname = each_residue.name
        chain = each_residue.chain
        current_chainindex = each_residue.chain.index
        if chain.id in residue_variants:
            if resid_in_chain in residue_variants[chain.id]:
                residue_states.append(residue_variants[chain.id][resid_in_chain])
                FLAGLABEL = f"-- This residue will be changed to: {residue_variants[chain.id][resid_in_chain]} --"
            else:
                residue_states.append(None)  # Note: we add None since we don't want to influence addHydrogens
                FLAGLABEL = ""
        else:
            residue_states.append(None)  # Note: we add None since we don't want to influence addHydrogens
            FLAGLABEL = ""

        logger.info(f"  {resid:<12}{resname:<13}{chain.index:<13}{chain.id:<13}{resid_in_chain:<13}       {FLAGLABEL}")
    return residue_states


def _add_solvent_or_membrane(
    modeller: openmm.app.Modeller,
    *,
    forcefield_obj: openmm.app.ForceField,
    implicit: bool,
    implicit_solvent_xmlfile: str | None,
    membrane: bool,
    membrane_lipidtype: str,
    membrane_padding: float,
    membrane_center_z: float,
    modeller_solvent_name: str,
    watermodel: str | None,
    waterxmlfile: str | None,
    solvent_boxdims: Sequence[float] | None,
    solvent_padding: float,
    ionicstrength: float,
    pos_iontype: str,
    neg_iontype: str,
    residue_templates: Mapping[openmm.app.topology.Residue, str],
) -> tuple[bool, Fragment, str | None]:
    if implicit is True:
        periodic = False
        logger.info("We are doing implicit solvation")
        logger.debug("Setting periodic to False")
        logger.info("Available implicit solvent models:")
        logger.info(
            "implicit/gbn2.xml, implicit/hct.xml, implicit/obc1.xml, implicit/obc2.xml, implicit/gbn.xml, "
            "implicit/gbn2.xml"
        )
        fragment = Fragment(pdbfile="system_afterH.pdb")
        if implicit_solvent_xmlfile is None:
            logger.debug("No XMLfile for implicit water selected (implicit_solvent_xmlfile keyword)")
            logger.info("Choosing : implicit/obc2.xml")
            implicit_solvent_xmlfile = "implicit/obc2.xml"
            waterxmlfile = implicit_solvent_xmlfile
    elif membrane is True:
        logger.info("We are doing membrane-addition and solvation")
        logger.debug("Setting periodic to True")
        periodic = True
        logger.debug("Adding membrane-lipid type (membrane_lipidtype keyword): %s", membrane_lipidtype)
        logger.debug("Adding solvent, modeller_solvent_name: %s", modeller_solvent_name)
        logger.info("Actual solvent name: %s", watermodel)
        logger.info("Actual solvent file: %s", waterxmlfile)
        modeller.addMembrane(
            forcefield_obj,
            lipidType=membrane_lipidtype,
            positiveIon=pos_iontype,
            negativeIon=neg_iontype,
            ionicStrength=ionicstrength * openmm_unit.molar,
            neutralize=True,
            membraneCenterZ=membrane_center_z * openmm_unit.angstrom,
            minimumPadding=membrane_padding * openmm_unit.angstrom,
        )

        write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "system_aftersolvent_ions.pdb")
        # NOTE: Had to remove separate ion-add step due to OpenMM 8.1 change
        print_systemsize(modeller)
        fragment = Fragment(pdbfile="system_aftersolvent_ions.pdb")
    else:
        logger.info("We are doing explicit solvation")
        logger.debug("Setting periodic to True")
        periodic = True
        logger.debug("Adding solvent, modeller_solvent_name: %s", modeller_solvent_name)
        logger.info("Actual solvent name: %s", watermodel)
        logger.info("Actual solvent file: %s", waterxmlfile)
        if solvent_boxdims is not None:
            logger.info(f"Solvent boxdimension provided: {solvent_boxdims} Å")
            logger.debug("Adding ionic strength: %s M, using ions: %s and %s", ionicstrength, pos_iontype, neg_iontype)
            modeller.addSolvent(
                forcefield_obj,
                boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1], solvent_boxdims[2]) * openmm_unit.angstrom,
                neutralize=True,
                positiveIon=pos_iontype,
                negativeIon=neg_iontype,
                ionicStrength=ionicstrength * openmm_unit.molar,
                residueTemplates=residue_templates,
            )
        else:
            logger.info(f"Using solvent padding (solvent_padding=X keyword): {solvent_padding} Å")
            logger.debug("Adding ionic strength: %s M, using ions: %s and %s", ionicstrength, pos_iontype, neg_iontype)
            logger.info("residueTemplates: %s", residue_templates)
            modeller.addSolvent(
                forcefield_obj,
                padding=solvent_padding * openmm_unit.angstrom,
                model=modeller_solvent_name,
                neutralize=True,
                positiveIon=pos_iontype,
                negativeIon=neg_iontype,
                ionicStrength=ionicstrength * openmm_unit.molar,
                residueTemplates=residue_templates,
            )
        write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "system_aftersolvent_ions.pdb")

        # NOTE: Had to remove separate ion-add step due to OpenMM 8.1 change
        print_systemsize(modeller)
        fragment = Fragment(pdbfile="system_aftersolvent_ions.pdb")
    return periodic, fragment, waterxmlfile


def _log_output_files_and_usage(
    *,
    systemxmlfile: str | os.PathLike[str],
    xmlfile: str | None,
    waterxmlfile: str | None,
    extraxmlfile: str | None,
    nonstandard_xmlfile: str | None,
    residuetemplate_choice: Mapping[str, str] | None,
    periodic: bool,
) -> None:
    logger.info("\n\nFiles written to disk:")
    logger.info("system_afteratlocfixes.pdb")
    logger.info("system_afterfixes.pdb")
    logger.info("system_afterfixes2.pdb")
    logger.info("system_afterH.pdb")
    logger.info("system_aftersolvent.pdb")
    logger.info("system_afterions.pdb and finalsystem.pdb (same)")
    if nonstandard_xmlfile is not None:
        logger.info("%s (forcefill-generated ligand parameters - keep with the other XML files)", nonstandard_xmlfile)
    logger.info("\nFinal files:")
    logger.info("finalsystem.pdb  (PDB file)")
    logger.info("finalsystem.cif  (PDBx/mmCIF file)")
    logger.info("finalsystem.frag  (fragment file)")
    logger.info("finalsystem.xyz   (XYZ coordinate file)")
    logger.info(f"{systemxmlfile}   (System XML file)")
    logger.info("\n\n OpenMM_Modeller done! System has been fully set up!\n")
    logger.warning("Strongly recommended: Check finalsystem.pdb carefully for correctness!")
    logger.info("\nTo use this system setup to define a future OpenMMTheory object you can either do:\n")

    xml_list = ", ".join(f'"{x}"' for x in (xmlfile, waterxmlfile, extraxmlfile, nonstandard_xmlfile) if x is not None)
    logger.info("1. Define using separate forcefield XML files and PDB-file (for topology):")
    logger.info(f'omm = OpenMMTheory(xmlfiles=[{xml_list}], pdbfile="finalsystem.pdb", periodic={periodic})')
    logger.info("2. Define using separate forcefield XML files and PDBx/mmCIF file (instead of PDB):")
    logger.info(f'omm = OpenMMTheory(xmlfiles=[{xml_list}], pdbxfile="finalsystem.cif", periodic={periodic})')
    logger.info(
        "3. Use forcefield object file :\n %s",
        f'omm = OpenMMTheory(topoforce=True, forcefield=forcefield_object, pdbfile="finalsystem.pdb", '
        f"topology=modeller.topology, periodic={periodic})",
    )
    if residuetemplate_choice is not None:
        logger.warning(
            "A residuetemplate_choice option was provided to OpenMM_Modeller. This means that you will have "
            "to provide this also when defining an OpenMMTheory object."
        )
        logger.info(
            f'E.g. like this: omm = OpenMMTheory(xmlfiles=[{xml_list}], pdbfile="finalsystem.pdb", '
            f"periodic={periodic}, residuetemplate_choice={residuetemplate_choice})"
        )
