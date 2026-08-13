import contextlib
import logging
import os
import time

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
    FileFormatError,
    InputError,
    MissingDependencyError,
)
from openmmqmmm.openbabel import xyz_to_pdb_with_connectivity
from openmmqmmm.openmm.theory import OpenMMTheory
from openmmqmmm.singlepoint import single_point
from openmmqmmm.utils import (
    find_replace_string_in_file,
    log_time_since,
    main_header,
    pygrep,
    writelisttofile,
)

logger = logging.getLogger(__name__)


def print_systemsize(modeller):
    logger.info(f"System size: {len(modeller.getPositions())} atoms\n")


def openmm_minimize(
    fragment=None,
    theory=None,
    maxiter=1000,
    tolerance=1,
    enforce_periodic_box=True,
    traj_frequency=100,
    use_reporter=True,
) -> "Fragment | bool":
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
        logger.warning("Autoconstraints have not been set in OpenMMTheory object definition.")
        logger.info("This means that by default no bonds are constrained in the optimization.")
        logger.info("Will continue...")
    if (openmmobject.rigidwater is True and len(openmmobject.user_frozen_atoms) != 0) or (
        openmmobject.autoconstraints is not None and len(openmmobject.user_frozen_atoms) != 0
    ):
        logger.warning(
            "WARNING: Frozen_atoms options selected but there are general constraints defined in "
            "the OpenMM object (either rigidwater=True or autoconstraints is not None)\n"
            "OpenMM will crash if constraints and frozen atoms involve the same atoms"
        )

    openmmobject.set_simulation_parameters(timestep=0.001, temperature=1, integrator="VerletIntegrator")

    simulation = openmmobject.create_simulation()

    logger.info("Simulation created.")

    # New in OpenMM 8.1: reporters for minimizer
    if version.parse(openmm.__version__) >= version.parse("8.1") and use_reporter is True:

        class Reporter(openmm.openmm.MinimizationReporter):
            def report(self, iteration, x, grad, args):
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

            def write_traj(self, x):
                if self.totaliter % traj_frequency == 0:
                    logger.info("%s", "-" * 40)
                    logger.info("Now writing to trajectory file")
                    logger.info("%s", "-" * 40)
                    pos = 10 * np.array(x).reshape(-1, 3)
                    write_xyzfile(fragment.elems, pos, "OpenMMOpt_traj", writemode="a")

            def print_energy(self, args):
                system_energy = args["system energy"] / openmmqmmm.constants.hartokj
                restraint_energy = args["restraint energy"] / openmmqmmm.constants.hartokj
                logger.info("System energy: %s", system_energy)
                logger.info("Restraint energy: %s", restraint_energy)
                logger.info("Restraint strength: %s", args["restraint strength"])
                logger.info("Max constraint error: %s", args["max constraint error"])

            def get_forces(self, grad):
                g = np.array(grad).reshape(-1, 3)  # To confirm
                kjmolnm_to_atomic_factor = -49614.752589207
                self.forces_init = g / kjmolnm_to_atomic_factor
                self.rms_force = np.sqrt(sum(n * n for n in self.forces_init.flatten()) / len(forces_init.flatten()))
                self.max_force = self.forces_init.max()

            def print_forces(self):
                logger.info(f"RMS force (w restraints): {self.rms_force} Eh/Bohr")
                logger.info(f"Max force (w restraints): {self.max_force} Eh/Bohr")
                logger.info("")

        reporter = Reporter()

    logger.info("Now adding coordinates")
    openmmobject.set_positions(fragment.coords, simulation)

    logger.info("")
    state = simulation.context.getState(getEnergy=True, getForces=True, enforcePeriodicBox=enforce_periodic_box)
    potE_init = (
        state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system) / openmmqmmm.constants.hartokj
    )
    logger.info(f"Initial potential energy is: {potE_init} Eh")
    kjmolnm_to_atomic_factor = -49614.752589207
    forces_init = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_init.flatten()) / len(forces_init.flatten()))
    logger.info(f"Initial RMS force: {rms_force} Eh/Bohr (w/o restraints)")
    logger.info(f"Initial Max force: {forces_init.max()} Eh/Bohr (w/o restraints)")
    logger.info("")
    logger.info("Starting minimization.")
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

    logger.info("")
    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=enforce_periodic_box
    )
    logger.info(
        "%s",
        f"Final Potential energy is: "
        f"{state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system) / openmmqmmm.constants.hartokj} "
        f"Eh",
    )
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
    logger.info("")
    logger.info("Updating coordinates in fragment.")
    fragment.coords = newcoords

    logger.info("All Done!")
    log_time_since(module_init_time, "OpenMM_Opt")

    return fragment


def openmm_modeller(
    pdbfile=None,
    forcefield_object=None,
    forcefield=None,
    xmlfile=None,
    waterxmlfile=None,
    watermodel=None,
    ph=7.0,
    solvent_padding=10.0,
    solvent_boxdims=None,
    extraxmlfile=None,
    residue_variants=None,
    ionicstrength=0.1,
    pos_iontype="Na+",
    neg_iontype="Cl-",
    use_higher_occupancy=False,
    platform="CPU",
    use_pdbfixer=True,
    implicit=False,
    implicit_solvent_xmlfile=None,
    membrane=False,
    membrane_lipidtype="POPC",
    membrane_padding=10.0,
    membrane_center_z=0.0,
    residuetemplate_choice=None,
) -> tuple:
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
    try:
        import pdbfixer
    except ImportError:
        raise MissingDependencyError(
            "Problem importing pdbfixer. Install first via conda:\nconda install -c conda-forge pdbfixer"
        ) from None

    if pdbfile is None:
        raise InputError("You must provide a pdbfile keyword argument")

    if residue_variants is None:
        residue_variants = {}

    if forcefield is not None:
        logger.info("Forcefield: %s", forcefield)
        if forcefield in {"Amber99", "Amber99sb"}:
            xmlfile = "amber99sb.xml"
        elif forcefield == "Amber99sb-ildn":
            xmlfile = "amber99sbildn.xml"
        elif forcefield == "Amber96":
            xmlfile = "amber96.xml"
        elif forcefield == "Amber03":
            xmlfile = "amber03.xml"
        elif forcefield == "Amber10":
            xmlfile = "amber10.xml"
        elif forcefield == "Amber14":
            xmlfile = "amber14-all.xml"
        elif forcefield == "CHARMM36":
            xmlfile = "charmm36.xml"
        elif forcefield == "CHARMM2013":
            xmlfile = "charmm_polar_2013.xml"
        elif forcefield == "Amoeba2013":
            xmlfile = "amoeba2013.xml"
        elif forcefield == "Amoeba2009":
            xmlfile = "amoeba2009.xml"
        else:
            raise InputError("Unknown forcefield")

        if "CHARMM" in forcefield:
            # Using specific CHARMM36 version of TIP3P
            if watermodel is None:
                logger.info("No watermodel selected.")
                if waterxmlfile is None:
                    logger.info("No waterxmlfile selected either")
                    logger.info("Selecting automatically recommended CHARMM-style TIP3P")
                    watermodel = "tip3p"

            logger.info("watermodel: %s", watermodel)
            if watermodel.lower() == "tip3p":
                modeller_solvent_name = "tip3p"  # Used when adding solvent
                waterxmlfile = "charmm36/water.xml"
            logger.info("Waterxmlfile selected: %s", waterxmlfile)

        if "Amber" in forcefield:
            if watermodel is None:
                logger.info("No watermodel selected.")
                if waterxmlfile is None:
                    logger.info("No waterxmlfile selected either")
                    logger.info("Selecting automatically recommended TIP3P-4B (watermodel='tip3pfb')")
                    logger.info("This is a reparameterized version of TIP3P")
                    watermodel = "tip3pfb"
            logger.info("watermodel: %s", watermodel)
            # Using specific Amber FB version of TIP3P
            if watermodel.lower() == "tip3pfb" or watermodel.lower() == "tip3p-fb":
                modeller_solvent_name = "tip3p"  # Used when adding solvent
                waterxmlfile = "amber14/tip3pfb.xml"  # NOTE: this is not actually TIP3P but a reparaterized version
            elif watermodel.lower() == "tip3p":
                modeller_solvent_name = "tip3p"
                waterxmlfile = "amber14/tip3p.xml" if forcefield == "Amber14" else "tip3p.xml"
            logger.info("Waterxmlfile selected: %s", waterxmlfile)

    if xmlfile is not None:
        logger.info("XMfile: %s", xmlfile)
        logger.info("Water model: %s", watermodel)
        logger.info("Water xmlfile: %s", waterxmlfile)
        if extraxmlfile is not None:
            logger.info("Using extra XML file: %s", extraxmlfile)
            if os.path.isfile(extraxmlfile) is not True:
                raise InputError(f"File {extraxmlfile} can not be found. Exiting.")
        logger.info("Now creating forcefield object")
        if extraxmlfile is None and waterxmlfile is None:
            forcefield_obj = openmm_app.forcefield.ForceField(xmlfile)
        elif extraxmlfile is not None and waterxmlfile is None:
            forcefield_obj = openmm_app.forcefield.ForceField(xmlfile, extraxmlfile)
        elif extraxmlfile is None and waterxmlfile is not None:
            forcefield_obj = openmm_app.forcefield.ForceField(xmlfile, waterxmlfile)
        elif extraxmlfile is not None and waterxmlfile is not None:
            forcefield_obj = openmm_app.forcefield.ForceField(xmlfile, extraxmlfile, waterxmlfile)
    elif forcefield_object is not None:
        logger.info("Using forcefield object provided")
        forcefield_obj = forcefield_object

        if watermodel is not None or waterxmlfile is not None:
            logger.warning("Ignoring watermodel/waterxmlfile: a forcefield_object was supplied")
    else:
        raise InputError("You must provide a forcefield name, forcefieldobject or xmlfile keywords!")

    logger.info("PDBfile: %s", pdbfile)
    logger.info("pH: %s", ph)
    logger.info("User-provided dictionary of residue_variants: %s", residue_variants)
    logger.info("\nNow checking PDB-file for alternate locations, i.e. multiple occupancies:\n")

    # Check PDB-file whether it contains alternate locations of residue atoms (multiple occupations)
    # Default behaviour:
    # - if no multiple occupancies return input PDBfile and go on
    # - if multiple occupancies, print list of residues and tell user to fix them. Exiting
    # - if use_higher_occupancy is set to True, user higher occupancy location, write new PDB_file and use
    pdbfile = find_alternate_locations_residues(pdbfile, use_higher_occupancy=use_higher_occupancy)

    logger.info("Using PDB-file %s", pdbfile)

    # Fix basic mistakes in PDB by PDBFixer
    # This will e.g. fix bad terminii
    if use_pdbfixer is True:
        logger.info("\nRunning PDBFixer")
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
        logger.info("PDBFixer done.")
        logger.warning(
            "Warning: PDBFixer can create unreasonable orientations of residues if residues "
            "are missing or multiple occupancies are present.\n         "
            "You should inspect the created PDB-file to be sure."
        )
        logger.info("Wrote PDBfile: system_afterfixes.pdb")
        pdbfile_for_modeller = "system_afterfixes.pdb"
    else:
        logger.info("Skipping PDBFixer")
        pdbfile_for_modeller = pdbfile

    pdb = openmm_app.PDBFile(pdbfile_for_modeller)
    logger.info("\n\nNow loading Modeller.")
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
        logger.info(
            f"This is chain {chain_x.index}, it has {len(chain_x._residues)} residues and they are: "
            f"{chain_x._residues}\n"
        )
    logger.info("\n")

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

    with open("system_afterfixes2.pdb", "w") as pdbfh:
        openmm_app.PDBFile.writeFile(modeller.topology, modeller.positions, pdbfh)

    if len(residue_states) != numresidues:
        raise InputError("residue_states != numresidues. Something went wrong")

    # This is were missing residue/atom errors will come
    logger.info("")
    logger.info("Adding hydrogens for pH: %s", ph)
    logger.warning("OpenMM Modeller will fail in this step if residue information is missing")
    logger.info("residue_states: %s", residue_states)

    residueTemplates = {}  # initisal
    if residuetemplate_choice is not None:
        logger.info("Found user-specified residuetemplate_choice")
        logger.info("Will generate residueTemplates based on residuetemplate_choice: %s", residuetemplate_choice)
        logger.info("Note: residuetemplate_choice should be a dict like this: residuetemplate_choice={'FER':'FE2'}   ")
        residueTemplates = {}
        for resname, choice in residuetemplate_choice.items():
            residueTemplates = {res: choice for res in modeller.topology.residues() if res.name == resname}
    logger.info("residueTemplates: %s", residueTemplates)

    logger.info("\nNow checking if we have problems with unmatched residues")
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
    logger.info("No problem with unmatched residues found. Continuing")

    try:
        logger.info("residueTemplates: %s", residueTemplates)
        modeller.addHydrogens(forcefield_obj, pH=ph, variants=residue_states, residueTemplates=residueTemplates)
    except ValueError as errormessage:
        logger.error("\nOpenMM modeller.addHydrogens signalled a ValueError")
        logger.info(
            "This is a common error and suggests a problem in PDB-file or missing residue information in the "
            "forcefield."
        )
        logger.info(
            "Non-standard inorganic/organic residues require providing an additional XML-file via extraxmlfile= option"
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

    if implicit is True:
        periodic = False
        logger.info("We are doing implicit solvation")
        logger.info("Setting periodic to False")
        logger.info("Available implicit solvent models:")
        logger.info(
            "implicit/gbn2.xml, implicit/hct.xml, implicit/obc1.xml, implicit/obc2.xml, implicit/gbn.xml, "
            "implicit/gbn2.xml"
        )
        fragment = Fragment(pdbfile="system_afterH.pdb")
        if implicit_solvent_xmlfile is None:
            logger.info("No XMLfile for implicit water selected (implicit_solvent_xmlfile keyword)")
            logger.info("Choosing : implicit/obc2.xml")
            implicit_solvent_xmlfile = "implicit/obc2.xml"
            waterxmlfile = implicit_solvent_xmlfile
    elif membrane is True:
        logger.info("We are doing membrane-addition and solvation")
        logger.info("Setting periodic to True")
        periodic = True
        logger.info("Adding membrane-lipid type (membrane_lipidtype keyword): %s", membrane_lipidtype)
        logger.info("Adding solvent, modeller_solvent_name: %s", modeller_solvent_name)
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
        logger.info("Setting periodic to True")
        periodic = True
        logger.info("Adding solvent, modeller_solvent_name: %s", modeller_solvent_name)
        logger.info("Actual solvent name: %s", watermodel)
        logger.info("Actual solvent file: %s", waterxmlfile)
        if solvent_boxdims is not None:
            logger.info(f"Solvent boxdimension provided: {solvent_boxdims} Å")
            logger.info(f"Adding ionic strength: {ionicstrength} M, using ions: {pos_iontype} and {neg_iontype}")
            modeller.addSolvent(
                forcefield_obj,
                boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1], solvent_boxdims[2]) * openmm_unit.angstrom,
                neutralize=True,
                positiveIon=pos_iontype,
                negativeIon=neg_iontype,
                ionicStrength=ionicstrength * openmm_unit.molar,
                residueTemplates=residueTemplates,
            )
        else:
            logger.info(f"Using solvent padding (solvent_padding=X keyword): {solvent_padding} Å")
            logger.info(f"Adding ionic strength: {ionicstrength} M, using ions: {pos_iontype} and {neg_iontype}")
            logger.info("residueTemplates: %s", residueTemplates)
            modeller.addSolvent(
                forcefield_obj,
                padding=solvent_padding * openmm_unit.angstrom,
                model=modeller_solvent_name,
                neutralize=True,
                positiveIon=pos_iontype,
                negativeIon=neg_iontype,
                ionicStrength=ionicstrength * openmm_unit.molar,
                residueTemplates=residueTemplates,
            )
        write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "system_aftersolvent_ions.pdb")

        # NOTE: Had to remove separate ion-add step due to OpenMM 8.1 change
        print_systemsize(modeller)
        fragment = Fragment(pdbfile="system_aftersolvent_ions.pdb")

    write_pdbfile_openmm_topology(modeller.topology, modeller.positions, "finalsystem.pdb")
    write_pdbxfile_openmm_topology(modeller.topology, modeller.positions, "finalsystem.cif")
    fragment.print_system(filename="finalsystem.frag")
    fragment.write_xyzfile(xyzfilename="finalsystem.xyz")

    logger.info("\nOpenMM_Modeller used the following XML-files to define system:")
    logger.info("General forcefield XML file: %s", xmlfile)
    logger.info("Solvent forcefield XML file: %s", waterxmlfile)
    logger.info("Extra forcefield XML file: %s", extraxmlfile)

    # Creating new OpenMM object from forcefield so that we can write out system XMLfile
    logger.info("Creating OpenMMTheory object")
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

    logger.info("\n\nFiles written to disk:")
    logger.info("system_afteratlocfixes.pdb")
    logger.info("system_afterfixes.pdb")
    logger.info("system_afterfixes2.pdb")
    logger.info("system_afterH.pdb")
    logger.info("system_aftersolvent.pdb")
    logger.info("system_afterions.pdb and finalsystem.pdb (same)")
    logger.info("\nFinal files:")
    logger.info("finalsystem.pdb  (PDB file)")
    logger.info("finalsystem.cif  (PDBx/mmCIF file)")
    logger.info("finalsystem.frag  (fragment file)")
    logger.info("finalsystem.xyz   (XYZ coordinate file)")
    logger.info(f"{systemxmlfile}   (System XML file)")
    logger.info("\n\n OpenMM_Modeller done! System has been fully set up!\n")
    logger.warning("Strongly recommended: Check finalsystem.pdb carefully for correctness!")
    logger.info("\nTo use this system setup to define a future OpenMMTheory object you can either do:\n")

    logger.info("1. Define using separate forcefield XML files and PDB-file (for topology):")
    if extraxmlfile is None:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="finalsystem.pdb", '
            f"periodic={periodic})"
        )
    else:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}", "{extraxmlfile}"], '
            f'pdbfile="finalsystem.pdb", periodic={periodic})'
        )
    logger.info("2. Define using separate forcefield XML files and PDBx/mmCIF file (instead of PDB):")
    if extraxmlfile is None:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbxfile="finalsystem.cif", '
            f"periodic={periodic})"
        )
    else:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}", "{extraxmlfile}"], '
            f'pdbxfile="finalsystem.cif", periodic={periodic})'
        )
    logger.info(
        "3. Use forcefield object file :\n %s",
        f'omm = OpenMMTheory(topoforce=True, forcefield=forcefield_object, pdbfile="finalsystem.pdb", '
        f"topology=modeller.topology, periodic={periodic})",
    )
    logger.info("")
    logger.info("")
    if residuetemplate_choice is not None:
        logger.warning(
            "Warning: A residuetemplate_choice option was provided to OpenMM_Modeller. This means that you will have "
            "to provide this also when defining an OpenMMTheory object."
        )
        logger.info(
            f'E.g. like this: omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="finalsystem.pdb", '
            f"periodic={periodic}, residuetemplate_choice={residuetemplate_choice})"
        )
    logger.info("\nNow running single-point MM job to check for bad contacts")
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


def write_pdbfile_openmm_topology(topology, positions, filename, connectivity_dict=None):
    if connectivity_dict is not None:
        logger.info("Connectivity passed to write_pdbfile_openMM")
        openmm_add_bonds_to_topology(topology, connectivity_dict)

    with open(filename, "w") as pdbfh:
        openmm.app.PDBFile.writeFile(topology, positions, file=pdbfh)
    logger.info("Wrote PDB-file: %s", filename)


def write_pdbxfile_openmm_topology(topology, positions, filename, connectivity_dict=None):
    if connectivity_dict is not None:
        logger.info("Connectivity passed to write_pdbxfile_openMM")
        openmm_add_bonds_to_topology(topology, connectivity_dict)

    with open(filename, "w") as pdbfh:
        openmm.app.PDBxFile.writeFile(topology, positions, file=pdbfh)
    logger.info("Wrote PDBx-file: %s", filename)


def openmm_add_bonds_to_topology(topology, connectivity):
    atoms = list(topology.atoms())
    for conatom, conlist in connectivity.items():
        for conl in conlist:
            topology.addBond(atoms[conatom], atoms[conl])


def solvate_small_molecule(
    fragment=None,
    charge=None,
    mult=None,
    watermodel=None,
    solvent_boxdims=None,
    xmlfile=None,
    lj_treatment=None,
    skip_xmlfile=False,
) -> tuple:
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
            "No xmlfile was provided. You must provide one\nIf you need a forcefield for the solute then try :\n       "
            "       small_molecule_parameterizer"
        )

    # Read XML-file and check for LJ treatment
    if skip_xmlfile is False:
        logger.info("Checking xmlfile for LJ treatment")
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
        logger.info("Creating forcefield using XML-files: %s", waterxmlfile)
        forcefield = openmm_app.forcefield.ForceField(*[waterxmlfile])
    else:
        logger.info("Creating forcefield using XML-files: %s %s", xmlfile, waterxmlfile)
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
    logger.info("Loading Modeller.")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    logger.info(f"Modeller topology has {modeller.topology.getNumResidues()} residues.")

    logger.info("Adding solvent, watermodel: %s", watermodel)

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
    logger.info("")
    logger.info("\nTo use this system setup to define a future OpenMMTheory object you can  do:\n")

    logger.info(
        f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="system_aftersolvent.pdb", '
        f"periodic=True, rigidwater=True)"
    )
    logger.info("")
    logger.info("")

    return forcefield, modeller.topology, newfragment


def find_alternate_locations_residues(pdbfile, use_higher_occupancy=False):
    if use_higher_occupancy is True:
        logger.info("Will keep higher occupancy atoms for alternate locations")

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

    def find_index_of_sublist_with_max_col(rows, index):
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
        logger.warning("\nFound residues in PDB-file that have alternate location labels i.e. multiple occupancies:")
        for chain, residues in bad_resids_dict.items():
            logger.info(f"\nChain {chain}:")
            for res in residues:
                logger.info("%s", res)
        logger.warning("\nThese residues should be manually inspected and fixed in the PDB-file before continuing")
        if use_higher_occupancy is True:
            logger.warning("\n Use higher-occupancy location opton was selected, so continuing.")
            writelisttofile(finalpdblines, "system_afteratlocfixes.pdb", separator="")
            return "system_afteratlocfixes.pdb"
        raise InputError(
            "You should delete either the labelled A or B location of the residue-atom/atoms and then remove the "
            "A/B label from column 17 in the file\nAlternatively, you can choose use_higher_occupancy=True keyword "
            "in OpenMM_Modeller and openmmqmmm will keep the higher occupied form and go on \nMake sure that there "
            "is always an A or B form present.\nExiting."
        )

    return pdbfile


def merge_pdb_files(pdbfile_1, pdbfile_2, outputname="merged.pdb") -> str:
    """Merge two PDB files into one (e.g. protein plus ligand)."""
    pdb1 = openmm.app.PDBFile(pdbfile_1)
    pdb2 = openmm.app.PDBFile(pdbfile_2)

    modeller = openmm.app.Modeller(pdb1.topology, pdb1.positions)  # Add pdbfile1
    modeller.add(pdb2.topology, pdb2.positions)  # Add pdbfile2
    mergedPositions = modeller.positions  # merging positions

    write_pdbfile_openmm_topology(modeller.topology, mergedPositions, outputname)
    logger.info("Wrote merged PDB file: %s", outputname)

    return outputname


def small_molecule_parameterizer(
    charge=None,
    xyzfile=None,
    pdbfile=None,
    molfile=None,
    sdffile=None,
    smiles_string=None,
    resname="LIG",
    forcefield_option="GAFF",
    gaffversion="gaff-2.11",
    openff_file="openff-2.0.0.offxml",
    expected_coul14=0.8333333333333334,
    expected_lj14=0.5,
    allow_undefined_stereo=None,
) -> tuple:
    """Generate an OpenMM forcefield XML for a small molecule with GAFF or OpenFF."""
    logger.info(main_header("SmallMolecule Parameterizor"))
    logger.info("Input options: xyzfile, pdbfile, molfile, sdffile, smiles_string")
    logger.info("Forcefield options: GAFF, OpenFF")
    if charge is None:
        raise InputError(
            "You have to specify a formal total charge of the molecule via the charge keyword (e.g. charge=0)"
        )
    if forcefield_option == "GAFF":
        logger.info("Using GAFF forcefield")
        logger.info("Options:")
    elif forcefield_option == "OpenFF":
        logger.info("Using OpenFF forcefield")
        logger.info(
            "OpenFF forcefield options are Sage (version 2.Y.Z) and Parsley (version 1.Y.Z)  (see https://github.com/openforcefield/openff-forcefields)"
        )
        logger.info("Chosen forcefield is: %s", openff_file)
    else:
        raise InputError("Unknown forcefield_option")

    try:
        from openmm.app import ForceField
    except ModuleNotFoundError:
        raise MissingDependencyError("OpenMM is required but could not be imported") from None

    try:
        import parmed
    except ImportError:
        raise MissingDependencyError(
            "Problem importing parmed Python library\nParmed can be installed using pip: pip install parmed"
        ) from None
    logger.info(f"Parmed version {parmed.__version__} imported")
    try:
        import openff
        from openff.toolkit.topology import Molecule
        from openmmforcefields.generators import GAFFTemplateGenerator
    except ImportError as errormessage:
        raise MissingDependencyError(
            f"OpenFF and openmmforcefields libraries are required but could not be imported\nYou can install like "
            f"this:   conda install --yes -c conda-forge openmmforcefields\nPython import error message: {errormessage}"
        ) from errormessage
    logger.info("")

    if molfile:
        # NOTE: Not well tested.
        logger.info("Mol file provided: %s", molfile)
        molecule = Molecule.from_file(molfile)
    elif sdffile:
        # NOTE: Not well tested.
        logger.info("SDF file provided %s", sdffile)
        molecule = Molecule.from_file(sdffile)
    elif smiles_string:
        logger.info("SMILES string provided: %s", smiles_string)
        molecule = Molecule.from_smiles(smiles_string, allow_undefined_stereo=allow_undefined_stereo)
        logger.info(
            "A SMILES string input means that no coordinate information is available. PDB-file created will have dummy "
            "coordinates that you have to fill in yourself."
        )
    elif xyzfile:
        logger.info("XYZ file provided: %s", xyzfile)
        if os.path.isfile(xyzfile) is False:
            raise FileFormatError("File does not exist. Exiting")
        logger.info("Will use RDKit to convert XYZ file to an RDKit Mol object and then to OpenFF Molecule object")
        # Now using rdkit for more reliable XYZ-Mol conversion (handles total charges and bond orders)
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds

        raw_mol = Chem.MolFromXYZFile(xyzfile)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        smiles_string = Chem.MolToSmiles(mol)
        logger.info("RDKit-determined Smiles_string is: %s", smiles_string)
        molecule = Molecule.from_rdkit(mol)
    elif pdbfile:
        logger.info("PDB-file provided: %s", pdbfile)
        logger.info("Will use RDKit to convert PDB file to an RDKit Mol object and then to OpenFF Molecule object")
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds

        raw_mol = Chem.MolFromPDBFile(pdbfile, removeHs=False)
        mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        smiles_string = Chem.MolToSmiles(mol)
        logger.info("RDKit-determined Smiles_string is: %s", smiles_string)
        molecule = Molecule.from_rdkit(mol)
    else:
        raise InputError("No inputfile provided. Exiting")

    # Affects both PDB-file and XML-file
    for atom in molecule.atoms:
        atom.metadata["residue_name"] = resname
    logger.info("Conversion to OpenFF molecule object successful")
    # NOTE: problem writing proper PDB-file here. Using OpenMM instead below

    logger.info("Now creating an Amber14 compatible OpenMM ForceField object")
    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml")

    if forcefield_option == "GAFF":
        logger.info("GAFF forcefield chosen")
        gaff = GAFFTemplateGenerator(molecules=molecule, forcefield=gaffversion)
        logger.info("GAFF version used: %s", gaff.gaff_version)

        logger.info("Now registering the GAFF template generator in Forcefield object")
        forcefield.registerTemplateGenerator(gaff.generator)

        # Parameterize an OpenMM Topology object that contains the specified molecule.
        # Forcefield will load the appropriate GAFF parameters when needed, and antechamber
        # will be used to generate small molecule parameters on the fly.

        topology = openff.toolkit.topology.Topology.from_molecules([molecule])
        topology_openmm = topology.to_openmm()
        topology = topology_openmm

        # Creating OpenMM system both to check that things works and for passing to Parmed for XML writing
        system = forcefield.createSystem(topology)

        final_xmlfilename = f"gaff_{resname}.xml"
        write_xmlfile_parmed(topology, system, final_xmlfilename)
    elif forcefield_option == "OpenFF":
        import openff
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield=openff_file)

        forcefield = ForceField(
            "amber/protein.ff14SB.xml", "amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml"
        )
        forcefield.registerTemplateGenerator(smirnoff.generator)

        topology = openff.toolkit.topology.Topology.from_molecules([molecule])
        topology_openmm = topology.to_openmm()
        topology = topology_openmm

        # Creating OpenMM system both to check that things works and for passing to Parmed for XML writing
        system = forcefield.createSystem(topology)

        final_xmlfilename = f"openff_{resname}.xml"
        write_xmlfile_parmed(topology, system, final_xmlfilename)

    logger.info("Now creating a PDB-file that matches the XML-file")
    pos = [openmm.Vec3(i[0]._magnitude, i[1]._magnitude, i[2]._magnitude) for i in molecule._conformers[0]]
    with open(f"{resname}.pdb", "w") as pdbfh:
        openmm.app.PDBFile.writeFile(topology, pos * openmm.unit.angstrom, pdbfh)

    logger.info("")
    logger.info("")
    logger.info("%s", "-" * 100)
    logger.info("A new XML-file for molecule has been created: %s", final_xmlfilename)
    logger.info(
        f"Modifying 1-4 scaling parameters in XML-file to match Amber14 FF (coul14={expected_coul14}  and "
        f"lj14={expected_lj14})"
    )
    find_replace_string_in_file(final_xmlfilename, 'coulomb14scale="1.0"', f'coulomb14scale="{expected_coul14}"')
    find_replace_string_in_file(final_xmlfilename, 'lj14scale="1.0"', f'lj14scale="{expected_lj14}"')

    logger.info("Now checking whether the 1-4 scaling is consistent in the XML-file vs. OpenMM system")
    system_from_xml = create_sys_and_check_14_scaling_nonbonding(
        topology=topology, xml_file=final_xmlfilename, expected_coul14=expected_coul14, expected_lj14=expected_lj14
    )
    logger.info("system_from_xml: %s", system_from_xml)
    coulomb_xml, lj_xml = calc_nonbonding_energy_exceptions(system=system_from_xml)
    coulomb_sys, lj_sys = calc_nonbonding_energy_exceptions(system=system)
    logger.info("")
    logger.info("Coulomb_xml: %s", coulomb_xml)
    logger.info("LJ_xml: %s", lj_xml)
    logger.info("")
    logger.info("Coulomb_sys: %s", coulomb_sys)
    logger.info("LJ_sys: %s", lj_sys)
    logger.info("")
    if abs(coulomb_xml - coulomb_sys) > 1e-5:
        raise InputError(
            f"abs(coulomb_xml - coulomb_sys): {abs(coulomb_xml - coulomb_sys)}\nProblem with Coulomb-14 scaling in "
            f"XML-file"
        )
    if abs(lj_xml - lj_sys) > 1e-5:
        raise InputError(f"abs(lj_xml - lj_system): {abs(lj_xml - lj_sys)}\nProblem with LJ-14 scaling in XML-file")
    logger.info("XML-file and forcefield objects are consistent. All good!")
    logger.info("Now returning a Forcefield object containing ligand compatible with the Amber14 FF.\n")
    logger.info(
        "You can feed this object into OpenMM_Modeller like this:\n\
          OpenMM_Modeller(pdbfile=full_pdbfile, forcefield_object=forcefield"
    )

    logger.info(
        "or feed it into OpenMMTheory like this:\n\
          OpenMM_Theory(pdbfile=full_pdbfile, forcefield=forcefield"
    )
    logger.info("")
    logger.info(
        f"The XML-file just created: {final_xmlfilename} can also be used directly (recommended only together with "
        f"Amber14)\n"
    )
    logger.info(
        f"You can use it in OpenMM_Modeller like this:\n\
          OpenMM_Modeller(pdbfile=full_pdbfile, forcefield='Amber14', extraxmlfile=\"{final_xmlfilename}\")"
    )

    logger.info(
        f'or in OpenMMTheory like this:\n\
          OpenMMTheory(xmlfiles=["amber14-all.xml", "amber14/tip3pfb.xml", "{final_xmlfilename}"])'
    )
    logger.info("")
    logger.warning(
        "\nWarning: Make sure that the ligand has the same atom order in the large-system PDB-file \nas in the \
file that was used in this function."
    )
    logger.info("Additionally the ligand requires correct CONECT record lines in that same PDB-file")
    logger.info(f"A {resname}.pdb file has been created that is compatible with the XML-file")
    logger.info("%s", "-" * 100)
    return forcefield


def create_sys_and_check_14_scaling_nonbonding(
    topology=None, xml_file=None, expected_coul14=0.833333, expected_lj14=0.5
):
    logger.info("Creating system from XML-file and topology")
    if topology is None:
        raise InputError("Error: topology is required if system is not provided")
    if xml_file is None:
        raise InputError("Error: xml_file is required if system is not provided")
    forcefield_from_xmlfile = openmm.app.ForceField(xml_file)
    system_from_xmlfile = forcefield_from_xmlfile.createSystem(topology)

    for force in system_from_xmlfile.getForces():
        if isinstance(force, openmm.NonbondedForce):
            break

    for exception_index in range(force.getNumExceptions()):
        atom1, atom2, qq, _sigma, epsilon = force.getExceptionParameters(exception_index)
        # if 0.0 then should be 1-2 or 1-3 interaction
        if epsilon._value == 0.0:
            continue
        q1, sigma1, epsilon1 = force.getParticleParameters(atom1)
        q2, sigma2, epsilon2 = force.getParticleParameters(atom2)

        expected_qq = expected_coul14 * q1 * q2
        expected_epsilon = expected_lj14 * (epsilon1 * epsilon2) ** 0.5

        if abs(qq - expected_qq).value_in_unit(openmm.unit.elementary_charge**2) > 1e-5:
            logger.info("Problem with LJ-14 scaling")
            logger.info("Actual qq: %s", qq)
            logger.info("expected_qq: %s", expected_qq)
            logger.info("expected_epsilon: %s", expected_epsilon)
            logger.info(f"q1: {q1} sigma1:{sigma1} epsilon1:{epsilon1}")
            logger.info(f"q2: {q2} sigma2:{sigma2} epsilon2:{epsilon2}")
        if abs(epsilon - expected_epsilon).value_in_unit(openmm.unit.kilojoule_per_mole) > 1e-5:
            logger.info("Problem with LJ-14 scaling")
            logger.info("Actual epsilon: %s", epsilon)
            logger.info("expected_qq: %s", expected_qq)
            logger.info("expected_epsilon: %s", expected_epsilon)

    return system_from_xmlfile


def calc_nonbonding_energy_exceptions(system=None):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            break
    coulomb_energy = 0.0
    lj_energy = 0.0

    for exception_index in range(force.getNumExceptions()):
        _atom1, _atom2, qq, _sigma, epsilon = force.getExceptionParameters(exception_index)
        # if 0.0 then should be 1-2 or 1-3 interaction
        if epsilon._value == 0.0:
            continue

        coulomb_energy += qq.value_in_unit(openmm.unit.elementary_charge**2)
        lj_energy += epsilon.value_in_unit(openmm.unit.kilojoule_per_mole)

    return coulomb_energy, lj_energy


def write_xmlfile_parmed(topology, system, xmlfilename):
    logger.info("Using Parmed to read topologyfiles")
    try:
        import parmed
    except ImportError:
        raise MissingDependencyError(
            "Problem importing parmed Python library\nMake sure parmed is present in your Python.\nParmed can be "
            "installed using pip: pip install parmed"
        ) from None
    st = parmed.openmm.load_topology(topology, system=system)
    w = parmed.amber.parameters.ParameterSet.from_structure(st)
    ww = parmed.openmm.parameters.OpenMMParameterSet.from_parameterset(w)
    ww.residues.update(parmed.modeller.ResidueTemplateContainer.from_structure(st).to_library())
    ww.write(xmlfilename)
    logger.info("Wrote XML-file: %s", xmlfilename)
