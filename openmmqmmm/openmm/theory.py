import logging
import os
import time

import numpy as np
from packaging import version

from openmmqmmm.exceptions import (
    FileFormatError,
    InputError,
    InternalError,
)

try:
    import openmm
    import openmm.app
    import openmm.unit
except ImportError:
    raise ImportError(
        "OpenMMTheory requires the OpenMM library. Try: conda install -c conda-forge openmm "
        "(see http://docs.openmm.org/latest/userguide/application.html)"
    ) from None


import openmmqmmm.constants
import openmmqmmm.parallel
from openmmqmmm.coords import (
    Fragment,
    define_dummy_topology,
    distance_between_atoms,
)
from openmmqmmm.coords_pbc import cell_params_to_vectors
from openmmqmmm.utils import (
    log_time_since,
    main_header,
    small_header,
    sub_header,
)

logger = logging.getLogger(__name__)

# Bonds constrained automatically at system creation, with the log line each choice prints.
AUTOCONSTRAINTS = {
    "HBonds": (openmm.app.HBonds, "HBonds option: X-H bond lengths will automatically be constrained"),
    "AllBonds": (openmm.app.AllBonds, "AllBonds option: All bond lengths will automatically be constrained"),
    "HAngles": (
        openmm.app.HAngles,
        "HAngles option: All bond lengths and H-X-H and H-O-X angles will automatically be constrained",
    ),
    None: (None, "No automatic constraints"),
}

# Long-range electrostatics treatments accepted for periodic systems.
NONBONDED_METHODS_PBC = {
    "PME": openmm.app.PME,
    "Ewald": openmm.app.Ewald,
    "LJPME": openmm.app.LJPME,
    "CutoffPeriodic": openmm.app.CutoffPeriodic,
}

# Non-periodic counterparts. CutoffPeriodic is deliberately absent: it is rejected with a
# dedicated message rather than silently accepted.
NONBONDED_METHODS_NO_PBC = {
    "NoCutoff": openmm.app.NoCutoff,
    "CutoffNonPeriodic": openmm.app.CutoffNonPeriodic,
}


class OpenMMTheory:
    """Interface to the OpenMM molecular-mechanics library."""

    def __init__(
        self,
        *,
        platform="CPU",
        numcores=1,
        topoforce=False,
        forcefield=None,
        topology=None,
        charmm_files=False,
        psffile=None,
        charmmtopfile=None,
        charmmprmfile=None,
        label="OpenMM",
        gromacs_files=False,
        gromacstopfile=None,
        grofile=None,
        gromacstopdir=None,
        amber_files=False,
        amberprmtopfile=None,
        properties=None,
        nonbonded_method_no_pbc="NoCutoff",
        nonbonded_cutoff_no_pbc=20,
        xmlfiles=None,
        pdbfile=None,
        pdbxfile=None,
        use_parmed=False,
        xmlsystemfile=None,
        do_energy_decomposition=False,
        periodic=False,
        periodic_cell_dimensions=None,
        pbc_vectors=None,
        periodic_cell_vectors=None,
        charmm_periodic_cell_dimensions=None,
        periodic_nonbonded_cutoff=12,
        dispersion_correction=True,
        nonbonded_method_pbc="PME",
        switching_function_distance=10.0,
        ewalderrortolerance=5e-4,
        pme_parameters=None,
        delete_qm1_mm1_bonded=False,
        applyconstraints_in_run=False,
        constraints=None,
        bondconstraints=None,
        restraints=None,
        frozen_atoms=None,
        fragment=None,
        dummysystem=False,
        autoconstraints="HBonds",
        hydrogenmass=1.5,
        rigidwater=True,
        changed_masses=None,
        residuetemplate_choice=None,
        rpmd_num_copies=32,
    ):
        logger.info(main_header("OpenMM Theory"))
        module_init_time = time.time()

        self._configure_platform(platform, numcores=numcores, properties=properties)

        self.theorytype = "MM"
        self.theorynamelabel = "OpenMM"
        self.analytic_hessian = False
        self.label = label
        self.fragment = fragment
        logger.info("Imported OpenMM library version: %s", openmm.__version__)
        if version.parse(openmm.__version__) < version.parse("8.1"):
            logger.warning("OpenMM version < 8.1. OpenMM 8.1 or higher is recommended")
            logger.info("Some features may not work as intended in older versions")

        if charmm_periodic_cell_dimensions is not None:
            raise InputError("charmm_periodic_cell_dimensions is deprecated. Use periodic_cell_dimensions instead")

        logger.info(sub_header("Defining OpenMM object"))
        self.system = None

        # Degrees of freedom of system (accounts for frozen atoms and constraints)
        self.dof = None

        self._configure_constraint_defaults(
            autoconstraints=autoconstraints, rigidwater=rigidwater, hydrogenmass=hydrogenmass
        )

        # Active when RPMDIntegrator is used
        self.rpmd_num_copies = rpmd_num_copies

        # Setting for controlling whether QM1-MM1 bonded terms are deleted or not in a QM/MM job
        # See modify_bonded_forces
        self.delete_qm1_mm1_bonded = delete_qm1_mm1_bonded

        # Whether to do energy decomposition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_decomposition = do_energy_decomposition

        self.coords = []
        self.charges = []
        self.periodic = periodic
        self.periodic_nonbonded_cutoff = periodic_nonbonded_cutoff
        self.nonbonded_cutoff_no_pbc = nonbonded_cutoff_no_pbc
        self.nonbonded_method_pbc = nonbonded_method_pbc
        self.nonbonded_method_no_pbc = nonbonded_method_no_pbc
        self.ewalderrortolerance = ewalderrortolerance

        # Whether to apply constraints or not when calculating MM energy via run method (does not apply to OpenMM MD)
        # NOTE: Should be False in general. Only True for special cases
        self.applyconstraints_in_run = applyconstraints_in_run

        # Residue names,ids,segments,atomtypes of all atoms of system.
        # Grabbed below from PSF-file. Information used to write PDB-file
        self.resnames = []
        self.resids = []
        self.segmentnames = []
        self.atomtypes = []
        self.atomnames = []
        self.mm_elements = []

        # Positions. Generally not used but can be if e.g. grofile has been read in.
        # Purpose: set virtual sites etc.
        self.positions = None
        logger.info(sub_header("Setting up force fields."))
        logger.info(
            "Note: OpenMM will fail in this step if parameters are missing in topology and\n"
            "      parameter files (e.g. nonbonded entries).\n"
        )

        pdb_pbc_vectors = None

        if pbc_vectors is not None:
            logger.warning("PBCvectors keyword is on its way out. Use periodic_cell_vectors instead")
            if periodic_cell_vectors is None:
                periodic_cell_vectors = pbc_vectors

        if charmm_files is True:
            self._load_charmm_files(psffile, charmmtopfile, charmmprmfile, use_parmed=use_parmed)
        elif gromacs_files is True:
            self._load_gromacs_files(grofile, gromacstopfile, gromacstopdir, use_parmed=use_parmed)
        elif amber_files is True:
            self._load_amber_files(
                amberprmtopfile,
                use_parmed=use_parmed,
                periodic_cell_vectors=periodic_cell_vectors,
                periodic_cell_dimensions=periodic_cell_dimensions,
            )
        elif topoforce is True:
            pdb_pbc_vectors = self._load_topology_forcefield(topology, forcefield, pdbfile)
        elif xmlsystemfile is not None:
            pdb_pbc_vectors = self._load_system_xml(xmlsystemfile, pdbfile)
        elif dummysystem is True:
            self._load_dummy_system(fragment)
        else:
            pdb_pbc_vectors = self._load_xml_forcefield(xmlfiles, pdbfile, pdbxfile)

        residueTemplates = {}  # initial
        if residuetemplate_choice is not None:
            logger.info("Found user-specified residuetemplate_choice")
            logger.info("Will generate residueTemplates based on residuetemplate_choice: %s", residuetemplate_choice)
            logger.info(
                "Note: residuetemplate_choice should be a dict like this: residuetemplate_choice={'FER':'FE2'}   "
            )
            residueTemplates = {}
            for resname, choice in residuetemplate_choice.items():
                residueTemplates = {res: choice for res in self.topology.residues() if res.name == resname}
        logger.info("residueTemplates: %s", residueTemplates)
        if self.system is None:
            if self.periodic is True:
                self._create_periodic_system(
                    periodic_cell_vectors=periodic_cell_vectors,
                    pdb_pbc_vectors=pdb_pbc_vectors,
                    periodic_cell_dimensions=periodic_cell_dimensions,
                    periodic_nonbonded_cutoff=periodic_nonbonded_cutoff,
                    switching_function_distance=switching_function_distance,
                    charmm_files=charmm_files,
                    gromacs_files=gromacs_files,
                    amber_files=amber_files,
                    use_parmed=use_parmed,
                    residue_templates=residueTemplates,
                    dispersion_correction=dispersion_correction,
                    pme_parameters=pme_parameters,
                )
            else:
                self._create_nonperiodic_system(
                    charmm_files=charmm_files, amber_files=amber_files, dummysystem=dummysystem
                )

        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                self.nonbonded_force = force

        # Set charges in OpenMMobject by taking from Force (used by QM/MM)
        logger.info("Setting charges")
        self.getatomcharges()

        self.numatoms = int(self.system.getNumParticles())
        self.allatoms = list(range(self.numatoms))
        logger.info("Number of atoms in OpenMM system: %s", self.numatoms)

        # Preserve original masses before any mass modifications or frozen atoms (set mass to 0)
        # NOTE: Creates list of Quantity objects (value, unit attributes)
        self.system_masses_original = [self.system.getParticleMass(i) for i in self.allatoms]

        # Note: constraints and bondconstraints are the same thing
        if constraints is not None:
            logger.info("constraints keyword specified is deprecated. Use bondconstraints instead")
            bondconstraints = constraints

        if bondconstraints or frozen_atoms or restraints:
            logger.info(sub_header("Adding user constraints, restraints or frozen atoms."))
        if bondconstraints is not None:
            self._apply_bondconstraints(bondconstraints, fragment=fragment, pdbfile=pdbfile)
        if frozen_atoms is not None:
            self._apply_frozen_atoms(frozen_atoms)
        if restraints is not None:
            self._apply_bondrestraints(restraints)

        if changed_masses is not None:
            logger.info("Modified masses")
            # changed_masses should be a dict of : atomindex: mass
            self.modify_masses(changed_masses=changed_masses)

        logger.info("\nSystem constraints defined upon system creation: %s", self.system.getNumConstraints())
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(self.system.getNumConstraints()):
                logger.info("Defined constraints: %s", self.system.getConstraintParameters(i))

        self.set_simulation_parameters()

        # Now calling function to compute the actual degrees of freedom.
        # NOTE: Needs to be called once, after system-create, constraints and frozen atoms are done.
        self.compute_dof()

        # Force run. Option to allow run even though constraints may be defined
        # Used by GentlewarmupMD etc. to get a basic gradient
        self.force_run = False

        # For energy decomposition we must create force groups
        # Must be done after system creation but before simulation creation
        if self.do_energy_decomposition is True:
            logger.info("Energy decomposition is active. Creating force groups")
            self.forcegroupify()

        log_time_since(module_init_time, "OpenMM object creation")

    def _configure_platform(self, platform, *, numcores, properties):
        # OpenMM also honours $OPENMM_CPU_THREADS from the shell; numcores sets it here.
        os.environ["OMP_NUM_THREADS"] = str(numcores)
        os.environ["OPENMM_CPU_THREADS"] = str(numcores)
        logger.info("OpenMM CPU threads set to: %s", os.environ["OMP_NUM_THREADS"])
        self.numcores = numcores  # Setting for general theory-interface compatibility

        self.platform_choice = platform
        self.properties = {} if properties is None else properties
        if self.platform_choice == "CPU":
            logger.info("Using platform: CPU")
            self.properties["Threads"] = str(numcores)
        else:
            logger.info("Using platform: %s", self.platform_choice)

    def _configure_constraint_defaults(self, *, autoconstraints, rigidwater, hydrogenmass):
        if autoconstraints == "None":
            autoconstraints = None
        try:
            self.autoconstraints, description = AUTOCONSTRAINTS[autoconstraints]
        except (KeyError, TypeError):
            raise InputError("Unknown autoconstraints option") from None
        logger.info(description)
        logger.info("AutoConstraint setting: %s", self.autoconstraints)

        self.user_frozen_atoms = []
        self.user_constraints = []
        self.user_restraints = []

        self.rigidwater = rigidwater
        logger.info("Rigidwater constraints: %s", self.rigidwater)
        self.hydrogenmass = None if hydrogenmass is None else hydrogenmass * openmm.unit.amu
        logger.info("Hydrogenmass option: %s", self.hydrogenmass)

    def _apply_bondconstraints(self, bondconstraints, *, fragment, pdbfile):
        logger.info(f"Before adding user constraints, system contains {self.system.getNumConstraints()} constraints")
        logger.info("")
        if len(bondconstraints) < 50:
            logger.info("User-constraints to add (bond) %s", bondconstraints)
        else:
            logger.info(f"{len(bondconstraints)} user-defined constraints to add.")

        if 2 in [len(con) for con in bondconstraints]:
            logger.info(
                "Missing distance value for some constraints. Can apply current-geometry distances if a\n"
                "fragment has been provided"
            )
            if fragment is None:
                logger.info(
                    "No fragment provided to OpenMMTheory. Will check if pdbfile is defined and use coordinates "
                    "from there"
                )
                if pdbfile is None:
                    logger.info(
                        "No PDBfile present either. Either fragment or PDBfile containing coordinates is "
                        "required for constraint definition"
                    )
                    raise InputError("Constraint definition requires a fragment or a PDB file with coordinates")
                fragment = Fragment(pdbfile=pdbfile)
            bondconstraints = clean_up_constraints_list(fragment=fragment, constraints=bondconstraints)
            self.add_bondconstraints(constraints=bondconstraints)

        self.user_constraints = bondconstraints
        logger.info(f"{len(self.user_constraints)} user-defined constraints added.")

    def _apply_frozen_atoms(self, frozen_atoms):
        self.user_frozen_atoms = frozen_atoms
        if len(self.user_frozen_atoms) < 50:
            logger.info("Frozen atoms to add: %s", str(frozen_atoms).strip("[]"))
        else:
            logger.info(f"{len(self.user_frozen_atoms)} user-defined frozen atoms to add.")
        self.freeze_atoms(frozen_atoms=frozen_atoms)

    def _apply_bondrestraints(self, restraints):
        # [[atom_i, atom_j, d, k]], e.g. [[700, 701, 1.05, 5.0]]; Angstrom and kcal/mol/Angstrom^2
        self.user_restraints = restraints
        if len(self.user_restraints) < 50:
            logger.info("User-restraints to add: %s", restraints)
        else:
            logger.info(f"{len(self.user_restraints)} user-defined restraints to add.")
        self.add_bondrestraints(restraints=restraints)

    def _create_periodic_system(
        self,
        *,
        periodic_cell_vectors,
        pdb_pbc_vectors,
        periodic_cell_dimensions,
        periodic_nonbonded_cutoff,
        switching_function_distance,
        charmm_files,
        gromacs_files,
        amber_files,
        use_parmed,
        residue_templates,
        dispersion_correction,
        pme_parameters,
    ):
        logger.info("System is periodic.")
        logger.info(sub_header("Setting up periodicity."))
        # Necessary for system creation with periodics (otherwise failure)
        self.set_periodics_before_system_creation(
            periodic_cell_vectors,
            pdb_pbc_vectors,
            periodic_cell_dimensions,
            charmm_files,
            amber_files,
            use_parmed,
        )

        try:
            nonb_method_PBC = NONBONDED_METHODS_PBC[self.nonbonded_method_pbc]
        except (KeyError, TypeError):
            raise InputError("Unknown nonbonded method") from None

        logger.info("Nonbonded PBC method selected: %s", nonb_method_PBC)

        smallest_boxdim = min(self.topology.getUnitCellDimensions()).value_in_unit(openmm.unit.angstroms)
        logger.info("Smallest_box dimension is: %s", smallest_boxdim)
        logger.info("periodic_nonbonded_cutoff: %s", periodic_nonbonded_cutoff)
        if smallest_boxdim < periodic_nonbonded_cutoff * 2:
            logger.warning(
                f"Warning: Smallest box dimension is less than 2*periodic_nonbonded_cutoff = "
                f"{2 * self.periodic_nonbonded_cutoff}"
            )
            logger.info(
                "This will not work. See https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#boxsize"
            )
            logger.info("Will now automatically set the cutoff to be 1/2 the smallest box dimension")
            self.periodic_nonbonded_cutoff = round(
                0.5 * min(self.topology.getUnitCellDimensions()).value_in_unit(openmm.unit.angstroms), 6
            )
            logger.info("periodic_nonbonded_cutoff is now: %s", self.periodic_nonbonded_cutoff)

        logger.info(f"Nonbonded cutoff is {self.periodic_nonbonded_cutoff} Angstrom.")
        # Parameters here are based on OpenMM DHFR example. Shared by all four
        # branches below, which differ only in what they add and what they pass
        # positionally: CHARMM hands createSystem its parsed parameter set, the
        # modeller/XML route hands it the topology, and GROMACS and Amber carry
        # their own (the forcefield object already holds the PBC information).
        pbc_system_kwargs = {
            "nonbondedMethod": nonb_method_PBC,
            "constraints": self.autoconstraints,
            "hydrogenMass": self.hydrogenmass,
            "rigidWater": self.rigidwater,
            "ewaldErrorTolerance": self.ewalderrortolerance,
            "nonbondedCutoff": self.periodic_nonbonded_cutoff * openmm.unit.angstroms,
        }
        if charmm_files is True:
            logger.info("Using CHARMM files.")
            self.system = self.forcefield.createSystem(
                self.params,
                switchDistance=switching_function_distance * openmm.unit.angstroms,
                **pbc_system_kwargs,
            )
        elif gromacs_files is True:
            # NOTE: Gromacs has read PBC info from Gro file already
            logger.info("Ewald Error tolerance: %s", self.ewalderrortolerance)
            # Note: no switchDistance. Not available for GROMACS?
            self.system = self.forcefield.createSystem(**pbc_system_kwargs)
        elif amber_files is True:
            # NOTE: PBC information should be in forcefield object already
            self.system = self.forcefield.createSystem(**pbc_system_kwargs)
        else:
            self.system = self.forcefield.createSystem(
                self.topology, residueTemplates=residue_templates, **pbc_system_kwargs
            )

        self.periodic_cell_vectors = np.array(
            [[v._value * 10 for v in vec] for vec in self.system.getDefaultPeriodicBoxVectors()]
        )
        logger.info("Periodic_cell_vectors (Å) %s", self.periodic_cell_vectors)

        self._log_nonbonded_force_settings(dispersion_correction=dispersion_correction, pme_parameters=pme_parameters)

    def _log_nonbonded_force_settings(self, *, dispersion_correction, pme_parameters):
        logger.info(small_header("OpenMM Forces defined:"))
        for force in self.system.getForces():
            logger.info("%s", force.getName())
            if isinstance(force, openmm.CustomNonbondedForce):
                # NOTE: This is only sometimes used: XML-CHARMM setup, GROMACS-files etc.
                pass
            elif isinstance(force, openmm.NonbondedForce):
                force.setUseDispersionCorrection(dispersion_correction)

                if pme_parameters is not None:
                    logger.info("Nonbonded force:  Changing PME parameters")
                    force.setPMEParameters(pme_parameters[0], pme_parameters[1], pme_parameters[2], pme_parameters[3])
                logger.info("Nonbonded force settings (after all modifications):")
                logger.info(f"   Periodic cutoff distance: {force.getCutoffDistance()}")
                logger.info(f"   Use SwitchingFunction: {force.getUseSwitchingFunction()}")
                if force.getUseSwitchingFunction() is True:
                    logger.info(f"   SwitchingFunction distance: {force.getSwitchingDistance()}")
                logger.info(f"   Use Long-range Dispersion correction: {force.getUseDispersionCorrection()}")
                logger.info("   PME Parameters: %s", force.getPMEParameters())
                logger.info("   Ewald error tolerance: %s", force.getEwaldErrorTolerance())
        logger.info(small_header("OpenMM system created."))

    def _create_nonperiodic_system(self, *, charmm_files, amber_files, dummysystem):
        if self.nonbonded_method_no_pbc == "CutoffPeriodic":
            raise InputError("nonbondedMethod_noPBC with CutoffPeriodic not currently allowed")
        try:
            noPBC_nonbondedMethod = NONBONDED_METHODS_NO_PBC[self.nonbonded_method_no_pbc]
        except (KeyError, TypeError):
            raise InputError("Unknown non-periodic nonbonded method") from None
        logger.info("System is non-periodic.")
        logger.info("nonbonded noPBC Method is: %s", noPBC_nonbondedMethod)

        logger.info("Nonbonded cutoff : %s Angstrom", self.nonbonded_cutoff_no_pbc)

        # No Ewald tolerance here: without PBC there is no Ewald sum.
        no_pbc_system_kwargs = {
            "nonbondedMethod": noPBC_nonbondedMethod,
            "constraints": self.autoconstraints,
            "rigidWater": self.rigidwater,
            "nonbondedCutoff": self.nonbonded_cutoff_no_pbc * openmm.unit.angstroms,
            "hydrogenMass": self.hydrogenmass,
        }
        if charmm_files is True:
            self.system = self.forcefield.createSystem(self.params, **no_pbc_system_kwargs)
        elif amber_files is True:
            self.system = self.forcefield.createSystem(**no_pbc_system_kwargs)
        elif dummysystem is True:
            # Dummy system: OpenMM's own defaults, no nonbonded settings applied
            self.system = self.forcefield.createSystem(self.topology)
        else:
            self.system = self.forcefield.createSystem(self.topology, **no_pbc_system_kwargs)
        logger.info(small_header("OpenMM system created."))
        logger.info("OpenMM Forces defined: %s", self.system.getForces())
        logger.info("")

    def _load_charmm_files(self, psffile, charmmtopfile, charmmprmfile, *, use_parmed=False):
        logger.info("Reading CHARMM files.")
        if use_parmed is True:
            import parmed

            logger.info("Using Parmed.")
            self.psf = parmed.charmm.CharmmPsfFile(psffile)
            # Removed , permissive=True, no longer in parmed
            self.params = parmed.charmm.CharmmParameterSet(charmmtopfile, charmmprmfile)
            # Note: OpenMM uses 0-indexing
            self.resnames = [self.psf.atoms[i].residue.name for i in range(len(self.psf.atoms))]
            self.resids = [self.psf.atoms[i].residue.idx for i in range(len(self.psf.atoms))]
            self.segmentnames = [self.psf.atoms[i].residue.segid for i in range(len(self.psf.atoms))]
            self.atomtypes = [i.type for i in self.psf.atoms]
            self.atomnames = [self.psf.atoms[i].name for i in range(len(self.psf.atoms))]
        else:
            self.psf = openmm.app.CharmmPsfFile(psffile)
            self.params = openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile, permissive=True)
            self.resnames = [self.psf.atom_list[i].residue.resname for i in range(len(self.psf.atom_list))]
            self.resids = [self.psf.atom_list[i].residue.idx for i in range(len(self.psf.atom_list))]
            self.segmentnames = [self.psf.atom_list[i].system for i in range(len(self.psf.atom_list))]
            self.atomtypes = [self.psf.atom_list[i].attype for i in range(len(self.psf.atom_list))]
            self.atomnames = [self.psf.atom_list[i].name for i in range(len(self.psf.atom_list))]
            self.define_mm_elements(self.psf.topology)

        self.topology = self.psf.topology
        self.forcefield = self.psf

    def _load_gromacs_files(self, grofile, gromacstopfile, gromacstopdir, *, use_parmed=False):
        logger.info("Reading Gromacs files.")
        if use_parmed is True:
            import parmed

            logger.info("Using Parmed.")
            logger.info("GROMACS top dir: %s", gromacstopdir)
            parmed.gromacs.GROMACS_TOPDIR = gromacstopdir
            logger.info("Reading GROMACS GRO file: %s", grofile)
            gmx_gro = parmed.gromacs.GromacsGroFile.parse(grofile)
            logger.info("Reading GROMACS topology file: %s", gromacstopfile)
            gmx_top = parmed.gromacs.GromacsTopologyFile(gromacstopfile)

            gmx_top.box = gmx_gro.box
            gmx_top.positions = gmx_gro.positions
            self.positions = gmx_top.positions

            self.topology = gmx_top.topology
            self.forcefield = gmx_top
        else:
            logger.info("Using built-in OpenMM routines to read GROMACS topology.")
            logger.warning("May fail if virtual sites present (e.g. TIP4P residues).")
            logger.info("Use 'parmed=True'  to avoid")
            gro = openmm.app.GromacsGroFile(grofile)
            self.grotop = openmm.app.GromacsTopFile(
                gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=gromacstopdir
            )

            self.topology = self.grotop.topology
            self.forcefield = self.grotop

        self.define_mm_elements(self.topology)

    def _load_amber_files(
        self,
        amberprmtopfile,
        *,
        use_parmed=False,
        periodic_cell_vectors=None,
        periodic_cell_dimensions=None,
    ):
        logger.info("Reading Amber files.")
        logger.warning("Only new-style Amber7 prmtop-file will work.")
        logger.warning("Will take periodic boundary conditions from prmtop file.")
        if use_parmed is True:
            import parmed

            logger.info("Using Parmed to read Amber files.")
            self.prmtop = parmed.load_file(amberprmtopfile)
        else:
            logger.info("Using built-in OpenMM routines to read Amber files.")
            # Note: Only new-style Amber7 prmtop files work
            # If PBC vectors provided and new OpenMM version
            # Note Jan 2024: Amber prmtop files sometimes have PBC vectors (ready by OpenMM parser), this is
            # deprecated behaviour though it seems
            # Generally recommended instead to get PBC info from inpcrd files that we typically don't use
            # Hence we need to override that info anyway
            # OpenMM 8.1 allows us to do this easily by constructor, older versions requires hacky workarounds
            # (see set_periodics_before_system_creation for those hacks)
            # Note: https://github.com/openmm/openmm/issues/4078

            if version.parse(openmm.__version__) >= version.parse("8.1"):
                temp_pbc_vecs = None if periodic_cell_vectors is None else periodic_cell_vectors * openmm.unit.angstrom
                # Angstrom units work here despite naming all three cell dimensions
                temp_pbc_cell_value = (
                    None if periodic_cell_dimensions is None else periodic_cell_dimensions * openmm.unit.angstrom
                )
                self.prmtop = openmm.app.AmberPrmtopFile(
                    amberprmtopfile, periodicBoxVectors=temp_pbc_vecs, unitCellDimensions=temp_pbc_cell_value
                )
            else:
                self.prmtop = openmm.app.AmberPrmtopFile(amberprmtopfile)
        self.topology = self.prmtop.topology
        logger.info("Amber PBC vectors read: %s", self.topology.getPeriodicBoxVectors())
        self.forcefield = self.prmtop

        # List of resids, resnames and mm_elements. Used by actregiondefine
        self.resids = [i.residue.index for i in self.prmtop.topology.atoms()]
        self.resnames = [i.residue.name for i in self.prmtop.topology.atoms()]
        self.define_mm_elements(self.prmtop.topology)
        self.atomnames = [i.name for i in self.prmtop.topology.atoms()]

    def _load_topology_forcefield(self, topology, forcefield, pdbfile):
        logger.info("Using forcefield info from topology and forcefield keyword.")
        pdb_pbc_vectors = None
        if topology is not None:
            logger.info("Topology provided as keyword")
            self.topology = topology
        else:
            logger.info("No topology provided as keyword")
            logger.info("Reading topology from PDB-file instead")
            pdb = openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            pdb_pbc_vectors = pdb.topology.getPeriodicBoxVectors()
        self.forcefield = forcefield
        self.define_mm_elements(self.topology)
        return pdb_pbc_vectors

    def _load_system_xml(self, xmlsystemfile, pdbfile):
        logger.info("Reading system XML file: %s", xmlsystemfile)
        with open(xmlsystemfile) as xmlfh:
            xmlsystemfileobj = xmlfh.read()
        logger.info("Now defining OpenMM system using information in file")
        logger.warning("File may contain hardcoded constraints that can not be overridden.")
        self.system = openmm.XmlSerializer.deserializeSystem(xmlsystemfileobj)
        # NOTE: Big drawback of xmlsystemfile is that constraints have been hardcoded and can

        logger.info("Reading topology from PDBfile: %s", pdbfile)
        pdb = openmm.app.PDBFile(pdbfile)
        self.topology = pdb.topology
        self.define_mm_elements(self.topology)
        return pdb.topology.getPeriodicBoxVectors()

    def _load_dummy_system(self, fragment):
        atomnames_full = [j + str(i) for i, j in enumerate(fragment.elems)]

        self.topology = define_dummy_topology(fragment.elems)

        xmlfile = write_xmlfile_nonbonded(
            filename="dummy.xml",
            resnames=["DUM"],
            atomnames_per_res=[atomnames_full],
            atomtypes_per_res=[fragment.elems],
            elements_per_res=[fragment.elems],
            masses_per_res=[fragment.masses],
            charges_per_res=[[0.0] * fragment.numatoms],
            sigmas_per_res=[[0.0] * fragment.numatoms],
            epsilons_per_res=[[0.0] * fragment.numatoms],
            skip_nb=False,
        )
        self.forcefield = openmm.app.ForceField(xmlfile)
        self.define_mm_elements(self.topology)

    def _load_xml_forcefield(self, xmlfiles, pdbfile, pdbxfile):
        logger.info("Reading OpenMM XML forcefield files and PDB (or PDBx) file")
        logger.info("xmlfiles: %s", str(xmlfiles).strip("[]"))
        logger.info("pdbfile: %s", pdbfile)
        logger.info("pdbxfile: %s", pdbxfile)
        if pdbfile is not None:
            pdb = openmm.app.PDBFile(pdbfile)
        elif pdbxfile is not None:
            pdb = openmm.app.PDBxFile(pdbxfile)
        else:
            raise InputError("Error: No pdbfile or pdbxfile input provided")

        pdb_pbc_vectors = pdb.topology.getPeriodicBoxVectors()

        self.topology = pdb.topology
        self.forcefield = openmm.app.ForceField(*xmlfiles)
        # Defining some things. resids is used by actregiondefine
        self.resids = [i.residue.index for i in self.topology.atoms()]
        self.resnames = [i.residue.name for i in self.topology.atoms()]
        self.atomnames = [i.name for i in self.topology.atoms()]
        self.define_mm_elements(self.topology)
        return pdb_pbc_vectors

    def define_mm_elements(self, topology):
        """Extract the element symbol of every atom from an OpenMM topology."""
        try:
            self.mm_elements = [i.element.symbol for i in topology.atoms()]
        except AttributeError:
            logger.info("Problem occurred while defining mm_elements.")
            logger.info("This may be due to virtual sites present")
            logger.info("mm_elements will be set to empty list")
            self.mm_elements = []

    def write_pdbfile(self, positions=None, outputname="system"):
        """Write a PDB file of the system using the stored OpenMM topology."""
        logger.info("Writing PDB-file using OpenMMTheory object")
        logger.info("Will be using defined topology.")
        logger.debug("Internal positions: %s", self.positions)
        # Explicit positions win: they were checked last, so passing minimized or
        # otherwise updated coordinates silently wrote the object's original ones.
        if positions is not None:
            logger.info("Using input positions")
            with open(f"{outputname}.pdb", "w") as pdbfh:
                openmm.app.PDBFile.writeFile(self.topology, positions, pdbfh)
        elif self.positions is not None:
            logger.info("Found positions in OpenMMTheory object. Using them to write PDB-file.")
            with open(f"{outputname}.pdb", "w") as pdbfh:
                openmm.app.PDBFile.writeFile(self.topology, self.positions, pdbfh)
        elif self.fragment is not None:
            logger.info("Found an fragment file referenced. Using coordinates in fragment to write PDB-file.")
            logger.debug("%s", self.fragment)
            coords_nm = self.fragment.coords * 0.1  # converting from Angstrom to nm
            pos = [
                openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in range(len(coords_nm))
            ] * openmm.unit.nanometer
            with open(f"{outputname}.pdb", "w") as pdbfh:
                openmm.app.PDBFile.writeFile(self.topology, pos, pdbfh)
        else:
            raise InputError("Found neither system positions defined or an fragment file. Can not write PDB-file.")

    def set_periodics_before_system_creation(
        self, periodic_cell_vectors, pdb_pbc_vectors, periodic_cell_dimensions, charmm_files, amber_files, use_parmed
    ):
        """Resolve the periodic box from the various possible sources."""
        logger.info("Inspecting periodicity input before system creation")
        logger.info("periodic_cell_vectors: %s", periodic_cell_vectors)
        logger.info("periodic_cell_dimensions: %s", periodic_cell_dimensions)
        logger.info("pdb_pbc_vectors: %s", pdb_pbc_vectors)
        # IF PBC vectors provided then we need to set them in the topology (otherwise system creation does not work)
        if periodic_cell_vectors is not None:
            logger.info("\nPBC vectors provided by user (in Angstrom): %s", periodic_cell_vectors)
            logger.info("Setting PBC vectors in topology object")
            self.topology.setPeriodicBoxVectors(periodic_cell_vectors * openmm.unit.angstroms)
            logger.info("Topology PBC vectors set: %s", self.topology.getPeriodicBoxVectors())
            logger.info("Setting PBC box vectors in forcefield object")
            if charmm_files is True:
                self.forcefield.box_vectors = periodic_cell_vectors * openmm.unit.angstrom
                logger.info("PBC box vectors set: %s", self.forcefield.box_vectors)
            elif amber_files is True and use_parmed is True:
                # Necessary for parmed object to define box_vectors in forcefield object
                self.forcefield.box_vectors = periodic_cell_vectors * openmm.unit.angstrom
                logger.info("PBC box vectors set: %s", self.forcefield.box_vectors)
            elif amber_files is True and use_parmed is False:
                # Not necessary to define box_vectors (grabbed from topology above) but we have to make sure PBC is on
                # Happens if no IFBOX defined in prmtop file but we still want periodicity
                # Hacky fix below
                logger.info("Amber-prmtop getIfBox: %s", self.forcefield._prmtop.getIfBox())
                self.forcefield._prmtop._raw_data["POINTERS"][27] = 1
                logger.info("Amber-prmtop getIfBox: %s", self.forcefield._prmtop.getIfBox())

                if version.parse(openmm.__version__) < version.parse("8.1"):
                    logger.warning("Amber prmtop file detected and OpenMM version < 8.0")
                    logger.warning("Will assume cubic box and set PBC vectors in a hacky way")
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"] = np.array([0.0, 0.0, 0.0, 0.0])
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][0] = 90.0
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][1] = periodic_cell_vectors[0][0]
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][2] = periodic_cell_vectors[1][1]
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][3] = periodic_cell_vectors[2][2]
        elif periodic_cell_dimensions is not None:
            logger.info("\nPBC cell dimensions provided by user: %s", periodic_cell_dimensions)
            self.topology.setUnitCellDimensions = [
                openmm.unit.Quantity(value=periodic_cell_dimensions[0], unit=openmm.unit.angstrom),
                openmm.unit.Quantity(value=periodic_cell_dimensions[1], unit=openmm.unit.angstrom),
                openmm.unit.Quantity(value=periodic_cell_dimensions[2], unit=openmm.unit.angstrom),
                openmm.unit.Quantity(value=periodic_cell_dimensions[3], unit=openmm.unit.degree),
                openmm.unit.Quantity(value=periodic_cell_dimensions[4], unit=openmm.unit.degree),
                openmm.unit.Quantity(value=periodic_cell_dimensions[5], unit=openmm.unit.degree),
            ]
            logger.info("Topology PBC dimensions set: %s", self.topology.getUnitCellDimensions())
            if self.topology.getUnitCellDimensions() is None:
                logger.warning("Problems with unitcell dimensions setting.")
                logger.warning("Will assume cubic box and set PBC vectors instead")
                self.topology.setPeriodicBoxVectors(
                    [
                        [periodic_cell_dimensions[0], 0, 0],
                        [0, periodic_cell_dimensions[1], 0],
                        [0, 0, periodic_cell_dimensions[2]],
                    ]
                    * openmm.unit.angstrom
                )
            logger.info("PeriodicBoxVectors:  %s", self.topology.getPeriodicBoxVectors())
            logger.info("Setting PBC box in forcefield object")
            self.forcefield.box = [
                openmm.unit.Quantity(value=periodic_cell_dimensions[0], unit=openmm.unit.angstrom),
                openmm.unit.Quantity(value=periodic_cell_dimensions[1], unit=openmm.unit.angstrom),
                openmm.unit.Quantity(value=periodic_cell_dimensions[2], unit=openmm.unit.angstrom),
                openmm.unit.Quantity(value=periodic_cell_dimensions[3], unit=openmm.unit.degree),
                openmm.unit.Quantity(value=periodic_cell_dimensions[4], unit=openmm.unit.degree),
                openmm.unit.Quantity(value=periodic_cell_dimensions[5], unit=openmm.unit.degree),
            ]
            logger.info("PBC box set: %s", self.forcefield.box)
            if charmm_files is True and use_parmed is False:
                self.forcefield.setBox(
                    openmm.unit.Quantity(value=periodic_cell_dimensions[0], unit=openmm.unit.angstrom),
                    openmm.unit.Quantity(value=periodic_cell_dimensions[1], unit=openmm.unit.angstrom),
                    openmm.unit.Quantity(value=periodic_cell_dimensions[2], unit=openmm.unit.angstrom),
                    alpha=openmm.unit.Quantity(value=periodic_cell_dimensions[3], unit=openmm.unit.degree),
                    beta=openmm.unit.Quantity(value=periodic_cell_dimensions[4], unit=openmm.unit.degree),
                    gamma=openmm.unit.Quantity(value=periodic_cell_dimensions[5], unit=openmm.unit.degree),
                )
                logger.info("PBC box set: %s", self.forcefield.box)
                logger.info("Set box vectors: %s", self.forcefield.box_vectors)
            if (charmm_files is True and use_parmed is True) or (amber_files is True and use_parmed is True):
                pass
            elif amber_files is True and use_parmed is False:
                logger.info("Amber ff getIfBox %s", self.forcefield._prmtop.getIfBox())
                # Hacky thing to make sure PBC is on for Amber.
                # PBCvectors will be grabbed from topology above
                # Happens if no IFBOX defined in prmtop file but we still want periodicity
                self.forcefield._prmtop._raw_data["POINTERS"][27] = 1

                if version.parse(openmm.__version__) < version.parse("8.1"):
                    logger.warning("Amber prmtop file detected and OpenMM version < 8.1")
                    logger.warning("Will assume cubic box and set PBC vectors in a hacky way")
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"] = np.array([0.0, 0.0, 0.0, 0.0])
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][0] = 90.0
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][1] = periodic_cell_dimensions[0]
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][2] = periodic_cell_dimensions[1]
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][3] = periodic_cell_dimensions[2]
        elif pdb_pbc_vectors is not None:
            logger.warning(
                "Warning: neither user keyword periodic_cell_vectors or periodic_cell_dimensions was set (None)"
            )
            logger.info(
                "However, we found PBC information inside PDB-topology of the PDB-file that was read in. Using this "
                "and continuing"
            )
        elif self.topology.getPeriodicBoxVectors() is not None:
            logger.info("Found PBC information in topology object. Using this and continuing")
        else:
            raise FileFormatError("Found no PBC information, yet periodicity is requested. Exiting!")

    def get_pbc_vectors(self):
        """Return the current periodic box vectors in Angstrom."""
        vectors_nm = list(self.topology.getPeriodicBoxVectors())
        a = list(vectors_nm[0].value_in_unit(openmm.unit.angstrom))
        b = list(vectors_nm[1].value_in_unit(openmm.unit.angstrom))
        c = list(vectors_nm[2].value_in_unit(openmm.unit.angstrom))
        return [a, b, c]

    def set_numcores(self, numcores):
        """Set the number of CPU threads OpenMM uses."""
        self.numcores = numcores

    def cleanup(self):
        """No-op: OpenMM keeps no scratch files that need removing between runs."""
        logger.info("Cleanup for OpenMMTheory called")

    # add force that restrains atoms to a fixed point:
    # https://github.com/openmm/openmm/issues/2568

    # To set positions in OpenMMobject (in nm) from np-array (Angstrom)
    def set_positions(self, coords, simulation):
        """Load coordinates into an OpenMM simulation."""
        logger.info("Setting coordinates of OpenMM object")
        coords_nm = coords * 0.1  # converting from Angstrom to nm
        pos = [
            openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in range(len(coords_nm))
        ] * openmm.unit.nanometer
        if isinstance(simulation.integrator, openmm.RPMDIntegrator):
            for copy_index in range(simulation.integrator.getNumCopies()):
                simulation.integrator.setPositions(copy_index, pos)
            logger.info("Coordinates set for all %s RPMD copies", simulation.integrator.getNumCopies())
        else:
            simulation.context.setPositions(pos)
        logger.info("Coordinates set")

    # Update cell using either periodic_cell_vectors or periodic_cell_dimensions
    # This method is called by Periodic optimizers
    def update_cell(self, periodic_cell_vectors=None, periodic_cell_dimensions=None):
        """Change the periodic box of the existing system."""
        logger.info("Updating cell vectors")
        logger.info("New periodic_cell_vectors are: %s", periodic_cell_vectors)
        if periodic_cell_vectors is not None:
            self.periodic_cell_vectors = periodic_cell_vectors
        elif periodic_cell_dimensions is not None:
            self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

        cellvecs_nm = self.periodic_cell_vectors / 10
        a = cellvecs_nm[0]
        b = cellvecs_nm[1]
        c = cellvecs_nm[2]

        # We may have to adjust the nonbonded cutoff.
        # Shortest box dimension (diagonal elements, safe estimate for triclinic)
        min_box_dim = min(cellvecs_nm[0, 0], cellvecs_nm[1, 1], cellvecs_nm[2, 2])
        hard_limit_cutoff = 0.499 * min_box_dim  # just under OpenMM's hard limit of 0.5

        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            if isinstance(force, openmm.NonbondedForce):
                current_cutoff = force.getCutoffDistance().value_in_unit(openmm.unit.nanometer)

                # Store the original intended cutoff the first time we see it
                if not hasattr(self, "_original_cutoff_nm"):
                    self._original_cutoff_nm = current_cutoff
                    logger.info(f"Storing original cutoff: {self._original_cutoff_nm:.3f} nm")

                # Desired cutoff: restore original if box allows, otherwise use hard limit
                desired_cutoff = min(self._original_cutoff_nm, hard_limit_cutoff)

                if abs(desired_cutoff - current_cutoff) > 1e-6:  # only update if actually changed
                    logger.info(
                        f"Adjusting cutoff from {current_cutoff:.3f} to {desired_cutoff:.3f} nm "
                        f"(box limit: {hard_limit_cutoff:.3f} nm, original: {self._original_cutoff_nm:.3f} nm)"
                    )
                    force.setCutoffDistance(desired_cutoff * openmm.unit.nanometer)
                break
        # Note we are modifying the system and topology itself because we are doing OpenMMTheory.run that creates new
        # sim and context each time
        self.system.setDefaultPeriodicBoxVectors(a, b, c)
        self.topology.setPeriodicBoxVectors(cellvecs_nm)

    # https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=161&t=10049&p=0&start=0&view=&sid=b844250e55b14682fb21b5f66a4d810f
    # https://github.com/openmm/openmm/issues/2262
    # Helpful for NPT simulations when solute is fixed
    def add_dummy_atom_to_restrain_solute(self, atomindices=None, forceconstant=100):
        """Add a massless dummy atom, harmonically tethered to a group of atoms."""
        logger.info("num particles %s", self.system.getNumParticles())
        self.system.addParticle(0)
        logger.info("num particles %s", self.system.getNumParticles())
        dummyatomindex = self.system.getNumParticles() - 1
        logger.info("dummyatomindex: %s", dummyatomindex)
        self.nonbonded_force.addParticle(0, 1, 0)
        chain = self.topology.addChain()
        residue = self.topology.addResidue("dummy", chain)
        dummy_element = openmm.app.element.Element(0, "Dummyel", "Dd", 0.0)
        self.topology.addAtom("Dum", dummy_element, residue)

        self.restraint = openmm.HarmonicBondForce()
        self.restraint.setUsesPeriodicBoundaryConditions(True)
        self.system.addForce(self.restraint)

        for i in atomindices:
            logger.info("Adding bond")
            self.restraint.addBond(i, dummyatomindex, 0, forceconstant)

    def remove_force(self, forceindex):
        """Remove a force by its index in the system."""
        logger.info(f"Removing force-index {forceindex}: {self.system.getForces()[forceindex].getName()}")
        self.system.removeForce(forceindex)

    def add_custom_bond_force(self, i, j, value, forceconstant):
        """Restrain the distance between two atoms harmonically."""
        logger.info(
            f"Adding custom bond force between atom index i={i} and j={j} with value: {value} Angstrom, "
            f"forceconstant={forceconstant} kcal/mol/Angstrom^2"
        )
        bond_force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
        bond_force.addGlobalParameter("k", forceconstant * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2)
        bond_force.addGlobalParameter("r0", value * openmm.unit.angstrom)
        bond_force.addBond(i, j)
        bond_force.setUsesPeriodicBoundaryConditions(False)
        self.system.addForce(bond_force)

    def add_custom_angle_force(self, i, j, k, value, forceconstant):
        """Restrain the i-j-k angle harmonically."""
        logger.info(
            f"Adding custom angle force for atoms: {i}, {j}, {k}  with value: {value} radians with "
            f"forceconstant={forceconstant}"
        )
        angle_force = openmm.CustomAngleForce("0.5*k*(theta-theta0)^2")
        angle_force.addGlobalParameter("k", forceconstant * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2)
        angle_force.addGlobalParameter("theta0", value * openmm.unit.radian)
        angle_force.addAngle(i, j, k)
        angle_force.setUsesPeriodicBoundaryConditions(False)
        self.system.addForce(angle_force)

    def add_custom_torsion_force(self, i, j, k, l, value, forceconstant):  # noqa: E741 - torsion atoms i-j-k-l
        """Restrain the i-j-k-l dihedral harmonically."""
        import math

        logger.info(f"Adding custom torsion force for atoms: {i}, {j}, {k}, {l}  with forceconstant={forceconstant}")
        torsion_force = openmm.CustomTorsionForce(
            "0.5*k*dtheta^2; dtheta = min(diff, 2*Pi-diff); diff = abs(theta - theta0)"
        )
        # Note: using global here, should be fine 1 torsion
        torsion_force.addGlobalParameter("Pi", math.pi)
        torsion_force.addGlobalParameter("k", forceconstant * openmm.unit.kilocalorie_per_mole / openmm.unit.radian**2)
        torsion_force.addGlobalParameter("theta0", value * openmm.unit.radian)
        torsion_force.addTorsion(i, j, k, l)
        logger.info("torsion_force getTorsionParameters: %s", torsion_force.getTorsionParameters(0))
        torsion_force.setUsesPeriodicBoundaryConditions(True)
        self.system.addForce(torsion_force)

    # This is custom external force that restrains group of atoms to center of system
    # Note: has flatbottom properties
    def add_centerforce(self, center_coords=None, atomindices=None, forceconstant=1.0, distance=5.0):
        """Tether atoms to a fixed point in space beyond a given distance."""
        logger.info("add_centerforce:")
        logger.info("Center coordinates: %s", center_coords)
        logger.info("Force acting on atomindices: %s", atomindices)
        logger.info(f"Forceconstant: {forceconstant} kcal/mol/Ang^2")
        logger.info(f"Force acting at values larger than {distance} Ang:")
        if self.periodic is True:
            centerforce = openmm.CustomExternalForce("0.5*k * max(0,periodicdistance(x, y, z, x0, y0, z0) - r0)^2")
        else:
            centerforce = openmm.CustomExternalForce("0.5*k * max(0,((x-x0)^2+(y-y0)^2+(z-z0)^2)-r0)^2")
        centerforce.addGlobalParameter(
            "k", forceconstant * openmm.unit.kilocalories_per_mole / openmm.unit.angstroms**2
        )
        centerforce.addGlobalParameter("r0", distance * openmm.unit.angstrom)
        centerforce.addPerParticleParameter("x0")
        centerforce.addPerParticleParameter("y0")
        centerforce.addPerParticleParameter("z0")
        center_x = center_coords[0] / 10
        center_y = center_coords[1] / 10
        center_z = center_coords[2] / 10
        for i in atomindices:
            centerforce.addParticle(i, openmm.Vec3(center_x, center_y, center_z))
        self.system.addForce(centerforce)
        logger.info("Added center force")
        return centerforce

    def add_custom_external_force(self):
        """Add the per-atom external force that carries the QM/MM gradient."""
        customforce = openmm.CustomExternalForce("-x*fx -y*fy -z*fz")
        customforce.addPerParticleParameter("fx")
        customforce.addPerParticleParameter("fy")
        customforce.addPerParticleParameter("fz")
        for i in range(self.system.getNumParticles()):
            customforce.addParticle(i, np.array([0.0, 0.0, 0.0]))
        self.system.addForce(customforce)
        # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CustomExternalForce.html

        logger.info("Added force")
        return customforce

    # NOTE: This setParticleParameters takes some time but not sure we can make this faster
    def update_custom_external_force(self, customforce, gradient, simulation):
        """Push a new gradient into the QM/MM external force."""
        logger.info("Updating custom external force")
        forces = -gradient * openmmqmmm.constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
        for i, f in enumerate(forces):
            customforce.setParticleParameters(i, i, f)
        customforce.updateParametersInContext(simulation.context)

    def add_bondrestraints(self, restraints=None):
        """Add harmonic distance restraints between atom pairs."""
        logger.info("Adding restraints: %s", restraints)

        new_restraints = openmm.HarmonicBondForce()
        for i, j, d, k in restraints:
            logger.info(
                f"Adding bond restraint between atoms {i} and {j}. Distance value: {d} Å. Force constant: {k} "
                f"kcal/mol*Å^-2"
            )
            new_restraints.addBond(
                i, j, d * openmm.unit.angstroms, k * openmm.unit.kilocalories_per_mole / openmm.unit.angstroms**2
            )
        self.system.addForce(new_restraints)

    def add_bondconstraints(self, constraints=None):
        """Constrain bond lengths rigidly (not harmonically)."""
        for i, j, d in constraints:
            logger.info(f"Adding bond constraint between atoms {i} and {j}. Distance value: {d:.4f} Å")
            self.system.addConstraint(i, j, d * openmm.unit.angstroms)

    def remove_all_constraints(self):
        """Remove every distance constraint from the system."""
        # Removing in reverse: each removal renumbers the constraints above it
        for index in reversed(range(self.system.getNumConstraints())):
            self.system.removeConstraint(index)

    def remove_constraints_for_atoms(self, atoms):
        """Remove every constraint involving any of the given atoms."""
        logger.info("Removing constraints in OpenMM object for atoms: %s", atoms)
        todelete = []
        for i in range(self.system.getNumConstraints()):
            con = self.system.getConstraintParameters(i)
            if con[0] in atoms or con[1] in atoms:
                todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Function to freeze atoms during OpenMM MD simulation. Sets masses to zero. Does not modify potential
    # energy-function.
    def freeze_atoms(self, frozen_atoms=None):
        """Freeze atoms by setting their mass to zero."""
        logger.info(f"Freezing {len(frozen_atoms)} atoms by setting particles masses to zero.")

        for i in frozen_atoms:
            self.system.setParticleMass(i, 0 * openmm.unit.daltons)

        # Also adding exceptions to nonbonded force to avoid interactions between frozen atoms (causes problems
        # otherwise in NPT)
        logger.info(
            "Also adding exceptions to nonbonded force for frozen atoms to avoid interactions between them (avoids "
            "problems in NPT)."
        )
        self.addexceptions(frozen_atoms)

    def modify_masses(self, changed_masses=None):
        """Set new masses for selected atoms, e.g. for hydrogen-mass repartitioning."""
        logger.info("Modify masses according:  %s", changed_masses)
        for am in changed_masses:
            self.system.setParticleMass(am, changed_masses[am] * openmm.unit.daltons)

    def unfreeze_atoms(self):
        """Restore the masses of atoms previously frozen by freeze_atoms."""
        for atom, mass in zip(self.allatoms, self.system_masses_original, strict=False):
            self.system.setParticleMass(atom, mass)

    # This removes interactions between particles in a region (e.g. QM-QM or frozen-frozen pairs)
    def addexceptions(self, atomlist):
        """Exclude the listed atoms from all nonbonded interactions with each other."""
        timeA = time.time()
        logger.info("Add exceptions/exclusions. Removing i-j interactions for list: %s atoms", len(atomlist))

        numexceptions = 0
        numexclusions = 0
        logger.debug("self.system.getForces()  %s", self.system.getForces())

        for force in self.system.getForces():
            logger.debug("force: %s", force)
            if isinstance(force, openmm.NonbondedForce):
                logger.info("Case Nonbondedforce. Adding Exception for ij pair.")
                for idx_i, i in enumerate(atomlist):
                    for j in atomlist[idx_i + 1 :]:
                        logger.debug(f"i,j : {i} and {j} ")
                        force.addException(i, j, 0, 0, 0, replace=True)

                        # NOTE: Case where there is also a CustomNonbonded force present (GROMACS interface).
                        # Then we have to add exclusion there too to avoid this issue: https://github.com/choderalab/perses/issues/357
                        # Basically both nonbonded forces have to have same exclusions (or exception where chargepro=0,

                        numexceptions += 1
            elif isinstance(force, openmm.CustomNonbondedForce):
                # Only applies to system with CustomNonbondedForce: GROMACS-setup, CHARMM-from-XML
                logger.info("Case CustomNonbondedforce. Adding Exclusion for kl pair.")
                # Using set of frozensets to get unique pairs
                all_exclusions = [
                    force.getExclusionParticles(exclindex) for exclindex in range(force.getNumExclusions())
                ]
                existing_exclusions = {frozenset(excl) for excl in all_exclusions}
                for idx_a, atom_a in enumerate(atomlist):
                    for atom_b in atomlist[idx_a + 1 :]:
                        if frozenset((atom_a, atom_b)) not in existing_exclusions:
                            existing_exclusions.add(frozenset([atom_a, atom_b]))
                            force.addExclusion(atom_a, atom_b)
                            numexclusions += 1
        logger.info("Number of exceptions (Nonbondedforce) added: %s", numexceptions)
        logger.info("Number of exclusions (CustomNonbondedforce) added: %s", numexclusions)
        logger.debug("self.system.getForces()  %s", self.system.getForces())
        log_time_since(timeA, "add exceptions")

    def set_simulation_parameters(
        self, timestep=0.001, coupling_frequency=1, temperature=300, integrator="VerletIntegrator"
    ):
        """Set the integrator and thermostat parameters used by create_simulation."""
        self.timestep = timestep
        self.coupling_frequency = coupling_frequency
        self.temperature = temperature
        self.integrator_name = integrator

    def create_integrator(self):
        # NOTE: Integrator definition has to be here (instead of set_simulation_parameters) as it has to be recreated
        # for each updated simulation
        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
        # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        """Create the OpenMM integrator from the stored simulation parameters."""
        if self.integrator_name == "VerletIntegrator":
            self.integrator = openmm.VerletIntegrator(self.timestep * openmm.unit.picoseconds)
        elif self.integrator_name == "VariableVerletIntegrator":
            self.integrator = openmm.VariableVerletIntegrator(self.timestep * openmm.unit.picoseconds)
        elif self.integrator_name == "LangevinIntegrator":
            self.integrator = openmm.LangevinIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.timestep * openmm.unit.picoseconds,
            )
        elif self.integrator_name == "LangevinMiddleIntegrator":
            # openmm recommended with 4 fs timestep, Hbonds 1/ps friction
            self.integrator = openmm.LangevinMiddleIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.timestep * openmm.unit.picoseconds,
            )
        elif self.integrator_name == "NoseHooverIntegrator":
            self.integrator = openmm.NoseHooverIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.timestep * openmm.unit.picoseconds,
            )
        # NOTE: Problem with Brownian, disabling
        elif self.integrator_name == "VariableLangevinIntegrator":
            self.integrator = openmm.VariableLangevinIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.timestep * openmm.unit.picoseconds,
            )
        elif self.integrator_name == "DrudeLangevinIntegrator":
            self.integrator = openmm.DrudeLangevinIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.temperature * openmm.unit.kelvin,
                self.timestep * openmm.unit.picoseconds,
                4,
            )
        elif self.integrator_name == "RPMDIntegrator":
            logger.info("RPMDIntegrator will be used")
            num_constraints = self.system.getNumConstraints()
            if num_constraints:
                raise InputError(
                    f"RPMDIntegrator does not support constraints, but the OpenMM System contains "
                    f"{num_constraints}. Create OpenMMTheory with autoconstraints=None, rigidwater=False, "
                    "and without bondconstraints."
                )
            logger.info(f"RPMD number of copies set to {self.rpmd_num_copies}. Use rpmd_num_copies keyword to change")
            self.integrator = openmm.RPMDIntegrator(
                self.rpmd_num_copies,
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.timestep * openmm.unit.picoseconds,
            )
        else:
            raise InputError(
                "Unknown integrator.\n Valid integrator keywords are: VerletIntegrator, VariableVerletIntegrator, "
                "LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VariableLangevinIntegrator, "
                "RPMDIntegrator"
            )

    def create_simulation(self, internal=False):
        """Build the OpenMM Simulation from the current system, topology and integrator."""
        timeA = time.time()

        logger.info(sub_header("Creating/updating OpenMM simulation object"))
        logger.info("Integrator name: %s", self.integrator_name)
        logger.info("Timestep: %s", self.timestep)
        logger.info("Temperature: %s", self.temperature)
        logger.info("Coupling frequency: %s", self.coupling_frequency)
        logger.info("Properties: %s", self.properties)
        logger.info("Topology: %s", self.topology)
        logger.debug("self.system.getForces()  %s", self.system.getForces())

        self.create_integrator()

        # Create simulation, either as part of OpenMMTheory (not picklable)
        # or not (used by run method)
        if internal is True:
            self.simulation = openmm.app.simulation.Simulation(
                self.topology,
                self.system,
                self.integrator,
                openmm.Platform.getPlatformByName(self.platform_choice),
                self.properties,
            )
            return None
        simulation = openmm.app.simulation.Simulation(
            self.topology,
            self.system,
            self.integrator,
            openmm.Platform.getPlatformByName(self.platform_choice),
            self.properties,
        )
        log_time_since(timeA, "creating/updating simulation")
        return simulation

    def forcegroupify(self):
        """Assign each force its own force group so their energies can be separated."""
        self.forcegroups = {}
        logger.info("inside forcegroupify")
        logger.debug("System forces: %s", self.system.getForces())
        logger.info("Number of forces:\n %s", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i

    def get_energy_decomposition(self, context):
        """Return the potential energy of each force group."""
        energies = {}
        for f, i in self.forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
        return energies

    def print_energy_decomposition(self, simulation):
        """Log the per-force energy breakdown of the current state."""
        timeA = time.time()
        # NOTE: Calling this is expensive (seconds)as the energy has to be recalculated.
        openmm_energy = {}
        energycomp = self.get_energy_decomposition(simulation.context)
        logger.info("")
        for comp in energycomp.items():
            openmm_energy[comp[0].getName()] = comp[1]

        sumofallcomponents = 0.0
        for val in openmm_energy.values():
            sumofallcomponents += val._value

        logger.info(f"{'Component':<20} | {'kJ/mol':<15} | {'kcal/mol':<15}")
        logger.info("%s", "-" * 56)
        for name in sorted(openmm_energy):
            logger.info(
                f"{name:<20} | {openmm_energy[name] / openmm.unit.kilojoules_per_mole:>15.2f} | "
                f"{openmm_energy[name] / openmm.unit.kilocalorie_per_mole:>15.2f}"
            )
        logger.info("%s", "-" * 56)
        sum_kcal = sumofallcomponents / openmmqmmm.constants.KCAL_TO_KJ
        logger.info(f"{'Sumcomponents':<20} | {sumofallcomponents:>15.2f} | {sum_kcal:>15.2f}")
        logger.info("")
        logger.info(
            f"{'Total':<20} | {self.energy * openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL:>15.2f} | "
            f"{self.energy * openmmqmmm.constants.HARTREE_TO_KCAL_PER_MOL:>15.2f}"
        )
        logger.info("")
        openmm_energy["Sum"] = sumofallcomponents
        log_time_since(timeA, "energy decomposition")

    def compute_dof(self):
        """Return the number of degrees of freedom, accounting for constraints and frozen atoms."""
        dof = 0
        for i in range(self.system.getNumParticles()):
            if self.system.getParticleMass(i) > 0 * openmm.unit.dalton:
                dof += 3
        for i in range(self.system.getNumConstraints()):
            p1, p2, _distance = self.system.getConstraintParameters(i)
            if (
                self.system.getParticleMass(p1) > 0 * openmm.unit.dalton
                or self.system.getParticleMass(p2) > 0 * openmm.unit.dalton
            ):
                dof -= 1
        if any(isinstance(self.system.getForce(i), openmm.CMMotionRemover) for i in range(self.system.getNumForces())):
            dof -= 3
        self.dof = dof

    def compute_cell_gradient_fd(self, context, eps=1e-4):
        """Compute the gradient with respect to the cell vectors by finite differences."""
        KJMOL_TO_EH = 1.0 / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
        eps_nm = eps * openmmqmmm.constants.BOHR_TO_NM

        state = context.getState(getEnergy=True, getPositions=True)
        E0 = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole) * KJMOL_TO_EH
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(openmm.unit.nanometer)  # (3,3) in nm
        logger.info("box: %s", box)

        # Only lower-triangular indices are valid for OpenMM triclinic box
        valid_indices = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]

        grad = np.zeros((3, 3))
        for i, j in valid_indices:
            box_pert = box.copy()
            box_pert[i, j] += eps_nm
            context.setPeriodicBoxVectors(*box_pert)
            E_plus = (
                context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                * KJMOL_TO_EH
            )
            grad[i, j] = (E_plus - E0) / eps  # dE[Eh] / dh[Bohr]
            context.setPeriodicBoxVectors(*box)  # restore
        return grad  # Eh/Bohr

    def get_cell_gradient(self):
        """Return the most recently computed cell gradient."""
        logger.info("Inside get_cell_gradient")
        # Using self.stored_context (should have been defined by .run call)
        self.cell_gradient = self.compute_cell_gradient_fd(self.stored_context, eps=1e-4)
        logger.info("OpenMM cell gradient: %s", self.cell_gradient)
        return self.cell_gradient

    # NOTE: Adding charge/mult/PC here to  be consistent with QM_theories. Not used
    def run(
        self,
        *,
        current_coords=None,
        elems=None,
        grad=False,
        fragment=None,
        qmatoms=None,
        label=None,
        charge=None,
        mult=None,
        pc=False,
        current_mm_coords=None,
        mm_charges=None,
        qm_elems=None,
        numcores=1,
    ):
        """Compute the MM energy (and gradient) of a geometry."""
        module_init_time = time.time()
        timeA = time.time()

        simulation = self.create_simulation()

        logger.info(sub_header("Running Single-point OpenMM Interface"))
        # If no coords given to run then a single-point job probably (not part of Optimizer or MD which would supply
        # coords). Then try if fragment object was supplied.
        # Otherwise internal coords if they exist
        if current_coords is None:
            if fragment is None:
                if len(self.coords) != 0:
                    logger.info("Using internal coordinates (from OpenMM object).")
                    current_coords = self.coords
                else:
                    raise FileFormatError("Found no coordinates!")
            else:
                current_coords = fragment.coords

        # IMPORTANT: Checking whether constraints have been defined in OpenMM object
        # Defined OpenMM constraints will not work within a Single-point run scheme
        # In fact forces will be all wrong. Thus checking before continuing
        # Constraints and frozen atoms have to instead by enforced by geomeTRICOptimizer, non-OpenMM dynamics module
        # etc.
        defined_constraints = self.system.getNumConstraints()
        logger.info("Number of OpenMM system constraints defined: %s", defined_constraints)

        if self.autoconstraints is not None or self.rigidwater:
            logger.error(
                "OpenMM autoconstraints (HBonds,AllBonds,HAngles) in OpenMMTheory are not compatible with "
                "OpenMMTheory.run()"
            )
            logger.warning("Please redefine OpenMMTheory object: autoconstraints=None, rigidwater=False")
            if self.force_run is True:
                logger.info("force_run is True. Will continue")
            else:
                raise InputError(
                    "OpenMMTheory constraints/frozen-atoms are incompatible with this run. "
                    "Redefine OpenMMTheory with autoconstraints=None, rigidwater=False (or pass force_run=True)"
                )

        if self.user_frozen_atoms or self.user_constraints or self.user_restraints:
            logger.info(
                "User-defined frozen atoms/constraints/restraints in OpemmTheory are not compatible with "
                "OpenMMTheory.run()"
            )
            logger.info(
                "Constraints must instead be defined inside the program that called OpenMMtheory.run(), e.g. "
                "geomeTRICOptimizer."
            )
            if self.force_run is True:
                logger.info("force_run is True. Will continue")
            else:
                raise InputError(
                    "OpenMMTheory constraints/frozen-atoms are incompatible with this run. "
                    "Redefine OpenMMTheory with autoconstraints=None, rigidwater=False (or pass force_run=True)"
                )
        if defined_constraints != 0:
            logger.error("OpenMM constraints not zero. Exiting.")
            if self.force_run is True:
                logger.info("force_run is True. Will continue")
            else:
                raise InputError(
                    "OpenMMTheory constraints/frozen-atoms are incompatible with this run. "
                    "Redefine OpenMMTheory with autoconstraints=None, rigidwater=False (or pass force_run=True)"
                )

        log_time_since(timeA, "OpenMMTheory.run: const-check")
        current_coords = np.array(current_coords)
        factor = -openmmqmmm.constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
        logger.info("Updating coordinates.")
        timeA = time.time()

        self.set_positions(current_coords, simulation)

        log_time_since(timeA, "Updating MM positions")
        timeA = time.time()
        # While these distance constraints should not matter, applying them makes the energy function agree with
        # previous benchmarking for bonded and nonbonded
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5549999/
        # Using 1e-6 hardcoded value since how used in paper
        # NOTE: Weirdly, applyconstraints is True result in constraints for TIP3P disappearing
        if self.applyconstraints_in_run is True:
            logger.info("Applying constraints before calculating MM energy.")
            simulation.context.applyConstraints(1e-6)
            log_time_since(timeA, "context: apply constraints")
            timeA = time.time()

        logger.info("Calling OpenMM getState.")
        if grad is True:
            state = simulation.context.getState(getEnergy=True, getForces=True)
            self.energy = (
                state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
            )
            self.gradient = np.array(state.getForces(asNumpy=True) / factor)
        else:
            state = simulation.context.getState(getEnergy=True, getForces=False)
            self.energy = (
                state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
            )

        log_time_since(timeA, "OpenMM getState")

        logger.info("OpenMM Energy: %s Eh", self.energy)
        logger.info("OpenMM Energy: %s kcal/mol", self.energy * openmmqmmm.constants.HARTREE_TO_KCAL_PER_MOL)

        # Do energy components or not. Can be turned off for e.g. MM MD simulation
        if self.do_energy_decomposition is True:
            self.print_energy_decomposition(simulation)
        logger.info(small_header("Ending OpenMM interface"))
        log_time_since(module_init_time, "OpenMM run")
        if grad is True:
            return self.energy, self.gradient
        return self.energy

    def getatomcharges(self):
        """Return the partial charge of every atom, in elementary charges."""
        chargelist = []
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for i in range(force.getNumParticles()):
                    charge = force.getParticleParameters(i)[0]
                    if isinstance(charge, openmm.unit.Quantity):
                        charge = charge / openmm.unit.elementary_charge
                        chargelist.append(charge)
                self.charges = chargelist
        return chargelist

    # Used to delete Coulomb interactions involving QM-QM and QM-MM atoms
    def delete_exceptions(self, atomlist):
        """Remove the nonbonded exceptions previously added for the listed atoms."""
        timeA = time.time()
        logger.info("Deleting Coulombexceptions for atomlist: %s", atomlist)
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                for exc in range(force.getNumExceptions()):
                    p1, p2, chargeprod, sigmaij, epsilonij = force.getExceptionParameters(exc)
                    if p1 in atomlist or p2 in atomlist:
                        chargeprod._value = 0.0
                        force.setExceptionParameters(exc, p1, p2, chargeprod, sigmaij, epsilonij)
        log_time_since(timeA, "delete_exceptions")

    # Updating LJ interactions in OpenMM object. Used to set LJ sites to zero e.g. so that they do not contribute
    # Can be used to get QM-MM LJ interaction energy
    def update_lj_epsilons(self, atomlist, epsilons):
        """Set new Lennard-Jones epsilon values for selected atoms."""
        timeA = time.time()
        logger.info("Updating LJ interaction strengths in OpenMM object.")
        if len(atomlist) != len(epsilons):
            raise InternalError("atomlist and epsilons size mismatch")
        for atomindex, newepsilon in zip(atomlist, epsilons, strict=False):
            charge, sigma, _oldepsilon = self.nonbonded_force.getParticleParameters(atomindex)
            if isinstance(self.nonbonded_force, openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [charge, sigma, newepsilon])
            elif isinstance(self.nonbonded_force, openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, charge, sigma, newepsilon)

        logger.debug("done here")
        log_time_since(timeA, "update_LJ_epsilons")

    # Taking list of atom-indices and list of charges (usually zero) and setting new charge
    # Note: Exceptions also needs to be dealt with (see delete_exceptions)
    def update_charges(self, atomlist, atomcharges):
        """Set new partial charges for selected atoms."""
        timeA = time.time()
        logger.info("Updating charges in OpenMM object.")
        if len(atomlist) != len(atomcharges):
            raise InternalError("atomlist and atomcharges size mismatch")
        for atomindex, newcharge in zip(atomlist, atomcharges, strict=False):
            self.charges[atomindex] = newcharge
            _oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(atomindex)
            if isinstance(self.nonbonded_force, openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [newcharge, sigma, epsilon])
            elif isinstance(self.nonbonded_force, openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, newcharge, sigma, epsilon)

        # Instead of recreating simulation we can just update like this:
        logger.info("Updating simulation object for modified Nonbonded force.")
        logger.debug("self.nonbonded_force: %s", self.nonbonded_force)
        logger.debug("Forces in system after charge update: %s", self.system.getForces())
        log_time_since(timeA, "update_charges")

    def modify_bonded_forces(self, atomlist):
        """Zero the bonded terms that lie entirely inside the given atom set."""
        timeA = time.time()
        logger.info("Modifying bonded forces.")
        logger.info("")
        # This is typically used by QM/MM object to set bonded forces to zero for qmatoms (atomlist)
        # Mimicking: https://github.com/openmm/openmm/issues/2792

        numharmbondterms_removed = 0
        numharmangleterms_removed = 0
        numpertorsionterms_removed = 0
        numcustomtorsionterms_removed = 0
        numcmaptorsionterms_removed = 0
        numcustombondterms_removed = 0

        for force in self.system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                logger.debug("HarmonicBonded force")
                logger.debug(f"There are {force.getNumBonds()} HarmonicBond terms defined.")
                logger.debug("")
                # REVISIT: Neglecting QM-QM and sQM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    p1, p2, length, k = force.getBondParameters(i)
                    # or: delete QM-QM and QM-MM
                    # and: delete QM-QM

                    if self.delete_qm1_mm1_bonded is True:
                        exclude = p1 in atomlist or p2 in atomlist
                    else:
                        exclude = p1 in atomlist and p2 in atomlist
                    if exclude is True:
                        logger.debug("exclude True")
                        logger.debug("atomlist: %s", atomlist)
                        logger.debug("i: %s", i)
                        logger.debug(f"Before p1: {p1} p2: {p2} length: {length} k: {k}")
                        force.setBondParameters(i, p1, p2, length, 0)
                        numharmbondterms_removed += 1
                        p1, p2, length, k = force.getBondParameters(i)
                        logger.debug(f"After p1: {p1} p2: {p2} length: {length} k: {k}")
                        logger.debug("")
            elif isinstance(force, openmm.HarmonicAngleForce):
                logger.debug("HarmonicAngle force")
                logger.debug(f"There are {force.getNumAngles()} HarmonicAngle terms defined.")
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    presence = [i in atomlist for i in [p1, p2, p3]]
                    # Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    if presence.count(True) >= 2:
                        logger.debug("presence.count(True): %s", presence.count(True))
                        logger.debug("exclude True")
                        logger.debug("atomlist: %s", atomlist)
                        logger.debug("i: %s", i)
                        logger.debug(f"Before p1: {p1} p2: {p2} p3: {p3} angle: {angle} k: {k}")
                        force.setAngleParameters(i, p1, p2, p3, angle, 0)
                        numharmangleterms_removed += 1
                        p1, p2, p3, angle, k = force.getAngleParameters(i)
                        logger.debug(f"After p1: {p1} p2: {p2} p3: {p3} angle: {angle} k: {k}")
            elif isinstance(force, openmm.PeriodicTorsionForce):
                logger.debug("PeriodicTorsionForce force")
                logger.debug(f"There are {force.getNumTorsions()} PeriodicTorsionForce terms defined.")
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    if presence.count(True) >= 3:
                        logger.debug("Found torsion in QM-region")
                        logger.debug("presence.count(True): %s", presence.count(True))
                        logger.debug("exclude True")
                        logger.debug("atomlist: %s", atomlist)
                        logger.debug("i: %s", i)
                        logger.debug(
                            f"Before p1: {p1} p2: {p2} p3: {p3} p4: {p4} periodicity: {periodicity} phase: {phase} k: "
                            f"{k}"
                        )
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed += 1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        logger.debug(
                            f"After p1: {p1} p2: {p2} p3: {p3} p4: {p4} periodicity: {periodicity} phase: {phase} k: "
                            f"{k}"
                        )
            elif isinstance(force, openmm.CustomTorsionForce):
                logger.debug("CustomTorsionForce force")
                logger.debug(f"There are {force.getNumTorsions()} CustomTorsionForce terms defined.")
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    if presence.count(True) >= 3:
                        logger.debug("Found torsion in QM-region")
                        logger.debug("presence.count(True): %s", presence.count(True))
                        logger.debug("exclude True")
                        logger.debug("atomlist: %s", atomlist)
                        logger.debug("i: %s", i)
                        logger.debug(f"Before p1: {p1} p2: {p2} p3: {p3} p4: {p4} pars {pars}")
                        force.setTorsionParameters(i, p1, p2, p3, p4, (0.0, 0.0))
                        numcustomtorsionterms_removed += 1
                        p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                        logger.debug(f"After p1: {p1} p2: {p2} p3: {p3} p4: {p4} pars {pars}")
            elif isinstance(force, openmm.CMAPTorsionForce):
                logger.debug("CMAPTorsionForce force")
                logger.debug(f"There are {force.getNumTorsions()} CMAP terms defined.")
                logger.debug(f"There are {force.getNumMaps()} CMAP maps defined")
                # Note (RB). CMAP is between pairs of backbone dihedrals.
                # Not sure if we can delete the terms:
                # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CMAPTorsionForce.html
                for i in range(force.getNumTorsions()):
                    jj, p1, p2, p3, p4, v1, v2, v3, v4 = force.getTorsionParameters(i)
                    presence = [i in atomlist for i in [p1, p2, p3, p4, v1, v2, v3, v4]]
                    if presence.count(True) >= 4:
                        logger.debug(
                            f"jj: {jj} p1: {p1} p2: {p2} p3: {p3} p4: {p4}      v1: {v1} v2: {v2} v3: {v3} v4: {v4}"
                        )
                        logger.debug("presence: %s", presence)
                        logger.debug("Found CMAP torsion partner in QM-region")
                        logger.debug("Not deleting. To be revisited...")
            elif isinstance(force, openmm.CustomBondForce):
                logger.debug("CustomBondForce")
                logger.debug(f"There are {force.getNumBonds()} force terms defined.")
                # Neglecting QM1-MM1 interactions. i.e if one atom in bond-pair is QM we neglect
                for i in range(force.getNumBonds()):
                    p1, p2, params = force.getBondParameters(i)
                    exclude = p1 in atomlist and p2 in atomlist
                    if exclude is True:
                        # NOTE: list of parameters now set to 0.0 for any number of parameters
                        force.setBondParameters(i, p1, p2, [0.0 for _ in params])
                        numcustombondterms_removed += 1
                        p1, p2, params = force.getBondParameters(i)

        logger.info("")
        logger.info("Number of bonded terms removed:")
        logger.info("Harmonic Bond terms: %s", numharmbondterms_removed)
        logger.info("Harmonic Angle terms: %s", numharmangleterms_removed)
        logger.info("Periodic Torsion terms: %s", numpertorsionterms_removed)
        logger.info("Custom Torsion terms: %s", numcustomtorsionterms_removed)
        logger.info("CMAP Torsion terms: %s", numcmaptorsionterms_removed)
        logger.info("CustomBond terms %s", numcustombondterms_removed)
        logger.info("")
        log_time_since(timeA, "modify_bonded_forces")


class ForceReporter:
    def __init__(self, file, report_interval, atomic_units=False):
        self._out = open(file, "w")  # noqa: SIM115 - reporter handle, closed in __del__
        self._reportInterval = report_interval
        self.atomic_units = atomic_units

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):  # noqa: N802 - OpenMM reporter API, do not rename
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        forces = state.getForces().value_in_unit(openmm.unit.kilojoules / openmm.unit.mole / openmm.unit.nanometer)
        if self.atomic_units:
            forces = np.array(forces) / -openmmqmmm.constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM

        self._out.write(f"{len(forces):g}\n{energy:g}\n")
        for f in forces:
            self._out.write(f"{f[0]:g} {f[1]:g} {f[2]:g}\n")
        self._out.flush()


def clean_up_constraints_list(fragment=None, constraints=None):
    logger.info("Checking defined constraints.")
    newconstraints = []
    for con in constraints:
        if len(con) == 3:
            newconstraints.append(con)
        elif len(con) == 2:
            distance = distance_between_atoms(fragment=fragment, atoms=[con[0], con[1]])
            logger.info(f"Adding missing distance definition between atoms {con[0]} and {con[1]}: {distance:.4f}")
            newcon = [con[0], con[1], distance]
            newconstraints.append(newcon)
    return newconstraints


def write_xmlfile_nonbonded(
    *,
    resnames=None,
    atomnames_per_res=None,
    atomtypes_per_res=None,
    elements_per_res=None,
    masses_per_res=None,
    charges_per_res=None,
    sigmas_per_res=None,
    epsilons_per_res=None,
    filename="system.xml",
    coulomb14scale=0.833333,
    lj14scale=0.5,
    skip_nb=False,
    charmm=False,
):
    logger.info("Inside write_xmlfile_nonbonded")

    if not (len(resnames) == len(atomnames_per_res) == len(atomtypes_per_res)):
        raise InternalError("Residue name/atomname/atomtype lists size mismatch")

    atomtypelines = []
    for _resname, atomtypelist, elemlist, masslist in zip(
        resnames, atomtypes_per_res, elements_per_res, masses_per_res, strict=False
    ):
        for atype, elem, mass in zip(atomtypelist, elemlist, masslist, strict=False):
            atomtypeline = f'<Type name="{atype}" class="{atype}" element="{elem}" mass="{mass!s}"/>\n'
            if atomtypeline not in atomtypelines:
                atomtypelines.append(atomtypeline)
    nonbondedlines = []
    LJforcelines = []
    for _resname, atomtypelist, chargelist, sigmalist, epsilonlist in zip(
        resnames, atomtypes_per_res, charges_per_res, sigmas_per_res, epsilons_per_res, strict=False
    ):
        for atype, charge, sigma, epsilon in zip(atomtypelist, chargelist, sigmalist, epsilonlist, strict=False):
            if charmm:
                # LJ parameters zero here
                nonbondedline = f'<Atom type="{atype}" charge="{charge}" sigma="{0.0}" epsilon="{0.0}"/>\n'
                ljline = f'<Atom type="{atype}" sigma="{sigma}" epsilon="{epsilon}"/>\n'
                if nonbondedline not in nonbondedlines:
                    nonbondedlines.append(nonbondedline)
                if ljline not in LJforcelines:
                    LJforcelines.append(ljline)
            else:
                nonbondedline = f'<Atom type="{atype}" charge="{charge}" sigma="{sigma}" epsilon="{epsilon}"/>\n'
                if nonbondedline not in nonbondedlines:
                    nonbondedlines.append(nonbondedline)

    with open(filename, "w") as xmlfile:
        xmlfile.write("<ForceField>\n")
        xmlfile.write("<AtomTypes>\n")
        for atomtypeline in atomtypelines:
            xmlfile.write(atomtypeline)
        xmlfile.write("</AtomTypes>\n")
        xmlfile.write("<Residues>\n")
        for resname, atomnamelist, atomtypelist in zip(resnames, atomnames_per_res, atomtypes_per_res, strict=False):
            xmlfile.write(f'<Residue name="{resname}">\n')
            xmlfile.writelines(
                f'<Atom name="{atomname}" type="{atomtype}"/>\n'
                for i, (atomname, atomtype) in enumerate(zip(atomnamelist, atomtypelist, strict=False))
            )
            xmlfile.write("</Residue>\n")
        xmlfile.write("</Residues>\n")
        # Write nonbonded block (even if skip_nb is True)
        xmlfile.write(f'<NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}">\n')
        if skip_nb is False:
            if charmm:
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
                xmlfile.write("</NonbondedForce>\n")
                xmlfile.write(f'<LennardJonesForce lj14scale="{lj14scale}">\n')
                for ljline in LJforcelines:
                    xmlfile.write(ljline)
                xmlfile.write("</LennardJonesForce>\n")
            else:
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
        xmlfile.write("</NonbondedForce>\n")
        xmlfile.write("</ForceField>\n")
    logger.info("Wrote XML-file: %s", filename)
    return filename
