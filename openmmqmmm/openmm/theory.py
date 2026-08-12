"""OpenMMTheory: the OpenMM system, its forces, and MM energies and gradients."""

import copy
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
import openmmqmmm.plotting
from openmmqmmm.coords import (
    Fragment,
    define_dummy_topology,
    distance_between_atoms,
)
from openmmqmmm.coords_pbc import cell_params_to_vectors, cell_vectors_to_params
from openmmqmmm.utils import (
    log_time_since,
    main_header,
    small_header,
    sub_header,
)

logger = logging.getLogger(__name__)


class OpenMMTheory:
    """Interface to the OpenMM molecular-mechanics library.

    The system is defined from forcefield XML files plus a PDB file
    (xmlfiles=/pdbfile=), an OpenMM XML system file (xmlsystemfile=), or
    Amber/GROMACS/CHARMM files. Periodic and non-periodic systems supported.
    """

    def __init__(
        self,
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
        customnonbondedforce=False,
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
        time.time()

        # CPU: Control either by provided numcores keyword, or by setting env variable:
        # $OPENMM_CPU_THREADS in shell
        # before running.
        os.environ["OMP_NUM_THREADS"] = str(numcores)
        os.environ["OPENMM_CPU_THREADS"] = str(numcores)
        logger.info("OpenMM CPU threads set to: %s", os.environ["OMP_NUM_THREADS"])
        self.numcores = numcores  # Setting for general theory-interface compatibility

        # Indicate that this is a MMtheory
        self.theorytype = "MM"
        self.theorynamelabel = "OpenMM"
        self.analytic_hessian = False
        self.label = label
        self.fragment = fragment
        logger.info("Imported OpenMM library version: %s", openmm.__version__)
        if version.parse(openmm.__version__) < version.parse("8.1"):
            logger.warning("OpenMM version < 8.1. OpenMM 8.1 or higher is recommended")
            logger.info("Some features may not work as intended in older versions")

        # Early exits
        # TODO: To be removed
        if charmm_periodic_cell_dimensions is not None:
            raise InputError("charmm_periodic_cell_dimensions is deprecated. Use periodic_cell_dimensions instead")

        # OpenMM variables
        logger.info(sub_header("Defining OpenMM object"))
        # Initialize system
        self.system = None

        # Degrees of freedom of system (accounts for frozen atoms and constraints)
        # Will be set by compute_DOF
        self.dof = None

        # Autoconstraints when creating MM system: Default: None,  Options: Hbonds, AllBonds, HAng
        if autoconstraints == "HBonds":
            logger.info("HBonds option: X-H bond lengths will automatically be constrained")
            self.autoconstraints = openmm.app.HBonds
        elif autoconstraints == "AllBonds":
            logger.info("AllBonds option: All bond lengths will automatically be constrained")
            self.autoconstraints = openmm.app.AllBonds
        elif autoconstraints == "HAngles":
            logger.info("HAngles option: All bond lengths and H-X-H and H-O-X angles will automatically be constrained")
            self.autoconstraints = openmm.app.HAngles
        elif autoconstraints is None or autoconstraints == "None":
            logger.info("No automatic constraints")
            self.autoconstraints = None
        else:
            raise InputError("Unknown autoconstraints option")
        logger.info("AutoConstraint setting: %s", self.autoconstraints)

        # User constraints, restraints and frozen atoms
        self.user_frozen_atoms = []
        self.user_constraints = []
        self.user_restraints = []

        # Rigidwater constraints are on by default. Can be turned off
        self.rigidwater = rigidwater
        logger.info("Rigidwater constraints: %s", self.rigidwater)
        # Modify hydrogenmass or not
        if hydrogenmass is not None:
            self.hydrogenmass = hydrogenmass * openmm.unit.amu
        else:
            self.hydrogenmass = None
        logger.info("Hydrogenmass option: %s", self.hydrogenmass)

        # RPMD PIMD: Number of copies in ring polymer MD
        # Active when RPMDIntegrator is used
        self.rpmd_num_copies = rpmd_num_copies

        # Setting for controlling whether QM1-MM1 bonded terms are deleted or not in a QM/MM job
        # See modify_bonded_forces
        # TODO: Move option to module_QMMM instead
        self.delete_qm1_mm1_bonded = delete_qm1_mm1_bonded
        self.platform_choice = platform

        if properties is None:
            self.properties = {}
        else:
            self.properties = properties
        if self.platform_choice == "CPU":
            logger.info("Using platform: CPU")
            self.properties["Threads"] = str(numcores)
            # if numcores > 1:
            #        os.environ["OPENMM_CPU_THREADS"]))
            #        "OPENMM_CPU_THREADS environment variable not set.\nOpenMM will choose number of physical cores "
            #        "present.")
        else:
            logger.info("Using platform: %s", self.platform_choice)
        # Whether to do energy decomposition of MM energy or not. Takes time. Can be turned off for MD runs
        self.do_energy_decomposition = do_energy_decomposition

        # Initializing
        self.coords = []
        self.charges = []
        self.periodic = periodic
        self.periodic_nonbonded_cutoff = periodic_nonbonded_cutoff
        self.nonbonded_cutoff_no_pbc = nonbonded_cutoff_no_pbc
        # Methods for nonbonded interactions, PBC and no-PBC
        self.nonbonded_method_pbc = nonbonded_method_pbc
        self.nonbonded_method_no_pbc = nonbonded_method_no_pbc
        self.ewalderrortolerance = ewalderrortolerance

        # Whether to apply constraints or not when calculating MM energy via run method (does not apply to OpenMM MD)
        # NOTE: Should be False in general. Only True for special cases
        self.applyconstraints_in_run = applyconstraints_in_run

        # Switching function distance in Angstrom
        self.switching_function_distance = switching_function_distance

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
        self.Forcefield = None
        # What type of forcefield files to read. Reads in different way.
        logger.info(sub_header("Setting up force fields."))
        logger.info(
            "Note: OpenMM will fail in this step if parameters are missing in topology and\n"
            "      parameter files (e.g. nonbonded entries).\n"
        )

        # Initializing
        pdb_pbc_vectors = None

        # Phasing out PBCvectors
        if pbc_vectors is not None:
            logger.warning("PBCvectors keyword is on its way out. Use periodic_cell_vectors instead")
            if periodic_cell_vectors is None:
                periodic_cell_vectors = pbc_vectors

        # #Always creates object we call self.forcefield that contains topology attribute
        if charmm_files is True:
            logger.info("Reading CHARMM files.")
            self.psffile = psffile
            if use_parmed is True:
                import parmed

                logger.info("Using Parmed.")
                self.psf = parmed.charmm.CharmmPsfFile(psffile)
                # Permissive True means less restrictive about atomtypes
                # Removed , permissive=True, no longer in parmed
                self.params = parmed.charmm.CharmmParameterSet(charmmtopfile, charmmprmfile)
                # Grab resnames from psf-object. Different for parmed object
                # Note: OpenMM uses 0-indexing
                self.resnames = [self.psf.atoms[i].residue.name for i in range(len(self.psf.atoms))]
                self.resids = [self.psf.atoms[i].residue.idx for i in range(len(self.psf.atoms))]
                self.segmentnames = [self.psf.atoms[i].residue.segid for i in range(len(self.psf.atoms))]
                self.atomtypes = [i.type for i in self.psf.atoms]
                # TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames = [self.psf.atoms[i].name for i in range(len(self.psf.atoms))]

                # TODO: Elements are unset here. Parmed parses things differently
                # NOTE: we could deduce element from atomname or mass
            else:
                # Load CHARMM PSF files via native routine.
                self.psf = openmm.app.CharmmPsfFile(psffile)
                self.params = openmm.app.CharmmParameterSet(charmmtopfile, charmmprmfile, permissive=True)
                # Grab resnames from psf-object
                self.resnames = [self.psf.atom_list[i].residue.resname for i in range(len(self.psf.atom_list))]
                self.resids = [self.psf.atom_list[i].residue.idx for i in range(len(self.psf.atom_list))]
                self.segmentnames = [self.psf.atom_list[i].system for i in range(len(self.psf.atom_list))]
                self.atomtypes = [self.psf.atom_list[i].attype for i in range(len(self.psf.atom_list))]
                # TODO: Note: For atomnames it seems OpenMM converts atomnames to its own. Perhaps not useful
                self.atomnames = [self.psf.atom_list[i].name for i in range(len(self.psf.atom_list))]
                self.define_mm_elements(self.psf.topology)

            self.topology = self.psf.topology
            self.forcefield = self.psf

        elif gromacs_files is True:
            logger.info("Reading Gromacs files.")
            # Reading grofile, not for coordinates but for periodic vectors
            if use_parmed is True:
                import parmed

                logger.info("Using Parmed.")
                logger.info("GROMACS top dir: %s", gromacstopdir)
                parmed.gromacs.GROMACS_TOPDIR = gromacstopdir
                logger.info("Reading GROMACS GRO file: %s", grofile)
                gmx_gro = parmed.gromacs.GromacsGroFile.parse(grofile)
                logger.info("Reading GROMACS topology file: %s", gromacstopfile)
                gmx_top = parmed.gromacs.GromacsTopologyFile(gromacstopfile)

                # Getting PBC parameters
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

            # TODO: Define resnames, resids, segmentnames, atomtypes, atomnames??
            self.define_mm_elements(self.topology)
        elif amber_files is True:
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

                # if float(openmm.__version__) >= 8.1:
                if version.parse(openmm.__version__) >= version.parse("8.1"):
                    if periodic_cell_vectors is None:
                        temp_pbc_vecs = None
                    else:
                        temp_pbc_vecs = periodic_cell_vectors * openmm.unit.angstrom  # Adding units
                    # If cell dims provided instead
                    if periodic_cell_dimensions is None:
                        temp_pbc_cell_value = None
                    else:
                        # This works despite specifying Angstrom units for all cell dimensions
                        temp_pbc_cell_value = periodic_cell_dimensions * openmm.unit.angstrom
                    # Providing PBC data upon prmtop object creaction (avoids some hazzles)
                    # PBC data is further handled later
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
            # NOTE: OpenMM does not grab Amber atomtypes for some reason. Feature request
            # TODO: Grab more topology information
            # TODO: Define segmentnames, atomtypes,

        elif topoforce is True:
            logger.info("Using forcefield info from topology and forcefield keyword.")
            if topology is not None:
                logger.info("Topology provided as keyword")
                self.topology = topology
            else:
                logger.info("No topology provided as keyword")
                logger.info("Reading topology from PDB-file instead")
                pdb = openmm.app.PDBFile(pdbfile)
                self.topology = pdb.topology
                # Check if PBC vectors in PDB-file
                pdb_pbc_vectors = pdb.topology.getPeriodicBoxVectors()
            self.forcefield = forcefield
            self.define_mm_elements(self.topology)

        # Load XMLfile for whole system
        elif xmlsystemfile is not None:
            logger.info("Reading system XML file: %s", xmlsystemfile)
            with open(xmlsystemfile) as xmlfh:
                xmlsystemfileobj = xmlfh.read()
            # Deserialize the XML text to create a System object.
            logger.info("Now defining OpenMM system using information in file")
            logger.warning("File may contain hardcoded constraints that can not be overridden.")
            self.system = openmm.XmlSerializer.deserializeSystem(xmlsystemfileobj)
            # NOTE: Big drawback of xmlsystemfile is that constraints have been hardcoded and can
            # NOTE: we could remove all present constraints using: self.remove_all_constraints()
            # NOTE: However, not sure how easy to enforce Hatom, rigidwater etc. constraints again without remaking
            # system object
            # NOTE: Maybe define system object using XmlSerializer, somehow create forcefield object from it.
            # NOTE: Then recreate system below. Not sure if possible

            # TODO: set further properties of system here, e.g. PME parameters
            # otherwise system is not completely set

            # We still need topology from somewhere to using pdbfile
            logger.info("Reading topology from PDBfile: %s", pdbfile)
            pdb = openmm.app.PDBFile(pdbfile)
            self.topology = pdb.topology
            self.define_mm_elements(self.topology)
            # Check if PBC vectors in PDB-file
            pdb_pbc_vectors = pdb.topology.getPeriodicBoxVectors()
        # Simple OpenMM system without any forcefield defined. Requires fragment
        # Used for OpenMM_MD with QM Hamiltonian
        elif dummysystem is True:
            # Create list of atomnames, used in PDB topology and XML file
            atomnames_full = [j + str(i) for i, j in enumerate(fragment.elems)]
            # Write PDB-file frag.pdb with dummy atomnames
            # Load PDB-file and create topology

            # Creating new
            self.topology = define_dummy_topology(fragment.elems)

            # Create dummy XML file
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
            # Create dummy forcefield
            self.forcefield = openmm.app.ForceField(xmlfile)
            self.define_mm_elements(self.topology)

        # Read topology from PDB-file or PDBx-file and XML-forcefield files to define forcefield
        else:
            logger.info("Reading OpenMM XML forcefield files and PDB (or PDBx) file")
            logger.info("xmlfiles: %s", str(xmlfiles).strip("[]"))
            logger.info("pdbfile: %s", pdbfile)
            logger.info("pdbxfile: %s", pdbxfile)
            # This would be regular OpenMM Forcefield definition requiring XML file
            # Topology from PDBfile annoyingly enough
            if pdbfile is not None:
                pdb = openmm.app.PDBFile(pdbfile)
            elif pdbxfile is not None:
                pdb = openmm.app.PDBxFile(pdbxfile)
            else:
                raise InputError("Error: No pdbfile or pdbxfile input provided")

            # Check if PBC vectors in PDB-file
            pdb_pbc_vectors = pdb.topology.getPeriodicBoxVectors()

            self.topology = pdb.topology
            self.forcefield = openmm.app.ForceField(*xmlfiles)
            # Defining some things. resids is used by actregiondefine
            self.resids = [i.residue.index for i in self.topology.atoms()]
            self.resnames = [i.residue.name for i in self.topology.atoms()]
            self.atomnames = [i.name for i in self.topology.atoms()]
            self.define_mm_elements(self.topology)

        # Dealing with possible user-defined residuetemplate_choice
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
        # NOW CREATE SYSTEM UNLESS already created (xmlsystemfile)
        if self.system is None:
            # Periodic or non-periodic ystem
            if self.periodic is True:
                logger.info("System is periodic.")
                logger.info(sub_header("Setting up periodicity."))
                # Inspect and set PBC in self.topology and self.forcefield
                # Necessary for system creation with periodics (otherwise failure)
                self.set_periodics_before_system_creation(
                    periodic_cell_vectors,
                    pdb_pbc_vectors,
                    periodic_cell_dimensions,
                    charmm_files,
                    amber_files,
                    use_parmed,
                )

                # Nonbonded method to use for PBC
                if self.nonbonded_method_pbc == "PME":
                    nonb_method_PBC = openmm.app.PME
                elif self.nonbonded_method_pbc == "Ewald":
                    nonb_method_PBC = openmm.app.Ewald
                elif self.nonbonded_method_pbc == "LJPME":
                    nonb_method_PBC = openmm.app.LJPME
                elif self.nonbonded_method_pbc == "CutoffPeriodic":
                    nonb_method_PBC = openmm.app.CutoffPeriodic
                else:
                    raise InputError("Unknown nonbonded method")

                logger.info("Nonbonded PBC method selected: %s", nonb_method_PBC)

                # Determining nonbonded cutoff strategy
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
                # Parameters here are based on OpenMM DHFR example
                if charmm_files is True:
                    logger.info("Using CHARMM files.")
                    self.system = self.forcefield.createSystem(
                        self.params,
                        nonbondedMethod=nonb_method_PBC,
                        constraints=self.autoconstraints,
                        hydrogenMass=self.hydrogenmass,
                        rigidWater=self.rigidwater,
                        ewaldErrorTolerance=self.ewalderrortolerance,
                        nonbondedCutoff=self.periodic_nonbonded_cutoff * openmm.unit.angstroms,
                        switchDistance=switching_function_distance * openmm.unit.angstroms,
                    )
                elif gromacs_files is True:
                    # NOTE: Gromacs has read PBC info from Gro file already
                    logger.info("Ewald Error tolerance: %s", self.ewalderrortolerance)
                    # Note: Turned off switchDistance. Not available for GROMACS?
                    #
                    self.system = self.forcefield.createSystem(
                        nonbondedMethod=nonb_method_PBC,
                        constraints=self.autoconstraints,
                        hydrogenMass=self.hydrogenmass,
                        rigidWater=self.rigidwater,
                        ewaldErrorTolerance=self.ewalderrortolerance,
                        nonbondedCutoff=self.periodic_nonbonded_cutoff * openmm.unit.angstroms,
                    )
                elif amber_files is True:
                    # NOTE: PBC information should be in forcefield object already
                    self.system = self.forcefield.createSystem(
                        nonbondedMethod=nonb_method_PBC,
                        constraints=self.autoconstraints,
                        hydrogenMass=self.hydrogenmass,
                        rigidWater=self.rigidwater,
                        ewaldErrorTolerance=self.ewalderrortolerance,
                        nonbondedCutoff=self.periodic_nonbonded_cutoff * openmm.unit.angstroms,
                    )

                else:
                    # Modeller and manual xmlfiles
                    self.system = self.forcefield.createSystem(
                        self.topology,
                        nonbondedMethod=nonb_method_PBC,
                        constraints=self.autoconstraints,
                        hydrogenMass=self.hydrogenmass,
                        rigidWater=self.rigidwater,
                        ewaldErrorTolerance=self.ewalderrortolerance,
                        nonbondedCutoff=self.periodic_nonbonded_cutoff * openmm.unit.angstroms,
                        residueTemplates=residueTemplates,
                    )

                # Setting as periodic_cell_vectors
                self.periodic_cell_vectors = np.array(
                    [[v._value * 10 for v in vec] for vec in self.system.getDefaultPeriodicBoxVectors()]
                )
                logger.info("Periodic_cell_vectors (Å) %s", periodic_cell_vectors)

                # Force modification here
                logger.info(small_header("OpenMM Forces defined:"))
                # Looping over forces
                for force in self.system.getForces():
                    logger.info("%s", force.getName())
                    # NONBONDED FORCE
                    if isinstance(force, openmm.CustomNonbondedForce):
                        # NOTE: This is only sometimes used: XML-CHARMM setup, GROMACS-files etc.
                        pass
                    elif isinstance(force, openmm.NonbondedForce):
                        # Turn Dispersion correction on/off depending on user
                        force.setUseDispersionCorrection(dispersion_correction)

                        # Modify PME Parameters if desired
                        if pme_parameters is not None:
                            logger.info("Nonbonded force:  Changing PME parameters")
                            force.setPMEParameters(
                                pme_parameters[0], pme_parameters[1], pme_parameters[2], pme_parameters[3]
                            )
                        # if switching_function is True:
                        #    #Switching distance in nm. To be looked at further
                        logger.info("Nonbonded force settings (after all modifications):")
                        logger.info(f"   Periodic cutoff distance: {force.getCutoffDistance()}")
                        logger.info(f"   Use SwitchingFunction: {force.getUseSwitchingFunction()}")
                        if force.getUseSwitchingFunction() is True:
                            logger.info(f"   SwitchingFunction distance: {force.getSwitchingDistance()}")
                        logger.info(f"   Use Long-range Dispersion correction: {force.getUseDispersionCorrection()}")
                        logger.info("   PME Parameters: %s", force.getPMEParameters())
                        logger.info("   Ewald error tolerance: %s", force.getEwaldErrorTolerance())
                logger.info(small_header("OpenMM system created."))

            # Non-Periodic
            else:
                if self.nonbonded_method_no_pbc == "NoCutoff":
                    noPBC_nonbondedMethod = openmm.app.NoCutoff
                elif self.nonbonded_method_no_pbc == "CutoffNonPeriodic":
                    noPBC_nonbondedMethod = openmm.app.CutoffNonPeriodic
                elif self.nonbonded_method_no_pbc == "CutoffPeriodic":
                    raise InputError("nonbondedMethod_noPBC with CutoffPeriodic not currently allowed")
                logger.info("System is non-periodic.")
                logger.info("nonbonded noPBC Method is: %s", noPBC_nonbondedMethod)

                logger.info("Nonbonded cutoff : %s Angstrom", self.nonbonded_cutoff_no_pbc)

                if charmm_files is True:
                    self.system = self.forcefield.createSystem(
                        self.params,
                        nonbondedMethod=noPBC_nonbondedMethod,
                        constraints=self.autoconstraints,
                        rigidWater=self.rigidwater,
                        nonbondedCutoff=self.nonbonded_cutoff_no_pbc * openmm.unit.angstroms,
                        hydrogenMass=self.hydrogenmass,
                    )
                elif amber_files is True:
                    self.system = self.forcefield.createSystem(
                        nonbondedMethod=noPBC_nonbondedMethod,
                        constraints=self.autoconstraints,
                        rigidWater=self.rigidwater,
                        nonbondedCutoff=self.nonbonded_cutoff_no_pbc * openmm.unit.angstroms,
                        hydrogenMass=self.hydrogenmass,
                    )
                # NOTE: might be unnecessary
                elif dummysystem is True:
                    self.system = self.forcefield.createSystem(self.topology)
                else:
                    self.system = self.forcefield.createSystem(
                        self.topology,
                        nonbondedMethod=noPBC_nonbondedMethod,
                        constraints=self.autoconstraints,
                        rigidWater=self.rigidwater,
                        nonbondedCutoff=self.nonbonded_cutoff_no_pbc * openmm.unit.angstroms,
                        hydrogenMass=self.hydrogenmass,
                    )
                logger.info(small_header("OpenMM system created."))
                logger.info("OpenMM Forces defined: %s", self.system.getForces())
                logger.info("")
                # for i,force in enumerate(self.system.getForces()):
                #    if isinstance(force, openmm.NonbondedForce):

                # Get charges from OpenMM object into self.charges

                # CASE CUSTOMNONBONDED FORCE
                # REPLACING REGULAR NONBONDED FORCE
                if customnonbondedforce is True:
                    raise InternalError("currently inactive")
                    # Create CustomNonbonded force
                    for i, force in enumerate(self.system.getForces()):
                        if isinstance(force, openmm.NonbondedForce):
                            custom_nonbonded_force, custom_bond_force = create_cnb(
                                self.system.getForces()[i], self.system.getNumParticles()
                            )
                    logger.info("1custom_nonbonded_force: %s", custom_nonbonded_force)
                    logger.info("num exclusions in customnonb: %s", custom_nonbonded_force.getNumExclusions())
                    logger.info("num 14 exceptions in custom_bond_force: %s", custom_bond_force.getNumBonds())

                    # TODO: Deal with frozen regions. NOT YET DONE
                    # Frozen-Act interaction
                    # Act-Act interaction

                    # Pointing self.nonbonded_force to CustomNonBondedForce instead of Nonbonded force
                    self.nonbonded_force = custom_nonbonded_force
                    logger.debug("Nonbonded force: %s", self.nonbonded_force)
                    self.custom_bondforce = custom_bond_force

                    # Update system with new forces and delete old force
                    self.system.addForce(self.nonbonded_force)
                    self.system.addForce(self.custom_bondforce)

                    # Remove oldNonbondedForce
                    for i, force in enumerate(self.system.getForces()):
                        if isinstance(force, openmm.NonbondedForce):
                            self.system.removeForce(i)

        # Defining nonbonded force
        for force in self.system.getForces():
            if isinstance(force, openmm.NonbondedForce):
                self.nonbonded_force = force

        # Set charges in OpenMMobject by taking from Force (used by QM/MM)
        logger.info("Setting charges")
        self.getatomcharges()

        # Storing numatoms and list of all atoms
        self.numatoms = int(self.system.getNumParticles())
        self.allatoms = list(range(self.numatoms))
        logger.info("Number of atoms in OpenMM system: %s", self.numatoms)

        # Preserve original masses before any mass modifications or frozen atoms (set mass to 0)
        # NOTE: Creates list of Quantity objects (value, unit attributes)
        self.system_masses_original = [self.system.getParticleMass(i) for i in self.allatoms]
        # List of currently used masses. Can be modified by self.modify_masses and self.freeze_atoms
        # NOTE: Regular list of floats
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

        # Note: constraints and bondconstraints are the same thing
        if constraints is not None:
            logger.info("constraints keyword specified is deprecated. Use bondconstraints instead")
            bondconstraints = constraints

        if bondconstraints or frozen_atoms or restraints:
            logger.info(sub_header("Adding user constraints, restraints or frozen atoms."))
        # Now adding user-defined system constraints (only bond-constraints supported for now)
        if bondconstraints is not None:
            if bondconstraints is None:
                bondconstraints = []
            tot_num_user_constraints = len(bondconstraints)

            logger.info(
                f"Before adding user constraints, system contains {self.system.getNumConstraints()} constraints"
            )
            logger.info("")

            if len(bondconstraints) < 50:
                logger.info("User-constraints to add (bond) %s", bondconstraints)
            else:
                logger.info(f"{tot_num_user_constraints} user-defined constraints to add.")

            # Cleaning up bondconstraint list. Adding distance if missing
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
                            "No PDBfile present either. Either fragment or PDBfile containing \
                                coordinates is required for constraint definition"
                        )
                        raise InputError("Constraint definition requires a fragment or a PDB file with coordinates")
                    else:
                        fragment = Fragment(pdbfile=pdbfile)
                # Cleaning up constraint list. Adding distance if missing
                bondconstraints = clean_up_constraints_list(fragment=fragment, constraints=bondconstraints)
                self.add_bondconstraints(constraints=bondconstraints)
            # Angle constraints
            # TODO
            # Dihedral constraints
            # TODO

            self.user_constraints = bondconstraints

            logger.info(f"{len(self.user_constraints)} user-defined constraints added.")
        # Now adding user-defined frozen atoms
        if frozen_atoms is not None:
            self.user_frozen_atoms = frozen_atoms
            if len(self.user_frozen_atoms) < 50:
                logger.info("Frozen atoms to add: %s", str(frozen_atoms).strip("[]"))
            else:
                logger.info(f"{len(self.user_frozen_atoms)} user-defined frozen atoms to add.")
            self.freeze_atoms(frozen_atoms=frozen_atoms)

        # Now adding user-defined restraints (only bond-restraints supported for now)
        if restraints is not None:
            # restraints is a list of lists defining bond restraints: constraints = [[atom_i,atom_j, d, k ]]
            # Example: [[700,701, 1.05, 5.0 ]] Unit is Angstrom and kcal/mol * Angstrom^-2
            self.user_restraints = restraints
            if len(self.user_restraints) < 50:
                logger.info("User-restraints to add: %s", restraints)
            else:
                logger.info(f"{len(self.user_restraints)} user-defined restraints to add.")
            self.add_bondrestraints(restraints=restraints)

        # Now changing masses if requested
        if changed_masses is not None:
            logger.info("Modified masses")
            # changed_masses should be a dict of : atomindex: mass
            self.modify_masses(changed_masses=changed_masses)

        logger.info("\nSystem constraints defined upon system creation: %s", self.system.getNumConstraints())
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(self.system.getNumConstraints()):
                logger.info("Defined constraints: %s", self.system.getConstraintParameters(i))
        time.time()

        # Set simulation parameters (here just default options)
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

    def define_mm_elements(self, topology):
        """Extract the element symbol of every atom from an OpenMM topology.

        Args:
            topology: OpenMM Topology object.
        """
        try:
            self.mm_elements = [i.element.symbol for i in topology.atoms()]
        except AttributeError:
            logger.info("Problem occurred while defining mm_elements.")
            logger.info("This may be due to virtual sites present")
            logger.info("mm_elements will be set to empty list")
            self.mm_elements = []

    # Function to write PDB-file if everything is available
    def write_pdbfile(self, positions=None, outputname="system"):
        """Write a PDB file of the system using the stored OpenMM topology.

        Coordinates are taken from the positions argument if given, otherwise from the
        object's own positions, otherwise from the referenced fragment.

        Args:
            positions: OpenMM positions to write; takes precedence over the stored ones.
            outputname: output name, without the .pdb extension.

        Raises:
            InputError: if no positions are available from any source.
        """
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

    # Function that handles periodicity in forcefield objects (for Amber, CHARMM). TODO: Test GROMACS and XML
    def set_periodics_before_system_creation(
        self, periodic_cell_vectors, pdb_pbc_vectors, periodic_cell_dimensions, charmm_files, amber_files, use_parmed
    ):
        """Resolve the periodic box from the various possible sources.

        The cell can come from an explicit argument, the PDB CRYST1 record, or the
        forcefield input files; this settles which one wins before the OpenMM System is built.

        Args:
            periodic_cell_vectors: explicit 3x3 cell vectors.
            pdb_pbc_vectors: cell vectors read from the PDB file.
            periodic_cell_dimensions: [a, b, c, alpha, beta, gamma] cell parameters.
            charmm_files: whether the system came from CHARMM input files.
            amber_files: whether the system came from an Amber prmtop.
            use_parmed: whether ParmEd was used to read the input files.
        """
        if use_parmed is True:
            pass
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
            # Setting PBC forcefield object
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
            # Openmm 7 and Amber problem only: Delete this at some point
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
            # Setting PBC forcefield object
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
            # Automatically set:
            # CHARMM without parmed: need to use setBox in forcefield (actually psf) object
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
                # Automatically set:
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
            # Should work automatically
        elif self.topology.getPeriodicBoxVectors() is not None:
            logger.info("Found PBC information in topology object. Using this and continuing")
        else:
            raise FileFormatError("Found no PBC information, yet periodicity is requested. Exiting!")

    # Get PBC vectors from topology of openmm object. Convenient in a script
    def get_pbc_vectors(self):
        # Get PBC vectors
        """Return the current periodic box vectors in Angstrom."""
        vectors_nm = list(self.topology.getPeriodicBoxVectors())
        a = list(vectors_nm[0].value_in_unit(openmm.unit.angstrom))
        b = list(vectors_nm[1].value_in_unit(openmm.unit.angstrom))
        c = list(vectors_nm[2].value_in_unit(openmm.unit.angstrom))
        # Return List of lists
        return [a, b, c]

    # Set numcores method: currently inactive. Included for completeness
    def set_numcores(self, numcores):
        """Set the number of CPU threads OpenMM uses.

        Args:
            numcores: thread count for the CPU platform.
        """
        self.numcores = numcores

    # Set numcores method
    def cleanup(self):
        """No-op: OpenMM keeps no scratch files that need removing between runs."""
        logger.info("Cleanup for OpenMMTheory called")

    # add force that restrains atoms to a fixed point:
    # https://github.com/openmm/openmm/issues/2568

    # To set positions in OpenMMobject (in nm) from np-array (Angstrom)
    def set_positions(self, coords, simulation):
        """Load coordinates into a simulation context.

        Args:
            coords: coordinates in Angstrom, one row per atom.
            simulation: OpenMM Simulation whose context is updated.
        """
        logger.info("Setting coordinates of OpenMM object")
        coords_nm = coords * 0.1  # converting from Angstrom to nm
        pos = [
            openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in range(len(coords_nm))
        ] * openmm.unit.nanometer
        simulation.context.setPositions(pos)
        logger.info("Coordinates set")

    # Update cell using either periodic_cell_vectors or periodic_cell_dimensions
    # This method is called by Periodic optimizers
    def update_cell(self, periodic_cell_vectors=None, periodic_cell_dimensions=None):
        """Change the periodic box of the existing system.

        Args:
            periodic_cell_vectors: new 3x3 cell vectors in Angstrom.
            periodic_cell_dimensions: new [a, b, c, alpha, beta, gamma] parameters.
        """
        logger.info("Updating cell vectors")
        logger.info("New periodic_cell_vectors are: %s", periodic_cell_vectors)
        if periodic_cell_vectors is not None:
            self.periodic_cell_vectors = periodic_cell_vectors
            self.periodic_cell_dimensions = cell_vectors_to_params(periodic_cell_vectors)
        elif periodic_cell_dimensions is not None:
            self.periodic_cell_dimensions = periodic_cell_dimensions
            self.periodic_cell_vectors = cell_params_to_vectors(periodic_cell_dimensions)

        # Now updating actual OpenMM objects
        # Converting to nm
        cellvecs_nm = self.periodic_cell_vectors / 10
        a = cellvecs_nm[0]
        b = cellvecs_nm[1]
        c = cellvecs_nm[2]

        # We may have to adjust the nonbonded cutoff.
        # Shortest box dimension (diagonal elements, safe estimate for triclinic)
        min_box_dim = min(cellvecs_nm[0, 0], cellvecs_nm[1, 1], cellvecs_nm[2, 2])
        hard_limit_cutoff = 0.499 * min_box_dim  # just under OpenMM's hard limit of 0.5

        # Find NonbondedForce and update cutoff only if the box has become too small
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
        # Topology
        self.topology.setPeriodicBoxVectors(cellvecs_nm)

    # Add dummy
    # https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=161&t=10049&p=0&start=0&view=&sid=b844250e55b14682fb21b5f66a4d810f
    # https://github.com/openmm/openmm/issues/2262
    # Helpful for NPT simulations when solute is fixed
    # TODO: Not quiteready. Not sure how to use best
    # Add dummy atom for each solute atom?
    # Or enought to add like a centroid atom and then bind each solute atom via restraint?
    def add_dummy_atom_to_restrain_solute(self, atomindices=None, forceconstant=100):
        """Add a massless dummy atom, harmonically tethered to a group of atoms.

        Used to keep a solute near the centre of the box during long simulations.

        Args:
            atomindices: atoms the dummy atom is tethered to.
            forceconstant: restraint force constant in kcal/mol/Angstrom**2.
        """
        logger.info("num particles %s", self.system.getNumParticles())
        # Adding dummy atom with mass 0
        self.system.addParticle(0)
        logger.info("num particles %s", self.system.getNumParticles())
        dummyatomindex = self.system.getNumParticles() - 1
        logger.info("dummyatomindex: %s", dummyatomindex)
        # Adding zero-charge and zero-epsilon to Nonbonded force (charge,sigma,epsilon)
        self.nonbonded_force.addParticle(0, 1, 0)
        # Adding dummy-atom to topology
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
        # for force in self.system.getForces():
        #    if isinstance(force,openmm.HarmonicBondForce):
        #        #Add harmonic bond between first atom in solute
        #        for i in atomindices:

    # Method to add any (compatible) force to system (could e.g. be a loaded TorchForce )
    def add_force(self, newforce):
        """Add a force to the system and recreate the simulation.

        Args:
            newforce: OpenMM Force object to add.
        """
        logger.info("Adding new force to system: %s", newforce)
        self.system.addForce(newforce)

    def remove_force(self, forceindex):
        """Remove a force by its index in the system.

        Args:
            forceindex: position of the force in system.getForces().
        """
        logger.info(f"Removing force-index {forceindex}: {self.system.getForces()[forceindex].getName()}")
        self.system.removeForce(forceindex)

    def remove_force_by_name(self, forcename):
        """Remove every force whose class name matches.

        Args:
            forcename: OpenMM force class name, e.g. "CMMotionRemover".
        """
        logger.info(f"Searching forces and removing a force name: {forcename}")
        for i, force in enumerate(self.system.getForces()):
            logger.info("force name: %s", force.getName())
            if force.getName() == forcename:
                logger.info(f"Removing force-index {i}: {forcename}")
                self.system.removeForce(i)

    # Bond restraint force, e.g. for umbrella sampling
    # TODO : unit check
    def add_custom_bond_force(self, i, j, value, forceconstant):
        """Restrain the distance between two atoms harmonically.

        Args:
            i: first atom index.
            j: second atom index.
            value: target distance in Angstrom.
            forceconstant: force constant in kcal/mol/Angstrom**2.
        """
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

    # For umbrella sampling e.g
    # TODO: unit check
    def add_custom_angle_force(self, i, j, k, value, forceconstant):
        """Restrain the i-j-k angle harmonically.

        Args:
            i: first atom index.
            j: central atom index.
            k: third atom index.
            value: target angle in degrees.
            forceconstant: force constant in kcal/mol/rad**2.
        """
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

    # Harmonic torsion bias for umbrella sampling e.g
    # TODO: unit check
    def add_custom_torsion_force(self, i, j, k, l, value, forceconstant):  # noqa: E741 - torsion atoms i-j-k-l
        """Restrain the i-j-k-l dihedral harmonically.

        Args:
            i: first atom index.
            j: second atom index.
            k: third atom index.
            l: fourth atom index.
            value: target dihedral in degrees.
            forceconstant: force constant in kcal/mol/rad**2.
        """
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
        """Tether atoms to a fixed point in space beyond a given distance.

        Args:
            center_coords: the centre the atoms are pulled towards, in Angstrom.
            atomindices: atoms the force applies to.
            forceconstant: force constant in kcal/mol/Angstrom**2.
            distance: radius inside which no force acts.
        """
        logger.info("add_centerforce:")
        logger.info("Center coordinates: %s", center_coords)
        logger.info("Force acting on atomindices: %s", atomindices)
        logger.info(f"Forceconstant: {forceconstant} kcal/mol/Ang^2")
        logger.info(f"Force acting at values larger than {distance} Ang:")
        # Distinguish periodic and nonperiodic scenarios:
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
        # Coordinates of system center
        center_x = center_coords[0] / 10
        center_y = center_coords[1] / 10
        center_z = center_coords[2] / 10
        for i in atomindices:
            centerforce.addParticle(i, openmm.Vec3(center_x, center_y, center_z))
        self.system.addForce(centerforce)
        logger.info("Added center force")
        return centerforce

    # e.g. for steered MD
    def add_custom_centroidbond_force(self, host_indices, guest_indices, forceconstant=1.0, r0=0.0):
        """Harmonically restrain the distance between two groups' centres of mass.

        Args:
            host_indices: atoms of the first group.
            guest_indices: atoms of the second group.
            forceconstant: force constant in kcal/mol/Angstrom**2.
            r0: target centre-of-mass separation in Angstrom.
        """
        logger.info(
            f"Adding CustomCentroidBondForce between centroid of host {host_indices}  and centroid of guest "
            f"{guest_indices} "
        )
        logger.info(f"Forceconstant : {forceconstant} kcal/mol/Å^2")

        force = openmm.CustomCentroidBondForce(2, "0.5*k*(distance(g1,g2)-r0)^2")
        force.addPerBondParameter("k")
        force.addGlobalParameter("r0", r0 * openmm.unit.angstroms)
        force.addGroup(host_indices)
        force.addGroup(guest_indices)
        force.addBond([0, 1], [forceconstant * openmm.unit.kilocalories_per_mole / openmm.unit.angstroms**2])
        self.system.addForce(force)
        logger.info("Added force")
        return force

    # Alternative version of a Flatbottom center force on small-molecule w.r.t. rest-of-system
    # Note: behaves differently with respect to PBC-wrapping, creating problems for QM/MM.
    def add_flatbottom_centerforce(self, mol_a_indices=None, mol_b_indices=None, distance=5.0, forceconstant=1.0):
        """Restrain two molecules' centres of mass with a flat-bottom potential.

        No force acts while the separation is below `distance`; beyond it the restraint is
        harmonic. Not well tested under periodic boundary conditions.

        Args:
            mol_a_indices: atoms of the first molecule.
            mol_b_indices: atoms of the second molecule.
            distance: flat-bottom radius in Angstrom.
            forceconstant: force constant in kcal/mol/Angstrom**2.
        """
        logger.info("Inside add_flatbottom_centerforce")
        logger.info("molA_indices size: %s", len(mol_a_indices))
        logger.info("molB_indices size: %s", len(mol_b_indices))
        logger.info("forceconstant: %s", forceconstant)
        logger.info("distance: %s", distance)
        # Define force
        centerforce = openmm.CustomCentroidBondForce(2, "0.5*k*max(0, distance(g1,g2)-r0)^2")
        # Periodic case (note: periodicdistance not available for CustomCentroidBondForce)
        if self.periodic is True:
            logger.warning("Using add_flatbottom_centerforce with PBC is not well tested")
            centerforce.setUsesPeriodicBoundaryConditions = True

        centerforce.addGlobalParameter(
            "k", forceconstant * openmm.unit.kilocalories_per_mole / openmm.unit.angstroms**2
        )
        centerforce.addGlobalParameter("r0", distance * openmm.unit.angstrom)
        g1 = mol_a_indices  # solute/ligand
        g2 = mol_b_indices  # rest
        centerforce.addGroup(g1)  # index will be 0
        centerforce.addGroup(g2)  # index will be 1
        centerforce.addBond([0, 1], [])  # no [] since global
        self.system.addForce(centerforce)
        logger.info("Added center force")
        return centerforce

    def add_custom_external_force(self):
        """Add the per-atom external force that carries the QM/MM gradient.

        QM/MM MD applies the QM contribution to the MM system through this force, updated
        each step by update_custom_external_force.

        Returns:
            The CustomExternalForce object that was added.
        """
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
        """Push a new gradient into the QM/MM external force.

        Args:
            customforce: the CustomExternalForce created by add_custom_external_force.
            gradient: gradient in Eh/Bohr, one row per atom.
            simulation: simulation whose context the parameters are updated in.
        """
        logger.info("Updating custom external force")
        # Convert Eh/Bohr gradient to force in kj/mol nm
        # *49614.501681716106452
        # NOTE: default conversion factor (49614.752589207) assumes input gradient in Eh/Bohr and converting to kJ/mol
        # nm
        forces = -gradient * 49614.752589207
        for i, f in enumerate(forces):
            customforce.setParticleParameters(i, i, f)
        customforce.updateParametersInContext(simulation.context)

    # Function to add restraints to system before MD
    def add_bondrestraints(self, restraints=None):
        """Add harmonic distance restraints between atom pairs.

        Args:
            restraints: list of [i, j, distance, forceconstant] entries.
        """
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

    # Z_cc: length of cone part
    # R_cylinder: radius of cylinder part
    # alpha: angle of cone in degrees
    # k_xy: force constant in kcal/mol/r2
    # force_group ??
    def add_funnel_restraint(
        self, host_index, guest_index, k_xy=10.0, z_cc=11.0, alpha=35.0, r_cylinder=1.0, force_group=10
    ):
        """Add a funnel-shaped restraint for funnel-metadynamics binding studies.

        Confines the guest to a cone near the binding site that widens into a cylinder in bulk.

        Args:
            host_index: index of a host reference atom.
            guest_index: index of a guest reference atom.
            k_xy: force constant of the lateral restraint.
            z_cc: distance along the funnel axis where the cone becomes a cylinder.
            alpha: cone opening angle in degrees.
            r_cylinder: radius of the cylindrical section.
            force_group: OpenMM force group the restraint is assigned to.
        """
        logger.info("Adding funnel restraint potential")
        # Funnel potential string expression
        funnel = openmm.CustomCentroidBondForce(
            2,
            "U_funnel + U_cylinder;"
            "U_funnel = step(z_cc - abs(r_z))*step(r_xy - R_funnel)*Wall;"
            "U_cylinder = step(abs(r_z) - z_cc)*step(r_xy - R_cylinder)*Wall;"
            "Wall = 0.5 * k_xy * r_xy^2;"
            "R_funnel = (z_cc-abs(r_z))*tan(alpha) + R_cylinder;"
            "r_xy = sqrt((x2 - x1)^2 + (y2 - y1)^2);"
            "r_z = z2 - z1;",
        )
        funnel.setUsesPeriodicBoundaryConditions(False)
        funnel.setForceGroup(force_group)

        # Funnel parameters
        funnel.addGlobalParameter("k_xy", k_xy * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2)
        funnel.addGlobalParameter("z_cc", z_cc * openmm.unit.angstrom)
        funnel.addGlobalParameter("alpha", alpha * openmm.unit.degrees)
        funnel.addGlobalParameter("R_cylinder", r_cylinder * openmm.unit.angstrom)

        # Add host and guest indices
        g1 = funnel.addGroup(host_index, [1.0 for i in range(len(host_index))])
        g2 = funnel.addGroup(guest_index, [1.0 for i in range(len(guest_index))])

        # Add bond
        funnel.addBond([g1, g2], [])

        # Add force to system
        self.system.addForce(funnel)

    # For restraining CVs, used by metadynamics
    # NOTE: Assuming Angstrom and kcal/mol^2 here like for regular restraints
    # NOTE: Dihedrals not supported (unclear if useful). Angles are and units are radians
    def add_cv_restraint(self, cvforce, restraint_par, cvtype):
        # Make copy of CVforce (otherwise we can not use it also in restraint)
        """Restrain a collective variable used in metadynamics.

        Args:
            cvforce: the force defining the collective variable.
            restraint_par: restraint parameters, interpreted according to cvtype.
            cvtype: kind of restraint, e.g. "upper_wall" or "lower_wall".
        """
        cvforce_copy = copy.copy(cvforce)
        # TODO: periodic CV vs non-periodic
        if cvtype == "dihedral" or cvtype == "torsion":
            raise InputError("Adding CV restraints for dihedrals is not available!")
            # Not sure whether there is ever a need
        elif cvtype == "angle":
            raise InputError("Adding CV restraints for angles is not available!")
            energy_expression = "(k/2)*max(0, var-var_max)^2"
            logger.info("CV type: angle")
            logger.info("Note: unit assumed to be in radians")
            var_unit = openmm.unit.radian
            var_unit_label = "radians"
        elif cvtype == "bond" or cvtype == "distance" or cvtype == "rmsd":
            energy_expression = "(k/2)*max(0, var-var_max)^2"
            logger.info("CV type: bond/rmsd")
            logger.info("Note: unit assumed be in Angstrom")
            var_unit = openmm.unit.angstroms
            var_unit_label = "Å"
        elif cvtype.lower() == "cn":
            energy_expression = "(k/2)*max(0, var-var_max)^2"
            logger.info("CV type: CN")
            var_unit = 1.0
            var_unit_label = " "
        else:
            raise InputError("Error: unknown cvtype for add_CV_restraint")
        # Energy unit
        energy_unit = openmm.unit.kilocalories_per_mole / openmm.unit.angstroms**2
        energy_unit_label = "kcal/mol*Å^-2"
        # Periodic:
        logger.info("Adding restraint with energy expression: %s", energy_expression)
        logger.info(f"Max value (var_max): {restraint_par[0]} {var_unit_label}")
        logger.info(f"Force constant (k) : {restraint_par[1]} {energy_unit_label}")
        restraint_force_CV = openmm.CustomCVForce(energy_expression)
        restraint_force_CV.addCollectiveVariable("var", cvforce_copy)
        restraint_force_CV.addGlobalParameter("var_max", restraint_par[0] * var_unit)
        restraint_force_CV.addGlobalParameter("k", restraint_par[1] * energy_unit)
        self.system.addForce(restraint_force_CV)

    # Write XML-file for full system
    def save_xml(self, xmlfile="system_full.xml"):
        """Serialize the OpenMM System to an XML file.

        Args:
            xmlfile: output file name.
        """
        serialized_system = openmm.XmlSerializer.serialize(self.system)
        with open(xmlfile, "w") as f:
            f.write(serialized_system)
        logger.info("Wrote system XML file: %s", xmlfile)

    # Function to add bond constraints to system before MD
    def add_bondconstraints(self, constraints=None):
        """Constrain bond lengths rigidly (not harmonically).

        Args:
            constraints: list of [i, j, distance] entries; distance in Angstrom.
        """
        for i, j, d in constraints:
            logger.info(f"Adding bond constraint between atoms {i} and {j}. Distance value: {d:.4f} Å")
            self.system.addConstraint(i, j, d * openmm.unit.angstroms)

    # Remove all defined constraints in system
    def remove_all_constraints(self):
        """Remove every distance constraint from the system."""
        todelete = []
        # Looping over all defined system constraints
        for i in range(self.system.getNumConstraints()):
            todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Remove specific constraints
    def remove_constraints(self, constraints):
        """Remove the listed constraints from the system.

        Args:
            constraints: list of [i, j] atom pairs whose constraint is removed.
        """
        todelete = []
        # Looping over all defined system constraints
        for i in range(self.system.getNumConstraints()):
            con = self.system.getConstraintParameters(i)
            for usercon in constraints:
                if all(elem in usercon for elem in [con[0], con[1]]):
                    todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Remove constraints for selected atoms. For example: QM atoms in QM/MM MD
    def remove_constraints_for_atoms(self, atoms):
        """Remove every constraint involving any of the given atoms.

        Used to free the QM region, whose internal geometry the QM code must control.

        Args:
            atoms: atom indices whose constraints are removed.
        """
        logger.info("Removing constraints in OpenMM object for atoms: %s", atoms)
        todelete = []
        # Looping over all defined system constraints
        for i in range(self.system.getNumConstraints()):
            con = self.system.getConstraintParameters(i)
            if con[0] in atoms or con[1] in atoms:
                todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Function to freeze atoms during OpenMM MD simulation. Sets masses to zero. Does not modify potential
    # energy-function.
    def freeze_atoms(self, frozen_atoms=None):
        """Freeze atoms by setting their mass to zero.

        The original masses are stored so unfreeze_atoms can restore them.

        Args:
            frozen_atoms: atom indices to freeze.
        """
        logger.info(f"Freezing {len(frozen_atoms)} atoms by setting particles masses to zero.")

        # Modify particle masses in system object. For freezing atoms
        for i in frozen_atoms:
            self.system.setParticleMass(i, 0 * openmm.unit.daltons)

        # Also adding exceptions to nonbonded force to avoid interactions between frozen atoms (causes problems
        # otherwise in NPT)
        logger.info(
            "Also adding exceptions to nonbonded force for frozen atoms to avoid interactions between them (avoids "
            "problems in NPT)."
        )
        self.addexceptions(frozen_atoms)

        # Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    # Changed masses according to user input dictionary
    def modify_masses(self, changed_masses=None):
        """Set new masses for selected atoms, e.g. for hydrogen-mass repartitioning.

        Args:
            changed_masses: dict of atom index to new mass in amu.
        """
        logger.info("Modify masses according:  %s", changed_masses)
        # Preserve original masses
        # Modify particle masses in system object.
        for am in changed_masses:
            self.system.setParticleMass(am, changed_masses[am] * openmm.unit.daltons)

        # Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    def unfreeze_atoms(self):
        # Looping over system_masses if frozen, otherwise empty list
        """Restore the masses of atoms previously frozen by freeze_atoms."""
        for atom, mass in zip(self.allatoms, self.system_masses_original, strict=False):
            self.system.setParticleMass(atom, mass)

        # Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    # This removes interactions between particles in a region (e.g. QM-QM or frozen-frozen pairs)
    # Give list of atom indices for which we will remove all pairs
    def addexceptions(self, atomlist):
        """Exclude the listed atoms from all nonbonded interactions with each other.

        This is how the QM-QM nonbonded terms are removed in QM/MM.

        Args:
            atomlist: atom indices whose mutual nonbonded interactions are switched off.
        """
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
                        # eps=0)
                        # TODO: This leads to : Exception: CustomNonbondedForce: Multiple exclusions are specified for
                        # particles
                        # Basically we have to inspect what is actually present in CustomNonbondedForce
                        # for force in self.system.getForces():
                        #    if isinstance(force, openmm.CustomNonbondedForce):

                        numexceptions += 1
            elif isinstance(force, openmm.CustomNonbondedForce):
                # Only applies to system with CustomNonbondedForce: GROMACS-setup, CHARMM-from-XML
                # Note: this code has been sped up quite a bit
                logger.info("Case CustomNonbondedforce. Adding Exclusion for kl pair.")
                # Get list of all present exclusions first
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
        """Set the integrator and thermostat parameters used by create_simulation.

        Args:
            timestep: MD timestep in ps.
            coupling_frequency: thermostat coupling frequency in 1/ps.
            temperature: target temperature in K.
            integrator: OpenMM integrator name, e.g. "VerletIntegrator" or "LangevinMiddleIntegrator".
        """
        self.timestep = timestep
        self.coupling_frequency = coupling_frequency
        self.temperature = temperature
        self.integrator_name = integrator

    # Create integrator.
    def create_integrator(self):
        # NOTE: Integrator definition has to be here (instead of set_simulation_parameters) as it has to be recreated
        # for each updated simulation
        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
        # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        """Create the OpenMM integrator from the stored simulation parameters.

        Made fresh each time: an integrator may only be bound to one context.
        """
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
            # TODO: options
            self.integrator = openmm.DrudeLangevinIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.temperature * openmm.unit.kelvin,
                self.timestep * openmm.unit.picoseconds,
                4,
            )
        elif self.integrator_name == "RPMDIntegrator":
            logger.info("RPMDIntegrator will be used")
            logger.warning("Autoconstraints, rigidwater and other contraints must have been disabled.")
            logger.info(f"RPMD number of copies set to {self.rpmd_num_copies}. Use RPMD_num_copies keyword to change")
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

    # Create simulation object (now not part of OpenMMTheory)
    def create_simulation(self, internal=False):
        """Build the OpenMM Simulation from the current system, topology and integrator.

        Args:
            internal: build the simulation for internal energy/gradient evaluation rather
                than for an MD run.
        """
        timeA = time.time()

        logger.info(sub_header("Creating/updating OpenMM simulation object"))
        logger.info("Integrator name: %s", self.integrator_name)
        logger.info("Timestep: %s", self.timestep)
        logger.info("Temperature: %s", self.temperature)
        logger.info("Coupling frequency: %s", self.coupling_frequency)
        logger.info("Properties: %s", self.properties)
        logger.info("Topology: %s", self.topology)
        logger.debug("self.system.getForces()  %s", self.system.getForces())

        # Create integrator object (needed for every update)
        self.create_integrator()

        # Create simulation, either as part of OpenMMTheory (not picklable)
        # or not (used by run method)
        if internal is True:
            # NOTE: Not sure if needed anymore
            self.simulation = openmm.app.simulation.Simulation(
                self.topology,
                self.system,
                self.integrator,
                openmm.Platform.getPlatformByName(self.platform_choice),
                self.properties,
            )
            return
        else:
            simulation = openmm.app.simulation.Simulation(
                self.topology,
                self.system,
                self.integrator,
                openmm.Platform.getPlatformByName(self.platform_choice),
                self.properties,
            )
            log_time_since(timeA, "creating/updating simulation")
            return simulation

    # Functions for energy decompositions
    def forcegroupify(self):
        """Assign each force its own force group so their energies can be separated.

        Required before get_energy_decomposition.
        """
        self.forcegroups = {}
        logger.info("inside forcegroupify")
        logger.debug("System forces: %s", self.system.getForces())
        logger.info("Number of forces:\n %s", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i

    def get_energy_decomposition(self, context):
        """Return the potential energy of each force group.

        Call forcegroupify first.

        Args:
            context: OpenMM Context to evaluate in.

        Returns:
            dict of force object to its potential energy.
        """
        energies = {}
        for f, i in self.forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
        return energies

    def print_energy_decomposition(self, simulation):
        """Log the per-force energy breakdown of the current state.

        Args:
            simulation: simulation whose context is evaluated.
        """
        timeA = time.time()
        # Energy decomposition
        # NOTE: Calling this is expensive (seconds)as the energy has to be recalculated.
        openmm_energy = {}
        energycomp = self.get_energy_decomposition(simulation.context)
        logger.info("")
        for comp in energycomp.items():
            openmm_energy[comp[0].getName()] = comp[1]

        # Sum all force-terms
        sumofallcomponents = 0.0
        for val in openmm_energy.values():
            sumofallcomponents += val._value

        # Print energy table
        logger.info(f"{'Component':<20} | {'kJ/mol':<15} | {'kcal/mol':<15}")
        logger.info("%s", "-" * 56)
        # TODO: Figure out better sorting of terms
        for name in sorted(openmm_energy):
            logger.info(
                f"{name:<20} | {openmm_energy[name] / openmm.unit.kilojoules_per_mole:>15.2f} | "
                f"{openmm_energy[name] / openmm.unit.kilocalorie_per_mole:>15.2f}"
            )
        logger.info("%s", "-" * 56)
        logger.info(f"{'Sumcomponents':<20} | {sumofallcomponents:>15.2f} | {sumofallcomponents / 4.184:>15.2f}")
        logger.info("")
        logger.info(
            f"{'Total':<20} | {self.energy * openmmqmmm.constants.hartokj:>15.2f} | "
            f"{self.energy * openmmqmmm.constants.harkcal:>15.2f}"
        )
        logger.info("")
        # Adding sum to table
        openmm_energy["Sum"] = sumofallcomponents
        self.energy_components = openmm_energy
        log_time_since(timeA, "energy decomposition")

    # Compute the number of degrees of freedom.
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

    # Compute cell gradient numerically
    def compute_cell_gradient_fd(self, context, eps=1e-4):
        # Conversion factors
        """Compute the gradient with respect to the cell vectors by finite differences.

        Args:
            context: OpenMM Context to evaluate in.
            eps: relative strain step used for the finite difference.

        Returns:
            The 3x3 cell gradient.
        """
        NM_TO_BOHR = 18.89726124  # 1 nm = 18.897... Bohr
        KJMOL_TO_EH = 1.0 / 2625.4996  # 1 kJ/mol = 1/2625.5 Hartree
        eps_nm = eps / NM_TO_BOHR  # convert eps to nm for OpenMM

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

    # Get cell gradient (called by an Optimizer e.g.)
    def get_cell_gradient(self):
        """Return the most recently computed cell gradient."""
        logger.info("Inside get_cell_gradient")
        # First compute the cell gradient numerically
        # Using self.stored_context (should have been defined by .run call)
        self.cell_gradient = self.compute_cell_gradient_fd(self.stored_context, eps=1e-4)
        logger.info("OpenMM cell gradient: %s", self.cell_gradient)
        return self.cell_gradient

    # NOTE: Adding charge/mult/PC here to  be consistent with QM_theories. Not used
    def run(
        self,
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
        """Compute the MM energy (and gradient) of a geometry.

        Args:
            current_coords: coordinates in Angstrom, one row per atom.
            elems: element symbols; unused, the topology defines the system.
            grad: also compute the gradient.
            fragment: fragment used for QM/MM bookkeeping.
            qmatoms: QM-region indices, for the QM/MM subtraction terms.
            label: unused; present for signature compatibility with the QM theories.
            charge: unused; MM charges come from the forcefield.
            mult: unused; MM has no spin state.
            pc: unused; MM does not take an external point-charge field.
            current_mm_coords: unused; present for signature compatibility.
            mm_charges: unused; present for signature compatibility.
            qm_elems: unused; present for signature compatibility.
            numcores: unused; the thread count is fixed when the theory is constructed.

        Returns:
            The energy in hartree, or (energy, gradient in Eh/Bohr) when grad=True.
        """
        module_init_time = time.time()
        timeA = time.time()

        # Need to call create_simulation here in order to get a simulation object
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
        # Making sure coords is np array and not list-of-lists
        current_coords = np.array(current_coords)
        factor = -49614.752589207
        logger.info("Updating coordinates.")
        timeA = time.time()

        # NOTE: THIS IS STILL RATHER SLOW
        current_coords_nm = current_coords * 0.1  # converting from Angstrom to nm
        pos = [
            openmm.Vec3(current_coords_nm[i, 0], current_coords_nm[i, 1], current_coords_nm[i, 2])
            for i in range(len(current_coords_nm))
        ] * openmm.unit.nanometer
        log_time_since(timeA, "Creating pos array")
        timeA = time.time()
        # THIS IS THE SLOWEST PART. Probably nothing to be done
        simulation.context.setPositions(pos)

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
                state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole) / openmmqmmm.constants.hartokj
            )
            self.gradient = np.array(state.getForces(asNumpy=True) / factor)
        else:
            state = simulation.context.getState(getEnergy=True, getForces=False)
            self.energy = (
                state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole) / openmmqmmm.constants.hartokj
            )

        log_time_since(timeA, "OpenMM getState")

        logger.info("OpenMM Energy: %s Eh", self.energy)
        logger.info("OpenMM Energy: %s kcal/mol", self.energy * openmmqmmm.constants.harkcal)

        # Do energy components or not. Can be turned off for e.g. MM MD simulation
        if self.do_energy_decomposition is True:
            self.print_energy_decomposition(simulation)
        logger.info(small_header("Ending OpenMM interface"))
        log_time_since(module_init_time, "OpenMM run")
        if grad is True:
            return self.energy, self.gradient
        else:
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

    # Delete selected exceptions. Only for Coulomb.
    # Used to delete Coulomb interactions involving QM-QM and QM-MM atoms
    def delete_exceptions(self, atomlist):
        """Remove the nonbonded exceptions previously added for the listed atoms.

        Args:
            atomlist: atom indices whose exceptions are deleted.
        """
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
        """Set new Lennard-Jones epsilon values for selected atoms.

        Zeroing these removes the QM region's LJ interactions in QM/MM.

        Args:
            atomlist: atom indices to update.
            epsilons: new epsilon values, one per atom in atomlist.
        """
        timeA = time.time()
        logger.info("Updating LJ interaction strengths in OpenMM object.")
        if len(atomlist) != len(epsilons):
            raise InternalError("atomlist and epsilons size mismatch")
        for atomindex, newepsilon in zip(atomlist, epsilons, strict=False):
            charge, sigma, _oldepsilon = self.nonbonded_force.getParticleParameters(atomindex)
            # Different depending on type of NonbondedForce
            if isinstance(self.nonbonded_force, openmm.CustomNonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, [charge, sigma, newepsilon])
            elif isinstance(self.nonbonded_force, openmm.NonbondedForce):
                self.nonbonded_force.setParticleParameters(atomindex, charge, sigma, newepsilon)

        logger.debug("done here")
        log_time_since(timeA, "update_LJ_epsilons")

    # Updating charges in OpenMM object. Used to set QM charges to 0 for example
    # Taking list of atom-indices and list of charges (usually zero) and setting new charge
    # Note: Exceptions also needs to be dealt with (see delete_exceptions)
    def update_charges(self, atomlist, atomcharges):
        """Set new partial charges for selected atoms.

        QM/MM zeroes the QM-region charges this way before building the point-charge field.

        Args:
            atomlist: atom indices to update.
            atomcharges: new charges, one per atom in atomlist.
        """
        timeA = time.time()
        logger.info("Updating charges in OpenMM object.")
        if len(atomlist) != len(atomcharges):
            raise InternalError("atomlist and atomcharges size mismatch")
        for atomindex, newcharge in zip(atomlist, atomcharges, strict=False):
            # Updating big chargelist of OpenMM object.
            # TODO: Is this actually used?
            self.charges[atomindex] = newcharge
            _oldcharge, sigma, epsilon = self.nonbonded_force.getParticleParameters(atomindex)
            # Different depending on type of NonbondedForce
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
        """Zero the bonded terms that lie entirely inside the given atom set.

        Removes the MM description of the QM region's internal geometry, which the QM code
        provides instead.

        Args:
            atomlist: atom indices treated quantum-mechanically.
        """
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
                # NOTE: Attempt at disabling as maybe not needed
            elif isinstance(force, openmm.HarmonicAngleForce):
                logger.debug("HarmonicAngle force")
                logger.debug(f"There are {force.getNumAngles()} HarmonicAngle terms defined.")
                for i in range(force.getNumAngles()):
                    p1, p2, p3, angle, k = force.getAngleParameters(i)
                    # Are angle-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3]]
                    # Excluding if 2 or 3 QM atoms. i.e. a QM2-QM1-MM1 or QM3-QM2-QM1 term
                    # Originally set to 2
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
                # NOTE: Attempt at disabling as maybe not needed
            elif isinstance(force, openmm.PeriodicTorsionForce):
                logger.debug("PeriodicTorsionForce force")
                logger.debug(f"There are {force.getNumTorsions()} PeriodicTorsionForce terms defined.")
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4]]
                    # Excluding if 3 or 4 QM atoms. i.e. a QM3-QM2-QM1-MM1 or QM4-QM3-QM2-QM1 term
                    # Originally set to 3
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
                # NOTE: Attempt at disabling as maybe not needed
            elif isinstance(force, openmm.CustomTorsionForce):
                logger.debug("CustomTorsionForce force")
                logger.debug(f"There are {force.getNumTorsions()} CustomTorsionForce terms defined.")
                for i in range(force.getNumTorsions()):
                    p1, p2, p3, p4, pars = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
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
                # NOTE: Attempt at disabling as maybe not needed
            elif isinstance(force, openmm.CMAPTorsionForce):
                logger.debug("CMAPTorsionForce force")
                logger.debug(f"There are {force.getNumTorsions()} CMAP terms defined.")
                logger.debug(f"There are {force.getNumMaps()} CMAP maps defined")
                # Note (RB). CMAP is between pairs of backbone dihedrals.
                # Not sure if we can delete the terms:
                # http://docs.openmm.org/latest/api-c++/generated/OpenMM.CMAPTorsionForce.html
                #
                for i in range(force.getNumTorsions()):
                    jj, p1, p2, p3, p4, v1, v2, v3, v4 = force.getTorsionParameters(i)
                    # Are torsion-atoms in atomlist?
                    presence = [i in atomlist for i in [p1, p2, p3, p4, v1, v2, v3, v4]]
                    # NOTE: Not sure how to use count properly here when dealing with torsion atoms in QM-region
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
                # NOTE: Attempt at disabling as maybe not needed

            elif isinstance(force, (openmm.CMMotionRemover, openmm.CustomNonbondedForce, openmm.NonbondedForce)):
                pass
            else:
                pass

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
            forces = np.array(forces) / -49614.752589207

        self._out.write(f"{len(forces):g}\n{energy:g}\n")
        for f in forces:
            self._out.write(f"{f[0]:g} {f[1]:g} {f[2]:g}\n")
        self._out.flush()


def create_cnb(original_nbforce, system_numparticles):
    """Create a CustomNonbondedForce that mimics the original nonbonded force.

    Also creates a CustomBondForce to handle the 1-4 exceptions.
    """
    # Next, create a CustomNonbondedForce with LJ and Coulomb terms
    ONE_4PI_EPS0 = 138.935456
    # TODO: Not sure whether sqrt should be present or not in epsilon???
    energy_expression = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r;"
    # sqrt ??
    energy_expression += "epsilon = sqrt(epsilon1*epsilon2);"
    energy_expression += "sigma = 0.5*(sigma1+sigma2);"
    energy_expression += f"ONE_4PI_EPS0 = {ONE_4PI_EPS0:f};"  # already in OpenMM units
    energy_expression += "chargeprod = charge1*charge2;"
    custom_nonbonded_force = openmm.CustomNonbondedForce(energy_expression)
    custom_nonbonded_force.addPerParticleParameter("charge")
    custom_nonbonded_force.addPerParticleParameter("sigma")
    custom_nonbonded_force.addPerParticleParameter("epsilon")
    # Configure force
    custom_nonbonded_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    custom_nonbonded_force.setUseLongRangeCorrection(False)
    logger.info("Adding particles to custom force.")
    for index in range(system_numparticles):
        [charge, sigma, epsilon] = original_nbforce.getParticleParameters(index)
        custom_nonbonded_force.addParticle([charge, sigma, epsilon])
    # For CustomNonbondedForce we need (unlike NonbondedForce) to create exclusions that correspond to the automatic
    # exceptions in NonbondedForce
    # These are interactions that are skipped for bonded atoms
    numexceptions = original_nbforce.getNumExceptions()
    logger.info("numexceptions in original_nbforce:  %s", numexceptions)

    # Turn exceptions from NonbondedForce into exclusions in CustombondedForce
    # except 1-4 which are not zeroed but are scaled. These are added to Custombondforce
    exceptions_14 = []
    numexclusions = 0
    for i in range(numexceptions):
        # Get exception parameters (indices)
        p1, p2, charge, sigma, epsilon = original_nbforce.getExceptionParameters(i)
        # If 0.0 then these are CHARMM 1-2 and 1-3 interactions set to zero
        if charge._value == 0.0 and epsilon._value == 0.0:
            # Set corresponding exclusion in customnonbforce
            custom_nonbonded_force.addExclusion(p1, p2)
            numexclusions += 1
        else:
            exceptions_14.append([p1, p2, charge, sigma, epsilon])
            # [798, 801, Quantity(value=-0.0684, unit=elementary charge**2), Quantity(value=0.2708332103146632,
            # unit=nanometer), Quantity(value=0.2672524882578271, unit=kilojoule/mole)]

    logger.info("len exceptions_14 %s", len(exceptions_14))
    logger.info("numexclusions: %s", numexclusions)

    # Creating custombondforce to handle these special exceptions
    # Now defining pair parameters
    # https://github.com/openmm/openmm/issues/2698
    energy_expression = "(4*epsilon*((sigma/r)^12 - (sigma/r)^6) + ONE_4PI_EPS0*chargeprod/r);"
    energy_expression += f"ONE_4PI_EPS0 = {ONE_4PI_EPS0:f};"  # already in OpenMM units
    custom_bond_force = openmm.CustomBondForce(energy_expression)
    custom_bond_force.addPerBondParameter("chargeprod")
    custom_bond_force.addPerBondParameter("sigma")
    custom_bond_force.addPerBondParameter("epsilon")

    for exception in exceptions_14:
        idx = exception[0]
        jdx = exception[1]
        c = exception[2]
        sig = exception[3]
        eps = exception[4]
        custom_bond_force.addBond(idx, jdx, [c, sig, eps])

    logger.info("Number of defined 14 bonds in custom_bond_force: %s", custom_bond_force.getNumBonds())

    return custom_nonbonded_force, custom_bond_force


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
    # Always list of lists now

    if not (len(resnames) == len(atomnames_per_res) == len(atomtypes_per_res)):
        raise InternalError("Residue name/atomname/atomtype lists size mismatch")
    # Get list of all unique atomtypes, elements, masses

    # Create list of all AtomTypelines (unique)
    atomtypelines = []
    for _resname, atomtypelist, elemlist, masslist in zip(
        resnames, atomtypes_per_res, elements_per_res, masses_per_res, strict=False
    ):
        for atype, elem, mass in zip(atomtypelist, elemlist, masslist, strict=False):
            atomtypeline = f'<Type name="{atype}" class="{atype}" element="{elem}" mass="{mass!s}"/>\n'
            if atomtypeline not in atomtypelines:
                atomtypelines.append(atomtypeline)
    # Create list of all nonbonded lines (unique)
    nonbondedlines = []
    LJforcelines = []
    for _resname, atomtypelist, chargelist, sigmalist, epsilonlist in zip(
        resnames, atomtypes_per_res, charges_per_res, sigmas_per_res, epsilons_per_res, strict=False
    ):
        for atype, charge, sigma, epsilon in zip(atomtypelist, chargelist, sigmalist, epsilonlist, strict=False):
            if charmm:
                # LJ parameters zero here
                nonbondedline = f'<Atom type="{atype}" charge="{charge}" sigma="{0.0}" epsilon="{0.0}"/>\n'
                # Here we set LJ parameters
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
            # All other atoms
            xmlfile.write("</Residue>\n")
        xmlfile.write("</Residues>\n")
        # Write nonbonded block (even if skip_nb is True)
        xmlfile.write(f'<NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}">\n')
        if skip_nb is False:
            if charmm:
                # Writing both Nonbnded force block and also LennardJonesForce block
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
                xmlfile.write("</NonbondedForce>\n")
                xmlfile.write(f'<LennardJonesForce lj14scale="{lj14scale}">\n')
                for ljline in LJforcelines:
                    xmlfile.write(ljline)
                xmlfile.write("</LennardJonesForce>\n")
            else:
                # Only NonbondedForce block
                for nonbondedline in nonbondedlines:
                    xmlfile.write(nonbondedline)
        # Close nonbondedforce block
        xmlfile.write("</NonbondedForce>\n")
        xmlfile.write("</ForceField>\n")
    logger.info("Wrote XML-file: %s", filename)
    return filename
