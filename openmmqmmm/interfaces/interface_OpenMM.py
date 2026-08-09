import copy
import logging
import os
import time
from sys import stdout

import numpy as np
from packaging import version

from openmmqmmm.exceptions import (
    FileFormatError,
    InputError,
    InternalError,
    MissingDependencyError,
)

try:
    import openmm
    import openmm.app
    import openmm.app as openmm_app
    import openmm.unit
    import openmm.unit as openmm_unit
except ImportError:
    raise ImportError(
        "OpenMMTheory requires the OpenMM library. Try: conda install -c conda-forge openmm "
        "(see http://docs.openmm.org/latest/userguide/application.html)"
    ) from None

import contextlib

import openmmqmmm.constants
import openmmqmmm.functions.functions_parallel
import openmmqmmm.modules.module_plotting
from openmmqmmm.functions.functions_general import (
    create_conn_dict,
    find_replace_string_in_file,
    log_time_since,
    main_header,
    pygrep,
    small_header,
    sub_header,
    writelisttofile,
    writestringtofile,
)
from openmmqmmm.interfaces.interface_mdtraj import MDtraj_imagetraj, MDtraj_import, MDtraj_RMSF
from openmmqmmm.interfaces.interface_openbabel import xyz_to_pdb_with_connectivity
from openmmqmmm.modules.module_coords import (
    Fragment,
    change_origin_to_centroid,
    check_charge_mult,
    check_gradient_for_bad_atoms,
    define_dummy_topology,
    distance_between_atoms,
    get_centroid,
    write_pdbfile,
    write_xyzfile,
)
from openmmqmmm.modules.module_coords_PBC import cell_params_to_vectors, cell_vectors_to_params
from openmmqmmm.modules.module_singlepoint import Singlepoint

logger = logging.getLogger(__name__)


class OpenMMTheory:
    def __init__(
        self,
        platform="CPU",
        numcores=1,
        topoforce=False,
        forcefield=None,
        topology=None,
        CHARMMfiles=False,
        psffile=None,
        charmmtopfile=None,
        charmmprmfile=None,
        label="OpenMM",
        GROMACSfiles=False,
        gromacstopfile=None,
        grofile=None,
        gromacstopdir=None,
        Amberfiles=False,
        amberprmtopfile=None,
        properties=None,
        nonbondedMethod_noPBC="NoCutoff",
        nonbonded_cutoff_noPBC=20,
        xmlfiles=None,
        pdbfile=None,
        pdbxfile=None,
        use_parmed=False,
        xmlsystemfile=None,
        do_energy_decomposition=False,
        periodic=False,
        periodic_cell_dimensions=None,
        PBCvectors=None,
        periodic_cell_vectors=None,
        charmm_periodic_cell_dimensions=None,
        customnonbondedforce=False,
        periodic_nonbonded_cutoff=12,
        dispersion_correction=True,
        nonbondedMethod_PBC="PME",
        switching_function_distance=10.0,
        ewalderrortolerance=5e-4,
        PMEparameters=None,
        delete_QM1_MM1_bonded=False,
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
        RPMD_num_copies=32,
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
        self.numcores = numcores  # Setting for general ASH compatibility

        # Indicate that this is a MMtheory
        self.theorytype = "MM"
        self.theorynamelabel = "OpenMM"
        self.analytic_hessian = False
        self.label = label
        self.fragment = fragment
        logger.info("Imported OpenMM library version: %s", openmm.__version__)
        if version.parse(openmm.__version__) < version.parse("8.1"):
            logger.info("Warning: OpenMM version < 8.1. OpenMM 8.1 or higher is recommended")
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
        self.RPMD_num_copies = RPMD_num_copies

        # Setting for controlling whether QM1-MM1 bonded terms are deleted or not in a QM/MM job
        # See modify_bonded_forces
        # TODO: Move option to module_QMMM instead
        self.delete_QM1_MM1_bonded = delete_QM1_MM1_bonded
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
        self.nonbonded_cutoff_noPBC = nonbonded_cutoff_noPBC
        # Methods for nonbonded interactions, PBC and no-PBC
        self.nonbondedMethod_PBC = nonbondedMethod_PBC
        self.nonbondedMethod_noPBC = nonbondedMethod_noPBC
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
        if PBCvectors is not None:
            logger.info("Warning: PBCvectors keyword is on its way out. Use periodic_cell_vectors instead")
            if periodic_cell_vectors is None:
                periodic_cell_vectors = PBCvectors

        # #Always creates object we call self.forcefield that contains topology attribute
        if CHARMMfiles is True:
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

        elif GROMACSfiles is True:
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
                logger.info("WARNING: may fail if virtual sites present (e.g. TIP4P residues).")
                logger.info("Use 'parmed=True'  to avoid")
                gro = openmm.app.GromacsGroFile(grofile)
                self.grotop = openmm.app.GromacsTopFile(
                    gromacstopfile, periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir=gromacstopdir
                )

                self.topology = self.grotop.topology
                self.forcefield = self.grotop

            # TODO: Define resnames, resids, segmentnames, atomtypes, atomnames??
            self.define_mm_elements(self.topology)
        elif Amberfiles is True:
            logger.info("Reading Amber files.")
            logger.info("WARNING: Only new-style Amber7 prmtop-file will work.")
            logger.info("WARNING: Will take periodic boundary conditions from prmtop file.")
            if use_parmed is True:
                import parmed

                logger.info("Using Parmed to read Amber files.")
                self.prmtop = parmed.load_file(amberprmtopfile)
            else:
                logger.info("Using built-in OpenMM routines to read Amber files.")
                # Note: Only new-style Amber7 prmtop files work
                # If PBC vectors provided and new OpenMM version
                # Note Jan 2024: Amber prmtop files sometimes have PBC vectors (ready by OpenMM parser), this is deprecated behaviour though it seems
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
            logger.info("Warning: file may contain hardcoded constraints that can not be overridden.")
            self.system = openmm.XmlSerializer.deserializeSystem(xmlsystemfileobj)
            # NOTE: Big drawback of xmlsystemfile is that constraints have been hardcoded and can
            # NOTE: we could remove all present constraints using: self.remove_all_constraints()
            # NOTE: However, not sure how easy to enforce Hatom, rigidwater etc. constraints again without remaking system object
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
        # Simple OpenMM system without any forcefield defined. Requires ASH fragment
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
                    CHARMMfiles,
                    Amberfiles,
                    use_parmed,
                )

                # Nonbonded method to use for PBC
                if self.nonbondedMethod_PBC == "PME":
                    nonb_method_PBC = openmm.app.PME
                elif self.nonbondedMethod_PBC == "Ewald":
                    nonb_method_PBC = openmm.app.Ewald
                elif self.nonbondedMethod_PBC == "LJPME":
                    nonb_method_PBC = openmm.app.LJPME
                elif self.nonbondedMethod_PBC == "CutoffPeriodic":
                    nonb_method_PBC = openmm.app.CutoffPeriodic
                else:
                    raise InputError("Unknown nonbonded method")

                logger.info("Nonbonded PBC method selected: %s", nonb_method_PBC)

                # Determining nonbonded cutoff strategy
                smallest_boxdim = min(self.topology.getUnitCellDimensions()).value_in_unit(openmm.unit.angstroms)
                logger.info("Smallest_box dimension is: %s", smallest_boxdim)
                logger.info("periodic_nonbonded_cutoff: %s", periodic_nonbonded_cutoff)
                if smallest_boxdim < periodic_nonbonded_cutoff * 2:
                    logger.info(
                        f"Warning: Smallest box dimension is less than 2*periodic_nonbonded_cutoff = {2 * self.periodic_nonbonded_cutoff}"
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
                if CHARMMfiles is True:
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
                elif GROMACSfiles is True:
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
                elif Amberfiles is True:
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
                        if PMEparameters is not None:
                            logger.info("Nonbonded force:  Changing PME parameters")
                            force.setPMEParameters(
                                PMEparameters[0], PMEparameters[1], PMEparameters[2], PMEparameters[3]
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
                if self.nonbondedMethod_noPBC == "NoCutoff":
                    noPBC_nonbondedMethod = openmm.app.NoCutoff
                elif self.nonbondedMethod_noPBC == "CutoffNonPeriodic":
                    noPBC_nonbondedMethod = openmm.app.CutoffNonPeriodic
                elif self.nonbondedMethod_noPBC == "CutoffPeriodic":
                    raise InputError("nonbondedMethod_noPBC with CutoffPeriodic not currently allowed")
                logger.info("System is non-periodic.")
                logger.info("nonbonded noPBC Method is: %s", noPBC_nonbondedMethod)

                logger.info("Nonbonded cutoff : %s Angstrom", self.nonbonded_cutoff_noPBC)

                if CHARMMfiles is True:
                    self.system = self.forcefield.createSystem(
                        self.params,
                        nonbondedMethod=noPBC_nonbondedMethod,
                        constraints=self.autoconstraints,
                        rigidWater=self.rigidwater,
                        nonbondedCutoff=self.nonbonded_cutoff_noPBC * openmm.unit.angstroms,
                        hydrogenMass=self.hydrogenmass,
                    )
                elif Amberfiles is True:
                    self.system = self.forcefield.createSystem(
                        nonbondedMethod=noPBC_nonbondedMethod,
                        constraints=self.autoconstraints,
                        rigidWater=self.rigidwater,
                        nonbondedCutoff=self.nonbonded_cutoff_noPBC * openmm.unit.angstroms,
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
                        nonbondedCutoff=self.nonbonded_cutoff_noPBC * openmm.unit.angstroms,
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
                    logger.info("self.nonbonded_force: %s", self.nonbonded_force)
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
                    "Missing distance value for some constraints. Can apply current-geometry distances if ASH\n"
                    "fragment has been provided"
                )
                if fragment is None:
                    logger.info(
                        "No ASH fragment provided to OpenMMTheory. Will check if pdbfile is defined and use coordinates from there"
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
        self.compute_DOF()

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
        try:
            self.mm_elements = [i.element.symbol for i in topology.atoms()]
        except AttributeError:
            logger.info("Problem occurred while defining mm_elements.")
            logger.info("This may be due to virtual sites present")
            logger.info("mm_elements will be set to empty list")
            self.mm_elements = []

    # Function to write PDB-file if everything is available
    def write_pdbfile(self, positions=None, outputname="system"):

        logger.info("Writing PDB-file using OpenMMTheory object")
        logger.info("Will be using defined topology.")
        logger.info("Internal positions: %s", self.positions)
        if self.positions is not None:
            logger.info("Found positions in OpenMMTheory object. Using them to write PDB-file.")
            with open(f"{outputname}.pdb", "w") as pdbfh:
                openmm.app.PDBFile.writeFile(self.topology, self.positions, pdbfh)
        elif self.fragment is not None:
            logger.info("Found an ASH fragment file referenced. Using coordinates in fragment to write PDB-file.")
            logger.info("%s", self.fragment)
            coords_nm = self.fragment.coords * 0.1  # converting from Angstrom to nm
            pos = [
                openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in range(len(coords_nm))
            ] * openmm.unit.nanometer
            with open(f"{outputname}.pdb", "w") as pdbfh:
                openmm.app.PDBFile.writeFile(self.topology, pos, pdbfh)
        # NOTE: If pdb-file is defined we could grab coordinates from there. However, they will be the same so what is the point
        elif positions is not None:
            logger.info("Using input positions")
            with open(f"{outputname}.pdb", "w") as pdbfh:
                openmm.app.PDBFile.writeFile(self.topology, positions, pdbfh)
        else:
            raise InputError("Found neither system positions defined or an ASH fragment file. Can not write PDB-file.")

    # Function that handles periodicity in forcefield objects (for Amber, CHARMM). TODO: Test GROMACS and XML
    def set_periodics_before_system_creation(
        self, periodic_cell_vectors, pdb_pbc_vectors, periodic_cell_dimensions, CHARMMfiles, Amberfiles, use_parmed
    ):

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
            if CHARMMfiles is True:
                self.forcefield.box_vectors = periodic_cell_vectors * openmm.unit.angstrom
                logger.info("PBC box vectors set: %s", self.forcefield.box_vectors)
            elif Amberfiles is True and use_parmed is True:
                # Necessary for parmed object to define box_vectors in forcefield object
                self.forcefield.box_vectors = periodic_cell_vectors * openmm.unit.angstrom
                logger.info("PBC box vectors set: %s", self.forcefield.box_vectors)
            elif Amberfiles is True and use_parmed is False:
                # Not necessary to define box_vectors (grabbed from topology above) but we have to make sure PBC is on
                # Happens if no IFBOX defined in prmtop file but we still want periodicity
                # Hacky fix below
                logger.info("Amber-prmtop getIfBox: %s", self.forcefield._prmtop.getIfBox())
                self.forcefield._prmtop._raw_data["POINTERS"][27] = 1
                logger.info("Amber-prmtop getIfBox: %s", self.forcefield._prmtop.getIfBox())

                if version.parse(openmm.__version__) < version.parse("8.1"):
                    logger.info("Warning: Amber prmtop file detected and OpenMM version < 8.0")
                    logger.info("Warning: Will assume cubic box and set PBC vectors in a hacky way")
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
                logger.info("Warning: problems with unitcell dimensions setting.")
                logger.info("Warning: Will assume cubic box and set PBC vectors instead")
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
            if CHARMMfiles is True and use_parmed is False:
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
            if (CHARMMfiles is True and use_parmed is True) or (Amberfiles is True and use_parmed is True):
                pass
            elif Amberfiles is True and use_parmed is False:
                logger.info("Amber ff getIfBox %s", self.forcefield._prmtop.getIfBox())
                # Hacky thing to make sure PBC is on for Amber.
                # PBCvectors will be grabbed from topology above
                # Happens if no IFBOX defined in prmtop file but we still want periodicity
                self.forcefield._prmtop._raw_data["POINTERS"][27] = 1

                if version.parse(openmm.__version__) < version.parse("8.1"):
                    logger.info("Warning: Amber prmtop file detected and OpenMM version < 8.1")
                    logger.info("Warning: Will assume cubic box and set PBC vectors in a hacky way")
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"] = np.array([0.0, 0.0, 0.0, 0.0])
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][0] = 90.0
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][1] = periodic_cell_dimensions[0]
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][2] = periodic_cell_dimensions[1]
                    self.forcefield._prmtop._raw_data["BOX_DIMENSIONS"][3] = periodic_cell_dimensions[2]
        elif pdb_pbc_vectors is not None:
            logger.info(
                "Warning: neither user keyword periodic_cell_vectors or periodic_cell_dimensions was set (None)"
            )
            logger.info(
                "However, we found PBC information inside PDB-topology of the PDB-file that was read in. Using this and continuing"
            )
            # Should work automatically
        elif self.topology.getPeriodicBoxVectors() is not None:
            logger.info("Found PBC information in topology object. Using this and continuing")
        else:
            raise FileFormatError("Found no PBC information, yet periodicity is requested. Exiting!")

    # Get PBC vectors from topology of openmm object. Convenient in a script
    def get_PBC_vectors(self):

        # Get PBC vectors
        vectors_nm = list(self.topology.getPeriodicBoxVectors())
        a = list(vectors_nm[0].value_in_unit(openmm.unit.angstrom))
        b = list(vectors_nm[1].value_in_unit(openmm.unit.angstrom))
        c = list(vectors_nm[2].value_in_unit(openmm.unit.angstrom))
        # Return List of lists
        return [a, b, c]

    # Set numcores method: currently inactive. Included for completeness
    def set_numcores(self, numcores):
        self.numcores = numcores

    # Set numcores method
    def cleanup(self):
        logger.info("Cleanup for OpenMMTheory called")

    # add force that restrains atoms to a fixed point:
    # https://github.com/openmm/openmm/issues/2568

    # To set positions in OpenMMobject (in nm) from np-array (Angstrom)
    def set_positions(self, coords, simulation):

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
        # Note we are modifying the system and topology itself because we are doing OpenMMTheory.run that creates new sim and context each time
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
        logger.info("Adding new force to system: %s", newforce)
        self.system.addForce(newforce)

    def remove_force(self, forceindex):
        logger.info(f"Removing force-index {forceindex}: {self.system.getForces()[forceindex].getName()}")
        self.system.removeForce(forceindex)

    def remove_force_by_name(self, forcename):
        logger.info(f"Searching forces and removing a force name: {forcename}")
        for i, force in enumerate(self.system.getForces()):
            logger.info("force name: %s", force.getName())
            if force.getName() == forcename:
                logger.info(f"Removing force-index {i}: {forcename}")
                self.system.removeForce(i)

    # Bond restraint force, e.g. for umbrella sampling
    # TODO : unit check
    def add_custom_bond_force(self, i, j, value, forceconstant):

        logger.info(
            f"Adding custom bond force between atom index i={i} and j={j} with value: {value} Angstrom, forceconstant={forceconstant} kcal/mol/Angstrom^2"
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

        logger.info(
            f"Adding custom angle force for atoms: {i}, {j}, {k}  with value: {value} radians with forceconstant={forceconstant}"
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
        logger.info(
            f"Adding CustomCentroidBondForce between centroid of host {host_indices}  and centroid of guest {guest_indices} "
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
    def add_flatbottom_centerforce(self, molA_indices=None, molB_indices=None, distance=5.0, forceconstant=1.0):

        logger.info("Inside add_flatbottom_centerforce")
        logger.info("molA_indices size: %s", len(molA_indices))
        logger.info("molB_indices size: %s", len(molB_indices))
        logger.info("forceconstant: %s", forceconstant)
        logger.info("distance: %s", distance)
        # Define force
        centerforce = openmm.CustomCentroidBondForce(2, "0.5*k*max(0, distance(g1,g2)-r0)^2")
        # Periodic case (note: periodicdistance not available for CustomCentroidBondForce)
        if self.periodic is True:
            logger.info("Warning: add_flatbottom_centerforce with PBC is not well tested")
            centerforce.setUsesPeriodicBoundaryConditions = True

        centerforce.addGlobalParameter(
            "k", forceconstant * openmm.unit.kilocalories_per_mole / openmm.unit.angstroms**2
        )
        centerforce.addGlobalParameter("r0", distance * openmm.unit.angstrom)
        g1 = molA_indices  # solute/ligand
        g2 = molB_indices  # rest
        centerforce.addGroup(g1)  # index will be 0
        centerforce.addGroup(g2)  # index will be 1
        centerforce.addBond([0, 1], [])  # no [] since global
        self.system.addForce(centerforce)
        logger.info("Added center force")
        return centerforce

    def add_custom_external_force(self):

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
        logger.info("Updating custom external force")
        # Convert Eh/Bohr gradient to force in kj/mol nm
        # *49614.501681716106452
        # NOTE: default conversion factor (49614.752589207) assumes input gradient in Eh/Bohr and converting to kJ/mol nm
        forces = -gradient * 49614.752589207
        for i, f in enumerate(forces):
            customforce.setParticleParameters(i, i, f)
        customforce.updateParametersInContext(simulation.context)

    # Function to add restraints to system before MD
    def add_bondrestraints(self, restraints=None):
        logger.info("Adding restraints: %s", restraints)

        new_restraints = openmm.HarmonicBondForce()
        for i, j, d, k in restraints:
            logger.info(
                f"Adding bond restraint between atoms {i} and {j}. Distance value: {d} Å. Force constant: {k} kcal/mol*Å^-2"
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
        self, host_index, guest_index, k_xy=10.0, z_cc=11.0, alpha=35.0, R_cylinder=1.0, force_group=10
    ):

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
        funnel.addGlobalParameter("R_cylinder", R_cylinder * openmm.unit.angstrom)

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
    def add_CV_restraint(self, cvforce, restraint_par, cvtype):

        # Make copy of CVforce (otherwise we can not use it also in restraint)
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
    def saveXML(self, xmlfile="system_full.xml"):

        serialized_system = openmm.XmlSerializer.serialize(self.system)
        with open(xmlfile, "w") as f:
            f.write(serialized_system)
        logger.info("Wrote system XML file: %s", xmlfile)

    # Function to add bond constraints to system before MD
    def add_bondconstraints(self, constraints=None):

        for i, j, d in constraints:
            logger.info(f"Adding bond constraint between atoms {i} and {j}. Distance value: {d:.4f} Å")
            self.system.addConstraint(i, j, d * openmm.unit.angstroms)

    # Remove all defined constraints in system
    def remove_all_constraints(self):
        todelete = []
        # Looping over all defined system constraints
        for i in range(self.system.getNumConstraints()):
            todelete.append(i)
        for d in reversed(todelete):
            self.system.removeConstraint(d)

    # Remove specific constraints
    def remove_constraints(self, constraints):
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

        logger.info(f"Freezing {len(frozen_atoms)} atoms by setting particles masses to zero.")

        # Modify particle masses in system object. For freezing atoms
        for i in frozen_atoms:
            self.system.setParticleMass(i, 0 * openmm.unit.daltons)

        # Also adding exceptions to nonbonded force to avoid interactions between frozen atoms (causes problems otherwise in NPT)
        logger.info(
            "Also adding exceptions to nonbonded force for frozen atoms to avoid interactions between them (avoids problems in NPT)."
        )
        self.addexceptions(frozen_atoms)

        # Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    # Changed masses according to user input dictionary
    def modify_masses(self, changed_masses=None):

        logger.info("Modify masses according:  %s", changed_masses)
        # Preserve original masses
        # Modify particle masses in system object.
        for am in changed_masses:
            self.system.setParticleMass(am, changed_masses[am] * openmm.unit.daltons)

        # Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    def unfreeze_atoms(self):
        # Looping over system_masses if frozen, otherwise empty list
        for atom, mass in zip(self.allatoms, self.system_masses_original, strict=False):
            self.system.setParticleMass(atom, mass)

        # Update list of current masses
        self.system_masses = [self.system.getParticleMass(i)._value for i in self.allatoms]

    # This removes interactions between particles in a region (e.g. QM-QM or frozen-frozen pairs)
    # Give list of atom indices for which we will remove all pairs
    def addexceptions(self, atomlist):

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
                        # Basically both nonbonded forces have to have same exclusions (or exception where chargepro=0, eps=0)
                        # TODO: This leads to : Exception: CustomNonbondedForce: Multiple exclusions are specified for particles
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
        self.timestep = timestep
        self.coupling_frequency = coupling_frequency
        self.temperature = temperature
        self.integrator_name = integrator

    # Create integrator.
    def create_integrator(self):

        # NOTE: Integrator definition has to be here (instead of set_simulation_parameters) as it has to be recreated for each updated simulation
        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
        # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
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
            logger.info("here1")
            # TODO: options
            self.integrator = openmm.DrudeLangevinIntegrator(
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.temperature * openmm.unit.kelvin,
                self.timestep * openmm.unit.picoseconds,
                4,
            )
            logger.info("here2")
        elif self.integrator_name == "RPMDIntegrator":
            logger.info("RPMDIntegrator will be used")
            logger.info("Warning: Autoconstraints, rigidwater and other contraints must have been disabled.")
            logger.info(f"RPMD number of copies set to {self.RPMD_num_copies}. Use RPMD_num_copies keyword to change")
            self.integrator = openmm.RPMDIntegrator(
                self.RPMD_num_copies,
                self.temperature * openmm.unit.kelvin,
                self.coupling_frequency / openmm.unit.picosecond,
                self.timestep * openmm.unit.picoseconds,
            )
        else:
            raise InputError(
                "Unknown integrator.\n Valid integrator keywords are: VerletIntegrator, VariableVerletIntegrator, LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VariableLangevinIntegrator, RPMDIntegrator"
            )

    # Create simulation object (now not part of OpenMMTheory)
    def create_simulation(self, internal=False):
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
        self.forcegroups = {}
        logger.info("inside forcegroupify")
        logger.info("self.system.getForces() %s", self.system.getForces())
        logger.info("Number of forces:\n %s", self.system.getNumForces())
        for i in range(self.system.getNumForces()):
            force = self.system.getForce(i)
            force.setForceGroup(i)
            self.forcegroups[force] = i

    def getEnergyDecomposition(self, context):
        energies = {}
        for f, i in self.forcegroups.items():
            energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
        return energies

    def printEnergyDecomposition(self, simulation):

        timeA = time.time()
        # Energy decomposition
        # NOTE: Calling this is expensive (seconds)as the energy has to be recalculated.
        openmm_energy = {}
        energycomp = self.getEnergyDecomposition(simulation.context)
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
    def compute_DOF(self):

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
        Grad=False,
        fragment=None,
        qmatoms=None,
        label=None,
        charge=None,
        mult=None,
        PC=False,
        current_MM_coords=None,
        MMcharges=None,
        qm_elems=None,
        numcores=1,
    ):
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
        # Constraints and frozen atoms have to instead by enforced by geomeTRICOptimizer, non-OpenMM dynamics module etc.
        defined_constraints = self.system.getNumConstraints()
        logger.info("Number of OpenMM system constraints defined: %s", defined_constraints)

        if self.autoconstraints is not None or self.rigidwater:
            logger.error(
                "OpenMM autoconstraints (HBonds,AllBonds,HAngles) in OpenMMTheory are not compatible with OpenMMTheory.run()"
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
                "User-defined frozen atoms/constraints/restraints in OpemmTheory are not compatible with OpenMMTheory.run()"
            )
            logger.info(
                "Constraints must instead be defined inside the program that called OpenMMtheory.run(), e.g. geomeTRICOptimizer."
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
        if Grad is True:
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
            self.printEnergyDecomposition(simulation)
        logger.info(small_header("Ending OpenMM interface"))
        log_time_since(module_init_time, "OpenMM run")
        if Grad is True:
            return self.energy, self.gradient
        else:
            return self.energy

    def getatomcharges(self):

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
    def update_LJ_epsilons(self, atomlist, epsilons):

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
        # Making sure that there still is a nonbonded force present in system (in case deleted)
        for i, force in enumerate(self.system.getForces()):
            logger.debug(f"i is {i} and force is {force}")
            if isinstance(force, openmm.NonbondedForce):
                logger.debug("here")
                # NOTE: Attempt at disabling
            if isinstance(force, openmm.CustomNonbondedForce):
                pass
        logger.debug("done here")
        log_time_since(timeA, "update_charges")

    def modify_bonded_forces(self, atomlist):

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

                    if self.delete_QM1_MM1_bonded is True:
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
                            f"Before p1: {p1} p2: {p2} p3: {p3} p4: {p4} periodicity: {periodicity} phase: {phase} k: {k}"
                        )
                        force.setTorsionParameters(i, p1, p2, p3, p4, periodicity, phase, 0)
                        numpertorsionterms_removed += 1
                        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                        logger.debug(
                            f"After p1: {p1} p2: {p2} p3: {p3} p4: {p4} periodicity: {periodicity} phase: {phase} k: {k}"
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


# Reporter for forces similar to xyz format
class ForceReporter:
    def __init__(self, file, reportInterval, atomic_units=False):
        self._out = open(file, "w")  # noqa: SIM115 - reporter handle, closed in __del__
        self._reportInterval = reportInterval
        self.atomic_units = atomic_units

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
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


# For frozen systems we use Customforce in order to specify interaction groups
# if len(self.frozen_atoms) > 0:

# https://ahy3nz.github.io/posts/2019/30/openmm2/
# http://www.maccallumlab.org/news/2015/1/23/testing


# Comes close to NonbondedForce results (after exclusions) but still not correct
# The issue is most likely that the 1-4 LJ interactions should not be excluded but rather scaled.
# See https://github.com/openmm/openmm/issues/1200
# https://github.com/openmm/openmm/issues/1696
# How to do:
# 1. Keep nonbonded force for only those interactions and maybe also electrostatics?
# Mimic this??: https://github.com/openmm/openmm/blob/master/devtools/forcefield-scripts/processCharmmForceField.py
# Or do it via Parmed? Better supported for future??
# 2. Go through the 1-4 interactions and not exclude but scale somehow manually. But maybe we can't do that in
# CustomNonbonded Force?
# Presumably not but maybe can add a special force object just for 1-4 interactions. We
def create_cnb(original_nbforce, system_numparticles):
    """Creates a CustomNonbondedForce object that mimics the original nonbonded force
    and also a Custombondforce to handle 14 exceptions
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
            # [798, 801, Quantity(value=-0.0684, unit=elementary charge**2), Quantity(value=0.2708332103146632, unit=nanometer), Quantity(value=0.2672524882578271, unit=kilojoule/mole)]

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


# TODO: Look into: https://github.com/ParmEd/ParmEd/blob/7e411fd03c7db6977e450c2461e065004adab471/parmed/structure.py#L2554


# Clean up list of lists of constraint definition. Add distance if missing
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


def OpenMM_Opt(
    fragment=None,
    theory=None,
    maxiter=1000,
    tolerance=1,
    enforcePeriodicBox=True,
    traj_frequency=100,
    use_reporter=True,
):

    module_init_time = time.time()
    logger.info(main_header("OpenMM Optimization"))

    if fragment is None:
        raise InputError("No fragment object. Exiting.")

    # Distinguish between OpenMM theory or QM/MM theory
    if isinstance(theory, OpenMMTheory):
        openmmobject = theory
    else:
        raise InputError("Only OpenMMTheory allowed in OpenMM_Opt. Exiting.")

    logger.info("Number of atoms: %s", fragment.numatoms)
    logger.info("Max iterations: %s", maxiter)
    logger.info(f"Tolerance: {tolerance} kj/mol/nm:")
    # if float(openmm.__version__) >= 8.1:
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
        logger.info("WARNING: Autoconstraints have not been set in OpenMMTheory object definition.")
        logger.info("This means that by default no bonds are constrained in the optimization.")
        logger.info("Will continue...")
    if (openmmobject.rigidwater is True and len(openmmobject.user_frozen_atoms) != 0) or (
        openmmobject.autoconstraints is not None and len(openmmobject.user_frozen_atoms) != 0
    ):
        logger.info(
            "WARNING: Frozen_atoms options selected but there are general constraints defined in "
            "the OpenMM object (either rigidwater=True or autoconstraints is not None)\n"
            "OpenMM will crash if constraints and frozen atoms involve the same atoms"
        )

    openmmobject.set_simulation_parameters(timestep=0.001, temperature=1, integrator="VerletIntegrator")

    # CREATE SIMULATION OBJECT
    simulation = openmmobject.create_simulation()

    logger.info("Simulation created.")

    #############################################
    # New in OpenMM 8.1: reporters for minimizer
    # StateDataReporter
    #############################################
    # if float(openmm.__version__ ) >= 8.1 and use_reporter is True:
    if version.parse(openmm.__version__) >= version.parse("8.1") and use_reporter is True:

        class Reporter(openmm.openmm.MinimizationReporter):
            def report(self, iteration, x, grad, args):
                if not hasattr(self, "totaliter"):
                    self.totaliter = -1

                # Counting total iterations
                self.totaliter += 1

                self.get_forces(grad)
                short = False

                if short is True:
                    # Possible short mode: not finished
                    logger.info(f"Iteration {iteration} ")
                else:
                    logger.info("TOTAL iteration: %s", self.totaliter)
                    logger.info(f"Micro Iteration {iteration}")
                    self.print_energy(args)
                    self.print_forces()
                    self.write_traj(x, iteration)
                # Once maxiter reached
                if iteration == maxiter - 1:
                    logger.info("Max iterations reached. Now modifying restraints and restarting")
                    return True

                return False

            def write_traj(self, x, iteration):
                if self.totaliter % traj_frequency == 0:
                    logger.info("%s", "-" * 40)
                    logger.info("Now writing to trajectory file")
                    logger.info("%s", "-" * 40)
                    # Reshaping and converting to Angstrom
                    pos = 10 * np.array(x).reshape(-1, 3)
                    write_xyzfile(fragment.elems, pos, "OpenMMOpt_traj", writemode="a")

            def print_energy(self, args):
                system_energy = args["system energy"] / openmmqmmm.constants.hartokj
                restraint_energy = args["restraint energy"] / openmmqmmm.constants.hartokj
                args["restraint strength"]
                args["max constraint error"]
                logger.info("System energy: %s", system_energy)
                logger.info("Restraint energy: %s", restraint_energy)

            def get_forces(self, grad):
                # Reshaping
                g = np.array(grad).reshape(-1, 3)  # To confirm
                kjmolnm_to_atomic_factor = -49614.752589207
                self.forces_init = g / kjmolnm_to_atomic_factor
                self.rms_force = np.sqrt(sum(n * n for n in self.forces_init.flatten()) / len(forces_init.flatten()))
                self.max_force = self.forces_init.max()

            def print_forces(self):
                logger.info(f"RMS force (w restraints): {self.rms_force} Eh/Bohr")
                logger.info(f"Max force (w restraints): {self.max_force} Eh/Bohr")
                logger.info("")

            def get_state(self):
                logger.info("")
                self.state = simulation.context.getState(
                    getEnergy=True, getForces=True, enforcePeriodicBox=enforcePeriodicBox
                )

        reporter = Reporter()

    # Context: settings positions of simulation object
    logger.info("Now adding coordinates")
    openmmobject.set_positions(fragment.coords, simulation)

    logger.info("")
    state = simulation.context.getState(getEnergy=True, getForces=True, enforcePeriodicBox=enforcePeriodicBox)
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
        # Removing possible old traj file
        with contextlib.suppress(OSError):
            os.remove("OpenMMOpt_traj.xyz")
        simulation.minimizeEnergy(maxIterations=maxiter, tolerance=tolerance, reporter=reporter)
        logger.info("Minimization done.")
        logger.info("OpenMM_Opt trajectory was written to: OpenMMOpt_traj.xyz")
    else:
        simulation.minimizeEnergy(maxIterations=maxiter, tolerance=tolerance)
        logger.info("Minimization done.")

    #####################################
    logger.info("")
    state = simulation.context.getState(
        getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=enforcePeriodicBox
    )
    logger.info(
        "%s",
        f"Final Potential energy is: {state.getPotentialEnergy().value_in_unit_system(openmm.unit.md_unit_system) / openmmqmmm.constants.hartokj} Eh",
    )
    forces_final = np.array(state.getForces(asNumpy=True)) / kjmolnm_to_atomic_factor
    rms_force = np.sqrt(sum(n * n for n in forces_final.flatten()) / len(forces_final.flatten()))
    logger.info(f"Final RMS force: {rms_force} Eh/Bohr (w/o restraints)")
    logger.info(f"Final Max force: {forces_final.max()} Eh/Bohr (w/o restraints)")

    # Writing final PDB-file. If system is non-periodic (according to OpenMMTheory settings) then we set enforcePeriodicBox to False
    # to avoid some strange geometry translation
    if openmmobject.periodic is True:
        logger.info(f"Writing final PDB file (enforcePeriodicBox={enforcePeriodicBox})")
        positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=enforcePeriodicBox).getPositions()
    else:
        logger.info("Writing final PDB file (enforcePeriodicBox=False)")
        positions = simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions()
    write_pdbfile_openMM(openmmobject.topology, positions, "frag-minimized.pdb")

    # Get coordinates to update fragment
    # Strange bug before:
    newcoords = (
        simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        .getPositions(asNumpy=True)
        .value_in_unit(openmm.unit.angstrom)
    )
    logger.info("")
    logger.info("Updating coordinates in ASH fragment.")
    fragment.coords = newcoords

    logger.info("All Done!")
    log_time_since(module_init_time, "OpenMM_Opt")

    return fragment


# Convenient
def print_systemsize(modeller):
    logger.info(f"System size: {len(modeller.getPositions())} atoms\n")


def OpenMM_Modeller(
    pdbfile=None,
    forcefield_object=None,
    forcefield=None,
    xmlfile=None,
    waterxmlfile=None,
    watermodel=None,
    pH=7.0,
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
    membraneCenterZ=0.0,
    residuetemplate_choice=None,
):
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

    # Water model. May be overridden by forcefield below
    # if watermodel == "tip3p":
    #    # Possible Problem: this only has water, no ions.
    #    # Problem: we need to define watermodel also

    # Forcefield options
    if forcefield is not None:
        logger.info("Forcefield: %s", forcefield)
        if forcefield == "Amber99" or forcefield == "Amber99sb":
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

        # Water model selection for CHARMM forcefields
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

        # Water model selection for AMber forcefields
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

    ############
    # Define a forcefield if using XML-files
    if xmlfile is not None:
        logger.info("XMfile: %s", xmlfile)
        logger.info("Water model: %s", watermodel)
        logger.info("Water xmlfile: %s", waterxmlfile)
        # Basic checks
        if extraxmlfile is not None:
            logger.info("Using extra XML file: %s", extraxmlfile)
            # Checking if file exists first before continuing
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
            logger.info("Warning: watermodel/waterxmlfile ignored when forcefield_object is supplied")

    else:
        raise InputError("You must provide a forcefield name, forcefieldobject or xmlfile keywords!")

    logger.info("PDBfile: %s", pdbfile)
    logger.info("pH: %s", pH)
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
            "Warning: PDBFixer can create unreasonable orientations of residues if residues are missing or multiple occupancies are present.\n \
        You should inspect the created PDB-file to be sure."
        )
        logger.info("Wrote PDBfile: system_afterfixes.pdb")
        pdbfile_for_modeller = "system_afterfixes.pdb"
    else:
        logger.info("Skipping PDBFixer")
        pdbfile_for_modeller = pdbfile

    # Load fixed PDB-file and create Modeller object
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
    # Getting residues for each chain
    for chain_x in modeller_chains:
        logger.info(
            f"This is chain {chain_x.index}, it has {len(chain_x._residues)} residues and they are: {chain_x._residues}\n"
        )
    logger.info("\n")

    # PRINTING big table of residues
    logger.info("User defined residue variants per chain:")
    for rv_key, rv_vals in residue_variants.items():
        logger.info(f"Chain {rv_key} : {rv_vals}")
    logger.info("\nMODELLER TOPOLOGY - RESIDUES TABLE\n")
    logger.info(
        "%s",
        "  {:<12}{:<13}{:<13}{:<13}{:<13}       {}".format(
            "ASH-resid", "Resname", "Chain-index", "Chain-name", "ResID-in-chain", "User-modification"
        ),
    )
    logger.info("%s", "-" * 100)
    current_chainindex = 0
    # Also using loop to get residue_states list that we pass on to modeller.addHydrogens
    residue_states = []
    for each_residue in modeller_residues:
        # Division line between chains
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

    # NOTE: to be deleted
    if len(residue_states) != numresidues:
        raise InputError("residue_states != numresidues. Something went wrong")

    # Adding hydrogens feeding in residue_states
    # This is were missing residue/atom errors will come
    logger.info("")
    logger.info("Adding hydrogens for pH: %s", pH)
    logger.info("Warning: OpenMM Modeller will fail in this step if residue information is missing")
    logger.info("residue_states: %s", residue_states)

    # Dealing with possible user-defined residuetemplate_choice
    residueTemplates = {}  # initisal
    if residuetemplate_choice is not None:
        logger.info("Found user-specified residuetemplate_choice")
        logger.info("Will generate residueTemplates based on residuetemplate_choice: %s", residuetemplate_choice)
        logger.info("Note: residuetemplate_choice should be a dict like this: residuetemplate_choice={'FER':'FE2'}   ")
        residueTemplates = {}
        for resname, choice in residuetemplate_choice.items():
            residueTemplates = {res: choice for res in modeller.topology.residues() if res.name == resname}
    logger.info("residueTemplates: %s", residueTemplates)

    # Checking if we have problems with unmatched residues
    logger.info("\nNow checking if we have problems with unmatched residues")
    # NOTE: We would get exception in addHydrogens anyway
    try:
        forcefield_obj.getUnmatchedResidues(modeller.topology, residueTemplates=residueTemplates)
    except Exception as e:
        logger.info("Exception found during forcefield_obj.getUnmatchedResidues.")
        logger.info("Exception: %s", e)
        logger.info(
            "\nASH interpretation. you probably have multiple matching templates in the forcefield XML-file for a residue"
        )
        raise InputError(
            "This occurs e.g. for the case of Fe2+ vs Fe3+ ion in the Amber FF.\nTo deal with this problem, you have to provide a residuetemplate_choice dictionary to the ASH interface\nExample: residuetemplate_choice should be a dict like this: residuetemplate_choice={'FER':'FE2'}   \n   where FER is here the name of the residue (in PDB-file) and FE2 is the name of the desired template in the forcefield XML-file"
        ) from e
    logger.info("No problem with unmatched residues found. Continuing")

    try:
        logger.info("residueTemplates: %s", residueTemplates)
        modeller.addHydrogens(forcefield_obj, pH=pH, variants=residue_states, residueTemplates=residueTemplates)
    except ValueError as errormessage:
        logger.error("\nError: OpenMM modeller.addHydrogens signalled a ValueError")
        logger.info(
            "This is a common error and suggests a problem in PDB-file or missing residue information in the forcefield."
        )
        logger.info(
            "Non-standard inorganic/organic residues require providing an additional XML-file via extraxmlfile= option"
        )
        logger.info("Note that C-terminii require the dangling O-atom to be named OXT ")
        raise InputError(
            f"Read the ASH documentation or the OpenMM documentation on dealing with this problem.\n\nFull error message from OpenMM:\n{errormessage}"
        ) from errormessage

    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_afterH.pdb")
    print_systemsize(modeller)

    # If using Residuetemplates then we have to remade after systemchange (addHydrogens)
    if residuetemplate_choice is not None:
        for resname, choice in residuetemplate_choice.items():
            residueTemplates = {res: choice for res in modeller.topology.residues() if res.name == resname}

    # Adding Solvent
    if implicit is True:
        periodic = False
        logger.info("We are doing implicit solvation")
        logger.info("Setting periodic to False")
        logger.info("Available implicit solvent models:")
        logger.info(
            "implicit/gbn2.xml, implicit/hct.xml, implicit/obc1.xml, implicit/obc2.xml, implicit/gbn.xml, implicit/gbn2.xml"
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
            membraneCenterZ=membraneCenterZ * openmm_unit.angstrom,
            minimumPadding=membrane_padding * openmm_unit.angstrom,
        )

        write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent_ions.pdb")
        # Ions
        # NOTE: Had to remove separate ion-add step due to OpenMM 8.1 change
        print_systemsize(modeller)
        # Create ASH fragment and write to disk
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
        write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent_ions.pdb")

        # Ions
        # NOTE: Had to remove separate ion-add step due to OpenMM 8.1 change
        print_systemsize(modeller)
        # Create ASH fragment and write to disk
        fragment = Fragment(pdbfile="system_aftersolvent_ions.pdb")

    write_pdbfile_openMM(modeller.topology, modeller.positions, "finalsystem.pdb")
    write_pdbxfile_openMM(modeller.topology, modeller.positions, "finalsystem.cif")
    fragment.print_system(filename="finalsystem.ygg")
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
    # Write out System XMLfile
    # TODO: Disable ?
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
    logger.info("finalsystem.ygg  (ASH fragment file)")
    logger.info("finalsystem.xyz   (XYZ coordinate file)")
    logger.info(f"{systemxmlfile}   (System XML file)")
    logger.info("\n\n OpenMM_Modeller done! System has been fully set up!\n")
    logger.warning("Strongly recommended: Check finalsystem.pdb carefully for correctness!")
    logger.info("\nTo use this system setup to define a future OpenMMTheory object you can either do:\n")

    logger.info("1. Define using separate forcefield XML files and PDB-file (for topology):")
    if extraxmlfile is None:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="finalsystem.pdb", periodic={periodic})'
        )
    else:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}", "{extraxmlfile}"], pdbfile="finalsystem.pdb", periodic={periodic})'
        )
    logger.info("2. Define using separate forcefield XML files and PDBx/mmCIF file (instead of PDB):")
    if extraxmlfile is None:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbxfile="finalsystem.cif", periodic={periodic})'
        )
    else:
        logger.info(
            f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}", "{extraxmlfile}"], pdbxfile="finalsystem.cif", periodic={periodic})'
        )
    logger.info(
        "3. Use forcefield object file :\n %s",
        f'omm = OpenMMTheory(topoforce=True, forcefield=forcefield_object, pdbfile="finalsystem.pdb", topology=modeller.topology, periodic={periodic})',
    )
    logger.info("")
    logger.info("")
    if residuetemplate_choice is not None:
        logger.info(
            "Warning: A residuetemplate_choice option was provided to OpenMM_Modeller. This means that you will have to provide this also when defining an OpenMMTheory object."
        )
        logger.info(
            f'E.g. like this: omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="finalsystem.pdb", periodic={periodic}, residuetemplate_choice={residuetemplate_choice})'
        )
    # Check system for atoms with large gradient and print warning
    # TODO: Can we avoid re-creating the omm object ?
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
    SP_result = Singlepoint(theory=omm, fragment=fragment, Grad=True)
    check_gradient_for_bad_atoms(fragment=fragment, gradient=SP_result.gradient, threshold=45000)

    log_time_since(module_init_time, "OpenMM_Modeller")

    # Return openmmobject. Could be used directly
    return openmmobject, fragment


def write_pdbfile_openMM(topology, positions, filename, connectivity_dict=None):

    if connectivity_dict is not None:
        logger.info("Connectivity passed to write_pdbfile_openMM")
        openmm_add_bonds_to_topology(topology, connectivity_dict)

    with open(filename, "w") as pdbfh:
        openmm.app.PDBFile.writeFile(topology, positions, file=pdbfh)
    logger.info("Wrote PDB-file: %s", filename)


def write_pdbxfile_openMM(topology, positions, filename, connectivity_dict=None):

    if connectivity_dict is not None:
        logger.info("Connectivity passed to write_pdbxfile_openMM")
        openmm_add_bonds_to_topology(topology, connectivity_dict)

    with open(filename, "w") as pdbfh:
        openmm.app.PDBxFile.writeFile(topology, positions, file=pdbfh)
    logger.info("Wrote PDBx-file: %s", filename)


# Take OpenMM topology and connectivity dictionary and add bonds to topology
# in order for OpenMM PDBFile.writeFile to write CONECT lines
def openmm_add_bonds_to_topology(topology, connectivity):
    atoms = list(topology.atoms())
    for conatom, conlist in connectivity.items():
        for conl in conlist:
            topology.addBond(atoms[conatom], atoms[conl])


# Assumes all atoms of small molecule present (including hydrogens)
def solvate_small_molecule(
    fragment=None,
    charge=None,
    mult=None,
    watermodel=None,
    solvent_boxdims=None,
    xmlfile=None,
    LJ_treatment=None,
    skip_xmlfile=False,
):
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

    # Check if fragment is provided
    if fragment is None:
        raise InputError("No fragment object provided. Exiting.")

    # Check charge/mult
    charge, mult = check_charge_mult(charge, mult, "QM", fragment, "solvate_small_molecule")

    # Check xmlfile

    if xmlfile is None and skip_xmlfile is False:
        raise InputError(
            "No xmlfile was provided. You must provide one\nIf you need a forcefield for the solute then try :\n              small_molecule_parameterizer"
        )

    # Read XML-file and check for LJ treatment
    if skip_xmlfile is False:
        logger.info("Checking xmlfile for LJ treatment")
        if pygrep('coulomb14scale="0.83333', xmlfile):
            logger.info("Found Amber-style scaling parameter.")
            LJ_treatment = "amber"
        elif pygrep("LennardJonesForce", xmlfile):
            logger.info("Found CHARMM-style format.")
            LJ_treatment = "charmm"
        else:
            raise InputError(
                "Unknown LJ14 scaling type in XML-file: neither CHARMM nor Amber format was recognized\nSolvation requires an Amber- or CHARMM-style forcefield XML-file"
            )

        logger.info("LJ_treatment: %s", LJ_treatment)

    # Now selecting watermodel XML-file based on whether CHARMM, Amber etc.
    if watermodel == "tip3p" or watermodel == "TIP3P":
        logger.info("Using watermodel=TIP3P")
        if LJ_treatment == "amber":
            waterxmlfile = "amber14/tip3p.xml"
        elif LJ_treatment == "charmm":
            waterxmlfile = "charmm36/water.xml"
        else:
            raise InputError(f"Unsupported LJ_treatment ({LJ_treatment}): must be 'amber' or 'charmm'")
    else:
        raise InputError("Only TIP3P water supported for now")

    # Create forcefield object
    if skip_xmlfile is True:
        logger.info("Creating forcefield using XML-files: %s", waterxmlfile)
        forcefield = openmm_app.forcefield.ForceField(*[waterxmlfile])
    else:
        logger.info("Creating forcefield using XML-files: %s %s", xmlfile, waterxmlfile)
        forcefield = openmm_app.forcefield.ForceField(*[xmlfile, waterxmlfile])

    # WRITE PDB-file
    # Check if xmlfile contains bonded parameters
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

    # Load PDB-file and create Modeller object
    pdb = openmm_app.PDBFile(pdbfile)
    logger.info("Loading Modeller.")
    modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
    logger.info(f"Modeller topology has {modeller.topology.getNumResidues()} residues.")

    # Solvent+Ions
    logger.info("Adding solvent, watermodel: %s", watermodel)

    # NOTE: modeller.addsolvent will automatically add ions to neutralize any excess charge
    logger.info("Warning: Modeller will automatically neutralize system with ions if system is charged")
    if solvent_boxdims is not None:
        logger.info(f"Solvent boxdimension provided: {solvent_boxdims} Å")
        modeller.addSolvent(
            forcefield,
            boxSize=openmm.Vec3(solvent_boxdims[0], solvent_boxdims[1], solvent_boxdims[2]) * openmm_unit.angstrom,
        )

    # Write out solvated system coordinates
    logger.info("Creating PDB-file: system_aftersolvent.pdb")
    write_pdbfile_openMM(modeller.topology, modeller.positions, "system_aftersolvent.pdb")
    print_systemsize(modeller)

    # Create ASH fragment and write to disk
    newfragment = Fragment(pdbfile="system_aftersolvent.pdb")
    newfragment.write_xyzfile(xyzfilename="system_aftersolvent.xyz")
    logger.info("Creating XYZ-file: system_aftersolvent.xyz")
    logger.info("")
    logger.info("\nTo use this system setup to define a future OpenMMTheory object you can  do:\n")

    logger.info(
        f'omm = OpenMMTheory(xmlfiles=["{xmlfile}", "{waterxmlfile}"], pdbfile="system_aftersolvent.pdb", periodic=True, rigidwater=True)'
    )
    logger.info("")
    logger.info("")

    # Return forcefield object,  topology object and ASH fragment
    return forcefield, modeller.topology, newfragment


# Simple XML-writing function. Will only write nonbonded parameters
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


# TODO: Move elsewhere?


def read_NPT_statefile(npt_output):
    import csv
    from collections import defaultdict

    # Read in CSV file of last NPT simulation and store in lists
    columns = defaultdict(list)

    with open(npt_output) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                columns[k].append(v)
    # Extract step number, volume and density and cast as floats
    steps = np.array(columns['#"Step"'])
    volume = np.array(columns["Box Volume (nm^3)"]).astype(float)
    density = np.array(columns["Density (g/mL)"]).astype(float)

    resultdict = {"steps": steps, "volume": volume, "density": density}
    return resultdict


# Wrapper function for OpenMM_MDclass
def OpenMM_MD(
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    restartfile_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
    barostat=None,
    pressure=1,
    trajectory_file_option="DCD",
    trajfilename="trajectory",
    specialtraj_frequency=1000,
    specialatoms=None,
    energy_file_option=None,
    force_file_option=None,
    atomic_units_force_reporter=False,
    coupling_frequency=1,
    charge=None,
    mult=None,
    hydrogenmass=1.5,
    force_periodic=None,
    periodic_cell_dimensions=None,
    anderson_thermostat=False,
    platform="CPU",
    constraints=None,
    restraints=None,
    enforcePeriodicBox=True,
    special_wrapping=False,
    special_wrapping_updatepos=False,
    wrapping_atoms=None,
    dummyatomrestraint=False,
    center_on_atoms=None,
    solute_indices=None,
    datafilename=None,
    dummy_MM=False,
    plumed_object=None,
    add_centerforce=False,
    centerforce_atoms=None,
    centerforce_constant=1.0,
    centerforce_distance=10.0,
    centerforce_center=None,
    barostat_frequency=25,
    chkfile=None,
    statefile=None,
):
    logger.info(main_header("OpenMM MD wrapper function"))
    md = OpenMM_MDclass(
        fragment=fragment,
        theory=theory,
        charge=charge,
        mult=mult,
        timestep=timestep,
        traj_frequency=traj_frequency,
        restartfile_frequency=restartfile_frequency,
        temperature=temperature,
        integrator=integrator,
        barostat=barostat,
        pressure=pressure,
        trajectory_file_option=trajectory_file_option,
        specialtraj_frequency=specialtraj_frequency,
        specialatoms=specialatoms,
        energy_file_option=energy_file_option,
        force_file_option=force_file_option,
        atomic_units_force_reporter=atomic_units_force_reporter,
        constraints=constraints,
        restraints=restraints,
        force_periodic=force_periodic,
        periodic_cell_dimensions=periodic_cell_dimensions,
        coupling_frequency=coupling_frequency,
        anderson_thermostat=anderson_thermostat,
        platform=platform,
        enforcePeriodicBox=enforcePeriodicBox,
        special_wrapping=special_wrapping,
        special_wrapping_updatepos=special_wrapping_updatepos,
        wrapping_atoms=wrapping_atoms,
        dummyatomrestraint=dummyatomrestraint,
        center_on_atoms=center_on_atoms,
        solute_indices=solute_indices,
        datafilename=datafilename,
        dummy_MM=dummy_MM,
        hydrogenmass=hydrogenmass,
        plumed_object=plumed_object,
        add_centerforce=add_centerforce,
        trajfilename=trajfilename,
        centerforce_atoms=centerforce_atoms,
        centerforce_constant=centerforce_constant,
        centerforce_distance=centerforce_distance,
        centerforce_center=centerforce_center,
        barostat_frequency=barostat_frequency,
        chkfile=chkfile,
        statefile=statefile,
    )
    if simulation_steps is not None:
        md.run(simulation_steps=simulation_steps)
    elif simulation_time is not None:
        md.run(simulation_time=simulation_time)
    else:
        raise InputError("Either simulation_steps or simulation_time need to be defined (not both).")

    # Now calling finalize_simulation: writing final files etc.
    md.finalize_simulation()

    # TODO: Return an ASH Results object here?
    return


class OpenMM_MDclass:
    def __init__(
        self,
        fragment=None,
        theory=None,
        charge=None,
        mult=None,
        timestep=0.001,
        traj_frequency=1000,
        restartfile_frequency=1000,
        temperature=300,
        integrator="LangevinMiddleIntegrator",
        barostat=None,
        pressure=1,
        trajectory_file_option="DCD",
        trajfilename="trajectory",
        specialtraj_frequency=1000,
        specialatoms=None,
        energy_file_option=None,
        force_file_option=None,
        atomic_units_force_reporter=False,
        coupling_frequency=1,
        platform="CPU",
        anderson_thermostat=False,
        hydrogenmass=1.5,
        constraints=None,
        restraints=None,
        force_periodic=False,
        periodic_cell_dimensions=None,
        enforcePeriodicBox=True,
        special_wrapping=False,
        special_wrapping_updatepos=False,
        wrapping_atoms=None,
        dummyatomrestraint=False,
        center_on_atoms=None,
        solute_indices=None,
        datafilename=None,
        dummy_MM=False,
        plumed_object=None,
        add_centerforce=False,
        centerforce_atoms=None,
        centerforce_constant=1.0,
        centerforce_distance=10.0,
        centerforce_center=None,
        barostat_frequency=25,
        chkfile=None,
        statefile=None,
    ):
        module_init_time = time.time()

        logger.info(main_header("OpenMM Molecular Dynamics Initialization"))

        if fragment is None:
            raise InputError("No fragment object. Exiting.")
        else:
            self.fragment = fragment

        # Check charge/mult
        self.charge, self.mult = check_charge_mult(
            charge, mult, theory.theorytype, fragment, "OpenMM_MD", theory=theory
        )

        # External QM option off by default
        self.externalqm = False

        # Trajectory filename. Used for trajs in DCD, PDB etc. format, also single PDB snapshots
        self.trajfilename = trajfilename

        # Specialatoms and specialtraj_frequency for special printing
        self.specialatoms = specialatoms
        self.specialtraj_frequency = specialtraj_frequency

        # Delete previous special and wrapping trajectory file
        if os.path.exists("wrapped_special_traj.xyz"):
            os.remove("wrapped_special_traj.xyz")
        if os.path.exists("OpenMMMD_traj_wrapped.xyz"):
            os.remove("OpenMMMD_traj_wrapped.xyz")

        # Distinguish between OpenMM theory QM/MM theory or QM theory
        self.dummy_MM = dummy_MM

        # Printlevel

        # Determine centroid of original fragment coordinates
        self.centroid_system = get_centroid(fragment.coords)

        # Theory_runtype
        self.theory_runtype = None

        self.openmmobject = None
        self.QM_MM_object = None
        logger.info("Analyzing theory input to OpenMM_MDclass")
        if isinstance(theory, OpenMMTheory):
            logger.info("This is an OpenMMTheory object")
            self.openmmobject = theory
            self.QM_MM_object = None
            if self.dummy_MM is True:
                self.theory_runtype = "dummy_MM"
            else:
                self.theory_runtype = "MM"
        # Case: QM/MM theory with OpenMM mm_theory
        elif isinstance(theory, openmmqmmm.QMMMTheory):
            logger.info("This is an QMMMTheory object")
            self.QM_MM_object = theory
            self.openmmobject = theory.mm_theory
            self.theory_runtype = "QMMM"

            # Making sure QM/MM object will exit before calculating MM part
            self.QM_MM_object.exit_after_customexternalforce_update = True
            logger.info("Turning on externalforce option.")
            self.QM_MM_object.openmm_externalforce = True
            # NOTE: Now creating externalforceobject as part of this MD object instead (previously QM/MM object)
            self.openmm_externalforceobject = self.openmmobject.add_custom_external_force()
            # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
            # (it would calculate OpenMM energy twice) so turning on in case forgotten
        # Case: OpenMM with external QM
        else:
            # NOTE: Recognize QM theories here ??
            logger.info("Unrecognized theory.")
            logger.info("Will assume to be QM theory and will continue")
            logger.info("QM-program forces will be added as a custom external force to OpenMM")
            self.externalqm = True
            logger.info("Now creating OpenMMTheory object")
            logger.info("OpenMM platform: %s", platform)
            # Creating dummy OpenMMTheory (basic topology, particle masses, no forces except CMMRemoval)
            self.openmmobject = OpenMMTheory(
                fragment=fragment,
                dummysystem=True,
                platform=platform,
                hydrogenmass=hydrogenmass,
                constraints=constraints,
                periodic=force_periodic,
                periodic_cell_dimensions=periodic_cell_dimensions,
            )  # NOTE: might add more options here
            logger.info("Creating new OpenMM custom external force for external QM theory.")
            self.openmm_externalforceobject = self.openmmobject.add_custom_external_force()
            self.QM_MM_object = None
            self.qmtheory = theory
            self.theory_runtype = "QM"

        # Basic restraints (bond,angle,torsion)
        if restraints is not None:
            logger.info("Restraints defined. Will add to OpenMMTheory object")
            logger.info("All restraints: %s", restraints)
            for restraint in restraints:
                logger.info("Restraint: %s", restraint)
                if len(restraint) == 4:
                    logger.info("Bond restraint assumed")
                    logger.info(
                        f"Atoms: {restraint[0]} {restraint[1]} Value: {restraint[2]} Force-constant: {restraint[3]} kcal/mol/Angstrom^2"
                    )
                    self.openmmobject.add_custom_bond_force(restraint[0], restraint[1], restraint[2], restraint[3])
                elif len(restraint) == 5:
                    logger.info("Angle restraint assumed")
                    logger.info(
                        f"Atoms: {restraint[0]} {restraint[1]} {restraint[2]} Value: {restraint[3]} Force-constant: {restraint[4]} kcal/mol/radian^2"
                    )
                    self.openmmobject.add_custom_angle_force(
                        restraint[0], restraint[1], restraint[2], restraint[3], restraint[4]
                    )
                elif len(restraint) == 6:
                    logger.info("Torsion restraint assumed")
                    logger.info(
                        f"Atoms: {restraint[0]} {restraint[1]} {restraint[2]} {restraint[3]} Value: {restraint[4]} Force-constant: {restraint[5]} kcal/mol/radian^2"
                    )
                    self.openmmobject.add_custom_torsion_force(
                        restraint[0], restraint[1], restraint[2], restraint[3], restraint[4], restraint[5]
                    )

        # RESTART options
        self.chkfile = chkfile
        self.statefile = statefile

        # Assigning some basic variables
        self.temperature = temperature
        self.pressure = pressure
        self.integrator = integrator
        self.coupling_frequency = coupling_frequency
        self.timestep = timestep
        self.traj_frequency = int(traj_frequency)
        self.restartfile_frequency = restartfile_frequency
        self.plumed_object = plumed_object
        self.barostat_frequency = barostat_frequency
        self.trajectory_file_option = trajectory_file_option
        self.force_file_option = force_file_option  # Gradients/forces as a file
        self.energy_file_option = energy_file_option  # Energies as a file
        self.atomic_units_force_reporter = atomic_units_force_reporter  # Forces in atomic units
        self.user_cvforce1 = None  # Initializing possibility of user CV object
        self.user_biasvar1 = None  # Initializing possibility of user biasvariable
        self.user_cvforce2 = None  # Initializing possibility of user CV object
        self.user_biasvar2 = None  # Initializing possibility of user biasvariable
        # PERIODIC or not
        if self.openmmobject.periodic is True:
            # Generally we want True except sometimes we do our own wrapping
            self.enforcePeriodicBox = enforcePeriodicBox
        else:
            logger.info("System is non-periodic. Setting enforcePeriodicBox to False")
            # Non-periodic. Setting enforcePeriodicBox to False (otherwise nonsense)
            self.enforcePeriodicBox = False

        # Optional wrapping_atoms (anchoratoms)
        self.special_wrapping = special_wrapping
        self.special_wrapping_updatepos = (
            special_wrapping_updatepos  # Testing: update positions in simulation object after wrapping
        )
        self.wrapping_atoms = wrapping_atoms

        logger.info(small_header("MD system parameters"))
        logger.info(f"Temperature: {self.temperature} K")
        logger.info("OpenMM autoconstraints: %s", self.openmmobject.autoconstraints)
        logger.info("OpenMM hydrogenmass: %s", self.openmmobject.hydrogenmass)
        logger.info("OpenMM rigidwater constraints: %s", self.openmmobject.rigidwater)
        logger.info("User Constraints: %s", self.openmmobject.user_constraints)
        logger.info("User Restraints: %s", self.openmmobject.user_restraints)
        logger.info("Number of atoms: %s", self.fragment.numatoms)
        logger.info("Number of frozen atoms: %s", len(self.openmmobject.user_frozen_atoms))
        if len(self.openmmobject.user_frozen_atoms) < 50:
            logger.info("Frozen atoms %s", self.openmmobject.user_frozen_atoms)
        logger.info("Integrator: %s", self.integrator)
        logger.info(f"Timestep: {self.timestep} ps")
        logger.info("Anderon Thermostat: %s", anderson_thermostat)
        logger.info(f"coupling_frequency: {self.coupling_frequency} ps^-1 (for Nose-Hoover and Langevin integrators)")
        logger.info("Barostat: %s", barostat)

        logger.info("")
        logger.info("Will write trajectory in format: %s", self.trajectory_file_option)
        logger.info("Trajectory write frequency: %s", self.traj_frequency)
        logger.info("enforcePeriodicBox: %s", self.enforcePeriodicBox)
        logger.info("special_wrapping: %s", self.special_wrapping)
        logger.info("special_wrapping_updatepos: %s", special_wrapping_updatepos)
        logger.info("wrapping_atoms: %s", self.wrapping_atoms)
        logger.info("")

        if self.openmmobject.autoconstraints is None:
            logger.info("""
                WARNING: Autoconstraints have not been set in OpenMMTheory object definition. This means that by
                         default no bonds are constrained in the MD simulation. This usually requires a small
                         timestep: 0.5 fs or so.
                         autoconstraints='HBonds' is recommended for 2 fs timesteps with LangevinIntegrator and 4fs with LangevinMiddleIntegrator).
                         autoconstraints='AllBonds' or autoconstraints='HAngles' allows even larger timesteps to be used.
                         See : https://github.com/openmm/openmm/pull/2754 and https://github.com/openmm/openmm/issues/2520
                         for recommended simulation settings in OpenMM.
                         """)
            logger.info("Will continue...")
        if (self.openmmobject.rigidwater is True and len(self.openmmobject.user_frozen_atoms) != 0) or (
            self.openmmobject.autoconstraints is not None and len(self.openmmobject.user_frozen_atoms) != 0
        ):
            logger.info(
                "WARNING: Frozen_atoms options selected but there are general constraints defined in "
                "the OpenMM object (either rigidwater=True or autoconstraints is not None)"
                "\nOpenMM will crash if constraints and frozen atoms involve the same atoms"
            )
        logger.info("")

        logger.info("Defining atom positions from fragment")
        # Note: using self.positions as we may add dummy atoms (e.g. dummyatomrestraint below)
        self.positions = self.fragment.coords

        # Dummy-atom restraint to deal with NPT simulations that contain constraints/restraints/frozen_atoms
        self.dummyatomrestraint = dummyatomrestraint
        if self.dummyatomrestraint is True:
            if solute_indices is None:
                raise InputError("Dummyatomrestraint requires solute_indices to be set")
            logger.warning(
                "Warning: Using dummyatomrestraints. This means that we will add a dummy atom to topology and OpenMM coordinates"
            )
            logger.info("We do not add the dummy atom to ASH-fragment")
            logger.info(
                "Affects visualization of trajectory (make sure to use PDB-file that contains the dummy-atom, printed in the end)"
            )
            # Should be centroid of solute or something rather
            solute_coords = np.take(self.fragment.coords, solute_indices, axis=0)
            dummypos = get_centroid(solute_coords)
            logger.info("Dummy atom will be added to position: %s", dummypos)
            # Adding dummy-atom coordinates to self.positions
            self.positions = np.append(self.positions, [dummypos], axis=0)
            logger.info("len self.pos %s", len(self.positions))
            logger.info("len self.fragment.coords %s", len(self.fragment.coords))

            # Restraining solute atoms to dummy-atom
            self.openmmobject.add_dummy_atom_to_restrain_solute(atomindices=solute_indices)

        # TRANSLATE solute: #https://github.com/openmm/openmm/issues/1854
        # Translate solute to geometric center on origin
        if center_on_atoms is not None:
            solute_coords = np.take(self.fragment.coords, solute_indices, axis=0)
            changed_origin_coords = change_origin_to_centroid(self.fragment.coords, subsetcoords=solute_coords)
            logger.info("changed_origin_coords %s", changed_origin_coords)

        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        # Set up system with chosen barostat, thermostat, integrator
        if barostat is not None:
            logger.info("Checking for barostat")
            if "MonteCarloBarostat" not in forceclassnames:
                montecarlobarostat = openmm.MonteCarloBarostat(
                    self.pressure * openmm.unit.bar, self.temperature * openmm.unit.kelvin
                )
                # Setting barostat frequency to chosen value or default (25)
                montecarlobarostat.setFrequency(self.barostat_frequency)
                self.openmmobject.system.addForce(montecarlobarostat)
                logger.info("Barostat added")
            else:
                logger.info("Barostat already present. Skipping.")

            self.integrator = "LangevinMiddleIntegrator"
            logger.info("Barostat requires using integrator: %s", self.integrator)
            self.openmmobject.set_simulation_parameters(
                timestep=self.timestep,
                temperature=self.temperature,
                integrator=self.integrator,
                coupling_frequency=self.coupling_frequency,
            )
        elif anderson_thermostat is True:
            logger.info("Anderson thermostat is on.")
            if "AndersenThermostat" not in forceclassnames:
                self.openmmobject.system.addForce(
                    openmm.AndersenThermostat(self.temperature * openmm.unit.kelvin, 1 / openmm.unit.picosecond)
                )
            self.integrator = "VerletIntegrator"
            logger.info("Now using integrator: %s", integrator)
            self.openmmobject.set_simulation_parameters(
                timestep=self.timestep,
                temperature=self.temperature,
                integrator=self.integrator,
                coupling_frequency=self.coupling_frequency,
            )
        else:
            # Deleting barostat and Andersen thermostat if present from previous sims
            for i, forcename in enumerate(forceclassnames):
                if forcename == "MonteCarloBarostat" or forcename == "AndersenThermostat":
                    logger.info("Removing old force: %s", forcename)
                    self.openmmobject.system.removeForce(i)

            # Regular thermostat or integrator without barostat
            # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
            # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
            self.openmmobject.set_simulation_parameters(
                timestep=self.timestep,
                temperature=self.temperature,
                integrator=self.integrator,
                coupling_frequency=self.coupling_frequency,
            )

        if barostat is not None:
            self.volume = self.density = True
        else:
            self.volume = self.density = False

        # If statedatareporter filename set:
        self.datafilename = datafilename
        if self.datafilename is not None:
            # Remove old file
            # Added because of problems (19 May 2023 by CVS) in read NPT data file (OpenMM box equilibration) as header is printed each time
            # Now removing file before starting. Possibly better to put this elsewhere as we may sometimes
            # want to keep running simulation while appending to datafile
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.datafilename)

            # Now doing open file object in append mode instead of just filename.
            # Just filename does not play nice when running simulation step by step
            # Future OpenMM update may do this automatically?
            self.dataoutputoption = open(self.datafilename, "a")  # noqa: SIM115 - handed to OpenMM reporter
            logger.info("Will write data to file: %s", self.datafilename)
        # otherwise stdout:
        else:
            self.dataoutputoption = stdout

        # NOTE: Better to use OpenMM-plumed interface instead??
        if plumed_object is not None:
            logger.info("Plumed active")
            # Create new OpenMM custom external force
            logger.info("Creating new OpenMM custom external force for Plumed.")
            self.plumedcustomforce = self.openmmobject.add_custom_external_force()

        # QM/MM MD
        # if self.QM_MM_object is not None:
        #    #True sometimes means we end up with solute in corner of box (wrong for nonPBC QM code)
        #
        #    #Making sure QM/MM object will exit before calculating MM part
        #
        #    # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
        #    # (it would calculate OpenMM energy twice) so turning on in case forgotten
        #    if self.QM_MM_object.openmm_externalforce is False:
        #        #NOTE: Now creating externalforceobject as part of this MD object instead (previously QM/MM object)
        # CENTER COORDINATES HERE on SOLUTE HERE ??
        # NOTE: Deprecated most likely
        # if centercoordinates is True:
        #    # Solute atoms assumed to be QM-region

        # Adding (flat-bottom) center force acting on solute
        if add_centerforce is True:
            logger.info("Centerforce option active")
            if centerforce_atoms is None:
                logger.info("centerforce_atoms unset. Trying to use QM atoms: %s", self.QM_MM_object.qmatoms)
                centerforce_atoms = self.QM_MM_object.qmatoms
            if centerforce_center is None:
                logger.info("No center coordinates set. Using geometric center of whole fragment.")
                # Get geometric center of system (Angstrom)
                centerforce_center = self.fragment.get_coordinate_center()
                logger.info("centerforce_center: %s", centerforce_center)
            # Alternative (PBC wrapping issues, however)
            # self.openmmobject.add_flatbottom_centerforce(molA_indices=centerforce_atoms, molB_indices=rest_system,
            #                                              forceconstant=centerforce_constant, distance=centerforce_distance)
            self.openmmobject.add_centerforce(
                center_coords=centerforce_center,
                atomindices=centerforce_atoms,
                forceconstant=centerforce_constant,
                distance=centerforce_distance,
            )

        # After adding possible QM/MM force, possible Plumed force, possible center force
        # Let's list all OpenMM object system forces for sanity
        logger.info("enforcePeriodicBox: %s", self.enforcePeriodicBox)
        logger.info("OpenMM Forces defined: %s", self.openmmobject.system.getForces())

        log_time_since(module_init_time, "OpenMM_MD setup")

    # Set sim reporters. Needs to be done after simulation is created and not modified anymore
    def set_sim_reporters(self, simulation, restart=False):

        # CheckpointReporter
        logger.info("Creating CheckpointReporter that will write a restartable checkpointfile every X steps")
        checkpointfilename = "OpenMM_MD.chk"
        simulation.reporters.append(openmm.app.CheckpointReporter(checkpointfilename, self.traj_frequency * 1))
        # StateDataReporter
        logger.info("Creating StateDataReporter that will write to stdout")
        statedatareporter_stdout = openmm.app.StateDataReporter(
            stdout,
            self.traj_frequency,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            volume=self.volume,
            density=self.density,
            temperature=True,
            separator=",",
        )
        simulation.reporters.append(statedatareporter_stdout)
        # Another reporter for writing to file
        if self.dataoutputoption != stdout:
            logger.info("Creating StateDataReporter that will write to file: %s", self.datafilename)
            logger.info("restart: %s", restart)
            statedatareporter_file = openmm.app.StateDataReporter(
                self.dataoutputoption,
                self.traj_frequency,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                volume=self.volume,
                density=self.density,
                temperature=True,
                separator=",",
                append=restart,
            )
            simulation.reporters.append(statedatareporter_file)
            self.dataoutputoption = open(self.datafilename, "a")  # noqa: SIM115 - handed to OpenMM reporter

        # TODO: See if this can be made to work for simulations with step-by-step
        if self.trajectory_file_option == "PDB":
            simulation.reporters.append(
                openmm.app.PDBReporter(
                    self.trajfilename + ".pdb", self.traj_frequency, enforcePeriodicBox=self.enforcePeriodicBox
                )
            )
        elif self.trajectory_file_option == "DCD":
            # Note: using append keyword here if restarting
            # Check first if file exists for restart (OpenMM errors otherwise)
            if restart is True and os.path.isfile(f"{self.trajfilename}.dcd") is False:
                logger.info("Warning: restart option was active but trajectory file not existing. Will create new file")
                restart = False

            simulation.reporters.append(
                openmm.app.DCDReporter(
                    self.trajfilename + ".dcd",
                    self.traj_frequency,
                    append=restart,
                    enforcePeriodicBox=self.enforcePeriodicBox,
                )
            )
            logger.info("DCDReporter added")
        elif self.trajectory_file_option == "NetCDFReporter":
            logger.info("NetCDFReporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = MDtraj_import()
            simulation.reporters.append(mdtraj.reporters.NetCDFReporter(self.trajfilename + ".nc", self.traj_frequency))
        elif self.trajectory_file_option == "HDF5Reporter":
            logger.info("HDF5Reporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = MDtraj_import()
            simulation.reporters.append(
                mdtraj.reporters.HDF5Reporter(
                    self.trajfilename + ".lh5", self.traj_frequency, enforcePeriodicBox=self.enforcePeriodicBox
                )
            )
        elif self.trajectory_file_option == "XYZ":
            logger.info("XYZ trajectory format selected (not available for classical MD). Warning: not very fast")
            logger.info("Deleting possible old trajectory-file (OpenMMMD_traj.xyz)")
            with contextlib.suppress(OSError):
                os.remove("OpenMMMD_traj.xyz")
            # Done manually by write_xyzfile

        if self.force_file_option is not None:
            logger.info("ForceReporter traj format selected.")
            simulation.reporters.append(
                ForceReporter(
                    self.trajfilename + "_force.txt", self.traj_frequency, atomic_units=self.atomic_units_force_reporter
                )
            )
        if self.energy_file_option is not None:
            logger.info("Energyfile  selected.")
            with contextlib.suppress(OSError):
                os.remove(self.energy_file_option)
        logger.info("simulation.reporters: %s", simulation.reporters)

    # For OpenMM native MTD
    def mtd_step(self, step, meta_object, metadyn_settings):
        checkpoint = time.time()
        cv1scaling = 1
        cv2scaling = 1
        meta_object.step(self.simulation, 1)
        log_time_since(checkpoint, "mtd sim step")
        checkpoint = time.time()

        # getCollectiveVariables
        if step % metadyn_settings["saveFrequency"] * metadyn_settings["frequency"] == 0:
            logger.info("MTD: Writing current collective variables to disk")
            current_cv = meta_object.getCollectiveVariables(self.simulation)
            if (
                metadyn_settings["CV1_type"] == "distance"
                or metadyn_settings["CV1_type"] == "bond"
                or metadyn_settings["CV1_type"] == "rmsd"
            ):
                cv1scaling = 10
            elif (
                metadyn_settings["CV1_type"] == "dihedral"
                or metadyn_settings["CV1_type"] == "torsion"
                or metadyn_settings["CV1_type"] == "angle"
            ):
                cv1scaling = 180 / np.pi
            if (
                metadyn_settings["CV2_type"] == "distance"
                or metadyn_settings["CV2_type"] == "bond"
                or metadyn_settings["CV2_type"] == "rmsd"
            ):
                cv2scaling = 10
            elif (
                metadyn_settings["CV2_type"] == "dihedral"
                or metadyn_settings["CV2_type"] == "torsion"
                or metadyn_settings["CV2_type"] == "angle"
            ):
                cv2scaling = 180 / np.pi
            currtime = step * self.timestep  # Time in ps
            with open("colvar", "a") as f:
                if metadyn_settings["numCVs"] == 2:
                    f.write(f"{currtime} {current_cv[0] * cv1scaling} {current_cv[1] * cv2scaling}\n")
                elif metadyn_settings["numCVs"] == 1:
                    f.write(f"{currtime} {current_cv[0] * cv1scaling}\n")
        log_time_since(checkpoint, "mtd colvar-flush")
        checkpoint = time.time()
        return

    def write_state_and_chk_files(self, step):
        # Saving state and chkfile to disk
        logger.info(
            f"Step {step}. Saving a statefile and checkpointfile : OpenMM_MD_state.xml and OpenMM_MD_checkpoint.chk"
        )
        logger.info(
            "Can be used to restart a simulation (statefile and chkfile keywords) using the same coordinates and velocities."
        )
        self.simulation.saveState("OpenMM_MD_state.xml")
        self.simulation.saveCheckpoint("OpenMM_MD_checkpoint.chk")

    # Simulation loop.
    # NOTE: process_id passed by Simple_parallel function when doing multiprocessing, e.g. Plumed multiwalker metadynamics
    def run(
        self,
        simulation_steps=None,
        simulation_time=None,
        metadynamics=False,
        metadyn_settings=None,
        plumedinput=None,
        process_id=None,
        workerdir=None,
        restraints=None,
        restart=False,
        chkfile=None,
        statefile=None,
    ):
        module_init_time = time.time()
        logger.info(main_header("OpenMM Molecular Dynamics Run"))

        if simulation_steps is None and simulation_time is None:
            raise InputError("Either simulation_steps or simulation_time needs to be set.")
        if simulation_time is not None:
            simulation_steps = int(simulation_time / self.timestep)
        if simulation_steps is not None:
            simulation_time = simulation_steps * self.timestep

        # Checking whether chkfile has been provided to run method or init
        if chkfile is None and self.chkfile is not None:
            logger.info("chkfile provided to init. Will use this for restart.")
            chkfile = self.chkfile
        if statefile is None and self.statefile is not None:
            logger.info("statefile provided to init. Will use this for restart.")
            statefile = self.statefile

        ##################################
        # CREATE SIMULATION OBJECT
        ##################################

        # Parallelization handling
        if process_id is None:
            process_id = 0
        if workerdir is not None:
            logger.info(f"Workerdir: {workerdir} provided. Entering dir")
            os.chdir(workerdir)

        # If using Plumed then now we add Plumed-force to system from plumedinput string
        if plumedinput is not None:
            import openmmplumed

            logger.info("Plumed active. Adding Plumedforce to system")
            if process_id is not None:
                logger.info(f"process_id ({process_id}) passed to md.run. Assuming multiwalker Plumed MD run")
                logger.info("plumedinput: %s", plumedinput)
                plumedinput = plumedinput.replace("WALKERID", str(process_id))
                logger.info("plumedinput: %s", plumedinput)
                writestringtofile(plumedinput, "plumedinput.in")
            self.openmmobject.system.addForce(openmmplumed.PlumedForce(plumedinput))

        # Case native OpenMM metadynamcis
        if metadynamics is True:
            biasdir = metadyn_settings["biasdir"]
            with contextlib.suppress(OSError):
                os.remove("colvar")
            # Reference positions for RMSD. Currently limited to starting position
            if metadyn_settings["CV1_type"] == "rmsd" or metadyn_settings["CV2_type"] == "rmsd":
                if metadyn_settings["reference_xyzfile"] is None:
                    logger.info("No reference_xyzfile was provided for RMSD-CV. Using input coordinates as reference")
                    coords_nm = self.fragment.coords * 0.1  # converting from Angstrom to nm
                    reference_pos = [
                        openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in range(len(coords_nm))
                    ] * openmm.unit.nanometer
                else:
                    logger.info("A reference_xyzfile was provided for RMSD-CV. Using")
                    logger.info("Reading XYZ-file: %s", metadyn_settings["reference_xyzfile"])
                    ref_frag = Fragment(xyzfile=metadyn_settings["reference_xyzfile"])
                    coords_nm = ref_frag.coords * 0.1  # converting from Angstrom to nm
                    reference_pos = [
                        openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in range(len(coords_nm))
                    ] * openmm.unit.nanometer
            else:
                reference_pos = None
            # Creating meta_object from settings provided
            if metadyn_settings["numCVs"] == 2:
                # Creating CV biasvariables and forces
                CV1_bias, cvforce_1 = create_CV_bias(
                    metadyn_settings["CV1_type"],
                    metadyn_settings["CV1_atoms"],
                    metadyn_settings["CV1_biaswidth"],
                    CV_range=metadyn_settings["CV1_range"],
                    reference_pos=reference_pos,
                    reference_particles=metadyn_settings["CV1_atoms"],
                    user_cvforce=self.user_cvforce1,
                    user_biasvar=self.user_biasvar1,
                    CV_parameters=metadyn_settings["CV1_parameters"],
                )
                CV2_bias, cvforce_2 = create_CV_bias(
                    metadyn_settings["CV2_type"],
                    metadyn_settings["CV2_atoms"],
                    metadyn_settings["CV2_biaswidth"],
                    CV_range=metadyn_settings["CV2_range"],
                    reference_pos=reference_pos,
                    reference_particles=metadyn_settings["CV2_atoms"],
                    user_cvforce=self.user_cvforce2,
                    user_biasvar=self.user_biasvar2,
                    CV_parameters=metadyn_settings["CV2_parameters"],
                )

                # Gridwidth and min/max values now set. Adding to dict
                metadyn_settings["CV1_gridwidth"] = CV1_bias.gridWidth
                metadyn_settings["CV2_gridwidth"] = CV2_bias.gridWidth
                metadyn_settings["CV1_minvalue"] = CV1_bias.minValue
                metadyn_settings["CV1_maxvalue"] = CV1_bias.maxValue
                metadyn_settings["CV2_minvalue"] = CV2_bias.minValue
                metadyn_settings["CV2_maxvalue"] = CV2_bias.maxValue
                ##Possible flatbottom or other restraint accompanying CV
                if metadyn_settings["flatbottom_restraint_CV1"] is not None:
                    logger.info("Adding flatbottom restraint for CV1")
                    self.openmmobject.add_CV_restraint(
                        cvforce_1, metadyn_settings["flatbottom_restraint_CV1"], metadyn_settings["CV2_type"]
                    )
                if metadyn_settings["flatbottom_restraint_CV2"] is not None:
                    logger.info("Adding flatbottom restraint for CV2")
                    self.openmmobject.add_CV_restraint(
                        cvforce_2, metadyn_settings["flatbottom_restraint_CV2"], metadyn_settings["CV2_type"]
                    )

                meta_object = openmm.app.Metadynamics(
                    self.openmmobject.system,
                    [CV1_bias, CV2_bias],
                    metadyn_settings["temperature"],
                    metadyn_settings["biasfactor"],
                    metadyn_settings["height"],
                    metadyn_settings["frequency"],
                    saveFrequency=metadyn_settings["saveFrequency"],
                    biasDir=metadyn_settings["biasdir"],
                )
            elif metadyn_settings["numCVs"] == 1:
                # Creating CV biasvariable and force
                CV1_bias, cvforce_1 = create_CV_bias(
                    metadyn_settings["CV1_type"],
                    metadyn_settings["CV1_atoms"],
                    metadyn_settings["CV1_biaswidth"],
                    CV_range=metadyn_settings["CV1_range"],
                    reference_pos=reference_pos,
                    reference_particles=metadyn_settings["CV1_atoms"],
                    user_cvforce=self.user_cvforce1,
                    user_biasvar=self.user_biasvar1,
                    CV_parameters=metadyn_settings["CV1_parameters"],
                )
                # Gridwidth and min/max values now set. Adding to dict
                metadyn_settings["CV1_gridwidth"] = CV1_bias.gridWidth
                metadyn_settings["CV1_minvalue"] = CV1_bias.minValue
                metadyn_settings["CV1_maxvalue"] = CV1_bias.maxValue
                metadyn_settings["CV2_gridwidth"] = None
                ##Possible flatbottom or other restraint accompanying CV
                if metadyn_settings["flatbottom_restraint_CV1"] is not None:
                    logger.info("Adding flatbottom restraint for CV1")
                    self.openmmobject.add_CV_restraint(
                        cvforce_1, metadyn_settings["flatbottom_restraint_CV1"], metadyn_settings["CV1_type"]
                    )

                meta_object = openmm.app.Metadynamics(
                    self.openmmobject.system,
                    [CV1_bias],
                    metadyn_settings["temperature"],
                    metadyn_settings["biasfactor"],
                    metadyn_settings["height"],
                    metadyn_settings["frequency"],
                    saveFrequency=metadyn_settings["saveFrequency"],
                    biasDir=metadyn_settings["biasdir"],
                )

            # Writing metadyn_settings dict to disk
            import json

            with open(f"{biasdir}/ASH_MTD_parameters.txt", "w") as mtdfh:
                json.dump(metadyn_settings, mtdfh)

        # Possible restraints added
        if restraints is not None:
            logger.info("Adding restraints")
            self.openmmobject.add_bondrestraints(restraints=restraints)

        # Creating simulation object and
        if chkfile is not None:
            self.simulation = self.openmmobject.create_simulation()
            logger.info("Checkpoint file provided. Restarting simulation using position and velocity data in file")
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities before: %s", state.getVelocities(asNumpy=True))
            self.simulation.loadCheckpoint(chkfile)
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities after loading checkpoint file: %s", state.getVelocities(asNumpy=True))
        elif statefile is not None:
            self.simulation = self.openmmobject.create_simulation()
            logger.info("State file provided. Restarting simulation using position and velocity data in file")
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities before: %s", state.getVelocities(asNumpy=True))
            self.simulation.loadState(statefile)
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities after loading statefile: %s", state.getVelocities(asNumpy=True))
        elif restart is True:
            logger.info("Restart true. Reusing already-defined simulation object")
        else:
            logger.info("Restart false and no chkfile/statefile set. This is a new simulation")
            self.simulation = self.openmmobject.create_simulation()
            logger.info("Simulation created.")
        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        ##################################
        # PRINT BASICS
        ##################################
        logger.info(small_header("MD run parameters"))
        logger.info(f"Simulation time: {simulation_time} ps")
        logger.info(f"Simulation steps: {simulation_steps}")
        logger.info(f"Timestep: {self.timestep} ps")
        logger.info(f"Set temperature: {self.temperature} K")
        logger.info("OpenMM integrator: %s", self.openmmobject.integrator_name)
        logger.info("")
        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        logger.info("OpenMM System forces present before run: %s", forceclassnames)

        # Printing PBCs
        if self.openmmobject.periodic is True:
            logger.info("Checking Initial PBC vectors.")
            self.state = self.simulation.context.getState()
            a, b, c = self.state.getPeriodicBoxVectors()
            logger.info("A:  %s", a)
            logger.info("B:  %s", b)
            logger.info("C:  %s", c)
            boxlength = a[0].value_in_unit(openmm.unit.angstrom)  # Box length in Angstrom
            logger.info(f"Boxlength: {boxlength} Angstrom")
        else:
            logger.info("System is not periodic")
        # Delete old traj
        ## Crashes when permissions not present or file is folder. Should never occur.
        #    pass

        # Make sure file associated with StateDataReporter is open
        if restart is True:
            logger.info("Restart true. Reusing simulation reporters")
            # if self.datafilename is not None:
            # Setting simulation reporters
            # Seems to be necessary to do this again after restart
            # restart option means that StateDatareport and DCDReporter will append to files
            self.set_sim_reporters(self.simulation, restart=True)
        elif statefile is not None:
            logger.info("statefile is used")
            # if self.datafilename is not None:
            # Setting simulation reporters
            # Seems to be necessary to do this again after restart
            # restart option means that StateDatareport and DCDReporter will append to files
            self.set_sim_reporters(self.simulation, restart=True)
        elif chkfile is not None:
            logger.info("chkfile is used")
            # if self.datafilename is not None:
            # Setting simulation reporters
            # Seems to be necessary to do this again after restart
            # restart option means that StateDatareport and DCDReporter will append to files
            self.set_sim_reporters(self.simulation, restart=True)
        else:
            logger.info("Restart false")
            if self.datafilename is not None:
                # RB addition: Delete file after each run
                logger.info("Deleting old datafile: %s", self.datafilename)
                with contextlib.suppress(OSError):
                    os.remove(self.datafilename)
                self.dataoutputoption = open(self.datafilename, "a")  # noqa: SIM115 - handed to OpenMM reporter
            # Setup data and simulation reporters for simulation object
            self.set_sim_reporters(self.simulation)

            # Setting coordinates of OpenMM object from current fragment.coords
            self.openmmobject.set_positions(self.positions, self.simulation)
        logger.info("")

        ###########################################
        # PBC and Wrapping
        ###########################################
        # Defining boxvectors in case we need
        if self.openmmobject.periodic is True:
            logger.info("Periodic Boundary Conditions used.")

            if self.enforcePeriodicBox is True:
                logger.info("EnforcePeriodic Box is True. Wrapping enforced by OpenMM.")
                logger.info(
                    "Warning: in case of problematic wrapping for e.g. QM/MM, try enabling special_wrapping=True"
                )
            # Wrapping handled by mdtraj
            if self.special_wrapping is True:
                logger.info("special_wrapping is True. Wrapping will be handled in each step by mdtraj library")
                logger.info("Importing mdtraj")
                try:
                    import mdtraj
                except ImportError:
                    raise MissingDependencyError(
                        "Error: mdtraj not found, needs to be installed (pip install mdtraj)"
                    ) from None
                # Defining boxvectors for wrapping
                boxvectors = self.simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
                # Convert topology from openmm format to mdtraj format
                mdtrajtopology = mdtraj.Topology.from_openmm(self.openmmobject.topology)
                # Choosing wrapping_atoms depending on theory-type
                if self.wrapping_atoms is None:
                    logger.info("No wrapping_atoms keyword has been set to center on.")
                    if self.theory_runtype == "QMMM":
                        logger.info("Theory-runtype is QMMM. Using QMatoms as wrapping_atoms")
                        wrapping_atoms = self.QM_MM_object.qmatoms
                    elif self.theory_runtype == "QM":
                        raise InputError("Theory_runtype is QM but no wrapping_atoms have been set.\nExiting")
                    elif self.theory_runtype == "dummy_MM":
                        raise InputError("Theory_runtype is dummy_MM but no wrapping_atoms have been set.\nExiting")
                    elif self.theory_runtype == "MM":
                        logger.info("Theory_runtype is MM. No achoratoms needed")
                        wrapping_atoms = None
                    logger.info("wrapping_atoms have been set to: %s", wrapping_atoms)
                else:
                    wrapping_atoms = self.wrapping_atoms
                    logger.info(f"Will use atoms {wrapping_atoms} for wrapping")

        ########################################
        # Writing intial frame to disk as PDB.
        ########################################
        pdb_filename = self.trajfilename + "_firstframe.pdb"
        logger.info("Writing intial frame to disk as PDB-file: %s", pdb_filename)
        blastate = self.simulation.context.getState(
            getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforcePeriodicBox
        )
        with open(pdb_filename, "w") as f:
            openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbfile.PDBFile.writeModel(
                self.openmmobject.topology, blastate.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
            openmm.app.pdbfile.PDBFile.writeFooter(self.openmmobject.topology, f)
        # PDBx/mmCIF
        pdbx_filename = self.trajfilename + "_firstframe.cif"
        logger.info("Writing intial frame to disk as PDBx/mmCIF-file: %s", pdbx_filename)
        with open(pdbx_filename, "w") as f:
            openmm.app.pdbxfile.PDBxFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbxfile.PDBxFile.writeModel(
                self.openmmobject.topology, blastate.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )

        ###############################################################################
        # MD LOOP for each Theory-Runtype: QMMM, QM, dummy_MM, MM
        ###############################################################################
        if self.theory_runtype == "QMMM":
            # if self.QM_MM_object is not None:
            logger.info("QM/MM MD run beginning")
            # CASE: QM/MM. Custom external force needs to have been created in OpenMMTheory (should be handled by init)

            # Get connectivity from OpenMM topology
            connectivity = []
            for resi in self.openmmobject.topology.residues():
                resatoms = [i.index for i in list(resi.atoms())]
                connectivity.append(resatoms)
            # Convert to dict
            create_conn_dict(connectivity)

            # MD LOOP
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                logger.debug("Step: %s", step)
                if step % self.traj_frequency == 0:
                    logger.info("Step: %s", step)

                # Get state of simulation. Gives access to coords, velocities, forces, energy etc.
                current_state = self.simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=self.enforcePeriodicBox, getEnergy=True
                )
                log_time_since(checkpoint, "get OpenMM state")
                checkpoint = time.time()
                # Get current coordinates from state to use for QM/MM step
                current_coords = np.array(current_state.getPositions(asNumpy=True)) * 10
                checkpoint = time.time()
                log_time_since(checkpoint, "get current_coords")

                # Periodic wrapping handling
                if self.openmmobject.periodic is True and self.special_wrapping is True:
                    logger.info("special_wrapping is True. Wrapping handled by mdtraj")
                    checkpoint = time.time()
                    # Wrapping
                    current_coords = diff_wrap_box_coords(
                        current_coords / 10.0, boxvectors, mdtrajtopology, wrapping_atoms
                    )
                    log_time_since(checkpoint, "wrapping via diff_wrap_box_coords")
                    checkpoint = time.time()
                    # Optional position update
                    if self.special_wrapping_updatepos is True:
                        logger.info("special_wrapping_update is True. Updating positions")
                        self.openmmobject.set_positions(current_coords, self.simulation)
                        log_time_since(checkpoint, "set positions update")
                        checkpoint = time.time()

                # Run QM/MM step to get full system QM+PC gradient.
                self.QM_MM_object.run(
                    current_coords=current_coords,
                    elems=self.fragment.elems,
                    Grad=True,
                    exit_after_customexternalforce_update=True,
                    charge=self.charge,
                    mult=self.mult,
                )
                log_time_since(checkpoint, "QM/MM run")
                checkpoint = time.time()

                if step % self.restartfile_frequency == 0:
                    # Writing state and chk files
                    self.write_state_and_chk_files(step)

                # Printing step-info or write-trajectory at regular intervals
                # NOTE: Manual per-step info is not possible here because the MM-energy has not been
                # calculated yet when using the customexternalforceupdate option
                if step % self.traj_frequency == 0:
                    logger.info("Writing wrapped coords to trajfile: OpenMMMD_traj_wrapped.xyz (for debugging)")
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj_wrapped", writemode="a")

                if self.specialatoms is not None and step % self.specialtraj_frequency == 0:
                    specialelems = [self.fragment.elems[i] for i in self.specialatoms]
                    special_coords = np.take(current_coords, self.specialatoms, axis=0)
                    logger.info("Writing wrapped coords to trajfile: only for special atoms")
                    write_xyzfile(specialelems, special_coords, "wrapped_special_traj", writemode="a")

                # Now need to update OpenMM external force with new QM-PC force
                # The QM_PC gradient (link-atom projected, from QM_MM object) is provided to OpenMM external force
                CheckpointTime = time.time()
                self.openmmobject.update_custom_external_force(
                    self.openmm_externalforceobject, self.QM_MM_object.QM_PC_gradient, self.simulation
                )
                log_time_since(CheckpointTime, "update custom external force")

                # NOTE: Think about energy correction (currently skipped above)
                # Now take OpenMM step (E+G + displacement etc.)
                checkpoint = time.time()

                # OpenMM metadynamics
                if metadynamics is True:
                    logger.info("Now calling OpenMM native metadynamics and taking 1 step")
                    self.mtd_step(step, meta_object, metadyn_settings)
                else:
                    self.simulation.step(1)
                    log_time_since(checkpoint, "openmmobject sim step")
                    checkpoint = time.time()
                    log_time_since(checkpoint_begin_step, "Total sim step")

        # External QM for OpenMMtheory
        # TODO: Think about possible wrapping
        elif self.theory_runtype == "QM":
            logger.info("External QM with OpenMM option")
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                logger.info("Step: %s", step)
                # Get state of simulation. Gives access to coords, velocities, forces, energy etc.
                current_state = self.simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=self.enforcePeriodicBox, getEnergy=True
                )
                log_time_since(checkpoint, "get OpenMM state")
                checkpoint = time.time()
                # Get current coordinates from state to use for QM/MM step
                current_coords = np.array(current_state.getPositions(asNumpy=True)) * 10
                log_time_since(checkpoint, "get current coords")
                checkpoint = time.time()

                # Periodic wrapping handling
                if self.openmmobject.periodic is True and self.special_wrapping is True:
                    logger.info("special_wrapping is True. Wrapping handled by mdtraj")
                    checkpoint = time.time()
                    # Wrapping
                    current_coords = diff_wrap_box_coords(
                        current_coords / 10.0, boxvectors, mdtrajtopology, wrapping_atoms
                    )
                    log_time_since(checkpoint, "wrapping via diff_wrap_box_coords")
                    checkpoint = time.time()
                    # Optional position update
                    if self.special_wrapping_updatepos is True:
                        logger.info("special_wrapping_update is True. Updating positions")
                        self.openmmobject.set_positions(current_coords, self.simulation)
                        log_time_since(checkpoint, "set positions update")
                        checkpoint = time.time()

                # Run QM step to get full system QM gradient.
                # Updates OpenMM object with QM forces
                energy, gradient = self.qmtheory.run(
                    current_coords=current_coords,
                    elems=self.fragment.elems,
                    Grad=True,
                    charge=self.charge,
                    mult=self.mult,
                )
                logger.info("Energy: %s", energy)
                log_time_since(checkpoint, "QM run")
                self.openmmobject.update_custom_external_force(
                    self.openmm_externalforceobject, gradient, self.simulation
                )

                # Calculate energy associated with external force so that we can subtract it later
                # TODO: take this and QM energy and add to print_current_step_info
                extforce_energy = 3 * np.mean(sum(gradient * current_coords * 1.88972612546))
                logger.info("extforce_energy: %s", extforce_energy)

                # Printing step-info or write-trajectory at regular intervals
                if step % self.traj_frequency == 0:
                    # Manual step info option
                    print_current_step_info(step, current_state, self.openmmobject, qm_energy=energy)

                    if self.energy_file_option is not None:
                        with open(self.energy_file_option, "a") as f:
                            f.write(f"{energy}\n")

                    # Manual trajectory option
                    if self.trajectory_file_option == "XYZ":
                        write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", writemode="a")

                if self.specialatoms is not None and step % self.specialtraj_frequency == 0:
                    specialelems = [self.fragment.elems[i] for i in self.specialatoms]
                    special_coords = np.take(current_coords, self.specialatoms, axis=0)
                    logger.info("Writing wrapped coords to trajfile: only for special atoms")
                    write_xyzfile(specialelems, special_coords, "wrapped_special_traj", writemode="a")

                if step % self.restartfile_frequency == 0:
                    # Writing state and chk files
                    self.write_state_and_chk_files(step)

                # OpenMM metadynamics
                if metadynamics is True:
                    logger.info("Now calling OpenMM native metadynamics and taking 1 step")
                    meta_object.step(self.simulation, 1)

                    # getCollectiveVariables
                    cv1scaling = 1
                    cv2scaling = 1
                    if step % metadyn_settings["saveFrequency"] * metadyn_settings["frequency"] == 0:
                        logger.info("MTD: Writing current collective variables to disk")
                        current_cv = meta_object.getCollectiveVariables(self.simulation)
                        if (
                            metadyn_settings["CV1_type"] == "distance"
                            or metadyn_settings["CV1_type"] == "bond"
                            or metadyn_settings["CV1_type"] == "rmsd"
                        ):
                            cv1scaling = 10
                        elif (
                            metadyn_settings["CV1_type"] == "dihedral"
                            or metadyn_settings["CV1_type"] == "torsion"
                            or metadyn_settings["CV1_type"] == "angle"
                        ):
                            cv1scaling = 180 / np.pi
                        if (
                            metadyn_settings["CV2_type"] == "distance"
                            or metadyn_settings["CV2_type"] == "bond"
                            or metadyn_settings["CV2_type"] == "rmsd"
                        ):
                            cv2scaling = 10
                        elif (
                            metadyn_settings["CV2_type"] == "dihedral"
                            or metadyn_settings["CV2_type"] == "torsion"
                            or metadyn_settings["CV2_type"] == "angle"
                        ):
                            cv2scaling = 180 / np.pi
                        currtime = step * self.timestep  # Time in ps
                        with open("colvar", "a") as f:
                            if metadyn_settings["numCVs"] == 2:
                                f.write(f"{currtime} {current_cv[0] * cv1scaling} {current_cv[1] * cv2scaling}\n")
                            elif metadyn_settings["numCVs"] == 1:
                                f.write(f"{currtime} {current_cv[0] * cv1scaling}\n")
                else:
                    self.simulation.step(1)
                log_time_since(checkpoint, "OpenMM sim step")
                log_time_since(checkpoint_begin_step, "Total sim step")

        elif self.theory_runtype == "MM":
            logger.info("External QM with OpenMM option")
            # OpenMM metadynamics
            if metadynamics is True:
                logger.info("Now calling OpenMM native metadynamics")
                meta_object.step(self.simulation, simulation_steps)
            else:
                logger.info("Regular classical OpenMM MD option chosen.")
                # Running all steps in one go
                self.simulation.step(simulation_steps)
        else:
            raise InputError(
                f"Error: Unrecognized Theory runtype ({self.theory_runtype}) for MD. This might mean that this ASH Theory object is not yet supported for running MD. Exiting."
            )

        logger.info(small_header("OpenMM MD simulation finished!"))
        log_time_since(module_init_time, "OpenMM_MD run")

        return

    def finalize_simulation(self):
        logger.info("Finalizing simulation data")

        #######################
        # CLOSING OPEN FILES
        #######################
        # Close Statadatareporter file if open
        if self.datafilename is not None:
            self.dataoutputoption.close()

        # Close Plumed also if active. Flushes HILLS/COLVAR etc.
        if self.plumed_object is not None:
            self.plumed_object.close()

        # GETTING positions, forces and energy of final frame
        self.state = self.simulation.context.getState(
            getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforcePeriodicBox
        )

        ##########################
        # PERIODIC BOX VECTORS
        ##########################
        if self.openmmobject.periodic is True:
            logger.info("Checking PBC vectors:")
            a, b, c = self.state.getPeriodicBoxVectors()
            logger.info("A:  %s", a)
            logger.info("B:  %s", b)
            logger.info("C:  %s", c)
            logger.info("a 0 %s", a[0])
            # Set new PBC vectors since they may have changed
            logger.info("Updating PBC vectors in simulation.context, OpenMM system and OpenMM topology")
            # Context. Used?
            self.simulation.context.setPeriodicBoxVectors(a, b, c)
            # System. Necessary
            self.openmmobject.system.setDefaultPeriodicBoxVectors(a, b, c)
            # Topology (for header in PDB-files). Necessary
            self.openmmobject.topology.setPeriodicBoxVectors(self.state.getPeriodicBoxVectors())

        ################################################
        # Writing final frame to disk as PDB and PDBx
        ################################################
        pdb_filename = self.trajfilename + "_lastframe.pdb"
        logger.info("Writing final frame to disk as PDB-file: %s", pdb_filename)
        with open(pdb_filename, "w") as f:
            openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbfile.PDBFile.writeModel(
                self.openmmobject.topology, self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
            openmm.app.pdbfile.PDBFile.writeFooter(self.openmmobject.topology, f)
        logger.info(f"Trajectory : {self.trajfilename}.{self.trajectory_file_option}")
        # PDBx/mmCIF
        pdbx_filename = self.trajfilename + "_lastframe.cif"
        logger.info("Writing final frame to disk as PDBx/mmCIF-file: %s", pdbx_filename)
        with open(pdbx_filename, "w") as f:
            openmm.app.pdbxfile.PDBxFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbxfile.PDBxFile.writeModel(
                self.openmmobject.topology, self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
        logger.info(f"Trajectory : {self.trajfilename}.{self.trajectory_file_option}")

        # Saving state to disk
        # Can be used to restart using statefile option
        logger.info(
            "Saving a statefile and checkpointfile of the final frame of the simulation: OpenMM_MD_final_state.xml and OpenMM_MD_final_checkpoint.chk"
        )
        logger.info(
            "These file can be used to restart a simulation (statefile and chkfile keywords) using the same coordinates and velocities."
        )
        self.simulation.saveState("OpenMM_MD_final_state.xml")
        self.simulation.saveCheckpoint("OpenMM_MD_final_checkpoint.chk")

        ########################
        # Updating ASH fragment
        ########################
        newcoords = self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        logger.info("Updating coordinates in ASH fragment.")
        self.fragment.coords = newcoords
        # Updating positions array also in case we call run again
        self.positions = newcoords


#############################
#  Multi-step MD protocols  #
#############################


# Note: dummyatomrestraints necessary for NPT simulation when constraining atoms in space
def OpenMM_box_equilibration(
    fragment=None,
    theory=None,
    datafilename="nptsim.csv",
    numsteps_per_NPT=10000,
    max_NPT_cycles=10,
    pressure=1,
    volume_threshold=1.3,
    density_threshold=0.005,
    temperature=300,
    timestep=0.001,
    traj_frequency=100,
    trajfilename="equilibration_NPT",
    trajectory_file_option="DCD",
    coupling_frequency=1,
    enforcePeriodicBox=True,
    use_mdtraj=True,
    dummyatomrestraint=False,
    solute_indices=None,
    barostat_frequency=25,
):
    """NPT simulations until volume and density stops changing

    Args:
        fragment ([type], optional): [description]. Defaults to None.
        theory ([type], optional): [description]. Defaults to None.
        datafilename (str, optional): [description]. Defaults to "nptsim.csv".
        numsteps_per_NPT (int, optional): [description]. Defaults to 10000.
        volume_threshold (float, optional): [description]. Defaults to 1.0.
        density_threshold (float, optional): [description]. Defaults to 0.001.
        temperature (int, optional): [description]. Defaults to 300.
        timestep (float, optional): [description]. Defaults to 0.001.
        traj_frequency (int, optional): [description]. Defaults to 100.
        trajectory_file_option (str, optional): [description]. Defaults to 'DCD'.
        coupling_frequency (int, optional): [description]. Defaults to 1.
        barostat_frequency (int, optional): [description]. Defaults to 25 (timesteps).
    """

    logger.info(main_header("Periodic Box Size Equilibration"))
    module_init_time = time.time()

    if fragment is None or theory is None:
        raise InputError("Fragment and theory required.")

    if numsteps_per_NPT < traj_frequency:
        raise InputError(
            "Parameter 'numpsteps_per_NPT' must be greater than 'traj_frequency', otherwise no data will be written during the equilibration!"
        )

    numpoints_for_convergence_check = numsteps_per_NPT // traj_frequency

    logger.info(small_header("Equilibration Parameters"))
    logger.info("Steps per NPT cycle: %s", numsteps_per_NPT)
    logger.info("Max NPT cycles: %s", max_NPT_cycles)
    logger.info(f"Timestep: {timestep * 1000} fs")
    logger.info("Density threshold: %s", density_threshold)
    logger.info("Volume threshold: %s", volume_threshold)
    logger.info("Intermediate MD data file: %s", datafilename)
    logger.info("Number of datapoints used for convergence check in each cycle: %s", numpoints_for_convergence_check)

    # Number of points used in each cycle to calculate stdev

    if len(theory.user_frozen_atoms) > 0:
        logger.info("Frozen_atoms: %s", theory.user_frozen_atoms)
        logger.warning(
            "OpenMM object has frozen atoms defined. This is known to cause strange issues for NPT simulations."
        )
        logger.warning("Check the results carefully!")

    # Starting parameters
    steps = 0
    volume_std = 10
    density_std = 1

    md = OpenMM_MDclass(
        fragment=fragment,
        theory=theory,
        timestep=timestep,
        traj_frequency=traj_frequency,
        pressure=pressure,
        temperature=temperature,
        integrator="LangevinMiddleIntegrator",
        enforcePeriodicBox=enforcePeriodicBox,
        coupling_frequency=coupling_frequency,
        barostat="MonteCarloBarostat",
        trajfilename=trajfilename,
        datafilename=datafilename,
        trajectory_file_option=trajectory_file_option,
        dummyatomrestraint=dummyatomrestraint,
        solute_indices=solute_indices,
        barostat_frequency=barostat_frequency,
    )
    restart = False
    # while volume_std >= volume_threshold and density_std >= density_threshold:
    for i in range(max_NPT_cycles):
        logger.info("")
        logger.info("%s", "-" * 100)
        logger.info(f"Now starting  NPT cycle {i} with {numsteps_per_NPT} MD steps")
        logger.info(
            f"Simulation data (timestep, energy, temperature, volume,density etc.) is also written to {datafilename}"
        )
        if restart is False:
            # Call MD object run method for the first
            md.run(numsteps_per_NPT, restart=restart)
            # Setting restart to True for next iteration
            restart = True
        else:
            # Easier and safer to continue by call simulation step directly instead of md.run
            md.simulation.step(numsteps_per_NPT)

        steps += numsteps_per_NPT

        # Read reporter file and calculate stdev

        NPTresults = read_NPT_statefile(datafilename)
        volume = NPTresults["volume"][-numpoints_for_convergence_check:]
        density = NPTresults["density"][-numpoints_for_convergence_check:]
        logger.info("Total number of volume datapoints available: %s", len(NPTresults["volume"]))
        logger.info("Total number of density datapoints available: %s", len(NPTresults["density"]))
        logger.info(
            "Number of datapoints (last) used for convergence check in each cycle: %s", numpoints_for_convergence_check
        )
        volume_std = np.std(volume)
        density_std = np.std(density)

        logger.info(small_header("Equilibration Status"))
        logger.info("Total steps taken: %s", steps)
        logger.info(f"Total simulation time: {timestep * steps} ps")
        logger.info("Current Volume: %s", volume[-1])
        logger.info(f"Current Density: {density[-1]}")
        logger.info("")
        logger.info(f"Current Volume SD: {volume_std}   (threshold: {volume_threshold})")
        logger.info(f"Current Density SD: {density_std} (threshold: {density_threshold})")

        if volume_std < volume_threshold and density_std < density_threshold:
            logger.info(f"Equilibration of periodic box finished after {steps} and {timestep * steps} ps !\n")
            break

        if i == max_NPT_cycles - 1:
            logger.info(
                f"Warning: Max NPT cycles reached ({max_NPT_cycles}). Total steps taken: {steps} and {timestep * steps} ps !\n"
            )
            logger.info("Warning: the NPT simulation may not be properly converged")
            break

    # Finalizing simulation (writes and updates files)
    md.finalize_simulation()

    logger.info(f"Final PDB file: {trajfilename}.pdb")
    logger.info(f"NPT trajectory: {trajfilename}.{trajectory_file_option.lower()}")

    # Running mdtraj
    if use_mdtraj is True:
        logger.info("Trying to load mdtraj for reimaging trajectory")
        try:
            logger.info("Imaging trajectory")
            MDtraj_imagetraj(f"{trajfilename}.dcd", f"{trajfilename}_lastframe.pdb")
        except ImportError:
            logger.info("mdtraj library could not be imported. Skipping")
        except ValueError as e:
            logger.info(f"mdtraj reimaging failed. Skipping. Error: {e}")

    log_time_since(module_init_time, "OpenMM_box_equilibration")
    return md.state.getPeriodicBoxVectors()


# Used in OpenMM_MD when doing simulation step-by-step (e.g. QM-MD and QM/MM MD)
def print_current_step_info(step, state, openmmobject, qm_energy=None):

    # Kinetic energy directly from MD-state
    kinetic_energy = state.getKineticEnergy()
    kinetic_energy_eh = kinetic_energy.value_in_unit(openmm.unit.kilojoules_per_mole) / 2625.5002

    # Potential energy from ASH Theory level instead
    if qm_energy is not None:
        dummy_warning = "(correct)"
        pot_energy = qm_energy
    else:
        dummy_warning = "(dummy)"
        pot_energy = state.getPotentialEnergy()

    temp = (2 * kinetic_energy / (openmmobject.dof * openmm.unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(
        openmm.unit.kelvin
    )

    logger.info("%s", "=" * 50)
    logger.info(f"SIMULATION STATUS (STEP {step})")
    logger.info("%s", "_" * 50)
    logger.info(f"Time: {state.getTime()}")
    logger.info(f"Potential energy {dummy_warning}: {pot_energy} Eh")
    logger.info(f"Kinetic energy: {kinetic_energy_eh} Eh")
    logger.info(f"Temperature: {temp}")
    logger.info("%s", "=" * 50)


# CHECKING PDB-FILE FOR multiple occupations.
# Default behaviour:
# - if no multiple occupancies return input PDBfile and go on
# - if multiple occupancies, print list of residues and tell user to fix them. Exiting
# - if use_higher_occupancy is set to True, user higher occupancy location, write new PDB_file and use


def find_alternate_locations_residues(pdbfile, use_higher_occupancy=False):
    if use_higher_occupancy is True:
        logger.info("Will keep higher occupancy atoms for alternate locations")

    # List of ATOM/HETATM lines to grab from PDB-file
    pdb_atomlines = []
    # Dict of residues with alternate location labels
    bad_resids_dict = {}

    # Alternate location dict for atoms found
    altloc_dict = {}

    # Looping through PDB-file
    with open(pdbfile) as pfile:
        for line in pfile:
            if line.startswith(("ATOM", "HETATM")):
                altloc = line[16]
                # Adding info to dicts and adding marker if alternate location info present for atom
                if altloc != " ":
                    chain = line[21:22]
                    # New dict item with chain as key
                    if chain not in bad_resids_dict:
                        bad_resids_dict[chain] = []
                    resid = int(line[22:26].replace(" ", ""))
                    resname = line[17:20].replace(" ", "")
                    residue = resname + str(resid)
                    atomname = line[12:16].replace(" ", "")
                    occupancy = float(line[54:60])
                    # Atomstring contains only the atom-information (not alt-location label)
                    atomstring = chain + "_" + resname + "_" + str(resid) + "_" + atomname
                    # Adding residue to dict
                    if residue not in bad_resids_dict[chain]:
                        bad_resids_dict[chain].append(residue)
                    # Adding atom-info to dict
                    altloc_dict[(atomstring, altloc)] = [altloc, occupancy, line]
                    # Adding atomstring to list as a marker
                    if ["REPLACE_", atomstring] not in pdb_atomlines:
                        pdb_atomlines.append(["REPLACE_", atomstring])
                # Use unmodifed ATOM line
                else:
                    pdb_atomlines.append(line)
            else:
                # Still keeping unmodified line
                pdb_atomlines.append(line)

    # For debugging
    # for k,v in altloc_dict.items():
    def find_index_of_sublist_with_max_col(rows, index):
        max_val = 0
        result = None
        for i, s in enumerate(rows):
            if s[index] > max_val:
                max_val = s[index]
                result = i
        return result

    # Now going through pdb_atomlines, finding marker and looking up the best occupancy atom from altloc_dict
    finalpdblines = []
    for pdbline in pdb_atomlines:
        if pdbline[0] == "REPLACE_":
            logger.info("Alternate locations for atom: %s", pdbline[1])
            options = []
            # Looping through altloc_dict items
            for i, j in altloc_dict.items():
                # Matching atomstring
                if i[0] == pdbline[1]:
                    options.append([j[0], j[1], j[2]])
            for option_row in options:
                pdblinestring = "".join(map(str, option_row[2:]))
                logger.info("%s", pdblinestring)
            # Get max occupancy item
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
        # if alternatelocation_label != None:
        if use_higher_occupancy is True:
            logger.warning("\n Use higher-occupancy location opton was selected, so continuing.")
            writelisttofile(finalpdblines, "system_afteratlocfixes.pdb", separator="")
            return "system_afteratlocfixes.pdb"
        else:
            raise InputError(
                "You should delete either the labelled A or B location of the residue-atom/atoms and then remove the A/B label from column 17 in the file\nAlternatively, you can choose use_higher_occupancy=True keyword in OpenMM_Modeller and ASH will keep the higher occupied form and go on \nMake sure that there is always an A or B form present.\nExiting."
            )
    # Returning original pdbfile if all OK

    return pdbfile


################################
# Native OpenMM metadynamics
################################


# Metadynamics written as a wrapper function around OpenMM_MDclass
# TODO: Decide units for CV biaswidth range and Gaussian height
# NOTE: Restraints are in Angstrom and kcal/mol^2
def OpenMM_metadynamics(
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
    specialatoms=None,
    specialtraj_frequency=1000,
    barostat=None,
    pressure=1,
    trajectory_file_option="DCD",
    trajfilename="trajectory",
    coupling_frequency=1,
    charge=None,
    mult=None,
    platform="CPU",
    hydrogenmass=1.5,
    constraints=None,
    anderson_thermostat=False,
    restraints=None,
    flatbottom_restraint_CV1=None,
    flatbottom_restraint_CV2=None,
    funnel_restraint=None,
    funnel_parameters=None,
    enforcePeriodicBox=True,
    special_wrapping=False,
    special_wrapping_updatepos=False,
    wrapping_atoms=None,
    dummyatomrestraint=False,
    center_on_atoms=None,
    solute_indices=None,
    datafilename=None,
    dummy_MM=False,
    add_centerforce=False,
    centerforce_atoms=None,
    centerforce_distance=10.0,
    centerforce_constant=1.0,
    centerforce_center=None,
    barostat_frequency=25,
    CV1_atoms=None,
    CV2_atoms=None,
    CV1_type=None,
    CV2_type=None,
    biasfactor=6,
    height=1,
    reference_xyzfile=None,
    CV1_biaswidth=0.5,
    CV2_biaswidth=0.5,
    CV1_range=None,
    CV2_range=None,
    CV1_parameters=None,
    CV2_parameters=None,
    user_cvforce1=None,
    user_biasvar1=None,
    user_cvforce2=None,
    user_biasvar2=None,
    frequency=1,
    savefrequency=10,
    chkfile=None,
    statefile=None,
    biasdir=".",
    multiplewalkers=False,
    numcores=1,
    walkerid=None,
):
    logger.info(main_header("OpenMM metadynamics"))

    # Biasdirectory
    logger.info("biasdirectory chosen to be: %s", biasdir)
    biasdir_full_path = os.path.abspath(biasdir)
    logger.info("Full path to biasdirectory is: %s", biasdir_full_path)
    if not os.path.isdir(biasdir_full_path):
        raise FileFormatError(f"Error: Biasdirectory: {biasdir_full_path} does not exist")

    if CV2_type is None:
        logger.info("CV2 not specified. Assuming only 1 CV in simulation.")
        numCVs = 1
        if user_cvforce1 is None and (CV1_atoms is None or CV1_type is None):
            raise InputError("Error: You must specify both CV1_atoms and CV1_type keywords")
    else:
        numCVs = 2
        if user_cvforce1 is None and (CV1_atoms is None or CV1_type is None):
            raise InputError("Error: You must specify both CV1_atoms and CV1_type keywords")
        if user_cvforce2 is None and (CV2_atoms is None or CV2_type is None):
            raise InputError("Error: You must specify both CV2_atoms and CV2_type keywords")

    # Parallelization
    if multiplewalkers is True and numcores == 1:
        raise InputError("Error: For multiplewalkers=True  you must set numcores to the number of walkers")

    # Creating MDclass
    md = OpenMM_MDclass(
        fragment=fragment,
        theory=theory,
        charge=charge,
        mult=mult,
        timestep=timestep,
        traj_frequency=traj_frequency,
        temperature=temperature,
        integrator=integrator,
        constraints=constraints,
        specialatoms=specialatoms,
        specialtraj_frequency=specialtraj_frequency,
        barostat=barostat,
        pressure=pressure,
        trajectory_file_option=trajectory_file_option,
        coupling_frequency=coupling_frequency,
        anderson_thermostat=anderson_thermostat,
        enforcePeriodicBox=enforcePeriodicBox,
        special_wrapping=special_wrapping,
        special_wrapping_updatepos=special_wrapping_updatepos,
        wrapping_atoms=wrapping_atoms,
        dummyatomrestraint=dummyatomrestraint,
        center_on_atoms=center_on_atoms,
        solute_indices=solute_indices,
        datafilename=datafilename,
        dummy_MM=dummy_MM,
        platform=platform,
        hydrogenmass=hydrogenmass,
        add_centerforce=add_centerforce,
        trajfilename=trajfilename,
        chkfile=chkfile,
        statefile=statefile,
        centerforce_atoms=centerforce_atoms,
        centerforce_constant=centerforce_constant,
        centerforce_distance=centerforce_distance,
        centerforce_center=centerforce_center,
        barostat_frequency=barostat_frequency,
    )

    #
    if user_cvforce1 is not None:
        logger.info("User CV-force 1 was given: %s", user_cvforce1)
        md.user_cvforce1 = user_cvforce1
    if user_biasvar1 is not None:
        logger.info("User Biasvar CV1 was given: %s", user_biasvar1)
        md.user_biasvar1 = user_biasvar1
    if user_cvforce2 is not None:
        logger.info("User CV-force 2 was given: %s", user_cvforce2)
        md.user_cvforce2 = user_cvforce2
    if user_biasvar2 is not None:
        logger.info("User Biasvar CV2 was given: %s", user_biasvar2)
        md.user_biasvar2 = user_biasvar2

    # Load OpenMM.app

    # If RMSD CV
    if CV1_type == "rmsd" or CV2_type == "rmsd":
        # Reference position. For now just use initial cooordinates as reference positions
        # reference_pos = [openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in
        #       range(len(coords_nm))] * openmm.unit.nanometer
        logger.info("rmsd_CV1_reference_indices: %s", CV1_atoms)
        logger.info("rmsd_CV2_reference_indices: %s", CV2_atoms)
    else:
        pass
    # Setting up collective variables for native case
    native_MTD = True
    # Creating dictionary with MTD parameters that will be passed to MD function
    if numCVs == 1:
        # Create metadynamics dict for 1 CV
        metadyn_settings = {
            "numCVs": numCVs,
            "temperature": temperature,
            "biasfactor": biasfactor,
            "height": height,
            "frequency": frequency,
            "saveFrequency": savefrequency,
            "biasdir": biasdir_full_path,
            "CV1_type": CV1_type,
            "CV2_type": None,
            "reference_xyzfile": reference_xyzfile,
            "CV1_atoms": CV1_atoms,
            "CV2_atoms": CV2_atoms,
            "CV1_range": CV1_range,
            "CV2_range": CV2_range,
            "CV1_biaswidth": CV1_biaswidth,
            "CV2_biaswidth": CV2_biaswidth,
            "CV2_minvalue": None,
            "CV2_maxvalue": None,
            "CV1_parameters": CV1_parameters,
            "flatbottom_restraint_CV1": flatbottom_restraint_CV1,
            "flatbottom_restraint_CV2": flatbottom_restraint_CV2,
        }
    elif numCVs == 2:
        # Create metadynamics object for 2 CVs
        metadyn_settings = {
            "numCVs": numCVs,
            "temperature": temperature,
            "biasfactor": biasfactor,
            "height": height,
            "frequency": frequency,
            "saveFrequency": savefrequency,
            "biasdir": biasdir_full_path,
            "CV1_type": CV1_type,
            "CV2_type": CV2_type,
            "reference_xyzfile": reference_xyzfile,
            "CV1_range": CV1_range,
            "CV2_range": CV2_range,
            "CV1_parameters": CV1_parameters,
            "CV2_parameters": CV2_parameters,
            "CV1_atoms": CV1_atoms,
            "CV2_atoms": CV2_atoms,
            "CV1_biaswidth": CV1_biaswidth,
            "CV2_biaswidth": CV2_biaswidth,
            "flatbottom_restraint_CV1": flatbottom_restraint_CV1,
            "flatbottom_restraint_CV2": flatbottom_restraint_CV2,
        }

    # Add restraining funnel for funnel metadynamics
    if funnel_restraint is not None:
        if funnel_parameters is None:
            raise InputError(
                "Error: funnel_restraint requires passing a dictionary with funnel definition parameters.\nExample: funnel_parameters = {'ligand_indices':[0,1,2], 'k_xyz':10.0, 'z_cc':11.0, 'alpha':35.0, 'R_cylinder':1.0, 'force_group':10}"
            )

        # Getting atom indices for host and guess
        guest_indices = funnel_parameters["ligand_indices"]
        logger.info("guest_indices: %s", guest_indices)
        if "host_indices" in funnel_parameters:
            logger.info("Found host indices in funnel_parameters")
            host_indices = funnel_parameters["host_indices"]
            logger.info("host_indices: %s", host_indices)
        else:
            raise InputError("No host_indices found in funnel_parameters")

        md.openmmobject.add_funnel_restraint(
            host_indices,
            guest_indices,
            k_xy=funnel_parameters["k_xy"],
            z_cc=funnel_parameters["z_cc"],
            alpha=funnel_parameters["alpha"],
            R_cylinder=funnel_parameters["R_cylinder"],
            force_group=funnel_parameters["force_group"],
        )

    # Calling md.run with either native option active or false
    logger.info("Now starting metadynamics simulation")

    if multiplewalkers is True:
        raise InputError("{}\nError: Disabled".format(f"Now launching Metadynamics job with {numcores} walkers"))
        # Input parameters passed as dictionary to Simple_parallel
        # NOTE: multiprocess library (instead of multiprocessing) is necessary.
        # Otherwise pickling problem involving _io.TextIOWrapper
        openmmqmmm.functions.functions_parallel.Simple_parallel(
            jobfunction=md.run,
            parameter_dict={
                "simulation_steps": simulation_steps,
                "simulation_time": simulation_time,
                "metadynamics": native_MTD,
                "metadyn_settings": metadyn_settings,
            },
            numcores=numcores,
            version="multiprocess",
            separate_dirs=True,
            restraints=restraints,
        )
    else:
        md.run(
            simulation_steps=simulation_steps,
            simulation_time=simulation_time,
            metadynamics=native_MTD,
            metadyn_settings=metadyn_settings,
            restraints=restraints,
        )
    logger.info("Metadynamics simulation done")

    # Finalizing simulation (writes and updates files)
    md.finalize_simulation()

    # Data plotting
    logger.info("\nAll bias-files have been written to biasdirectory: %s", biasdir_full_path)
    logger.info("Dir also contains: ASH_MTD_parameters.txt")
    logger.info("Use function  get_free_energy_from_biasfiles  to create free-energy surface")
    logger.info("and function metadynamics_plot_data to plot the data")
    logger.info("")
    return


# Metadynamics-function that used OpenMM_Plumed interface
def OpenMM_MD_plumed(
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
    specialatoms=None,
    specialtraj_frequency=1000,
    barostat=None,
    pressure=1,
    trajectory_file_option="DCD",
    trajfilename="trajectory",
    coupling_frequency=1,
    charge=None,
    mult=None,
    platform="CPU",
    hydrogenmass=1.5,
    constraints=None,
    anderson_thermostat=False,
    restraints=None,
    enforcePeriodicBox=True,
    special_wrapping=False,
    special_wrapping_updatepos=False,
    wrapping_atoms=None,
    dummyatomrestraint=False,
    center_on_atoms=None,
    solute_indices=None,
    datafilename=None,
    dummy_MM=False,
    add_centerforce=False,
    centerforce_atoms=None,
    centerforce_distance=10.0,
    centerforce_constant=1.0,
    centerforce_center=None,
    barostat_frequency=25,
    chkfile=None,
    statefile=None,
    plumed_input_string=None,
    numcores=1,
):
    logger.info(main_header("OpenMM metadynamics using OpenMM-Plumed interface"))

    logger.info("Using metadynamics via OpenMM Plumed plugin")
    try:
        import openmmplumed
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "openmmplumed module plugin not found. See https://github.com/openmm/openmm-plumed \nYou can install via conda: \nconda install -c conda-forge openmm-plumed"
        ) from None

    # Creating MDclass
    md = OpenMM_MDclass(
        fragment=fragment,
        theory=theory,
        charge=charge,
        mult=mult,
        timestep=timestep,
        traj_frequency=traj_frequency,
        temperature=temperature,
        integrator=integrator,
        constraints=constraints,
        specialatoms=specialatoms,
        specialtraj_frequency=specialtraj_frequency,
        barostat=barostat,
        pressure=pressure,
        trajectory_file_option=trajectory_file_option,
        coupling_frequency=coupling_frequency,
        anderson_thermostat=anderson_thermostat,
        enforcePeriodicBox=enforcePeriodicBox,
        special_wrapping=special_wrapping,
        special_wrapping_updatepos=special_wrapping_updatepos,
        wrapping_atoms=wrapping_atoms,
        dummyatomrestraint=dummyatomrestraint,
        center_on_atoms=center_on_atoms,
        solute_indices=solute_indices,
        datafilename=datafilename,
        dummy_MM=dummy_MM,
        platform=platform,
        hydrogenmass=hydrogenmass,
        add_centerforce=add_centerforce,
        trajfilename=trajfilename,
        centerforce_atoms=centerforce_atoms,
        centerforce_constant=centerforce_constant,
        chkfile=chkfile,
        statefile=statefile,
        centerforce_distance=centerforce_distance,
        centerforce_center=centerforce_center,
        barostat_frequency=barostat_frequency,
    )

    # Load OpenMM.app

    logger.info("Setting up Plumed")
    # OPTION to provide the full Plumed input as string instead
    if plumed_input_string is not None:
        logger.info(
            "plumed_input_string provided. Will read all options from this string (make sure to provide atom indices in 1-based indexing)"
        )
        writestringtofile(plumed_input_string, "plumedinput.in")
        plumedinput = plumed_input_string

    logger.info("Now starting metadynamics simulation")
    md.run(
        simulation_steps=simulation_steps,
        simulation_time=simulation_time,
        restraints=restraints,
        plumedinput=plumedinput,
    )
    logger.info("Metadynamics simulation done")

    # Finalizing simulation (writes and updates files)
    md.finalize_simulation()

    os.path.dirname(os.path.dirname(os.path.dirname(openmmplumed.mm.pluginLoadedLibNames[0])))
    logger.info(
        "You can now analyze/plot the metadynamics data with plumed's own tools (requires presence of HILLS and COLVAR files in directory)"
    )
    logger.info("\n")

    return


def Gentle_warm_up_MD(
    theory=None,
    fragment=None,
    time_steps=None,
    steps=None,
    temperatures=None,
    check_gradient_first=True,
    gradient_threshold=100,
    use_mdtraj=True,
    trajfilename="warmup_MD",
    initial_opt=True,
    traj_frequencies=None,
    maxoptsteps=10,
    coupling_frequency=1,
):
    if traj_frequencies is None:
        traj_frequencies = [1, 1, 100]
    if temperatures is None:
        temperatures = [1, 10, 300]
    if steps is None:
        steps = [10, 50, 10000]
    if time_steps is None:
        time_steps = [0.0005, 0.001, 0.004]
    logger.info(main_header("Gentle_warm_up_MD"))
    module_init_time = time.time()
    logger.info("Trajectory filename: %s", trajfilename)
    if theory is None or fragment is None:
        raise InputError("Gentle_warm_up_MD requires theory (OpenMM object) and fragment")

    if len(time_steps) != len(steps) or len(time_steps) != len(temperatures):
        raise InputError("Error: Lists time_steps, steps and temperatures all need to be the same length. Exiting")

    # Gradient check before we proceed
    if check_gradient_first is True:
        logger.info("check_gradient_first is True")
        logger.info("Will run singlepoint gradient calculation to check for large forces")
        theory.force_run = True
        SP_result = Singlepoint(theory=theory, fragment=fragment, Grad=True)
        badindices = check_gradient_for_bad_atoms(
            fragment=fragment, gradient=SP_result.gradient, threshold=gradient_threshold
        )
        if len(badindices) > 0:
            logger.info(f"\nNumber of atoms with large forces: {len(badindices)}")
            logger.info("Suggests a bad system geometry or that atoms need constraints (might be present already)")
            logger.info("Gentle_warm_up_MD will go on")

    # Try a simple minimization first or simple MD

    if initial_opt is True:
        logger.info(f"\ninitial_opt is True (default). Will attempt initial {maxoptsteps}-step minimization first")
        logger.info("If this step runs forever something is wrong. Select initial_opt=False to avoid in this case")
        try:
            OpenMM_Opt(fragment=fragment, theory=theory, maxiter=maxoptsteps, tolerance=1)
            logger.info("Minimization successful")
        except Exception as e:  # noqa: BLE001 - MD warm-up continues even if pre-minimization fails
            logger.info("Problem minimizing system")
            logger.info("Error message: %s", e)
            logger.info("Will go on to do MD")

    logger.info(f"\n{len(steps)} MD-runs have been defined")
    for num, (ts, step, temp) in enumerate(zip(time_steps, steps, temperatures, strict=False)):
        logger.info(f"MD-step {num} Number of simulation steps: {step} with timestep: {ts} and temperature: {temp} K")

    logger.info("")
    logger.info("")
    # Gentle heating up protocol
    for num, (ts, step, temp, traj_frequency) in enumerate(
        zip(time_steps, steps, temperatures, traj_frequencies, strict=False)
    ):
        # Name of PDB and DCD filename: i.e. warmup_MD_cycle1.pdb and warmup_MD_cycle1.dcd
        MDcyclename = trajfilename + f"_cycle{num}"
        logger.info(
            f"\n\nNow running MD-run {num}. Number of steps: {step} with timestep:{ts} and temperature: {temp} K"
        )
        logger.info(f"Will write trajectory to file: {MDcyclename}.dcd")
        OpenMM_MD(
            fragment=fragment,
            theory=theory,
            timestep=ts,
            simulation_steps=step,
            traj_frequency=traj_frequency,
            temperature=temp,
            integrator="LangevinMiddleIntegrator",
            coupling_frequency=coupling_frequency,
            trajfilename=MDcyclename,
            trajectory_file_option="DCD",
        )

        # Running mdtraj after each sim
        if use_mdtraj is True:
            logger.info("Trying to load mdtraj for basic analysis of trajectory")
            try:
                logger.info("Imaging trajectory")
                MDtraj_imagetraj(f"{MDcyclename}.dcd", f"{MDcyclename}_lastframe.pdb")
                logger.info("\nRunning RMS Fluctuation analysis on trajectory")
                MDtraj_RMSF(
                    f"{MDcyclename}.dcd",
                    f"{MDcyclename}_lastframe.pdb",
                    print_largest_values=True,
                    threshold=0.005,
                    largest_values=10,
                )
            except ImportError:
                logger.info("mdtraj library could not be imported. Skipping")
            except ValueError as e:
                logger.info(f"mdtraj reimaging failed. Skipping. Error: {e}")

    logger.info("Gentle_warm_up_MD finished successfully!")
    log_time_since(module_init_time, "Gentle_warm_up_MD")
    return


# Function to create CV biases in native OpenMM metadynamics
def create_CV_bias(
    CV_type,
    CV_atoms,
    biaswidth_cv,
    CV_range=None,
    reference_pos=None,
    reference_particles=None,
    user_cvforce=None,
    user_biasvar=None,
    CV_parameters=None,
):

    logger.info("Inside create_CV_bias")
    logger.info("CV_type: %s", CV_type)
    logger.info("CV_atoms: %s", CV_atoms)
    # TODO: Try changing dihedrals/angles to deg units
    # Most of the time though there is no reason to specify CV min and max for these CVs as you want the full range
    # However the biaswidth is also in
    if CV_range is None:
        logger.info("Warning: No minx/max value range for CVchosen by user")
        logger.info("Will choose reasonable values based on CV type:")
        if CV_type == "dihedral" or CV_type == "torsion":
            CV_min_val = -np.pi
            CV_max_val = np.pi
            CV_unit = openmm.unit.radians
            CV_unit_label = "rad"
            biaswidth_cv_unit = openmm.unit.radians
            biaswidth_cv_unit_label = "rad"
        elif CV_type == "angle":
            CV_min_val = 0
            CV_max_val = np.pi
            CV_unit = openmm.unit.radians
            CV_unit_label = "rad"
            biaswidth_cv_unit = openmm.unit.radians
            biaswidth_cv_unit_label = "rad"
        elif CV_type == "distance" or CV_type == "bond" or CV_type == "rmsd":
            CV_min_val = 0.0
            CV_max_val = 5.0
            CV_unit = openmm.unit.angstroms
            CV_unit_label = "Å"
            biaswidth_cv_unit = openmm.unit.angstroms
            biaswidth_cv_unit_label = "Å"
        elif CV_type.lower() == "cn":
            CV_min_val = 0.0
            CV_max_val = 5.0  # TODO
            CV_unit = 1
            CV_unit_label = "CN"
            biaswidth_cv_unit = 1  # TODO
            biaswidth_cv_unit_label = "CN"  #
        elif CV_type.lower() == "custom":
            CV_min_val = 0.0
            CV_max_val = 5.0  # TODO
            CV_unit = 1
            CV_unit_label = "Custom"
            biaswidth_cv_unit = 1  # TODO
            biaswidth_cv_unit_label = "Custom"  #
    else:
        logger.info("CV range given.")
        CV_min_val = CV_range[0]
        CV_max_val = CV_range[1]
        if CV_type == "dihedral" or CV_type == "torsion" or CV_type == "angle":
            CV_unit = openmm.unit.radians
            CV_unit_label = "rad"
            biaswidth_cv_unit = openmm.unit.radians
            biaswidth_cv_unit_label = "rad"
        elif CV_type == "distance" or CV_type == "bond" or CV_type == "rmsd":
            CV_unit = openmm.unit.angstroms
            CV_unit_label = "Å"
            biaswidth_cv_unit = openmm.unit.angstroms
            biaswidth_cv_unit_label = "Å"
        elif CV_type.lower() == "cn":
            CV_unit = 1  # TODO
            CV_unit_label = "CN"
            biaswidth_cv_unit = 1  # TODO
            biaswidth_cv_unit_label = "CN"
        elif CV_type.lower() == "custom":
            CV_unit = 1  # TODO
            CV_unit_label = "Custom"
            biaswidth_cv_unit = 1  # TODO
            biaswidth_cv_unit_label = "Custom"
    logger.info(f"CV_min_val: {CV_min_val} and CV_max_val: {CV_max_val} {CV_unit_label}")
    logger.info(f"Biaswidth of CV: {biaswidth_cv} {biaswidth_cv_unit_label}")
    # Define collective variables for CV1 and CV2.
    if CV_type == "dihedral" or CV_type == "torsion":
        if len(CV_atoms) != 4:
            raise InputError("Error: CV_atoms list must contain 4 atom indices")
        cvforce = openmm.CustomTorsionForce("theta")
        cvforce.addTorsion(*CV_atoms)
        CV_bias = openmm.app.BiasVariable(
            cvforce, CV_min_val * CV_unit, CV_max_val * CV_unit, biaswidth_cv * biaswidth_cv_unit, periodic=True
        )
    elif CV_type == "angle":
        if len(CV_atoms) != 3:
            raise InputError("Error: CV_atoms list must contain 3 atom indices")
        cvforce = openmm.CustomAngleForce("theta")
        cvforce.addAngle(*CV_atoms)
        CV_bias = openmm.app.BiasVariable(
            cvforce, CV_min_val * CV_unit, CV_max_val * CV_unit, biaswidth_cv * biaswidth_cv_unit, periodic=False
        )
    elif CV_type == "distance" or CV_type == "bond":
        if len(CV_atoms) != 2:
            raise InputError("Error: CV_atoms list must contain 2 atom indices")
        cvforce = openmm.CustomBondForce("r")
        cvforce.addBond(*CV_atoms)
        CV_bias = openmm.app.BiasVariable(
            cvforce, CV_min_val * CV_unit, CV_max_val * CV_unit, biaswidth_cv * biaswidth_cv_unit, periodic=False
        )
    elif CV_type == "rmsd":
        # http://docs.openmm.org/development/api-python/generated/openmm.openmm.RMSDForce.html
        # reference_pos: A vector of atom positions
        # reference_particles: atom indices used to calculate RMSD
        cvforce = openmm.RMSDForce(reference_pos)
        cvforce.setParticles(reference_particles)
        CV_bias = openmm.app.BiasVariable(
            cvforce, CV_min_val * CV_unit, CV_max_val * CV_unit, biaswidth_cv * biaswidth_cv_unit, periodic=False
        )
    elif CV_type.lower() == "cn":
        logger.info("CV type is CN")

        if CV_parameters is None:
            raise InputError(
                "Error: CV-type coordination number requires a threshold value (when the distance should not longer be considered a bond)\nThis should be passed as a list using CV1_parameters/CV2_parameters, e.g. CV1_parameters=[2.0]"
            )
        logger.info(f"List CV_parameters contains: {CV_parameters}. Using {CV_parameters[0]} as threshold")
        # Defining custom cvforce
        energy_expression = "1/(1+x^6) ; x=r/threshold"
        cvforce = openmm.CustomBondForce(energy_expression)
        # Threshold that defines when a bond is present
        # Taking threshold from first number in CV_parameters list
        cvforce.addGlobalParameter("threshold", CV_parameters[0] * openmm.unit.angstrom)

        # Adding the atoms that define each bonds
        for bond_indices in CV_atoms:
            cvforce.addBond(*bond_indices)

        # Creating Biasvariable: forceobj, minval, maxval, biaswidth
        CV_bias = openmm.app.BiasVariable(
            cvforce, CV_min_val * CV_unit, CV_max_val * CV_unit, biaswidth_cv * biaswidth_cv_unit, periodic=False
        )

    elif CV_type.lower() == "custom":
        # User cv force with atoms already added
        # cvforce
        cvforce = user_cvforce
        CV_bias = user_biasvar
        logger.info("cvforce: %s", cvforce)
        logger.info("CV_bias: %s", CV_bias)
    else:
        raise InputError("unsupported CV_type for native OpenMM metadynamics implementation")
    return CV_bias, cvforce


# Calculate free-energy from total bias array
def free_energy_from_bias_array(temperature, biasFactor, totalBias):
    deltaT = temperature * (biasFactor - 1)
    kjpermoleconversion = 1
    free_energy = -((temperature + deltaT) / deltaT) * totalBias * kjpermoleconversion
    return free_energy


# Calculate free-energy from OpenMM biasfiles
def get_free_energy_from_biasfiles(temperature, biasfactor, CV1_gridwith, CV2_gridwith, directory="."):
    import glob

    # Checking gridwiths
    full_bias = np.zeros(CV1_gridwith) if CV2_gridwith is None else np.zeros((CV2_gridwith, CV1_gridwith))

    # Looping over bias-files
    logger.info("full_bias shape: %s", full_bias.shape)
    list_of_biases = []
    for biasfile in glob.glob(f"{directory}/*.npy"):
        logger.info("Loading biasfile: %s", biasfile)
        try:
            data = np.load(biasfile)
            logger.info("data shape: %s", data.shape)
            full_bias += data
            list_of_biases.append(data)
        except FileNotFoundError:
            logger.info("File not found error: Simulation probably still running. skipping file")

    # Get final free energy (sum of all)
    free_energy = free_energy_from_bias_array(temperature, biasfactor, full_bias)

    # Get free-energy per biasfile
    list_of_free_energies = []
    for bias_array in list_of_biases:
        fe = free_energy_from_bias_array(temperature, biasfactor, bias_array)
        list_of_free_energies.append(fe)

    # Return final free_energy array and also list of free-energy-arrays for each biasfile
    return free_energy, list_of_free_energies


# Simple plotting for native OpenMM metadynamics via ASH
# NOTE: plot_xlim/plot_ylim in final CV units (Ang for distance/rmsd and ° for dihedrals/angles)
# CV1_minvalue/CV1_maxvalue should be set before simulation
def metadynamics_plot_data(biasdir=None, dpi=200, imageformat="png", plot_xlim=None, plot_ylim=None):
    import json

    # Read mtd settings dict from file
    with open(f"{biasdir}/ASH_MTD_parameters.txt") as mtdfh:
        metadyn_settings = json.load(mtdfh)

    CV1_type = metadyn_settings["CV1_type"]
    CV2_type = metadyn_settings["CV2_type"]
    temperature = metadyn_settings["temperature"]
    biasfactor = metadyn_settings["biasfactor"]
    CV1_gridwidth = metadyn_settings["CV1_gridwidth"]
    logger.info("metadyn_settings: %s", metadyn_settings)
    CV2_gridwidth = metadyn_settings["CV2_gridwidth"]

    CV1_minvalue = metadyn_settings["CV1_minvalue"]
    CV1_maxvalue = metadyn_settings["CV1_maxvalue"]
    CV2_minvalue = metadyn_settings["CV2_minvalue"]
    CV2_maxvalue = metadyn_settings["CV2_maxvalue"]
    logger.info(f"Using CV1_minvalue:{CV1_minvalue} CV1_maxvalue:{CV1_maxvalue}")
    logger.info(f"Using CV2_minvalue:{CV2_minvalue} CV2_maxvalue:{CV2_maxvalue}")

    e_conversionfactor = 4.184  # kJ/mol to kcal/mol
    numCVs = 2 if CV2_type is not None else 1
    if numCVs == 2:
        cv1_conversionfactor = 1.0
        cv2_conversionfactor = 1.0
        if CV1_type == "dihedral" or CV1_type == "torsion" or CV1_type == "angle":
            cv1_conversionfactor = 180 / np.pi
            CV1_unit_label = "°"
        elif CV1_type == "bond" or CV1_type == "distance" or CV1_type == "rmsd":
            cv1_conversionfactor = 10.0
            CV1_unit_label = "Å"
        if CV2_type == "dihedral" or CV2_type == "angle" or CV1_type == "torsion":
            cv2_conversionfactor = 180 / np.pi
            CV2_unit_label = "°"
        elif CV2_type == "bond" or CV2_type == "distance" or CV2_type == "rmsd":
            cv2_conversionfactor = 10.0
            CV2_unit_label = "Å"
        else:
            CV1_unit_label = ""
            CV2_unit_label = ""

        # Get free energy surface from biasfiles
        free_energy, _list_of_fes_from_biasfiles = get_free_energy_from_biasfiles(
            temperature, biasfactor, CV1_gridwidth, CV2_gridwidth, directory=biasdir
        )
        # Relative free energy in kcal/mol
        rel_free_energy = (free_energy - np.min(free_energy)) / e_conversionfactor
        # Coordinates in correct unit
        xvalues = [
            cv1_conversionfactor * (CV1_minvalue + ((CV1_maxvalue - CV1_minvalue) / (CV1_gridwidth - 1)) * i)
            for i in range(CV1_gridwidth)
        ]
        yvalues = [
            cv2_conversionfactor * (CV2_minvalue + ((CV2_maxvalue - CV2_minvalue) / (CV2_gridwidth - 1)) * i)
            for i in range(CV2_gridwidth)
        ]
        np.savetxt("MTD_free_energy.txt", free_energy)
        np.savetxt("MTD_free_energy_rel.txt", rel_free_energy)
        np.savetxt("CV1_coord_values.txt", xvalues)
        np.savetxt("CV2_coord_values.txt", yvalues)

        # Plot
        logger.info("Now plotting:")
        try:
            import matplotlib.pyplot
        except ImportError:
            logger.info("Problem importing matplotlib")
            return
        # 2D CV plotting uisng scatter with colormap
        # Colormap to use in 2CV plots.
        # Perceptually uniform sequential: viridis, plasma, inferno, magma, cividis
        # Others: # RdYlBu_r
        # See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        colormap_option3 = "RdYlBu_r"
        X2, Y2 = np.meshgrid(xvalues, yvalues)
        option3fig, option3ax = matplotlib.pyplot.subplots()
        cm = matplotlib.pyplot.cm.get_cmap(colormap_option3)
        colorscatter = option3ax.scatter(X2, Y2, c=rel_free_energy, marker="o", linestyle="-", linewidth=1, cmap=cm)
        # Colorbar
        cbar = matplotlib.pyplot.colorbar(colorscatter)
        cbar.set_label("ΔG (kcal/mol)", fontweight="bold", fontsize="xx-small")
        # Limits
        if plot_xlim is not None:
            option3ax.set_xlim(plot_xlim[0], plot_xlim[1])
        if plot_ylim is not None:
            option3ax.set_ylim(plot_ylim[0], plot_ylim[1])
        option3ax.set_xlabel(f"CV1:{CV1_type}  ({CV1_unit_label})")
        option3ax.set_ylabel(f"CV2:{CV2_type}  ({CV2_unit_label})")
        option3fig.savefig("MTD_CV1_CV2_.png", format=imageformat, dpi=dpi)
        logger.info("Created file: MTD_CV1_CV2_.png")
        return

    elif numCVs == 1:
        cv1_conversionfactor = 1.0
        if CV1_type == "dihedral" or CV1_type == "torsion" or CV1_type == "angle":
            cv1_conversionfactor = 180 / np.pi
            CV1_unit_label = "°"
        elif CV1_type == "bond" or CV1_type == "distance" or CV1_type == "rmsd":
            cv1_conversionfactor = 10.0
            CV1_unit_label = "Ang"
        else:
            CV1_unit_label = ""
        free_energy, _bla = get_free_energy_from_biasfiles(
            temperature, biasfactor, CV1_gridwidth, None, directory=biasdir
        )

        # X-values
        full_range = CV1_maxvalue - CV1_minvalue
        increment = full_range / (CV1_gridwidth - 1)
        xvalues = [cv1_conversionfactor * (CV1_minvalue + increment * i) for i in range(CV1_gridwidth)]
        np.savetxt("CV1_coord_values.txt", xvalues)
        # Relative energy in kcal/mol
        rel_free_energy = (free_energy - min(free_energy)) / e_conversionfactor
        logger.info("rel_free_energy: %s", rel_free_energy)
        # Save stuff
        np.savetxt("MTD_free_energy.txt", free_energy)
        np.savetxt("MTD_free_energy_rel.txt", rel_free_energy)

        # Plot object
        logger.info("Now plotting:")
        CVlabel = f"{CV1_type} ({CV1_unit_label})"
        y_axislabel = "Energy (kcal(/mol))"
        eplot = openmmqmmm.modules.module_plotting.ASH_plot(
            "Metadynamics", num_subplots=1, x_axislabel=CVlabel, y_axislabel=y_axislabel
        )
        eplot.addseries(0, x_list=xvalues, y_list=rel_free_energy, legend=None, color="blue", line=True, scatter=False)
        eplot.savefig("MTD_CV1", imageformat=imageformat, dpi=dpi)
        return


# Function to wrap coordinates of whole molecules outside box


def diff_wrap_box_coords(coords_nm, boxvectors, mdtrajtopology, anchoratoms):
    # Import mdtraj library
    import mdtraj

    # Creating Trajectory object for geometry
    traj = mdtraj.Trajectory(coords_nm, mdtrajtopology)
    # Setting PBC vectors
    traj.unitcell_vectors = np.array(boxvectors).reshape(1, 3, 3)
    # Anchoratoms (usually QM-region or similar)
    anchors = [{traj.topology.atom(i) for i in anchoratoms}]
    # Re-imaging trajectory
    imaged = traj.image_molecules(anchor_molecules=anchors)
    return imaged._xyz[0] * 10.0


def merge_pdb_files(pdbfile_1, pdbfile_2, outputname="merged.pdb"):

    # Function to merge PDB-files (e.g. protein and ligand) while preserving and updating connectivity records
    # PDB inputfiles
    pdb1 = openmm.app.PDBFile(pdbfile_1)
    pdb2 = openmm.app.PDBFile(pdbfile_2)

    # Create modeller object
    modeller = openmm.app.Modeller(pdb1.topology, pdb1.positions)  # Add pdbfile1
    modeller.add(pdb2.topology, pdb2.positions)  # Add pdbfile2
    mergedPositions = modeller.positions  # merging positions

    # Write merged topology and positions to new PDB file
    write_pdbfile_openMM(modeller.topology, mergedPositions, outputname)
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
):
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

    # OpenMM
    try:
        from openmm.app import ForceField
    except ModuleNotFoundError:
        raise MissingDependencyError("OpenMM is required but could not be imported") from None

    # Parmed
    try:
        import parmed
    except ImportError:
        raise MissingDependencyError(
            "Problem importing parmed Python library\nParmed can be installed using pip: pip install parmed"
        ) from None
    logger.info(f"Parmed version {parmed.__version__} imported")
    # OpenMMForcefields stuff
    try:
        import openff
        from openff.toolkit.topology import Molecule
        from openmmforcefields.generators import GAFFTemplateGenerator
    except ImportError as errormessage:
        raise MissingDependencyError(
            f"OpenFF and openmmforcefields libraries are required but could not be imported\nYou can install like this:   conda install --yes -c conda-forge openmmforcefields\nPython import error message: {errormessage}"
        ) from errormessage
    logger.info("")

    # How to read in file
    if molfile:
        # NOTE: Not well tested.
        logger.info("Mol file provided: %s", molfile)
        molecule = Molecule.from_file(molfile)
    elif sdffile:
        # NOTE: Not well tested.
        logger.info("SDF file provided %s", sdffile)
        molecule = Molecule.from_file(sdffile)
    elif smiles_string:
        # NOTE:
        logger.info("SMILES string provided: %s", smiles_string)
        # Create an OpenFF Molecule object from SMILES string
        molecule = Molecule.from_smiles(smiles_string, allow_undefined_stereo=allow_undefined_stereo)
        logger.info(
            "A SMILES string input means that no coordinate information is available. PDB-file created will have dummy coordinates that you have to fill in yourself."
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

        # OLD silly way: convert XYZ-to-PDB and then PDB-to-SMILES
        # Create a SMILES string from PDB-file

        # Create an OpenFF Molecule object from SMILES string
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

        # Create a SMILES string from PDB-file
        # Create an OpenFF Molecule object from SMILES string
    else:
        raise InputError("No inputfile provided. Exiting")

    # Changing residue name in molecule object (for each atom)
    # Affects both PDB-file and XML-file
    for atom in molecule.atoms:
        atom.metadata["residue_name"] = resname
    logger.info("Conversion to OpenFF molecule object successful")
    # NOTE: problem writing proper PDB-file here. Using OpenMM instead below

    # Create an OpenMM ForceField object
    logger.info("Now creating an Amber14 compatible OpenMM ForceField object")
    forcefield = ForceField("amber/protein.ff14SB.xml", "amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml")

    if forcefield_option == "GAFF":
        logger.info("GAFF forcefield chosen")
        # Create the GAFF template generator
        gaff = GAFFTemplateGenerator(molecules=molecule, forcefield=gaffversion)
        logger.info("GAFF version used: %s", gaff.gaff_version)

        # Register the GAFF template generator
        logger.info("Now registering the GAFF template generator in Forcefield object")
        forcefield.registerTemplateGenerator(gaff.generator)

        # Parameterize an OpenMM Topology object that contains the specified molecule.
        # Forcefield will load the appropriate GAFF parameters when needed, and antechamber
        # will be used to generate small molecule parameters on the fly.

        # Create system from PDB topology
        # Topology:
        # Option 1: pdb_obj.topology
        # Option 2:
        topology = openff.toolkit.topology.Topology.from_molecules([molecule])
        topology_openmm = topology.to_openmm()
        topology = topology_openmm

        # Creating OpenMM system both to check that things works and for passing to Parmed for XML writing
        system = forcefield.createSystem(topology)

        # Write XML-file for ligand using Parmed
        final_xmlfilename = f"gaff_{resname}.xml"
        write_xmlfile_parmed(topology, system, final_xmlfilename)

    elif forcefield_option == "OpenFF":
        import openff
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator

        smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule, forcefield=openff_file)

        forcefield = ForceField(
            "amber/protein.ff14SB.xml", "amber/tip3p_standard.xml", "amber/tip3p_HFE_multivalent.xml"
        )
        # Register the SMIRNOFF template generator
        forcefield.registerTemplateGenerator(smirnoff.generator)

        # Alternative: Create system from PDB topology

        topology = openff.toolkit.topology.Topology.from_molecules([molecule])
        topology_openmm = topology.to_openmm()
        topology = topology_openmm

        # Creating OpenMM system both to check that things works and for passing to Parmed for XML writing
        system = forcefield.createSystem(topology)

        # Write XML-file for ligand using Parmed
        final_xmlfilename = f"openff_{resname}.xml"
        write_xmlfile_parmed(topology, system, final_xmlfilename)

    # Create PDB-file that matches xml-file
    logger.info("Now creating a PDB-file that matches the XML-file")
    # Getting Cartesian coordinates from molecule
    pos = [openmm.Vec3(i[0]._magnitude, i[1]._magnitude, i[2]._magnitude) for i in molecule._conformers[0]]
    with open(f"{resname}.pdb", "w") as pdbfh:
        openmm.app.PDBFile.writeFile(topology, pos * openmm.unit.angstrom, pdbfh)

    # Now we have created an OpenMM system based on ligand Forcefield and created an XML-file
    logger.info("")
    logger.info("")
    logger.info("%s", "-" * 100)
    # Modifying XML-files
    logger.info("A new XML-file for molecule has been created: %s", final_xmlfilename)
    logger.info(
        f"Modifying 1-4 scaling parameters in XML-file to match Amber14 FF (coul14={expected_coul14}  and lj14={expected_lj14})"
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
            f"abs(coulomb_xml - coulomb_sys): {abs(coulomb_xml - coulomb_sys)}\nProblem with Coulomb-14 scaling in XML-file"
        )
    if abs(lj_xml - lj_sys) > 1e-5:
        raise InputError(f"abs(lj_xml - lj_system): {abs(lj_xml - lj_sys)}\nProblem with LJ-14 scaling in XML-file")
    logger.info("XML-file and forcefield objects are consistent. All good!")
    #
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
        f"The XML-file just created: {final_xmlfilename} can also be used directly (recommended only together with Amber14)\n"
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
    logger.info(
        "\nWarning: Make sure that the ligand has the same atom order in the large-system PDB-file \nas in the \
file that was used in this function."
    )
    logger.info("Additionally the ligand requires correct CONECT record lines in that same PDB-file")
    logger.info(f"A {resname}.pdb file has been created that is compatible with the XML-file")
    logger.info("%s", "-" * 100)
    return forcefield


# Function to create system from XML-file and check whether the 14 scaling in the OpenMM system is consistent with expected scaling
def create_sys_and_check_14_scaling_nonbonding(
    topology=None, xml_file=None, system=None, expected_coul14=0.833333, expected_lj14=0.5
):

    logger.info("Creating system from XML-file and topology")
    if topology is None:
        raise InputError("Error: topology is required if system is not provided")
    if xml_file is None:
        raise InputError("Error: xml_file is required if system is not provided")
    forcefield_from_xmlfile = openmm.app.ForceField(xml_file)
    system_from_xmlfile = forcefield_from_xmlfile.createSystem(topology)

    # Find NonbondedForce in system_from_xmlfile
    for force in system_from_xmlfile.getForces():
        if isinstance(force, openmm.NonbondedForce):
            break

    # Looping over exceptions
    for exception_index in range(force.getNumExceptions()):
        # Get the pair parameters
        atom1, atom2, qq, _sigma, epsilon = force.getExceptionParameters(exception_index)
        # if 0.0 then should be 1-2 or 1-3 interaction
        if epsilon._value == 0.0:
            continue
        # Get the particle parameters in the pair
        q1, sigma1, epsilon1 = force.getParticleParameters(atom1)
        q2, sigma2, epsilon2 = force.getParticleParameters(atom2)

        # Calculate expected value based on expected scaling factors
        expected_qq = expected_coul14 * q1 * q2
        expected_epsilon = expected_lj14 * (epsilon1 * epsilon2) ** 0.5

        # Checking deviations
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


# Function to check the nonbonded energy of exceptions of an OpenMM system
def calc_nonbonding_energy_exceptions(system=None):

    # Find NonbondedForce in system
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            break
    # Coulomb and LJ energies
    coulomb_energy = 0.0
    lj_energy = 0.0

    # Looping over exceptions
    for exception_index in range(force.getNumExceptions()):
        # Get the pair parameters
        _atom1, _atom2, qq, _sigma, epsilon = force.getExceptionParameters(exception_index)
        # if 0.0 then should be 1-2 or 1-3 interaction
        if epsilon._value == 0.0:
            continue

        coulomb_energy += qq.value_in_unit(openmm.unit.elementary_charge**2)
        lj_energy += epsilon.value_in_unit(openmm.unit.kilojoule_per_mole)

        # Return Coulomb energy and LJ energy
    return coulomb_energy, lj_energy


# Function that uses parmed to write an XML-file topology and OpenMM system
# Warning: Nonbonded 14 scaling requires modification after writing
def write_xmlfile_parmed(topology, system, xmlfilename):
    # Load Parmed
    logger.info("Using Parmed to read topologyfiles")
    try:
        import parmed
    except ImportError:
        raise MissingDependencyError(
            "Problem importing parmed Python library\nMake sure parmed is present in your Python.\nParmed can be installed using pip: pip install parmed"
        ) from None
    st = parmed.openmm.load_topology(topology, system=system)
    w = parmed.amber.parameters.ParameterSet.from_structure(st)
    ww = parmed.openmm.parameters.OpenMMParameterSet.from_parameterset(w)
    ww.residues.update(parmed.modeller.ResidueTemplateContainer.from_structure(st).to_library())
    ww.write(xmlfilename)
    logger.info("Wrote XML-file: %s", xmlfilename)
