import contextlib
import glob
import logging
import os
import shutil
import subprocess as sp
import time

import numpy as np

import openmmqmmm.constants
import openmmqmmm.coords
import openmmqmmm.elstructure
import openmmqmmm.parallel
from openmmqmmm.coords import _print_internal_coordinate_table, check_charge_mult
from openmmqmmm.exceptions import (
    ExternalProgramError,
    FileFormatError,
    InputError,
)
from openmmqmmm.utils import (
    insert_line_into_file,
    listdiff,
    log_time_since,
    main_header,
    pygrep,
    pygrep2,
    search_list_of_lists_for_index,
)

logger = logging.getLogger(__name__)


# ORCA Theory object.
class ORCATheory:
    def __init__(
        self,
        orcadir=None,
        orcasimpleinput="",
        basis_per_element=None,
        extrabasisatoms=None,
        extrabasis=None,
        atom_specific_basis_dict=None,
        ecp_dict=None,
        tddft=False,
        tddft_roots=5,
        follow_root=1,
        orcablocks="",
        extraline="",
        first_iteration_input=None,
        brokensym=None,
        hs_mult=None,
        atomstoflip=None,
        numcores=1,
        nprocs=None,
        label="ORCA",
        moreadfile=None,
        moreadfile_always=False,
        bind_to_core_option=True,
        ignore_orca_error=False,
        autostart=True,
        propertyblock=None,
        save_output_with_label=False,
        keep_each_run_output=False,
        print_population_analysis=False,
        filename="orca",
        check_for_errors=True,
        check_for_warnings=True,
        fragment_indices=None,
        xdm=False,
        xdm_a1=None,
        xdm_a2=None,
        xdm_func=None,
        nmf=False,
        nmf_sigma=None,
        cpcm_radii=None,
        rohf_uhf_swap=False,
        delta_scf=False,
        delta_scf_pmom=False,
        delta_scf_confline=None,
        delta_scf_turn_off_automatically=True,
    ):
        logger.info(main_header("ORCATheory initialization"))

        self.theorynamelabel = "ORCA"
        self.theorytype = "QM"
        self.analytic_hessian = True

        # Making sure we have a working ORCA installation
        self.orcadir = find_orca(orcadir)
        # Checking OpenMPI
        if numcores != 1:
            logger.info(
                f"ORCA parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version (for the ORCA version) is available in your environment"
            )
            openmmqmmm.parallel.check_openmpi()

        # Bind to core option when calling ORCA: i.e. execute: /path/to/orca file.inp "--bind-to none"
        # TODO: Default False; make True?
        self.bind_to_core_option = bind_to_core_option
        logger.info("bind_to_core_option: %s", self.bind_to_core_option)

        # Checking if user added Opt, Freq keywords
        if " OPT" in orcasimpleinput.upper() or " FREQ" in orcasimpleinput.upper():
            raise InputError(
                f"Error. orcasimpleinput variable can not contain ORCA job-directives like: Opt, Freq, Numfreq\nString: {orcasimpleinput.upper()}\norcasimpleinput should only contain information on electronic-structure method (e.g. functional), basis set, grid, SCF convergence etc."
            )
        if "!" not in orcasimpleinput:
            raise InputError(
                "Error. orcasimpleinput should contain at least a '!' with method and basis set information"
            )

        # Whether to check ORCA outputfile for errors and warnings or not
        # Generally recommended. Could be disabled to speed up I/O a tiny bit
        self.check_for_errors = check_for_errors
        self.check_for_warnings = check_for_warnings

        # Counter for how often ORCATheory.run is called
        self.runcalls = 0

        # Whether to keep the ORCA outputfile for each run as orca_runX.out
        self.keep_each_run_output = keep_each_run_output
        # Whether to save ORCA outputfile with given label
        if save_output_with_label is True and label is None:
            raise InputError("Error: save_output_with_label option requires a label keyword also")
        else:
            self.save_output_with_label = save_output_with_label

        # Print population_analysis in each run
        self.print_population_analysis = print_population_analysis

        # Label to distinguish different ORCA objects
        self.label = label

        # Create inputfile with generic name
        self.filename = filename

        # Whether to exit ORCA if subprocess command faile
        self.ignore_orca_error = ignore_orca_error

        # MOREAD-file
        self.moreadfile = moreadfile
        self.moreadfile_always = moreadfile_always
        # Autostart
        self.autostart = autostart
        # Each ORCA calculation will save path to last GBW-file used in case we have switched directories
        # and we want to use last one
        self.path_to_last_gbwfile_used = None  # default None

        # Printlevel

        # TDDFT
        self.tddft = tddft
        self.tddft_roots = tddft_roots
        self.follow_root = follow_root

        # Setting numcores of object
        # NOTE: nprocs is deprecated but kept on for a bit
        if nprocs is None:
            self.numcores = numcores
        else:
            self.numcores = nprocs

        # Property block. Added after coordinates unless None
        self.propertyblock = propertyblock

        # Store optional properties of ORCA run job in a dict
        self.properties = {}

        # Adding NoAutostart keyword to extraline if requested
        if self.autostart is False:
            self.extraline = extraline + "\n! Noautostart\n"
        else:
            self.extraline = extraline

        # Inputfile definitions
        self.orcasimpleinput = orcasimpleinput
        self.orcablocks = orcablocks

        # Input-lines only for first run call
        if first_iteration_input is not None:
            self.first_iteration_input = first_iteration_input
        else:
            self.first_iteration_input = ""

        # BROKEN SYM OPTIONS
        self.brokensym = brokensym
        self.hs_mult = hs_mult
        if isinstance(atomstoflip, int):
            raise InputError(
                "Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer."
            )
        # Add UKS if not present for broken-symmetry jobs
        if self.brokensym is True and "UKS" not in self.orcasimpleinput and "UHF" not in self.orcasimpleinput:
            logger.info("Warning: UKS/UHF keyword not present in orcasimpleinput for BS job. Adding.")
            self.orcasimpleinput = self.orcasimpleinput + " UKS"
        if atomstoflip is not None:
            self.atomstoflip = atomstoflip
        else:
            self.atomstoflip = []
        # DELTASCF
        self.delta_scf = delta_scf
        self.delta_scf_pmom = delta_scf_pmom
        self.delta_scf_confline = delta_scf_confline
        self.delta_scf_turn_off_automatically = delta_scf_turn_off_automatically
        if self.delta_scf is True and self.delta_scf_confline is None:
            raise InputError("Error: DELTASCF is True but no deltaSCF_confline provided. Exiting")
        if self.delta_scf is True:
            logger.info("DeltaSCF True, turning on population analysis printing")
            self.print_population_analysis = True

        # Basis sets per element
        self.basis_per_element = basis_per_element
        if self.basis_per_element is not None:
            logger.info("Basis set dictionary for each element provided: %s", basis_per_element)

        # Extrabasis: add specific basis set keyword to certain atoms
        if extrabasisatoms is not None:
            self.extrabasisatoms = extrabasisatoms
            self.extrabasis = extrabasis
        else:
            self.extrabasisatoms = []
            self.extrabasis = ""
        # Atom-specific basis set options
        # Within ORCA inputfile, define a basis set for each and every atom. Requires a dictionary with element as key and basis set as value
        self.atom_specific_basis_dict = atom_specific_basis_dict
        self.ecp_dict = ecp_dict  # ECP dict that usually goes with atom_specific dict

        # Used in the case of counterpoise calculations
        self.ghostatoms = []  # Adds ":" in front of element in coordinate block. Have basis functions and grid points
        self.dummyatoms = []  # Adds DA instead of element. No real atom

        # For ORCA calculations that define fragments within molecule
        self.fragment_indices = fragment_indices

        # self.qmatoms need to be set for Flipspin to work for QM/MM job.
        # Overwritten by QMMMtheory, used in Flip-spin
        self.qmatoms = []

        # Whether to keep a copy of last output (filename_last.out) or not
        self.keep_last_output = True

        # NMF
        self.nmf = nmf
        if self.nmf is True:
            if nmf_sigma is None:
                raise InputError("NMF option requires setting NMF_sigma")
            self.nmf_sigma = nmf_sigma

            logger.info("NMF option is active. Will activate Fermi-smearing in ORCA input!")
            NMF_smeartemp = self.nmf_sigma / openmmqmmm.constants.R_gasconst
            logger.info(f"NMF_smeartemp = {NMF_smeartemp} calculated from NMF_sigma: {self.nmf_sigma}:")
            self.orcablocks = (
                self.orcablocks
                + f"""
%scf
fracocc true
smeartemp {NMF_smeartemp}
end
            """
            )

        # TDDFT option
        # If gradient requested by Singlepoint(Grad=True) or Optimizer then TDDFT gradient is calculated instead
        if self.tddft is True and "%tddft" not in self.orcablocks:
            self.orcablocks = (
                self.orcablocks
                + f"""
%tddft
nroots {self.tddft_roots}
IRoot {self.follow_root}
end
"""
            )
        # ROHF-UHF swap
        self.rohf_uhf_swap = rohf_uhf_swap

        # Specific CPCM radii. e.g. to use DRACO radii
        if cpcm_radii is not None:
            logger.info("CPCM radii provided: %s", cpcm_radii)
            # if len(cpcm_radii) != len(c:
            cpcm_block = "%cpcm\n"
            for i, radius in enumerate(cpcm_radii):
                cpcm_block = cpcm_block + f"AtomRadii({i},  {radius})\n"
            cpcm_block = cpcm_block + "end\n"
            logger.info("cpcm_block: %s", cpcm_block)
            self.orcablocks = self.orcablocks + cpcm_block

        # XDM: if True then we add !AIM to input
        self.xdm = False
        if xdm is True:
            self.xdm = True
            self.xdm_a1 = xdm_a1
            self.xdm_a2 = xdm_a2
            self.xdm_func = xdm_func
            self.orcasimpleinput = self.orcasimpleinput + " AIM"

        logger.info("")
        logger.info("Creating ORCA object")
        logger.info("ORCA dir: %s", self.orcadir)
        logger.info("%s", self.orcasimpleinput)
        logger.info("%s", self.orcablocks)
        logger.info("\nORCATheory object created!")

    # Set numcores method
    def set_numcores(self, numcores):
        self.numcores = numcores

    # Cleanup after run.
    def cleanup(self):
        logger.info("Cleaning up old ORCA files")
        list_files = []
        # Keeping outputfiles
        list_files.append(self.filename + ".gbw")
        list_files.append(self.filename + ".densities")
        list_files.append(self.filename + ".ges")
        list_files.append(self.filename + ".prop")
        list_files.append(self.filename + ".uco")
        list_files.append(self.filename + "_property.txt")
        list_files.append(self.filename + ".inp")
        list_files.append(self.filename + ".engrad")
        list_files.append(self.filename + ".cis")
        list_files.append(self.filename + "_last.out")
        list_files.append(self.filename + ".xyz")
        for file in list_files:
            with contextlib.suppress(FileNotFoundError):
                os.remove(file)
        try:
            for tmpfile in glob.glob("self.filename*tmp"):
                os.remove(tmpfile)
        except FileNotFoundError:
            pass

    # Do an ORCA-optimization instead of geomeTRIC optimization. Useful for gas-phase chemistry when ORCA-optimizer is better than geomeTRIC
    def opt(self, fragment=None, grad=None, hessian=None, numcores=None, charge=None, mult=None):

        module_init_time = time.time()
        logger.info("------------RUNNING INTERNAL ORCA OPTIMIZATION-------------")
        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:

        if fragment is None:
            raise InputError("No fragment provided to Opt.")
        else:
            logger.info("Fragment provided to Opt")

        current_coords = fragment.coords
        elems = fragment.elems
        # Check charge/mult
        charge, mult = check_charge_mult(charge, mult, self.theorytype, fragment, "ORCATheory.Opt", theory=self)

        if charge is None or mult is None:
            raise InputError("Error. charge and mult has not been defined for ORCATheory.Opt method")

        if numcores is None:
            numcores = self.numcores

        self.extraline = self.extraline + "\n! OPT "

        logger.info(f"Running ORCA with {numcores} cores available")
        logger.info("Object label: %s", self.label)

        logger.info("Creating inputfile: %s", self.filename + ".inp")
        logger.info("ORCA input:")
        logger.info("%s", self.orcasimpleinput)
        logger.info("%s", self.extraline)
        logger.info("%s", self.orcablocks)
        if self.propertyblock is not None:
            logger.info("%s", self.propertyblock)
        logger.info(f"Charge: {charge}  Mult: {mult}")

        # TODO: Make more general
        create_orca_input_plain(
            self.filename,
            elems,
            current_coords,
            self.orcasimpleinput,
            self.orcablocks,
            charge,
            mult,
            extraline=self.extraline,
            hs_mult=self.hs_mult,
            moreadfile=self.moreadfile,
        )
        logger.info(f"ORCA Calculation started using {numcores} CPU cores")
        run_orca_sp_parallel(
            self.orcadir,
            self.filename + ".inp",
            numcores=numcores,
            bind_to_core_option=self.bind_to_core_option,
            ignore_orca_error=self.ignore_orca_error,
        )
        logger.info("ORCA Calculation done.")

        outfile = self.filename + ".out"
        ORCAfinished, _iter = check_orca_finished(outfile)
        if ORCAfinished:
            logger.info("ORCA job finished")
            if check_orca_opt_finished(outfile):
                logger.info("ORCA geometry optimization finished")
                self.energy = grab_orca_final_energy(outfile)
                # Grab optimized coordinates from filename.xyz
                _opt_elems, opt_coords = openmmqmmm.coords.read_xyzfile(self.filename + ".xyz")
                logger.info("%s", opt_coords)

                fragment.replace_coords(fragment.elems, opt_coords)
            else:
                raise ExternalProgramError("ORCA optimization failed to converge. Check ORCA output")
        else:
            raise ExternalProgramError("Something happened with ORCA job. Check ORCA output")

        logger.info("ORCA optimized energy: %s", self.energy)
        logger.info("fragment updated: %s", fragment)
        fragment.print_coords()
        # Writing out fragment file and XYZ file
        fragment.print_system(filename="fragment_optimized.frag")
        fragment.write_xyzfile(xyzfilename="Fragment-optimized.xyz")

        # Printing internal coordinate table
        _print_internal_coordinate_table(fragment)
        log_time_since(module_init_time, "ORCA Opt-run")
        return

    # Method to grab dipole moment from an ORCA outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        dm = grab_dipole_moment(self.filename + ".out")
        logger.info("Dipole moment: %s", dm)
        return dm

    def get_polarizability_tensor(self):
        logger.info("here")
        logger.info("self.filename+'.out': %s", self.filename + ".out")
        polarizability, _diag_pz = grab_polarizability_tensor(self.filename + ".out")
        logger.info("polarizability: %s", polarizability)
        return polarizability

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(
        self,
        current_coords=None,
        charge=None,
        mult=None,
        current_mm_coords=None,
        mm_charges=None,
        qm_elems=None,
        elems=None,
        grad=False,
        hessian=False,
        pc=False,
        numcores=None,
        label=None,
    ):
        module_init_time = time.time()
        self.runcalls += 1
        logger.info("------------RUNNING ORCA INTERFACE-------------")
        logger.info("Object-label: %s", self.label)
        logger.info("Run-label: %s", label)
        # Coords provided to run
        if current_coords is not None:
            pass
        else:
            raise InputError("Error:no current_coords")

        # Checking if charge and mult has been provided
        if charge is None or mult is None:
            raise InputError("Error. charge and mult has not been defined for ORCATheory.run method")

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                raise InputError("No elems provided")
            else:
                qm_elems = elems

        # If QM/MM then atomindices lists like extrabasisatoms, atomstoflip and fragment_indices have to be updated
        if len(self.qmatoms) != 0:
            # Fragment indices need to be updated if QM/MM
            if self.fragment_indices is not None:
                fragment_indices = []
                for f in self.fragment_indices:
                    temp = [self.qmatoms.index(i) for i in f]
                    fragment_indices.append(temp)
            else:
                fragment_indices = self.fragment_indices
            # extrabasisatomindices if QM/MM
            qmatoms_extrabasis = [self.qmatoms.index(i) for i in self.extrabasisatoms]
            # new QM-region indices for atomstoflip if QM/MM
            try:
                qmatomstoflip = [self.qmatoms.index(i) for i in self.atomstoflip]
            except ValueError:
                raise InputError(
                    f"Atoms to flip: {self.atomstoflip}\nError: Atoms to flip are not all in QM-region"
                ) from None
        else:
            qmatomstoflip = self.atomstoflip
            qmatoms_extrabasis = self.extrabasisatoms
            fragment_indices = self.fragment_indices

        if numcores is None:
            numcores = self.numcores

        # Basis set definition per element from input dict
        if self.basis_per_element is not None:
            basisstring = ""
            for el, b in self.basis_per_element.items():
                basisstring += f'newgto {el} "{b}" end\n'
            basisblock = f"""
%basis
{basisstring}
end"""

            if basisblock not in self.orcablocks:
                self.orcablocks = self.orcablocks + basisblock

        # If ECP-dict provided (often goes with atom_specific_basis_dict)
        if self.ecp_dict is not None:
            bstring = ""
            for b in self.ecp_dict.values():
                for x in b:
                    bstring += f"{x}"
            ecpbasisblock = f"""
%basis
{bstring}
end"""
            if ecpbasisblock not in self.orcablocks:
                self.orcablocks = self.orcablocks + ecpbasisblock

        logger.info(f"Running ORCA with {numcores} cores available")

        # MOREAD. Checking file provided exists and determining what to do if not
        if self.moreadfile is not None:
            logger.info(f"Moreadfile option active. File path: {self.moreadfile}")
            if os.path.isfile(self.moreadfile) is True:
                logger.info(f"File exists in current directory: {os.getcwd()}")
            else:
                logger.info(f"File does not exist in current directory: {os.getcwd()}")
                if os.path.isabs(self.moreadfile) is True:
                    raise FileFormatError("Error: Absolute path provided but file does not exists. Exiting")
                else:
                    logger.info("Checking if file exists in parentdir instead:")
                    if os.path.isfile(f"../{self.moreadfile}") is True:
                        logger.info("Yes. Copying file to current dir")
                        shutil.copy(f"../{self.moreadfile}", f"./{self.moreadfile}")
        else:
            logger.info("Moreadfile option not active")
            if os.path.isfile(f"{self.filename}.gbw") is False:
                logger.info(f"No {self.filename}.gbw file is present in dir.")
                if self.path_to_last_gbwfile_used is not None:
                    logger.info(
                        f"Found a path ({self.path_to_last_gbwfile_used}) to last GBW-file used by this Theory object. Will try to copy this file do current dir"
                    )
                    try:
                        shutil.copy(self.path_to_last_gbwfile_used, f"./{self.filename}.gbw")
                    except FileNotFoundError:
                        logger.info("File was not found. May have been deleted")
                    if self.autostart is False:
                        logger.info("Autostart option is False. ORCA will ignore this file")
                    else:
                        logger.info("Autostart feature is active. ORCA will read GBW-file present.")
                else:
                    logger.info(f"Checking if a file {self.filename}.gbw exists in parentdir:")
                    if os.path.isfile(f"../{self.filename}.gbw") is True:
                        logger.info("Yes. Copying file from parentdir to current dir")
                        shutil.copy(f"../{self.filename}.gbw", f"./{self.filename}.gbw")
                    else:
                        logger.info("Found no file. ORCA will guess new orbitals")
            else:
                logger.info(f"A GBW-file with same basename : {self.filename}.gbw is present")
                if self.autostart is False:
                    logger.info("Autostart is False. ORCA will ignore any file present")
                else:
                    logger.info("Autostart feature is active. ORCA will read GBW-file present.")

        # If 1st runcall, add first_iteration_input to inputfile
        extraline = self.extraline + "\n" + self.first_iteration_input if self.runcalls == 1 else self.extraline

        logger.info("Creating inputfile: %s", self.filename + ".inp")
        logger.info("ORCA input:")
        logger.info("%s", self.orcasimpleinput)
        logger.info("%s", extraline)
        logger.info("%s", self.orcablocks)
        logger.info(f"Charge: {charge}  Mult: {mult}")
        # Printing extra options chosen:
        if self.brokensym is True:
            logger.info(f"Brokensymmetry SpinFlipping on! HSmult: {self.hs_mult}.")

            if self.hs_mult is None:
                raise InputError("Error:HSmult keyword in ORCATheory has not been set. This is required. Exiting.")
            if len(qmatomstoflip) == 0:
                raise InputError("Error: atomstoflip keyword needs to be set. This is required. Exiting.")

            for flipatom, qmflipatom in zip(self.atomstoflip, qmatomstoflip, strict=False):
                logger.info(f"Flipping atom: {flipatom} QMregionindex: {qmflipatom} Element: {qm_elems[qmflipatom]}")
        # DeltaSCF
        deltascfblock = None
        if self.delta_scf is True:
            logger.info("DeltaSCF option chosen. Will attempt MOM excited state SCF solution in first run")
            logger.info("DeltaSCF PMOM: %s", self.delta_scf_pmom)
            logger.info("Configuration line: %s", self.delta_scf_confline)
            if mult == 1 and "UKS" not in self.orcasimpleinput and "UHF" not in self.orcasimpleinput:
                logger.info("Warning: Singlet DeltaSCF calculation requested but no UKS/UHF keyword present.")
                logger.info("Only doubly excited SCF states can be found ")

            deltascfblock = f"! DELTASCF \n%scf\n PMOM {self.delta_scf_pmom} \n {self.delta_scf_confline}\nend"

        if self.extrabasis != "":
            logger.info(f"Using extra basis ({self.extrabasis}) on QM-region indices : {qmatoms_extrabasis}")
        if self.dummyatoms:
            logger.info("Dummy atoms defined: %s", self.dummyatoms)
        if self.ghostatoms:
            logger.info("Ghost atoms defined: %s", self.ghostatoms)
        if self.fragment_indices:
            logger.info("List of fragment indices defined: %s", fragment_indices)

        if pc is True:
            logger.info("Pointcharge embedding is on!")
            create_orca_pcfile(self.filename, current_mm_coords, mm_charges)
            if self.brokensym is True:
                create_orca_input_pc(
                    self.filename,
                    qm_elems,
                    current_coords,
                    self.orcasimpleinput,
                    self.orcablocks,
                    charge,
                    mult,
                    extraline=extraline,
                    hs_mult=self.hs_mult,
                    grad=grad,
                    hessian=hessian,
                    moreadfile=self.moreadfile,
                    atomstoflip=qmatomstoflip,
                    extrabasisatoms=qmatoms_extrabasis,
                    extrabasis=self.extrabasis,
                    propertyblock=self.propertyblock,
                    fragment_indices=fragment_indices,
                    atom_specific_basis_dict=self.atom_specific_basis_dict,
                    rohf_uhf_swap=self.rohf_uhf_swap,
                    delta_scf_block=deltascfblock,
                )
            else:
                create_orca_input_pc(
                    self.filename,
                    qm_elems,
                    current_coords,
                    self.orcasimpleinput,
                    self.orcablocks,
                    charge,
                    mult,
                    extraline=extraline,
                    grad=grad,
                    hessian=hessian,
                    moreadfile=self.moreadfile,
                    extrabasisatoms=qmatoms_extrabasis,
                    extrabasis=self.extrabasis,
                    propertyblock=self.propertyblock,
                    fragment_indices=fragment_indices,
                    atom_specific_basis_dict=self.atom_specific_basis_dict,
                    rohf_uhf_swap=self.rohf_uhf_swap,
                    delta_scf_block=deltascfblock,
                )
        else:
            if self.brokensym is True:
                create_orca_input_plain(
                    self.filename,
                    qm_elems,
                    current_coords,
                    self.orcasimpleinput,
                    self.orcablocks,
                    charge,
                    mult,
                    extraline=extraline,
                    hs_mult=self.hs_mult,
                    grad=grad,
                    hessian=hessian,
                    moreadfile=self.moreadfile,
                    atomstoflip=qmatomstoflip,
                    extrabasisatoms=qmatoms_extrabasis,
                    extrabasis=self.extrabasis,
                    propertyblock=self.propertyblock,
                    ghostatoms=self.ghostatoms,
                    dummyatoms=self.dummyatoms,
                    rohf_uhf_swap=self.rohf_uhf_swap,
                    fragment_indices=fragment_indices,
                    atom_specific_basis_dict=self.atom_specific_basis_dict,
                    delta_scf_block=deltascfblock,
                )
            else:
                create_orca_input_plain(
                    self.filename,
                    qm_elems,
                    current_coords,
                    self.orcasimpleinput,
                    self.orcablocks,
                    charge,
                    mult,
                    extraline=extraline,
                    grad=grad,
                    hessian=hessian,
                    moreadfile=self.moreadfile,
                    extrabasisatoms=qmatoms_extrabasis,
                    extrabasis=self.extrabasis,
                    propertyblock=self.propertyblock,
                    ghostatoms=self.ghostatoms,
                    dummyatoms=self.dummyatoms,
                    rohf_uhf_swap=self.rohf_uhf_swap,
                    fragment_indices=fragment_indices,
                    atom_specific_basis_dict=self.atom_specific_basis_dict,
                    delta_scf_block=deltascfblock,
                )

        # Run inputfile using ORCA parallelization. Take numcores argument.
        logger.info("ORCA Calculation starting.")

        run_orca_sp_parallel(
            self.orcadir,
            self.filename + ".inp",
            numcores=numcores,
            bind_to_core_option=self.bind_to_core_option,
            check_for_errors=self.check_for_errors,
            check_for_warnings=self.check_for_warnings,
            ignore_orca_error=self.ignore_orca_error,
        )
        logger.info("ORCA Calculation done.")

        outfile = self.filename + ".out"
        engradfile = self.filename + ".engrad"
        pcgradfile = self.filename + ".pcgrad"

        # Checking if finished.
        if self.ignore_orca_error is False:
            ORCAfinished, numiterations = check_orca_finished(outfile)
            # Check if ORCA finished or not. Exiting if so
            if ORCAfinished is False:
                logger.error("Problem with ORCA run")
                logger.info("------------ENDING ORCA-INTERFACE-------------")
                log_time_since(module_init_time, "ORCA run")
                raise ExternalProgramError(
                    f"ORCA calculation did not terminate normally - check the output file: {outfile}"
                )

            logger.info(f"ORCA converged in {numiterations} iterations")
        else:
            logger.info("There was an ORCA error that was ignored by user-input")

        if self.rohf_uhf_swap:
            logger.info("\nROHF UHF swap feature active.")
            logger.info("This means that a $new_job ORCA job was run with a ROHF-UHF noiter switch")
            logger.info(f"Note that the relevant GBW file is then: {self.filename}_job2.gbw\n")
            logger.info("Stored as self.gbwfile of this ORCATheory object")
            self.gbwfile = self.filename + "_job2.gbw"
        else:
            self.gbwfile = self.filename + ".gbw"

        # Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart, etc.)
        # using this theory object
        if self.brokensym is True:
            logger.info(
                "ORCA Flipspin calculation done. Now turning off brokensym in ORCA object for possible future calculations"
            )
            self.brokensym = False
        # Turning off deltaSCF for future calcs
        if self.delta_scf is True:
            logger.info("DeltaSCF calculation done.")
            if self.delta_scf_turn_off_automatically is True:
                logger.info(
                    "deltaSCF_turn_off_automatically option is True. Turning off DELTASCF for future calculations."
                )
                self.delta_scf = False
                deltascfblock = None
                if "nososcf" not in self.orcasimpleinput:
                    logger.info(
                        "Adding NOSOSCF to orcasimpleinput to avoid future calculations from falling back to ground-state"
                    )
                    self.orcasimpleinput = self.orcasimpleinput + " nososcf"
                if "nodamp" not in self.orcasimpleinput:
                    logger.info(
                        "Adding NODAMP to orcasimpleinput to avoid future calculations from falling back to ground-state"
                    )
                    self.orcasimpleinput = self.orcasimpleinput + " nodamp"
                if "nolshift" not in self.orcasimpleinput:
                    logger.info(
                        "Adding NOLSHIFT to orcasimpleinput to avoid future calculations from falling back to ground-state"
                    )
                    self.orcasimpleinput = self.orcasimpleinput + " nolshift"
            else:
                logger.info("deltaSCF_turn_off_automatically option if False. Will keep DeltaSCF settings")

        # Now that we have possibly run a ORCA job with moreadfile we now turn the moreadfile option off
        #  as we probably want to use the orbitals we created
        if self.moreadfile is not None:
            logger.info("First ORCATheory calculation finished.")
            # Now either keeping moreadfile or removing it. Default: removing
            if not self.moreadfile_always:
                logger.info("Now turning moreadfile option off.")
                self.moreadfile = None

        # Optional save ORCA output with filename according to label
        if self.save_output_with_label is True:
            shutil.copy(self.filename + ".out", self.filename + f"_{self.label}_{charge}_{mult}.out")

        # Keep outputfile from each run if requested
        if self.keep_each_run_output is True:
            logger.info("\nkeep_each_run_output is True")
            logger.info(
                "%s", "Copying {} to {}".format(self.filename + ".out", self.filename + f"_run{self.runcalls}" + ".out")
            )
            shutil.copy(self.filename + ".out", self.filename + f"_run{self.runcalls}" + ".out")

        # Always make copy of last output file
        if self.keep_last_output is True:
            shutil.copy(self.filename + ".out", self.filename + "_last.out")

        # Save path to last GBW-file (used if the run changes directories, e.g. goes from NumFreq)
        self.path_to_last_gbwfile_used = f"{os.getcwd()}/{self.filename}.gbw"

        # Print population analysis in each run if requested
        if self.print_population_analysis is True:
            logger.info("\nPrinting Mulliken Population analysis:")
            logger.info("%s", "-" * 30)
            charges = grab_orca_atom_charges("Mulliken", self.filename + ".out")
            spinpops = grab_orca_spin_populations("Mulliken", self.filename + ".out")
            self.properties["Mulliken_charges"] = charges
            self.properties["Mulliken_spinpops"] = spinpops
            if len(spinpops) == 0 and len(charges) != 0:
                logger.info("%s", "{:<2} {:<2}  {:>10}".format(" ", " ", "Charge"))
                for i, (el, ch) in enumerate(zip(qm_elems, charges, strict=False)):
                    logger.info(f"{i:<2} {el:<2}: {ch:>10.4f}")
                logger.info("")
            elif len(spinpops) != 0 and len(charges) != 0:
                logger.info("%s", "{:<2} {:<2}  {:>10} {:>10}".format(" ", " ", "Charge", "Spinpop"))
                for i, (el, ch, sp) in enumerate(zip(qm_elems, charges, spinpops, strict=False)):
                    logger.info(f"{i:<2} {el:<2}: {ch:>10.4f} {sp:>10.4f}")
                logger.info("")
            else:
                logger.info("Warning: No charges or spinpops were found in ORCA output. Continuing")
        # Grab energy
        if self.ignore_orca_error is False:
            self.energy = grab_orca_final_energy(outfile)
            logger.info("ORCA energy: %s", self.energy)
        else:
            self.energy = grab_orca_final_energy(outfile)

            if self.energy is None:
                logger.info("No energy could be found in ORCA outputfile.")
                logger.info("Setting energy to 0.0 and returning")
                return 0.0
        # NMF
        if self.nmf is True:
            logger.info("NMF option is active.")
            E_NMF = self.energy
            occupations = np.array(grab_scf_fod_occupations(outfile))
            logger.info("Fractional ccupations (Fermi distribution): %s", occupations)
            logger.info("Now also calculating correlation energy from the fractional occupation numbers")
            logger.info("Assuming Fermi distribution")
            Ec = openmmqmmm.elstructure.get_ec_entropy(occupations, self.nmf_sigma, method="fermi")
            logger.info("Ec: %s", Ec)
            self.properties["NMF_occupations"] = occupations
            self.properties["E_NMF"] = E_NMF
            self.properties["NMF_Ec"] = Ec
            self.energy = self.energy + Ec

        # Grab possible properties
        # ICE-CI
        try:
            E_PT2_rest = float(pygrep("'rest' energy", self.filename + ".out")[-1])
            num_genCFGs, num_selected_CFGs, num_after_SD_CFGs = grab_ice_wf_cfg_ci_size(self.filename + ".out")
            self.properties["E_var"] = self.energy
            self.properties["E_PT2_rest"] = E_PT2_rest
            self.properties["num_genCFGs"] = num_genCFGs
            self.properties["num_selected_CFGs"] = num_selected_CFGs
            self.properties["num_after_SD_CFGs"] = num_after_SD_CFGs
        except Exception:  # noqa: BLE001 - best-effort: ICE-CI properties only in ICE-CI outputs
            pass

        # TDDFT results
        if self.tddft is True:
            transition_energies = tddftgrab(f"{self.filename}.out")
            transition_intensities = tddftintens_grab(f"{self.filename}.out")

            self.properties["TDDFT_transition_energies"] = transition_energies
            self.properties["TDDFT_transition_intensities"] = transition_intensities

        # Grab timings from ORCA output
        orca_timings = grab_orca_timings(outfile)

        # Initializing zero gradient array
        self.grad = np.zeros((len(qm_elems), 3))
        self.dipole_moment = None

        # XDM option: WFX file should have been created.
        if self.xdm:
            dispE, dispgrad = openmmqmmm.elstructure.xdm_run(
                wfxfile=self.filename + ".wfx", a1=self.xdm_a1, a2=self.xdm_a2, functional=self.xdm_func
            )
            logger.info("XDM dispersion energy: %s", dispE)
            self.energy = self.energy + dispE
            logger.info("DFT+XDM energy: %s", self.energy)
            # TODO: dispgrad not yet done
            self.grad = self.grad + dispgrad

        # Grab Hessian if calculated
        if hessian is True:
            logger.info("Reading Hessian from file: %s", self.filename + ".hess")
            self.hessian = grab_hessian(self.filename + ".hess")
            self.ir_intensities = grab_ir_intensities(self.filename + ".hess")

        if grad is True:
            grad = grab_orca_gradient(engradfile)
            self.grad = self.grad + grad
            logger.debug("ORCA gradient: %s", self.grad)

            if pc:
                # Print time to calculate ORCA QM-PC gradient
                if "pc_gradient" in orca_timings:
                    logger.info(
                        "%s", "Time calculating QM-Pointcharge gradient: {} seconds".format(orca_timings["pc_gradient"])
                    )
                # Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                self.pcgrad = grab_orca_pc_gradient(pcgradfile)
                logger.info("------------ENDING ORCA-INTERFACE-------------")
                log_time_since(module_init_time, "ORCA run")
                return self.energy, self.grad, self.pcgrad
            else:
                logger.info("------------ENDING ORCA-INTERFACE-------------")
                log_time_since(module_init_time, "ORCA run")
                return self.energy, self.grad

        else:
            logger.info("Single-point ORCA energy: %s", self.energy)
            logger.info("------------ENDING ORCA-INTERFACE-------------")
            log_time_since(module_init_time, "ORCA run")
            return self.energy


###############################################
# ORCA program discovery
###############################################


def _looks_like_orca_dir(directory):
    # A real ORCA installation ships the orca binary alongside orca_* helper
    # binaries (orca_scf, orca_gtoint, ...). Their absence identifies unrelated
    # programs that happen to be called orca (e.g. the GNOME screen reader).
    return os.path.isfile(os.path.join(directory, "orca")) and len(glob.glob(os.path.join(directory, "orca_*"))) > 0


def _orca_binary_runs(directory):
    # Argument-less orca exits immediately asking for a parameterfile; the
    # timeout guards against non-ORCA programs that block (a daemon would
    # otherwise hang this probe forever).
    orca_binary = os.path.join(directory, "orca")
    try:
        completed = sp.run([orca_binary], stdout=sp.PIPE, stderr=sp.STDOUT, timeout=15, check=False)
    except (OSError, sp.TimeoutExpired) as err:
        logger.info(f"ORCA binary {orca_binary} could not be executed ({err})")
        return False
    output = completed.stdout.decode(errors="replace")
    if "parameterfile" not in output:
        logger.info(f"ORCA binary {orca_binary} does not behave like the ORCA quantum chemistry program")
        return False
    return True


def find_orca(orcadir=None, required=True):
    """Locate a working ORCA installation directory.

    Search order: the explicit orcadir argument, the OPENMMQMMM_ORCADIR
    environment variable, then the directory containing an orca binary found
    in PATH. Every candidate is validated (orca_* helper binaries present and
    the orca binary executes) before being accepted.

    An invalid explicit location (argument or environment variable) is an
    error: failing loudly beats silently falling back to a different
    installation. An invalid PATH hit is merely skipped, since it is the
    incidental-collision case.

    Returns the installation directory, or None if nothing was found and
    required=False.
    """
    for source, directory in (
        ("orcadir argument", orcadir),
        ("OPENMMQMMM_ORCADIR environment variable", os.environ.get("OPENMMQMMM_ORCADIR")),
    ):
        if not directory:
            continue
        directory = os.path.expanduser(directory)
        if _looks_like_orca_dir(directory) and _orca_binary_runs(directory):
            logger.info(f"Using ORCA installation: {directory} (from {source})")
            return directory
        if required:
            raise ExternalProgramError(
                f"The {source} points at {directory}, which is not a working ORCA installation "
                "(orca binary plus orca_* helper binaries expected)"
            )
        return None

    orca_in_path = shutil.which("orca")
    if orca_in_path is not None:
        directory = os.path.dirname(os.path.realpath(orca_in_path))
        if _looks_like_orca_dir(directory) and _orca_binary_runs(directory):
            logger.info(f"Using ORCA installation: {directory} (found via PATH)")
            return directory
        logger.info(f"Note: ignoring {orca_in_path} from PATH - not the ORCA quantum chemistry program")

    if required:
        raise ExternalProgramError(
            "Found no working ORCA installation.\nPass orcadir= , set the OPENMMQMMM_ORCADIR environment variable, or put the orca binary in PATH"
        )
    return None


# Run ORCA single-point job using ORCA parallelization. Will add pal-block if numcores >1.
def run_orca_sp_parallel(
    orcadir,
    inpfile,
    numcores=1,
    check_for_warnings=True,
    check_for_errors=True,
    bind_to_core_option=True,
    ignore_orca_error=False,
):
    if numcores > 1:
        palstring = f"%pal \nnprocs {numcores}\nend"
        with open(inpfile):
            insert_line_into_file(inpfile, "!", palstring, once=True)
    basename = inpfile.replace(".inp", "")

    # LD_LIBRARY_PATH enforce: https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=10118
    # "-x LD_LIBRARY_PATH -x PATH"

    with open(basename + ".out", "w") as ofile:
        try:
            if bind_to_core_option is True:
                sp.run(
                    [orcadir + "/orca", inpfile, "--bind-to none"],
                    check=True,
                    stdout=ofile,
                    stderr=ofile,
                    text=True,
                )
            else:
                sp.run([orcadir + "/orca", inpfile], check=True, stdout=ofile, stderr=ofile, text=True)
            if check_for_errors:
                grab_orca_errors(basename + ".out")
            if check_for_warnings:
                grab_orca_warnings(basename + ".out")
        except Exception as e:
            logger.info("Subprocess error! Exception message: %s", e)

            # We get an exception if
            logger.error("Problem running ORCA. Something went wrong, most likely ORCA ran into an error.")
            logger.error(f"Please check the ORCA outputfile: {basename + '.out'} for error messages")
            logger.info("")
            if check_for_errors:
                grab_orca_errors(basename + ".out")
            if check_for_warnings:
                grab_orca_warnings(basename + ".out")
            logger.info("ignore_ORCA_error: %s", ignore_orca_error)
            if ignore_orca_error is True:
                logger.info("ignore_ORCA_error here")
                return
            else:
                raise ExternalProgramError(f"ORCA run failed - check the output file: {basename}.out") from e


def grab_orca_warnings(filename):
    warning_lines = []
    # Error-words to search for
    # TODO: Avoid searching though file multiple times.
    # TODO: Write pygrep version that supports list of search-strings
    warning_strings = ["WARNING", "warning", "Warning"]
    for warnstring in warning_strings:
        warn_l = pygrep2(warnstring, filename, errors="ignore")
        warning_lines += warn_l

    warnings = []
    # Lines that are not useful warnings
    ignore_lines = [
        "                       Please study these wa",
        "                                        WARNINGS",
        "Warning: in a DFT calculation",
        "WARNING: Old DensityContainer",
        "WARNING: your system is open-shell",
    ]
    for warn in warning_lines:
        false_positive = any(warn.startswith(ign) for ign in ignore_lines)
        if false_positive is False:
            warnings.append(warn)
    if len(warnings):
        logger.info("Found warning messages in ORCA outputfile:")
        logger.info("%s", "".join(warnings))


def grab_orca_errors(filename):
    error_lines = []
    # Error-words to search for
    # TODO: Avoid searching though file multiple times.
    # TODO: Write pygrep version that supports list of search-strings
    error_strings = ["error", "Error", "ERROR", "aborting"]
    for errstring in error_strings:
        error_l = pygrep2(errstring, filename, errors="ignore")
        for e in error_l:
            if e not in error_lines:
                error_lines.append(e)

    errors = []
    # Lines that are not errors
    ignore_lines = [
        "   Iter.        energy            ||Error||_2",
        " WARNING: the maximum gradient error",
        "           *** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR",
        "   Startup",
        "   DIIS-Error",
        " DIIS",
        "sum of PNO error",
        "  Last DIIS Error",
        "    DIIS-Error",
        " Sum of total truncation errors",
        "  Sum of total UMP2 truncation",
    ]
    for err in error_lines:
        false_positive = any(err.startswith(ign) for ign in ignore_lines)
        if false_positive is False:
            errors.append(err)
    if len(errors):
        logger.info("Found possible error messages in ORCA outputfile:")
        logger.info("%s", "".join(errors))


# Check if ORCA finished.
# Todo: Use reverse-read instead to speed up?
def check_orca_finished(file):
    scf_iterations = None
    with open(file, errors="ignore") as f:
        for line in f:
            if "SCF CONVERGED AFTER" in line:
                scf_iterations = line.split()[-3]
            if "TOTAL RUN TIME:" in line:
                return True, scf_iterations
    return False, None


def check_orca_opt_finished(file):
    converged = False
    with open(file, errors="ignore") as f:
        for line in f:
            if "THE OPTIMIZATION HAS CONVERGED" in line:
                converged = True
            if converged and "***               (AFTER" in line:
                cycles = line.split()[2]
                logger.info(f"ORCA Optimization converged in {cycles} cycles")
        return converged


# Grab Final single point energy. Ignoring possible encoding errors in file
def grab_orca_final_energy(file, errors="ignore"):
    Energy = None
    with open(file, errors=errors) as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                if "Wavefunction not fully converged!" in line:
                    raise ExternalProgramError("ORCA WF not fully converged!\nNot using energy. Modify ORCA settings")
                else:
                    # Changing: sometimes ORCA adds info to the right of energy
                    Energy = float(line.split()[5]) if "(MM)" in line else float(line.split()[4])
    if Energy is None:
        logger.error("Found no energy in file: %s", file)
        logger.error("Something went wrong with ORCA run. Check ORCA outputfile: %s", file)
        logger.info("------------ENDING ORCA-INTERFACE-------------")
        return None
    return Energy


# Grab ORCA timings. Return dictionary
def grab_orca_timings(file):
    timings = {}  # in seconds
    try:
        with open(file, errors="ignore") as f:
            for line in f:
                if "Calculating one electron integrals" in line:
                    one_elec_integrals = float(line.split()[-2].replace("(", ""))
                    timings["one_elec_integrals"] = one_elec_integrals
                if "SCF Gradient evaluation         ..." in line:
                    time_scfgrad = float(line.split()[4])
                    timings["time_scfgrad"] = time_scfgrad
                if "SCF iterations                  ..." in line:
                    time_scfiterations = float(line.split()[3])
                    timings["time_scfiterations"] = time_scfiterations
                if "GTO integral calculation        ..." in line:
                    time_gtointegrals = float(line.split()[4])
                    timings["time_gtointegrals"] = time_gtointegrals
                if "SCF Gradient evaluation         ..." in line:
                    time_scfgrad = float(line.split()[4])
                    timings["time_scfgrad"] = time_scfgrad
                if "Sum of individual times         ...:" in line:
                    total_time = float(line.split()[4])
                    timings["total_time"] = total_time
                if "One electron gradient       ...." in line:
                    one_elec_gradient = float(line.split()[4])
                    timings["one_elec_gradient"] = one_elec_gradient
                if "RI-J Coulomb gradient       ...." in line:
                    rij_coulomb_gradient = float(line.split()[4])
                    timings["rij_coulomb_gradient"] = rij_coulomb_gradient
                if "XC gradient                 ...." in line:
                    xc_gradient = float(line.split()[3])
                    timings["xc_gradient"] = xc_gradient
                if "Point charge gradient       ...." in line:
                    pc_gradient = float(line.split()[4])
                    timings["pc_gradient"] = pc_gradient
    except (OSError, ValueError, IndexError):
        pass
    return timings


# Grab gradient from ORCA engrad file
def grab_orca_gradient(engradfile):
    grab = False
    numatomsgrab = False
    row = 0
    col = 0
    with open(engradfile) as gradfile:
        for line in gradfile:
            if numatomsgrab and "#" not in line:
                numatoms = int(line.split()[0])
                # Initializing array
                gradient = np.zeros((numatoms, 3))
                numatomsgrab = False
            if "# Number of atoms" in line:
                numatomsgrab = True
            if grab and "#" not in line:
                val = float(line.split()[0])
                gradient[row, col] = val
                if col == 2:
                    row += 1
                    col = 0
                else:
                    col += 1
            if "# The current gradient in Eh/bohr" in line:
                grab = True
            if "# The atomic numbers and " in line:
                grab = False
    return gradient


# Grab pointcharge gradient from ORCA pcgrad file
def grab_orca_pc_gradient(pcgradfile):
    with open(pcgradfile) as pgradfile:
        for count, line in enumerate(pgradfile):
            if count == 0:
                numatoms = int(line.split()[0])
                # Initializing array
                gradient = np.zeros((numatoms, 3))
            elif count > 0:
                val_x = float(line.split()[0])
                val_y = float(line.split()[1])
                val_z = float(line.split()[2])
                gradient[count - 1] = [val_x, val_y, val_z]
    return gradient


def grab_dipole_moment(outfile):
    dipole_moment = []
    with open(outfile) as f:
        for line in f:
            if "Total Dipole Moment    :" in line:
                dipole_moment.append(float(line.split()[-3]))
                dipole_moment.append(float(line.split()[-2]))
                dipole_moment.append(float(line.split()[-1]))
    return dipole_moment


def grab_polarizability_tensor(outfile):
    pz_tensor = np.zeros((3, 3))
    diag_pz_tensor = []
    count = 0
    grab = False
    grab2 = False
    grab3 = False
    with open(outfile) as f:
        for line in f:
            if grab3 is True:
                if len(line.split()) == 0:
                    grab2 = False
                else:
                    diag_pz_tensor.append(float(line.split()[0]))
                    diag_pz_tensor.append(float(line.split()[1]))
                    diag_pz_tensor.append(float(line.split()[2]))
                    grab = False
                    grab2 = False
                    grab3 = False
            if grab is True:
                if "The raw cartesian tensor" in line:
                    grab2 = True
                if "diagonalized tensor:" in line:
                    grab2 = False
                    grab3 = True
                if grab2 is True and len(line.split()) == 3:
                    pz_tensor[count, 0] = float(line.split()[0])
                    pz_tensor[count, 1] = float(line.split()[1])
                    pz_tensor[count, 2] = float(line.split()[2])
                    count += 1
            if "STATIC POLARIZABILITY TENSOR" in line:
                logger.info("grab True")
                grab = True
    return pz_tensor, diag_pz_tensor


# Grab TDDFT state energies from ORCA output
def tddftgrab(file):
    tddftstates = []
    tddft = True
    tddftgrab = False
    if tddft:
        with open(file) as f:
            for line in f:
                if tddftgrab and "STATE" in line:
                    if "eV" in line:
                        tddftstates.append(float(line.split()[5]))
                    tddftgrab = True
                if "the weight of the individual excitations" in line:
                    tddftgrab = True
    return tddftstates


# Grab TDDFT state intensities from ORCA output
def tddftintens_grab(file):
    intensities = []
    tddftgrab = False
    with open(file) as f:
        for line in f:
            if tddftgrab:
                if "->" in line:
                    intensities.append(float(line.split()[-5]))
                if len(line.split()) == 0:
                    tddftgrab = False
            if "fosc(D2)" in line:
                tddftgrab = True
    return intensities


# Grab TDDFT orbital pairs from ORCA output


def grab_ir_intensities(filename):
    grab = False
    intensities = []
    with open(filename) as f:
        for line in f:
            if grab and len(line.split()) == 6:
                intens = float(line.split()[2])
                intensities.append(intens)
            if "$ir_spectrum" in line:
                grab = True
    return intensities


# Function to write ORCA-style Hessian file


def write_orca_hessfile(hessian, coords, elems, masses, hessatoms, outputname):
    hessdim = hessian.shape[0]
    with open(outputname, "w") as orcahessfile:
        orcahessfile.write("$orca_hessian_file\n")
        orcahessfile.write("\n")
        orcahessfile.write("$hessian\n")
        orcahessfile.write(str(hessdim) + "\n")
        orcahesscoldim = 5
        index = 0
        tempvar = ""
        temp2var = ""
        chunks = hessdim // orcahesscoldim
        left = hessdim % orcahesscoldim
        if left > 0:
            chunks = chunks + 1
        for chunk in range(chunks):
            if chunk == chunks - 1:
                # If last chunk and cleft is exactly 0 then all 5 columns should be done
                if left == 0:
                    left = 5
                for temp in range(index, index + left):
                    temp2var = temp2var + "         " + str(temp)
            else:
                for temp in range(index, index + orcahesscoldim):
                    temp2var = temp2var + "         " + str(temp)
            orcahessfile.write(str(temp2var) + "\n")
            for i in range(hessdim):
                if chunk == chunks - 1:
                    for k in range(index, index + left):
                        tempvar = tempvar + "         " + str(hessian[i, k])
                else:
                    for k in range(index, index + orcahesscoldim):
                        tempvar = tempvar + "         " + str(hessian[i, k])
                orcahessfile.write("    " + str(i) + "   " + str(tempvar) + "\n")
                tempvar = ""
                temp2var = ""
            index += 5
        orcahessfile.write("\n")
        orcahessfile.write("# The atoms: label  mass x y z (in bohrs)\n")
        orcahessfile.write("$atoms\n")
        orcahessfile.write(str(len(elems)) + "\n")

        # Write coordinates and masses to Orca Hessian file
        # TODO. Note. Changed things. We now don't go through hessatoms and analyze atom indices for full system
        # Either full system lists were passed or partial-system lists
        # for atom, mass in zip(hessatoms, masses):
        orcahessfile.writelines(
            " "
            + el
            + "    "
            + str(mass)
            + "  "
            + str(coord[0] / openmmqmmm.constants.bohr2ang)
            + " "
            + str(coord[1] / openmmqmmm.constants.bohr2ang)
            + " "
            + str(coord[2] / openmmqmmm.constants.bohr2ang)
            + "\n"
            for el, mass, coord in zip(elems, masses, coords, strict=False)
        )
        orcahessfile.write("\n")
        orcahessfile.write("\n")
    logger.info("")
    logger.info("ORCA-style Hessian written to: %s", outputname)


# Function to grab Hessian from ORCA-Hessian file
def grab_hessian(hessfile):
    hesstake = False
    j = 0
    orcacoldim = 5
    shiftpar = 0
    lastchunk = False
    grabsize = False
    with open(hessfile) as hfile:
        for line in hfile:
            if "$vibrational_frequencies" in line:
                hesstake = False
                continue
            if hesstake and len(line.split()) == 1 and grabsize:
                grabsize = False
                hessdim = int(line.split()[0])

                hessarray2d = np.zeros((hessdim, hessdim))

            if hesstake and lastchunk:
                if len(line.split()) == hessdim - shiftpar + 1:
                    for i in range(hessdim - shiftpar):
                        hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                    j += 1
            elif hesstake and len(line.split()) == 5:
                continue
                # Headerline
            if hesstake and len(line.split()) == 6:
                # Hessianline
                for i in range(orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if "$hessian" in line:
                hesstake = True
                grabsize = True
        return hessarray2d


# Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Allows for extraline that could be another '!' line or block-inputline.
def create_orca_input_pc(
    name,
    elems,
    coords,
    orcasimpleinput,
    orcablockinput,
    charge,
    mult,
    grad=False,
    extraline="",
    hs_mult=None,
    atomstoflip=None,
    hessian=False,
    extrabasisatoms=None,
    extrabasis=None,
    atom_specific_basis_dict=None,
    extraspecialbasisatoms=None,
    extraspecialbasis=None,
    moreadfile=None,
    propertyblock=None,
    fragment_indices=None,
    rohf_uhf_swap=False,
    delta_scf_block=None,
):
    if extrabasisatoms is None:
        extrabasisatoms = []
    pcfile = name + ".pc"
    with open(name + ".inp", "w") as orcafile:
        orcafile.write(orcasimpleinput + "\n")
        if extraline != "":
            orcafile.write(extraline + "\n")
        if grad:
            orcafile.write("! Engrad" + "\n")
        if hessian:
            orcafile.write("! Freq" + "\n")
        if moreadfile is not None:
            logger.info("MOREAD option active. Will read orbitals from file: %s", moreadfile)
            orcafile.write("\n! MOREAD" + "\n")
            orcafile.write(f'%moinp "{moreadfile}"' + "\n")
        orcafile.write(f'%pointcharges "{pcfile}"\n')
        orcafile.write(orcablockinput + "\n")
        if atomstoflip is not None:
            atomstoflipstring = ",".join(map(str, atomstoflip))
            orcafile.write("%scf\n")
            orcafile.write(f"Flipspin {atomstoflipstring}" + "\n")
            orcafile.write(f"FinalMs {(mult - 1) / 2}" + "\n")
            orcafile.write("end  \n")
        orcafile.write("\n")
        # DELTASCF
        if delta_scf_block is not None:
            orcafile.write(delta_scf_block)
        orcafile.write("\n")
        if atomstoflip is not None:
            orcafile.write(f"*xyz {charge} {hs_mult}\n")
        else:
            orcafile.write(f"*xyz {charge} {mult}\n")
        # Writing coordinates. Adding extrabasis keyword for atom if option active
        for i, (el, c) in enumerate(zip(elems, coords, strict=False)):
            if i in extrabasisatoms:
                orcafile.write(f'{el} {c[0]} {c[1]} {c[2]} newgto "{extrabasis}" end\n')
            # Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                logger.info("Writing atom-specific basis for atom: %s", i)
                # Regular line
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
                orcafile.writelines(str(bline) for bline in atom_specific_basis_dict[(el, i)])
            # Adding fragment specification
            elif fragment_indices is not None:
                fragmentindex = search_list_of_lists_for_index(i, fragment_indices)
                # To prevent linkatoms:
                if fragmentindex is not None:
                    orcafile.write("{} {} {} {} \n".format(f"{el}({fragmentindex + 1})", c[0], c[1], c[2]))
            else:
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
        orcafile.write("*\n")
        if propertyblock is not None:
            orcafile.write(propertyblock)
        # For ROHF job, add newjob and switch to UHF noiter
        if rohf_uhf_swap:
            newjobline = f"""\n$new_job
{orcasimpleinput.replace("ROHF", "UHF noiter ")}
{orcablockinput}
* xyz {charge} {mult}
"""
            orcafile.write(newjobline)
            for el, c in zip(elems, coords, strict=False):
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
            orcafile.write("*\n")


# Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Allows for extraline that could be another '!' line or block-inputline.
def create_orca_input_plain(
    name,
    elems,
    coords,
    orcasimpleinput,
    orcablockinput,
    charge,
    mult,
    grad=False,
    hessian=False,
    extraline="",
    hs_mult=None,
    atomstoflip=None,
    extrabasis=None,
    extrabasisatoms=None,
    moreadfile=None,
    propertyblock=None,
    ghostatoms=None,
    dummyatoms=None,
    fragment_indices=None,
    atom_specific_basis_dict=None,
    rohf_uhf_swap=False,
    delta_scf_block=None,
):
    if extrabasisatoms is None:
        extrabasisatoms = []
    if ghostatoms is None:
        ghostatoms = []
    if dummyatoms is None:
        dummyatoms = []
    with open(name + ".inp", "w") as orcafile:
        orcafile.write(orcasimpleinput + "\n")
        if extraline != "":
            orcafile.write(extraline)
        if grad is True:
            orcafile.write("! Engrad" + "\n")
        if hessian is True:
            orcafile.write("! Freq" + "\n")
        if moreadfile is not None:
            logger.info("MOREAD option active. Will read orbitals from file: %s", moreadfile)
            orcafile.write("\n! MOREAD" + "\n")
            orcafile.write(f'%moinp "{moreadfile}"' + "\n")
        orcafile.write(orcablockinput)
        if atomstoflip is not None:
            atomstoflipstring = str(atomstoflip) if isinstance(atomstoflip, int) else ",".join(map(str, atomstoflip))
            orcafile.write("%scf\n")
            orcafile.write(f"Flipspin {atomstoflipstring}" + "\n")
            orcafile.write(f"FinalMs {(mult - 1) / 2}" + "\n")
            orcafile.write("end  \n")
            orcafile.write("\n")
        # DELTASCF
        if delta_scf_block is not None:
            orcafile.write(delta_scf_block)
        orcafile.write("\n")
        if atomstoflip is not None:
            orcafile.write(f"*xyz {charge} {hs_mult}\n")
        else:
            orcafile.write(f"*xyz {charge} {mult}\n")

        for i, (el, c) in enumerate(zip(elems, coords, strict=False)):
            # Extra basis on each atom
            if i in extrabasisatoms:
                orcafile.write(f'{el} {c[0]} {c[1]} {c[2]} newgto "{extrabasis}" end\n')
            # Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                logger.info("Writing atom-specific basis for atom: %s", i)
                # Regular line
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
                orcafile.writelines(str(bline) for bline in atom_specific_basis_dict[(el, i)])
            # Setting atom to be a ghost atom
            elif i in ghostatoms:
                orcafile.write("{}{} {} {} {} \n".format(el, ":", c[0], c[1], c[2]))
            elif i in dummyatoms:
                orcafile.write("{} {} {} {} \n".format("DA", c[0], c[1], c[2]))
            # Adding fragment specification
            elif fragment_indices is not None:
                fragmentindex = search_list_of_lists_for_index(i, fragment_indices)
                orcafile.write("{} {} {} {} \n".format(f"{el}({fragmentindex + 1})", c[0], c[1], c[2]))
            else:
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
        orcafile.write("*\n")
        if propertyblock is not None:
            orcafile.write(propertyblock)
        # For ROHF job, add newjob and switch to UHF noiter
        if rohf_uhf_swap:
            newjobline = f"""\n$new_job
{orcasimpleinput.replace("ROHF", "UHF noiter ")}
{orcablockinput}
* xyz {charge} {mult}
"""
            orcafile.write(newjobline)
            for el, c in zip(elems, coords, strict=False):
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
            orcafile.write("*\n")


# Create ORCA pointcharge file based on provided list of elems and coords (MM region elems and coords)
# and list of point charges of MM atoms
def create_orca_pcfile(name, coords, listofcharges):
    with open(name + ".pc", "w") as pcfile:
        pcfile.write(str(len(listofcharges)) + "\n")
        for p, c in zip(listofcharges, coords, strict=False):
            line = f"{p} {c[0]} {c[1]} {c[2]}"
            pcfile.write(line + "\n")


# Grabbing spin populations
def grab_orca_spin_populations(chargemodel, outputfile):
    grab = False
    spinpops = []
    BS = False  # if broken-symmetry job
    numatoms = int(pygrep("Number of atoms                             ...", outputfile)[-1])
    # if
    if len(pygrep2("WARNING: Broken symmetry calculations", outputfile)):
        BS = True

    if chargemodel == "Mulliken":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab is True:
                    if "Sum of atomic" in line:
                        grab = False
                    elif "------" not in line:
                        spinpops.append(float(line.split()[-1]))
                if "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS" in line:
                    grab = True
    elif chargemodel == "Loewdin":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab is True:
                    if "Sum of atomic" in line or len(line.replace(" ", "")) < 2:
                        grab = False
                    elif "------" not in line:
                        spinpops.append(float(line.split()[-1]))
                if "LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS" in line:
                    grab = True
    else:
        raise FileFormatError("Unknown chargemodel. Exiting...")
    # If BS then we have grabbed charges for both high-spin and BS solution
    if BS is True:
        logger.info("Broken-symmetry job detected. Only taking BS-state populations")
        if len(spinpops) != numatoms:
            spinpops = spinpops[-numatoms:]
    # if len(spinpops) == 0:
    return spinpops


def grab_orca_atom_charges(chargemodel, outputfile):
    grab = False
    coordgrab = False
    charges = []
    BS = False  # if broken-symmetry job
    column = None

    numatoms = int(pygrep("Number of atoms                             ...", outputfile)[-1])

    # if
    if len(pygrep2("WARNING: Broken symmetry calculations", outputfile)):
        BS = True

    if chargemodel == "NPA" or chargemodel == "NBO":
        logger.info("Warning: NPA/NBO charge-option in ORCA requires setting environment variable NBOEXE:")
        logger.info("e.g. export NBOEXE=/path/to/nbo7.exe")
        with open(outputfile) as ofile:
            for line in ofile:
                if grab:
                    if "=======" in line:
                        grab = False
                    elif "------" not in line:
                        charges.append(float(line.split()[2]))
                if "Atom No    Charge        Core      Valence    Rydberg      Total" in line:
                    grab = True
    elif chargemodel.upper() == "CHELPG":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab:
                    if "Total charge: " in line:
                        grab = False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-1]))
                if "CHELPG Charges" in line:
                    grab = True
                    # Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges = []
    elif chargemodel.upper() == "HIRSHFELD":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab:
                    if len(line) < 3:
                        grab = False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-2]))
                if "  ATOM     CHARGE      SPIN" in line:
                    grab = True
                    # Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges = []
    elif chargemodel.upper() == "CM5":
        elems = []
        coords = []
        with open(outputfile) as ofile:
            for line in ofile:
                # Getting coordinates as used in CM5 definition
                if coordgrab is True and "----------------------" not in line:
                    if len(line.split()) < 2:
                        coordgrab = False
                    else:
                        elems.append(line.split()[0])
                        coords_x = float(line.split()[1])
                        coords_y = float(line.split()[2])
                        coords_z = float(line.split()[3])
                        coords.append([coords_x, coords_y, coords_z])
                if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                    coordgrab = True
                if grab:
                    if len(line) < 3:
                        grab = False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-2]))
                if "  ATOM     CHARGE      SPIN" in line:
                    # Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges = []
                    grab = True
        logger.info("Hirshfeld charges : %s", charges)
        atomicnumbers = openmmqmmm.coords.elemstonuccharges(elems)
        charges = openmmqmmm.elstructure.calc_cm5(atomicnumbers, coords, charges)
        logger.info("CM5 charges : %s", list(charges))
    elif chargemodel.upper() == "MULLIKEN":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab:
                    if "Sum of atomic" in line:
                        grab = False
                    elif "------" not in line:
                        charges.append(float(line.split()[column]))
                if "MULLIKEN ATOMIC CHARGES" in line:
                    grab = True
                    column = -2 if "SPIN POPULATIONS" in line else -1

    elif chargemodel.upper() == "LOEWDIN":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab:
                    if "Sum of atomic" in line or len(line.replace(" ", "")) < 2:
                        grab = False
                    elif "------" not in line:
                        charges.append(float(line.split()[column]))
                if "LOEWDIN ATOMIC CHARGES" in line:
                    grab = True
                    column = -2 if "SPIN POPULATIONS" in line else -1
    elif chargemodel.upper() == "IAO":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab:
                    if "Sum of atomic" in line:
                        grab = False
                    elif "------" not in line and "Warning" not in line:
                        logger.info("line: %s", line)
                        charges.append(float(line.split()[-1]))
                if "IAO PARTIAL CHARGES" in line:
                    grab = True
    else:
        raise FileFormatError("Unknown chargemodel. Exiting...")

    # If BS then we have grabbed charges for both high-spin and BS solution
    if BS is True:
        logger.info("Broken-symmetry job detected. Only taking BS-state populations")
        if len(charges) != numatoms:
            charges = charges[numatoms:]
        logger.info("charges: %s", charges)
    return charges


# Wrapper around interactive orca_plot
# Todo: add TDDFT difference density, natural orbitals, MDCI spin density?


# Reading stability analysis from output. Returns true if stab-analysis good, otherwise falsee
# If no stability analysis present in output, then also return true


def grab_scf_fod_occupations(filename):
    occgrab = False
    occupations = []
    with open(filename) as f:
        for line in f:
            if occgrab is True:
                if "***********" in line:
                    return occupations
                if " SPIN DOWN" in line:
                    occgrab = False
                    return occupations
                if len(line.split()) == 4 and "  NO   OCC" not in line:
                    occupations.append(float(line.split()[1]))
            if "SPIN UP ORBITALS" in line or "ORBITAL ENERGIES" in line:
                occgrab = True
    return occupations


# Grab ICE-WF CFG info from CI job
def grab_ice_wf_cfg_ci_size(filename):
    num_after_SD_CFGs = 0
    num_genCFGs = 0
    num_selected_CFGs = 0
    with open(filename) as g:
        for line in g:
            if "# of configurations after S+D" in line:
                num_after_SD_CFGs = int(line.split()[-1])
            if "# of configurations after Selection" in line:
                num_selected_CFGs = int(line.split()[-1])
            if " # of generator configurations" in line:
                num_genCFGs = int(line.split()[5])
    return num_genCFGs, num_selected_CFGs, num_after_SD_CFGs


# Writes the ORCA-style .engrad file that the generated otool_external script
# (see write_otool_script below) produces for ORCA's ExtOpt driver.
def print_gradient_in_orca_format(energy, gradient, basename, extrabasename="_EXT"):
    numatoms = len(gradient)
    with open(basename + extrabasename + ".engrad", "w") as f:
        f.write("#\n")
        f.write("# Number of atoms\n")
        f.write("#\n")
        f.write(f"         {numatoms}\n")
        f.write("#\n")
        f.write("# The current total energy in E\n")
        f.write("#\n")
        f.write(f"     {energy}\n")
        f.write("#\n")
        f.write("# The current gradient in Eh/Bohr\n")
        f.write("#\n")
        for g in gradient:
            f.writelines(f"{gg}\n" for gg in g)
        f.write("#\n")
        f.write("# The atomic numbers and current coordinates in Bohr\n")
        f.write("#\n")


def write_otool_script(basename=None, theoryfile=None, scriptlocation=None, charge=None, mult=None):
    import stat

    with open(scriptlocation + "/otool_external", "w") as otool:
        otool.write("#!/usr/bin/env python3\n")
        otool.write("import pickle\n\n")
        otool.write("from openmmqmmm import Fragment, single_point\n")
        otool.write("from openmmqmmm.orca import print_gradient_in_orca_format\n\n")
        otool.write(f'frag = Fragment(xyzfile="{basename}.xyz")\n')
        otool.write(f'with open("{theoryfile}", "rb") as theoryfh:\n')
        otool.write("    theory = pickle.load(theoryfh)\n")
        otool.write(f"result = single_point(theory=theory, fragment=frag, grad=True, charge={charge}, mult={mult})\n")
        otool.write(f'print_gradient_in_orca_format(result.energy, result.gradient, "{basename}")\n')
    st = os.stat(scriptlocation + "/otool_external")
    os.chmod(scriptlocation + "/otool_external", st.st_mode | stat.S_IEXEC)


# Using ORCA as an external optimizer
# Will only work for theories that can be pickled.
def orca_external_optimizer(
    fragment=None,
    theory=None,
    orcadir=None,
    charge=None,
    mult=None,
    orca_jobkeyword="Opt",
    orca_blockinput="",
    actatoms=None,
):
    logger.info(main_header("ORCA_External_Optimizer"))
    if fragment is None or theory is None:
        raise InputError("ORCA_External_Optimizer requires fragment and theory keywords")

    if charge is None or mult is None:
        logger.warning("Warning: Charge/mult was not provided to ORCA_External_Optimizer")
        if fragment.charge is not None and fragment.mult is not None:
            logger.warning(
                f"Fragment contains charge/mult information: Charge: {fragment.charge} Mult: {fragment.mult} Using this instead"
            )
            logger.warning("Make sure this is what you want!")
            charge = fragment.charge
            mult = fragment.mult
        else:
            raise InputError("No charge/mult information present in fragment either. Exiting.")

    # Making sure we have a working ORCA installation
    orcadir = find_orca(orcadir)
    # Adding orcadir to PATH. Only required if ORCA not in PATH already
    os.environ["PATH"] += os.pathsep + orcadir

    # Pickle for serializing theory object
    import pickle

    # Serialize theory object for later use
    theoryfilename = "theory.saved"
    with open(theoryfilename, "wb") as theoryfh:
        pickle.dump(theory, theoryfh)

    # Write otool_script once in location that ORCA will launch. This is an energy+gradient calculator script
    # ORCA will call : otool_external test_EXT.extinp.tmp
    # ASH_otool creates basename_Ext.engrad that ORCA reads
    basename = "ORCAEXTERNAL"
    scriptlocation = "."
    os.environ["PATH"] += os.pathsep + "."
    write_otool_script(
        basename=basename, theoryfile=theoryfilename, scriptlocation=scriptlocation, charge=charge, mult=mult
    )

    # Create XYZ-file for ORCA-Extopt
    xyzfile = "orca_external.xyz"
    fragment.write_xyzfile(xyzfile)

    # Active atoms become inverted constraints
    constraintsblock = ""
    if actatoms is not None:
        logger.info("Activeatoms list was provided. This means that we need to provide constraints to ORCA")
        frozenatoms = listdiff(fragment.allatoms, actatoms)
        logger.info("Freezing the non-active atoms: %s", frozenatoms)
        cons = []
        for f in frozenatoms:
            cons.append(f"{{C {f} C}}\n")
        consstring = "".join(cons)
        constraintsblock = f"""%geom Constraints
{consstring}end
end
"""
    # ORCA input file
    with open(basename + ".inp", "w") as o:
        o.write(f"! ExtOpt {orca_jobkeyword}\n")
        o.write("\n")
        o.write(f"{orca_blockinput}")
        o.write(f"{constraintsblock}")
        o.write("%method\n")
        o.write('ProgExt "otool_external"\n')
        o.write("end\n")
        o.write(f"*xyzfile {charge} {mult} {xyzfile}\n")

    if "GOAT" in orca_jobkeyword.upper():
        logger.info("GOAT keyword found. ")

    # Call ORCA to do Opt/GOAT etc. job
    with open(basename + ".out", "w") as ofile:
        sp.run(["orca", basename + ".inp"], check=True, stdout=ofile, stderr=ofile, text=True)

    # Check if ORCA finished
    ORCAfinished, _iter = check_orca_finished(basename + ".out")
    if ORCAfinished is not True:
        raise ExternalProgramError("Something failed about external ORCA job")
    # Check if optimization completed
    if check_orca_opt_finished(basename + ".out") is not True:
        raise ExternalProgramError("ORCA external job failed. Check outputfile: {}".format(basename + ".out"))
    logger.info("ORCA external job finished")

    # Grabbing final geometry to update fragment object
    _elems, coords = openmmqmmm.coords.read_xyzfile(basename + ".xyz")
    fragment.coords = coords

    # Grabbing final energy
    energylines = pygrep2("FINAL SINGLE POINT ENERGY (From external program)", f"{basename}.out", errors="ignore")
    energy = float(energylines[-1].split()[-1])
    logger.info("Final energy from external ORCA job: %s", energy)

    return energy


# Get natural orbitals of any calculated density of an ORCA calculation
# Convenient when ORCA natural orbital printing is buggy
# NOTE: Not fully tested
