import copy
import glob
import multiprocessing as mp
import numpy as np
import os
import shutil
import subprocess as sp
import time

import openmmqmmm.constants
import openmmqmmm.functions.functions_elstructure
import openmmqmmm.functions.functions_parallel
import openmmqmmm.modules.module_coords
import openmmqmmm.settings_ash
from openmmqmmm.functions.functions_general import ashexit, insert_line_into_file, BC, print_time_rel, \
    print_line_with_mainheader, pygrep2, \
    pygrep, search_list_of_lists_for_index, print_if_level, writestringtofile, check_program_location, listdiff
from openmmqmmm.modules.module_coords import check_charge_mult, print_internal_coordinate_table_new
from openmmqmmm.modules.module_singlepoint import Singlepoint


# ORCA Theory object.
class ORCATheory:
    def __init__(self, orcadir=None, orcasimpleinput='', printlevel=2, basis_per_element=None, extrabasisatoms=None,
                 extrabasis=None, atom_specific_basis_dict=None, ecp_dict=None, TDDFT=False, TDDFTroots=5, FollowRoot=1,
                 orcablocks='', extraline='', first_iteration_input=None, brokensym=None, HSmult=None, atomstoflip=None,
                 numcores=1, nprocs=None, label="ORCA",
                 moreadfile=None, moreadfile_always=False, bind_to_core_option=True, ignore_ORCA_error=False,
                 autostart=True, propertyblock=None, save_output_with_label=False, keep_each_run_output=False,
                 print_population_analysis=False, filename="orca", check_for_errors=True, check_for_warnings=True,
                 fragment_indices=None, xdm=False, xdm_a1=None, xdm_a2=None, xdm_func=None, NMF=False, NMF_sigma=None,
                 cpcm_radii=None, ROHF_UHF_swap=False,
                 deltaSCF=False, deltaSCF_PMOM=False, deltaSCF_confline=None, deltaSCF_turn_off_automatically=True):
        print_line_with_mainheader("ORCATheory initialization")

        self.theorynamelabel = "ORCA"
        self.theorytype = "QM"
        self.analytic_hessian = True

        # Making sure we have a working ORCA location
        print("Checking for ORCA location")
        self.orcadir = check_ORCA_location(orcadir, modulename="ORCATheory")
        # Making sure ORCA binary works (and is not orca the screenreader)
        check_ORCAbinary(self.orcadir)
        # Checking OpenMPI
        if numcores != 1:
            print(
                f"ORCA parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version (for the ORCA version) is available in your environment")
            openmmqmmm.functions.functions_parallel.check_OpenMPI()

        # Bind to core option when calling ORCA: i.e. execute: /path/to/orca file.inp "--bind-to none"
        # TODO: Default False; make True?
        self.bind_to_core_option = bind_to_core_option
        print("bind_to_core_option:", self.bind_to_core_option)

        # Checking if user added Opt, Freq keywords
        if ' OPT' in orcasimpleinput.upper() or ' FREQ' in orcasimpleinput.upper():
            print(BC.FAIL,
                  "Error. orcasimpleinput variable can not contain ORCA job-directives like: Opt, Freq, Numfreq",
                  BC.END)
            print("String:", orcasimpleinput.upper())
            print(
                "orcasimpleinput should only contain information on electronic-structure method (e.g. functional), basis set, grid, SCF convergence etc.")
            ashexit()
        if '!' not in orcasimpleinput:
            print(BC.FAIL, "Error. orcasimpleinput should contain at least a '!' with method and basis set information",
                  BC.END)
            ashexit()

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
            print("Error: save_output_with_label option requires a label keyword also")
            ashexit()
        else:
            self.save_output_with_label = save_output_with_label

        # Print population_analysis in each run
        self.print_population_analysis = print_population_analysis

        # Label to distinguish different ORCA objects
        self.label = label

        # Create inputfile with generic name
        self.filename = filename

        # Whether to exit ORCA if subprocess command faile
        self.ignore_ORCA_error = ignore_ORCA_error

        # MOREAD-file
        self.moreadfile = moreadfile
        self.moreadfile_always = moreadfile_always
        # Autostart
        self.autostart = autostart
        # Each ORCA calculation will save path to last GBW-file used in case we have switched directories
        # and we want to use last one
        self.path_to_last_gbwfile_used = None  # default None

        # Printlevel
        self.printlevel = printlevel

        # TDDFT
        self.TDDFT = TDDFT
        self.TDDFTroots = TDDFTroots
        self.FollowRoot = FollowRoot

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
        self.HSmult = HSmult
        if isinstance(atomstoflip, int):
            print(BC.FAIL, "Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer.",
                  BC.END)
            ashexit()
        if self.brokensym is True:
            # Add UKS if not present
            if 'UKS' not in self.orcasimpleinput:
                if 'UHF' not in self.orcasimpleinput:
                    print("Warning: UKS/UHF keyword not present in orcasimpleinput for BS job. Adding.")
                    self.orcasimpleinput = self.orcasimpleinput + ' UKS'
        if atomstoflip is not None:
            self.atomstoflip = atomstoflip
        else:
            self.atomstoflip = []
        # DELTASCF
        self.deltaSCF = deltaSCF
        self.deltaSCF_PMOM = deltaSCF_PMOM
        self.deltaSCF_confline = deltaSCF_confline
        self.deltaSCF_turn_off_automatically = deltaSCF_turn_off_automatically
        if self.deltaSCF is True and self.deltaSCF_confline is None:
            print("Error: DELTASCF is True but no deltaSCF_confline provided. Exiting")
            ashexit()
        if self.deltaSCF is True:
            print("DeltaSCF True, turning on population analysis printing")
            self.print_population_analysis = True

        # Basis sets per element
        self.basis_per_element = basis_per_element
        if self.basis_per_element is not None:
            print("Basis set dictionary for each element provided:", basis_per_element)

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
        self.NMF = NMF
        if self.NMF is True:
            if NMF_sigma is None:
                print("NMF option requires setting NMF_sigma")
                ashexit()
            self.NMF_sigma = NMF_sigma

            print("NMF option is active. Will activate Fermi-smearing in ORCA input!")
            NMF_smeartemp = self.NMF_sigma / openmmqmmm.constants.R_gasconst
            print(f"NMF_smeartemp = {NMF_smeartemp} calculated from NMF_sigma: {self.NMF_sigma}:")
            self.orcablocks = self.orcablocks + f"""
%scf
fracocc true
smeartemp {NMF_smeartemp}
end
            """

        # TDDFT option
        # If gradient requested by Singlepoint(Grad=True) or Optimizer then TDDFT gradient is calculated instead
        if self.TDDFT is True:
            if '%tddft' not in self.orcablocks:
                self.orcablocks = self.orcablocks + f"""
%tddft
nroots {self.TDDFTroots}
IRoot {self.FollowRoot}
end
"""
        # ROHF-UHF swap
        self.ROHF_UHF_swap = ROHF_UHF_swap

        # Specific CPCM radii. e.g. to use DRACO radii
        if cpcm_radii is not None:
            print("CPCM radii provided:", cpcm_radii)
            # if len(cpcm_radii) != len(c:
            #    print("Error: Number of radii provided does not match number of elements in molecule")
            #    ashexit()
            cpcm_block = "%cpcm\n"
            for i, radius in enumerate(cpcm_radii):
                cpcm_block = cpcm_block + f"AtomRadii({i},  {radius})\n"
            cpcm_block = cpcm_block + "end\n"
            print("cpcm_block:", cpcm_block)
            self.orcablocks = self.orcablocks + cpcm_block

        # XDM: if True then we add !AIM to input
        self.xdm = False
        if xdm is True:
            self.xdm = True
            self.xdm_a1 = xdm_a1
            self.xdm_a2 = xdm_a2
            self.xdm_func = xdm_func
            self.orcasimpleinput = self.orcasimpleinput + ' AIM'

        if self.printlevel >= 2:
            print("")
            print("Creating ORCA object")
            print("ORCA dir:", self.orcadir)
            print(self.orcasimpleinput)
            print(self.orcablocks)
        print("\nORCATheory object created!")

    # Set numcores method
    def set_numcores(self, numcores):
        self.numcores = numcores

    # Cleanup after run.
    def cleanup(self):
        print("Cleaning up old ORCA files")
        list_files = []
        # Keeping outputfiles
        # list_files.append(self.filename + '.out')
        list_files.append(self.filename + '.gbw')
        list_files.append(self.filename + '.densities')
        list_files.append(self.filename + '.ges')
        list_files.append(self.filename + '.prop')
        list_files.append(self.filename + '.uco')
        list_files.append(self.filename + '_property.txt')
        list_files.append(self.filename + '.inp')
        list_files.append(self.filename + '.engrad')
        list_files.append(self.filename + '.cis')
        list_files.append(self.filename + '_last.out')
        list_files.append(self.filename + '.xyz')
        for file in list_files:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        try:
            for tmpfile in glob.glob("self.filename*tmp"):
                os.remove(tmpfile)
        except FileNotFoundError:
            pass

    # Do an ORCA-optimization instead of ASH optimization. Useful for gas-phase chemistry when ORCA-optimizer is better than geomeTRIC
    def Opt(self, fragment=None, Grad=None, Hessian=None, numcores=None, charge=None, mult=None):

        module_init_time = time.time()
        print(BC.OKBLUE, BC.BOLD, "------------RUNNING INTERNAL ORCA OPTIMIZATION-------------", BC.END)
        # Coords provided to run or else taken from initialization.
        # if len(current_coords) != 0:

        if fragment == None:
            print("No fragment provided to Opt.")
            ashexit()
        else:
            print("Fragment provided to Opt")

        current_coords = fragment.coords
        elems = fragment.elems
        # Check charge/mult
        charge, mult = check_charge_mult(charge, mult, self.theorytype, fragment, "ORCATheory.Opt", theory=self)

        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.Opt method", BC.END)
            ashexit()

        if numcores == None:
            numcores = self.numcores

        self.extraline = self.extraline + "\n! OPT "

        print("Running ORCA with {} cores available".format(numcores))
        print("Object label:", self.label)

        print("Creating inputfile:", self.filename + '.inp')
        print("ORCA input:")
        print(self.orcasimpleinput)
        print(self.extraline)
        print(self.orcablocks)
        if self.propertyblock != None:
            print(self.propertyblock)
        print("Charge: {}  Mult: {}".format(charge, mult))

        # TODO: Make more general
        create_orca_input_plain(self.filename, elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                charge, mult, extraline=self.extraline, HSmult=self.HSmult, moreadfile=self.moreadfile)
        print(BC.OKGREEN, f"ORCA Calculation started using {numcores} CPU cores", BC.END)
        run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', numcores=numcores,
                            bind_to_core_option=self.bind_to_core_option,
                            ignore_ORCA_error=self.ignore_ORCA_error)
        print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        outfile = self.filename + '.out'
        ORCAfinished, iter = checkORCAfinished(outfile)
        if ORCAfinished == True:
            print("ORCA job finished")
            if checkORCAOptfinished(outfile) == True:
                print("ORCA geometry optimization finished")
                self.energy = ORCAfinalenergygrab(outfile)
                # Grab optimized coordinates from filename.xyz
                opt_elems, opt_coords = openmmqmmm.modules.module_coords.read_xyzfile(self.filename + '.xyz')
                print(opt_coords)

                fragment.replace_coords(fragment.elems, opt_coords)
            else:
                print("ORCA optimization failed to converge. Check ORCA output")
                ashexit()
        else:
            print("Something happened with ORCA job. Check ORCA output")
            ashexit()

        print("ORCA optimized energy:", self.energy)
        print("ASH fragment updated:", fragment)
        fragment.print_coords()
        # Writing out fragment file and XYZ file
        fragment.print_system(filename='Fragment-optimized.ygg')
        fragment.write_xyzfile(xyzfilename='Fragment-optimized.xyz')

        # Printing internal coordinate table
        print_internal_coordinate_table_new(fragment)
        print_time_rel(module_init_time, modulename='ORCA Opt-run', moduleindex=2)
        return

    # Method to grab dipole moment from an ORCA outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        dm = grab_dipole_moment(self.filename + '.out')
        print("Dipole moment:", dm)
        return dm

    def get_polarizability_tensor(self):
        print("here")
        print("self.filename+'.out':", self.filename + '.out')
        polarizability, diag_pz = grab_polarizability_tensor(self.filename + '.out')
        print("polarizability:", polarizability)
        return polarizability

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, charge=None, mult=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            mm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, label=None):
        module_init_time = time.time()
        self.runcalls += 1
        if self.printlevel >= 2:
            print(BC.OKBLUE, BC.BOLD, "------------RUNNING ORCA INTERFACE-------------", BC.END)
            print("Object-label:", self.label)  # To distinguish multiple objects
            print("Run-label:", label)  # Primarily used in multiprocessing
        # Coords provided to run
        if current_coords is not None:
            pass
        else:
            print("Error:no current_coords")
            ashexit()

        # Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for ORCATheory.run method", BC.END)
            ashexit()

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems

        # If QM/MM then atomindices lists like extrabasisatoms, atomstoflip and fragment_indices have to be updated
        if len(self.qmatoms) != 0:

            # Fragment indices need to be updated if QM/MM
            if self.fragment_indices != None:
                fragment_indices = []
                for f in self.fragment_indices:
                    temp = [self.qmatoms.index(i) for i in f]
                    fragment_indices.append(temp)
            else:
                fragment_indices = self.fragment_indices
            # extrabasisatomindices if QM/MM
            # print("QM atoms :", self.qmatoms)
            qmatoms_extrabasis = [self.qmatoms.index(i) for i in self.extrabasisatoms]
            # new QM-region indices for atomstoflip if QM/MM
            try:
                qmatomstoflip = [self.qmatoms.index(i) for i in self.atomstoflip]
            except ValueError:
                print("Atoms to flip:", self.atomstoflip)
                print("Error: Atoms to flip are not all in QM-region")
                ashexit()
        else:
            qmatomstoflip = self.atomstoflip
            qmatoms_extrabasis = self.extrabasisatoms
            fragment_indices = self.fragment_indices

        if numcores == None:
            numcores = self.numcores

        # Basis set definition per element from input dict
        if self.basis_per_element != None:
            basisstring = ""
            for el, b in self.basis_per_element.items():
                basisstring += f"newgto {el} \"{b}\" end\n"
            basisblock = f"""
%basis
{basisstring}
end"""

            if basisblock not in self.orcablocks:
                self.orcablocks = self.orcablocks + basisblock

        # If ECP-dict provided (often goes with atom_specific_basis_dict)
        if self.ecp_dict != None:
            bstring = ""
            for el, b in self.ecp_dict.items():
                for x in b:
                    bstring += f"{x}"
            ecpbasisblock = f"""
%basis
{bstring}
end"""
            if ecpbasisblock not in self.orcablocks:
                self.orcablocks = self.orcablocks + ecpbasisblock

        if self.printlevel >= 2:
            print("Running ORCA with {} cores available".format(numcores))

        # MOREAD. Checking file provided exists and determining what to do if not
        if self.moreadfile != None:
            print_if_level(f"Moreadfile option active. File path: {self.moreadfile}", self.printlevel, 2)
            if os.path.isfile(self.moreadfile) is True:
                print_if_level(f"File exists in current directory: {os.getcwd()}", self.printlevel, 2)
            else:
                print_if_level(f"File does not exist in current directory: {os.getcwd()}", self.printlevel, 2)
                if os.path.isabs(self.moreadfile) is True:
                    print("Error: Absolute path provided but file does not exists. Exiting")
                    ashexit()
                else:
                    print_if_level("Checking if file exists in parentdir instead:", self.printlevel, 2)
                    if os.path.isfile(f"../{self.moreadfile}") is True:
                        print_if_level("Yes. Copying file to current dir", self.printlevel, 2)
                        shutil.copy(f"../{self.moreadfile}", f"./{self.moreadfile}")
        else:
            print_if_level(f"Moreadfile option not active", self.printlevel, 2)
            if os.path.isfile(f"{self.filename}.gbw") is False:
                print_if_level(f"No {self.filename}.gbw file is present in dir.", self.printlevel, 2)
                if self.path_to_last_gbwfile_used != None:
                    print_if_level(
                        f"Found a path ({self.path_to_last_gbwfile_used}) to last GBW-file used by this Theory object. Will try to copy this file do current dir",
                        self.printlevel, 2)
                    try:
                        shutil.copy(self.path_to_last_gbwfile_used, f"./{self.filename}.gbw")
                    except FileNotFoundError:
                        print_if_level("File was not found. May have been deleted", self.printlevel, 2)
                    if self.autostart is False:
                        print_if_level("Autostart option is False. ORCA will ignore this file", self.printlevel, 2)
                    else:
                        print_if_level("Autostart feature is active. ORCA will read GBW-file present.", self.printlevel,
                                       2)
                else:
                    print_if_level(f"Checking if a file {self.filename}.gbw exists in parentdir:", self.printlevel, 2)
                    if os.path.isfile(f"../{self.filename}.gbw") is True:
                        print_if_level("Yes. Copying file from parentdir to current dir", self.printlevel, 2)
                        shutil.copy(f"../{self.filename}.gbw", f"./{self.filename}.gbw")
                    else:
                        print_if_level("Found no file. ORCA will guess new orbitals", self.printlevel, 2)
            else:
                print_if_level(f"A GBW-file with same basename : {self.filename}.gbw is present", self.printlevel, 2)
                if self.autostart is False:
                    print_if_level("Autostart is False. ORCA will ignore any file present", self.printlevel, 2)
                else:
                    print_if_level("Autostart feature is active. ORCA will read GBW-file present.", self.printlevel, 2)

        # If 1st runcall, add this to inputfile
        if self.runcalls == 1:
            # first_iteration_input
            extraline = self.extraline + "\n" + self.first_iteration_input
        else:
            extraline = self.extraline

        if self.printlevel >= 2:
            print("Creating inputfile:", self.filename + '.inp')
            print("ORCA input:")
            print(self.orcasimpleinput)
            print(extraline)
            print(self.orcablocks)
            print("Charge: {}  Mult: {}".format(charge, mult))
        # Printing extra options chosen:
        if self.brokensym is True:
            if self.printlevel >= 2:
                print("Brokensymmetry SpinFlipping on! HSmult: {}.".format(self.HSmult))

            if self.HSmult is None:
                print("Error:HSmult keyword in ORCATheory has not been set. This is required. Exiting.")
                ashexit()
            if len(qmatomstoflip) == 0:
                print("Error: atomstoflip keyword needs to be set. This is required. Exiting.")
                ashexit()

            for flipatom, qmflipatom in zip(self.atomstoflip, qmatomstoflip):
                if self.printlevel >= 2:
                    print("Flipping atom: {} QMregionindex: {} Element: {}".format(flipatom, qmflipatom,
                                                                                   qm_elems[qmflipatom]))
        # DeltaSCF
        deltascfblock = None
        if self.deltaSCF is True:
            if self.printlevel >= 2:
                print("DeltaSCF option chosen. Will attempt MOM excited state SCF solution in first run")
                print("DeltaSCF PMOM:", self.deltaSCF_PMOM)
                print("Configuration line:", self.deltaSCF_confline)
            if mult == 1:
                if 'UKS' not in self.orcasimpleinput:
                    if 'UHF' not in self.orcasimpleinput:
                        print("Warning: Singlet DeltaSCF calculation requested but no UKS/UHF keyword present.")
                        print("Only doubly excited SCF states can be found ")

            deltascfblock = f"! DELTASCF \n%scf\n PMOM {self.deltaSCF_PMOM} \n {self.deltaSCF_confline}\nend"

        if self.extrabasis != "":
            if self.printlevel >= 2:
                print("Using extra basis ({}) on QM-region indices : {}".format(self.extrabasis, qmatoms_extrabasis))
        if self.dummyatoms:
            if self.printlevel >= 2:
                print("Dummy atoms defined:", self.dummyatoms)
        if self.ghostatoms:
            if self.printlevel >= 2:
                print("Ghost atoms defined:", self.ghostatoms)
        if self.fragment_indices:
            if self.printlevel >= 2:
                print("List of fragment indices defined:", fragment_indices)

        if PC is True:
            if self.printlevel >= 2:
                print("Pointcharge embedding is on!")
            create_orca_pcfile(self.filename, current_MM_coords, MMcharges)
            if self.brokensym is True:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                     charge, mult, extraline=extraline, HSmult=self.HSmult, Grad=Grad, Hessian=Hessian,
                                     moreadfile=self.moreadfile,
                                     atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis,
                                     extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                     fragment_indices=fragment_indices,
                                     atom_specific_basis_dict=self.atom_specific_basis_dict,
                                     ROHF_UHF_swap=self.ROHF_UHF_swap,
                                     deltaSCFblock=deltascfblock)
            else:
                create_orca_input_pc(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                     charge, mult, extraline=extraline, Grad=Grad, Hessian=Hessian,
                                     moreadfile=self.moreadfile,
                                     extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis,
                                     propertyblock=self.propertyblock,
                                     fragment_indices=fragment_indices,
                                     atom_specific_basis_dict=self.atom_specific_basis_dict,
                                     ROHF_UHF_swap=self.ROHF_UHF_swap,
                                     deltaSCFblock=deltascfblock)
        else:
            if self.brokensym is True:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        charge, mult, extraline=extraline, HSmult=self.HSmult, Grad=Grad,
                                        Hessian=Hessian, moreadfile=self.moreadfile,
                                        atomstoflip=qmatomstoflip, extrabasisatoms=qmatoms_extrabasis,
                                        extrabasis=self.extrabasis, propertyblock=self.propertyblock,
                                        ghostatoms=self.ghostatoms, dummyatoms=self.dummyatoms,
                                        ROHF_UHF_swap=self.ROHF_UHF_swap,
                                        fragment_indices=fragment_indices,
                                        atom_specific_basis_dict=self.atom_specific_basis_dict,
                                        deltaSCFblock=deltascfblock)
            else:
                create_orca_input_plain(self.filename, qm_elems, current_coords, self.orcasimpleinput, self.orcablocks,
                                        charge, mult, extraline=extraline, Grad=Grad, Hessian=Hessian,
                                        moreadfile=self.moreadfile,
                                        extrabasisatoms=qmatoms_extrabasis, extrabasis=self.extrabasis,
                                        propertyblock=self.propertyblock,
                                        ghostatoms=self.ghostatoms, dummyatoms=self.dummyatoms,
                                        ROHF_UHF_swap=self.ROHF_UHF_swap,
                                        fragment_indices=fragment_indices,
                                        atom_specific_basis_dict=self.atom_specific_basis_dict,
                                        deltaSCFblock=deltascfblock)

        # Run inputfile using ORCA parallelization. Take numcores argument.
        # print(BC.OKGREEN, "------------Running ORCA calculation-------------", BC.END)
        if self.printlevel >= 2:
            print(BC.OKGREEN, "ORCA Calculation starting.", BC.END)

        run_orca_SP_ORCApar(self.orcadir, self.filename + '.inp', numcores=numcores,
                            bind_to_core_option=self.bind_to_core_option,
                            check_for_errors=self.check_for_errors, check_for_warnings=self.check_for_warnings,
                            ignore_ORCA_error=self.ignore_ORCA_error)
        if self.printlevel >= 1:
            print(BC.OKGREEN, "ORCA Calculation done.", BC.END)

        outfile = self.filename + '.out'
        engradfile = self.filename + '.engrad'
        pcgradfile = self.filename + '.pcgrad'

        # Checking if finished.
        if self.ignore_ORCA_error is False:
            ORCAfinished, numiterations = checkORCAfinished(outfile)
            # Check if ORCA finished or not. Exiting if so
            if ORCAfinished is False:
                print(BC.FAIL, "Problem with ORCA run", BC.END)
                print(BC.OKBLUE, BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2)
                ashexit()

            if self.printlevel >= 1:
                print(f"ORCA converged in {numiterations} iterations")
        else:
            print("There was an ORCA error that was ignored by user-input")

        if self.ROHF_UHF_swap:
            print("\nROHF UHF swap feature active.")
            print("This means that a $new_job ORCA job was run with a ROHF-UHF noiter switch")
            print(f"Note that the relevant GBW file is then: {self.filename}_job2.gbw\n")
            print("Stored as self.gbwfile of this ORCATheory object")
            self.gbwfile = self.filename + '_job2.gbw'
        else:
            self.gbwfile = self.filename + '.gbw'

        # Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart, etc.)
        # using this theory object
        if self.brokensym is True:
            if self.printlevel >= 2:
                print(
                    "ORCA Flipspin calculation done. Now turning off brokensym in ORCA object for possible future calculations")
            self.brokensym = False
        # Turning off deltaSCF for future calcs
        if self.deltaSCF is True:
            print("DeltaSCF calculation done.")
            if self.deltaSCF_turn_off_automatically is True:
                print("deltaSCF_turn_off_automatically option is True. Turning off DELTASCF for future calculations.")
                self.deltaSCF = False
                deltascfblock = None
                if 'nososcf' not in self.orcasimpleinput:
                    print(
                        "Adding NOSOSCF to orcasimpleinput to avoid future calculations from falling back to ground-state")
                    self.orcasimpleinput = self.orcasimpleinput + ' nososcf'
                if 'nodamp' not in self.orcasimpleinput:
                    print(
                        "Adding NODAMP to orcasimpleinput to avoid future calculations from falling back to ground-state")
                    self.orcasimpleinput = self.orcasimpleinput + ' nodamp'
                if 'nolshift' not in self.orcasimpleinput:
                    print(
                        "Adding NOLSHIFT to orcasimpleinput to avoid future calculations from falling back to ground-state")
                    self.orcasimpleinput = self.orcasimpleinput + ' nolshift'
            else:
                print("deltaSCF_turn_off_automatically option if False. Will keep DeltaSCF settings")

        # Now that we have possibly run a ORCA job with moreadfile we now turn the moreadfile option off
        #  as we probably want to use the orbitals we created
        if self.moreadfile != None:
            print("First ORCATheory calculation finished.")
            # Now either keeping moreadfile or removing it. Default: removing
            if self.moreadfile_always == False:
                print("Now turning moreadfile option off.")
                self.moreadfile = None

        # Optional save ORCA output with filename according to label
        if self.save_output_with_label is True:
            shutil.copy(self.filename + '.out', self.filename + f'_{self.label}_{charge}_{mult}.out')

        # Keep outputfile from each run if requested
        if self.keep_each_run_output is True:
            print("\nkeep_each_run_output is True")
            print("Copying {} to {}".format(self.filename + '.out',
                                            self.filename + '_run{}'.format(self.runcalls) + '.out'))
            shutil.copy(self.filename + '.out', self.filename + '_run{}'.format(self.runcalls) + '.out')

        # Always make copy of last output file
        if self.keep_last_output is True:
            shutil.copy(self.filename + '.out', self.filename + '_last.out')

        # Save path to last GBW-file (used if ASH changes directories, e.g. goes from NumFreq)
        self.path_to_last_gbwfile_used = f"{os.getcwd()}/{self.filename}.gbw"

        # Print population analysis in each run if requested
        if self.print_population_analysis is True:
            if self.printlevel >= 2:
                print("\nPrinting Mulliken Population analysis:")
                print("-" * 30)
                charges = grabatomcharges_ORCA("Mulliken", self.filename + '.out')
                spinpops = grabspinpop_ORCA("Mulliken", self.filename + '.out')
                self.properties["Mulliken_charges"] = charges
                self.properties["Mulliken_spinpops"] = spinpops
                if len(spinpops) == 0 and len(charges) != 0:
                    print("{:<2} {:<2}  {:>10}".format(" ", " ", "Charge"))
                    for i, (el, ch) in enumerate(zip(qm_elems, charges)):
                        print("{:<2} {:<2}: {:>10.4f}".format(i, el, ch))
                    print()
                elif len(spinpops) != 0 and len(charges) != 0:
                    print("{:<2} {:<2}  {:>10} {:>10}".format(" ", " ", "Charge", "Spinpop"))
                    for i, (el, ch, sp) in enumerate(zip(qm_elems, charges, spinpops)):
                        print("{:<2} {:<2}: {:>10.4f} {:>10.4f}".format(i, el, ch, sp))
                    print()
                else:
                    print("Warning: No charges or spinpops were found in ORCA output. Continuing")
        # Grab energy
        if self.ignore_ORCA_error is False:
            self.energy = ORCAfinalenergygrab(outfile)
            if self.printlevel >= 1:
                print("ORCA energy:", self.energy)
        else:
            self.energy = ORCAfinalenergygrab(outfile)

            if self.energy is None:
                print("No energy could be found in ORCA outputfile.")
                print("Setting energy to 0.0 and returning")
                return 0.0
        # NMF
        if self.NMF is True:
            print("NMF option is active.")
            E_NMF = self.energy
            occupations = np.array(SCF_FODocc_grab(outfile))
            print("Fractional ccupations (Fermi distribution):", occupations)
            print("Now also calculating correlation energy from the fractional occupation numbers")
            print("Assuming Fermi distribution")
            Ec = openmmqmmm.functions.functions_elstructure.get_ec_entropy(occupations, self.NMF_sigma, method='fermi')
            print("Ec:", Ec)
            self.properties["NMF_occupations"] = occupations
            self.properties["E_NMF"] = E_NMF
            self.properties["NMF_Ec"] = Ec
            self.energy = self.energy + Ec

        # Grab possible properties
        # ICE-CI
        try:
            E_PT2_rest = float(pygrep("\'rest\' energy", self.filename + '.out')[-1])
            num_genCFGs, num_selected_CFGs, num_after_SD_CFGs = ICE_WF_CFG_CI_size(self.filename + '.out')
            self.properties["E_var"] = self.energy
            self.properties["E_PT2_rest"] = E_PT2_rest
            self.properties["num_genCFGs"] = num_genCFGs
            self.properties["num_selected_CFGs"] = num_selected_CFGs
            self.properties["num_after_SD_CFGs"] = num_after_SD_CFGs
        except:
            pass

        # TDDFT results
        if self.TDDFT is True:
            transition_energies = tddftgrab(f"{self.filename}.out")
            transition_intensities = tddftintens_grab(f"{self.filename}.out")

            self.properties["TDDFT_transition_energies"] = transition_energies
            self.properties["TDDFT_transition_intensities"] = transition_intensities

        # Grab timings from ORCA output
        orca_timings = ORCAtimingsgrab(outfile)

        # Initializing zero gradient array
        self.grad = np.zeros((len(qm_elems), 3))
        self.dipole_moment = None

        # XDM option: WFX file should have been created.
        if self.xdm == True:
            dispE, dispgrad = openmmqmmm.functions.functions_elstructure.xdm_run(wfxfile=self.filename + '.wfx',
                                                                          a1=self.xdm_a1, a2=self.xdm_a2,
                                                                          functional=self.xdm_func)
            if self.printlevel >= 2:
                print("XDM dispersion energy:", dispE)
            self.energy = self.energy + dispE
            if self.printlevel >= 2:
                print("DFT+XDM energy:", self.energy)
            # TODO: dispgrad not yet done
            self.grad = self.grad + dispgrad

        # Grab Hessian if calculated
        if Hessian is True:
            print("Reading Hessian from file:", self.filename + ".hess")
            self.hessian = Hessgrab(self.filename + ".hess")
            self.ir_intensities = grab_IR_intensities(self.filename + '.hess')

        if Grad is True:
            grad = ORCAgradientgrab(engradfile)
            self.grad = self.grad + grad
            if self.printlevel >= 3:
                print("ORCA gradient:", self.grad)

            if PC == True:
                # Print time to calculate ORCA QM-PC gradient
                if "pc_gradient" in orca_timings:
                    if self.printlevel >= 2:
                        print(
                            "Time calculating QM-Pointcharge gradient: {} seconds".format(orca_timings["pc_gradient"]))
                # Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                self.pcgrad = ORCApcgradientgrab(pcgradfile)
                if self.printlevel >= 2:
                    print(BC.OKBLUE, BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2, currprintlevel=self.printlevel,
                               currthreshold=1)
                return self.energy, self.grad, self.pcgrad
            else:
                if self.printlevel >= 2:
                    print(BC.OKBLUE, BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
                print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2, currprintlevel=self.printlevel,
                               currthreshold=1)
                return self.energy, self.grad

        else:
            if self.printlevel >= 2:
                print("Single-point ORCA energy:", self.energy)
                print(BC.OKBLUE, BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
            print_time_rel(module_init_time, modulename='ORCA run', moduleindex=2, currprintlevel=self.printlevel,
                           currthreshold=1)
            return self.energy


###############################################
# CHECKS FOR ORCA program
###############################################

def check_ORCA_location(orcadir, modulename="ORCATheory"):
    if orcadir != None:
        finalorcadir = orcadir
        print(BC.OKGREEN, f"Using orcadir path provided: {finalorcadir}", BC.END)
    else:
        print(BC.WARNING,
              f"No orcadir argument passed to {modulename}. Attempting to find orcadir variable in ASH settings file (~/ash_user_settings.ini)",
              BC.END)
        try:
            finalorcadir = openmmqmmm.settings_ash.settings_dict["orcadir"]
            print(BC.OKGREEN, "Using orcadir path provided from ASH settings file (~/ash_user_settings.ini): ",
                  finalorcadir, BC.END)
        except KeyError:
            print(BC.WARNING, "Found no orcadir variable in ASH settings file either.", BC.END)
            print(BC.WARNING, "Checking for ORCA in PATH environment variable.", BC.END)
            try:
                finalorcadir = os.path.dirname(shutil.which('orca'))
                print(BC.OKGREEN, "Found orca binary in PATH. Using the following directory:", finalorcadir, BC.END)
            except TypeError:
                print(BC.FAIL, "Found no orca binary in PATH environment variable either. Giving up.", BC.END)
                ashexit()
    return finalorcadir


def check_ORCAbinary(orcadir):
    """Checks if this is a proper working ORCA quantum chemistry binary
    Args:
        orcadir ([type]): [description]
    """
    print("Checking if ORCA binary works...", end="")
    try:
        p = sp.Popen([orcadir + "/orca"], stdout=sp.PIPE)
        out, err = p.communicate()
        if 'This program requires the name of a parameterfile' in str(out):
            print(BC.OKGREEN, "yes", BC.END)
            return True
        else:
            print(BC.FAIL, "Problem: ORCA binary: {} does not work. Exiting!".format(orcadir + '/orca'), BC.END)
            ashexit()
    except FileNotFoundError:
        print("ORCA binary was not found")
        ashexit()


# Once inputfiles are ready, organize them. We want open-shell calculation (e.g. oxidized) to reuse closed-shell GBW file
# https://www.machinelearningplus.com/python/parallel-processing-python/
# Good subprocess documentation: http://queirozf.com/entries/python-3-subprocess-examples
# https://shuzhanfan.github.io/2017/12/parallel-processing-python-subprocess/
# https://data-flair.training/blogs/python-multiprocessing/
# https://rsmith.home.xs4all.nl/programming/parallel-execution-with-python.html


# Run single-point ORCA calculation (Energy or Engrad). Assumes no ORCA parallelization.
# Function can be called by multiprocessing.


# Run ORCA single-point job using ORCA parallelization. Will add pal-block if numcores >1.
def run_orca_SP_ORCApar(orcadir, inpfile, numcores=1, check_for_warnings=True, check_for_errors=True,
                        bind_to_core_option=True, ignore_ORCA_error=False):
    if numcores > 1:
        palstring = '%pal \nnprocs {}\nend'.format(numcores)
        with open(inpfile) as ifile:
            insert_line_into_file(inpfile, '!', palstring, Once=True)
    basename = inpfile.replace('.inp', '')

    # LD_LIBRARY_PATH enforce: https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=10118
    # "-x LD_LIBRARY_PATH -x PATH"

    with open(basename + '.out', 'w') as ofile:
        try:
            if bind_to_core_option is True:
                # f"\"-x {orcadir} --bind-to none\""
                process = sp.run([orcadir + '/orca', inpfile, f"--bind-to none"], check=True, stdout=ofile,
                                 stderr=ofile, universal_newlines=True)
            else:
                process = sp.run([orcadir + '/orca', inpfile], check=True, stdout=ofile, stderr=ofile,
                                 universal_newlines=True)
            if check_for_errors:
                grab_ORCA_errors(basename + '.out')
            if check_for_warnings:
                grab_ORCA_warnings(basename + '.out')
        except Exception as e:
            print("Subprocess error! Exception message:", e)

            # We get an exception if
            print(BC.FAIL,
                  "ASH encountered a problem when running ORCA. Something went wrong, most likely ORCA ran into an error.",
                  BC.END)
            print(BC.FAIL, f"Please check the ORCA outputfile: {basename + '.out'} for error messages", BC.END)
            print()
            if check_for_errors:
                grab_ORCA_errors(basename + '.out')
            if check_for_warnings:
                grab_ORCA_warnings(basename + '.out')
            print("ignore_ORCA_error:", ignore_ORCA_error)
            if ignore_ORCA_error is True:
                print("ignore_ORCA_error here")
                return
            else:
                ashexit()


def grab_ORCA_warnings(filename):
    warning_lines = []
    # Error-words to search for
    # TODO: Avoid searching though file multiple times.
    # TODO: Write pygrep version that supports list of search-strings
    warning_strings = ['WARNING', 'warning', 'Warning']
    for warnstring in warning_strings:
        warn_l = pygrep2(warnstring, filename, errors="ignore")
        warning_lines += warn_l

    warnings = []
    # Lines that are not useful warnings
    ignore_lines = ['                       Please study these wa', '                                        WARNINGS',
                    'Warning: in a DFT calculation', 'WARNING: Old DensityContainer',
                    'WARNING: your system is open-shell']
    for warn in warning_lines:
        false_positive = any(warn.startswith(ign) for ign in ignore_lines)
        if false_positive is False:
            warnings.append(warn)
    if len(warnings):
        print("Found warning messages in ORCA outputfile:")
        print(*warnings)


def grab_ORCA_errors(filename):
    error_lines = []
    # Error-words to search for
    # TODO: Avoid searching though file multiple times.
    # TODO: Write pygrep version that supports list of search-strings
    error_strings = ['error', 'Error', 'ERROR', 'aborting']
    for errstring in error_strings:
        error_l = pygrep2(errstring, filename, errors="ignore")
        for e in error_l:
            if e not in error_lines:
                error_lines.append(e)

    errors = []
    # Lines that are not errors
    ignore_lines = ['   Iter.        energy            ||Error||_2', ' WARNING: the maximum gradient error',
                    '           *** ORCA-CIS/TD-DFT FINISHED WITHOUT ERROR', '   Startup', '   DIIS-Error', ' DIIS',
                    'sum of PNO error', '  Last DIIS Error', '    DIIS-Error', ' Sum of total truncation errors',
                    '  Sum of total UMP2 truncation', ]
    for err in error_lines:
        false_positive = any(err.startswith(ign) for ign in ignore_lines)
        if false_positive is False:
            errors.append(err)
    if len(errors):
        print("Found possible error messages in ORCA outputfile:")
        print(*errors)


# Check if ORCA finished.
# Todo: Use reverse-read instead to speed up?
def checkORCAfinished(file):
    iter = None
    with open(file, errors="ignore") as f:
        for line in f:
            if 'SCF CONVERGED AFTER' in line:
                iter = line.split()[-3]
            if 'TOTAL RUN TIME:' in line:
                return True, iter
    return False, None


def checkORCAOptfinished(file):
    converged = False
    with open(file, errors="ignore") as f:
        for line in f:
            if 'THE OPTIMIZATION HAS CONVERGED' in line:
                converged = True
            if converged == True:
                if '***               (AFTER' in line:
                    cycles = line.split()[2]
                    print("ORCA Optimization converged in {} cycles".format(cycles))
        return converged


# Grab Final single point energy. Ignoring possible encoding errors in file
def ORCAfinalenergygrab(file, errors='ignore'):
    Energy = None
    with open(file, errors=errors) as f:
        for line in f:
            if 'FINAL SINGLE POINT ENERGY' in line:
                if "Wavefunction not fully converged!" in line:
                    print("ORCA WF not fully converged!")
                    print("Not using energy. Modify ORCA settings")
                    ashexit()
                else:
                    # Changing: sometimes ORCA adds info to the right of energy
                    # Energy=float(line.split()[-1])
                    if "(MM)" in line:
                        Energy = float(line.split()[5])
                    else:
                        Energy = float(line.split()[4])
    if Energy is None:
        print(BC.FAIL, "ASH found no energy in file:", file, BC.END)
        print(BC.FAIL, "Something went wrong with ORCA run. Check ORCA outputfile:", file, BC.END)
        print(BC.OKBLUE, BC.BOLD, "------------ENDING ORCA-INTERFACE-------------", BC.END)
        return None
    return Energy


# Grab ORCA timings. Return dictionary
def ORCAtimingsgrab(file):
    timings = {}  # in seconds
    try:
        with open(file, errors="ignore") as f:
            for line in f:
                if 'Calculating one electron integrals' in line:
                    one_elec_integrals = float(line.split()[-2].replace("(", ""))
                    timings["one_elec_integrals"] = one_elec_integrals
                if 'SCF Gradient evaluation         ...' in line:
                    time_scfgrad = float(line.split()[4])
                    timings["time_scfgrad"] = time_scfgrad
                if 'SCF iterations                  ...' in line:
                    time_scfiterations = float(line.split()[3])
                    timings["time_scfiterations"] = time_scfiterations
                if 'GTO integral calculation        ...' in line:
                    time_gtointegrals = float(line.split()[4])
                    timings["time_gtointegrals"] = time_gtointegrals
                if 'SCF Gradient evaluation         ...' in line:
                    time_scfgrad = float(line.split()[4])
                    timings["time_scfgrad"] = time_scfgrad
                if 'Sum of individual times         ...:' in line:
                    total_time = float(line.split()[4])
                    timings["total_time"] = total_time
                if 'One electron gradient       ....' in line:
                    one_elec_gradient = float(line.split()[4])
                    timings["one_elec_gradient"] = one_elec_gradient
                if 'RI-J Coulomb gradient       ....' in line:
                    rij_coulomb_gradient = float(line.split()[4])
                    timings["rij_coulomb_gradient"] = rij_coulomb_gradient
                if 'XC gradient                 ....' in line:
                    xc_gradient = float(line.split()[3])
                    timings["xc_gradient"] = xc_gradient
                if 'Point charge gradient       ....' in line:
                    pc_gradient = float(line.split()[4])
                    timings["pc_gradient"] = pc_gradient
    except:
        pass
    return timings


# Grab gradient from ORCA engrad file
def ORCAgradientgrab(engradfile):
    grab = False
    numatomsgrab = False
    row = 0
    col = 0
    with open(engradfile) as gradfile:
        for line in gradfile:
            if numatomsgrab == True:
                if '#' not in line:
                    numatoms = int(line.split()[0])
                    # Initializing array
                    gradient = np.zeros((numatoms, 3))
                    numatomsgrab = False
            if '# Number of atoms' in line:
                numatomsgrab = True
            if grab == True:
                if '#' not in line:
                    val = float(line.split()[0])
                    gradient[row, col] = val
                    if col == 2:
                        row += 1
                        col = 0
                    else:
                        col += 1
            if '# The current gradient in Eh/bohr' in line:
                grab = True
            if '# The atomic numbers and ' in line:
                grab = False
    return gradient


# Grab pointcharge gradient from ORCA pcgrad file
def ORCApcgradientgrab(pcgradfile):
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
            if 'Total Dipole Moment    :' in line:
                dipole_moment.append(float(line.split()[-3]))
                dipole_moment.append(float(line.split()[-2]))
                dipole_moment.append(float(line.split()[-1]))
    return dipole_moment


def grab_polarizability_tensor(outfile):
    pz_tensor = np.zeros((3, 3))
    diag_pz_tensor = []
    count = 0
    grab = False;
    grab2 = False;
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
                    grab = False;
                    grab2 = False;
                    grab3 = False
            if grab is True:
                if 'The raw cartesian tensor' in line:
                    grab2 = True
                if 'diagonalized tensor:' in line:
                    grab2 = False
                    grab3 = True
                if grab2 is True and len(line.split()) == 3:
                    pz_tensor[count, 0] = float(line.split()[0])
                    pz_tensor[count, 1] = float(line.split()[1])
                    pz_tensor[count, 2] = float(line.split()[2])
                    count += 1
            if 'STATIC POLARIZABILITY TENSOR' in line:
                print("grab True")
                grab = True
    return pz_tensor, diag_pz_tensor


# Grab multiple Final single point energies in output. e.g. new_job calculation


# Grab SCF energy (non-dispersion corrected)


# Get reference energy and correlation energy from a single post-HF calculation
# Support regular CC, DLPNO-CC, CC-12, DLPNO-CC-F12
# Note: CC-12 untested


# Grab XES state energies and intensities from ORCA output


# Grab TDDFT state energies from ORCA output
def tddftgrab(file):
    tddftstates = []
    tddft = True
    tddftgrab = False
    if tddft == True:
        with open(file) as f:
            for line in f:
                if tddftgrab == True:
                    if 'STATE' in line:
                        if 'eV' in line:
                            tddftstates.append(float(line.split()[5]))
                        tddftgrab = True
                if 'the weight of the individual excitations' in line:
                    tddftgrab = True
    return tddftstates


# Grab TDDFT state intensities from ORCA output
def tddftintens_grab(file):
    intensities = []
    tddftgrab = False
    with open(file) as f:
        for line in f:
            if tddftgrab == True:
                if '->' in line:
                    intensities.append(float(line.split()[-5]))
                if len(line.split()) == 0:
                    tddftgrab = False
            if 'fosc(D2)' in line:
                tddftgrab = True
    return intensities


# Grab TDDFT orbital pairs from ORCA output


def grab_IR_intensities(filename):
    grab = False
    intensities = []
    with open(filename) as f:
        for line in f:
            if grab:
                if len(line.split()) == 6:
                    intens = float(line.split()[2])
                    intensities.append(intens)
            if '$ir_spectrum' in line:
                grab = True
    return intensities


# Grab energies from unrelaxed scan in ORCA (paras block type)


# TODO: Limited older version. Better version below


# Grab <S**2> expectation values from outputfile


# Function to grab masses and elements from ORCA Hessian file


# Function to write ORCA-style Hessian file

def write_ORCA_Hessfile(hessian, coords, elems, masses, hessatoms, outputname):
    hessdim = hessian.shape[0]
    orcahessfile = open(outputname, 'w')
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
        for i in range(0, hessdim):

            if chunk == chunks - 1:
                for k in range(index, index + left):
                    tempvar = tempvar + "         " + str(hessian[i, k])
            else:
                for k in range(index, index + orcahesscoldim):
                    tempvar = tempvar + "         " + str(hessian[i, k])
            orcahessfile.write("    " + str(i) + "   " + str(tempvar) + "\n")
            tempvar = "";
            temp2var = ""
        index += 5
    orcahessfile.write("\n")
    orcahessfile.write("# The atoms: label  mass x y z (in bohrs)\n")
    orcahessfile.write("$atoms\n")
    orcahessfile.write(str(len(elems)) + "\n")

    # Write coordinates and masses to Orca Hessian file
    # print("hessatoms", hessatoms)
    # print("masses ", masses)
    # print("elems ", elems)
    # print("coords", coords)
    # print(len(elems))
    # print(len(coords))
    # print(len(hessatoms))
    # print(len(masses))
    # TODO. Note. Changed things. We now don't go through hessatoms and analyze atom indices for full system
    # Either full system lists were passed or partial-system lists
    # for atom, mass in zip(hessatoms, masses):
    for el, mass, coord in zip(elems, masses, coords):
        # mass=atommass[elements.index(elems[atom-1].lower())]
        # print("atom:", atom)
        # print("mass:", mass)
        # print(str(elems[atom]))
        # print(str(mass))
        # print(str(coords[atom][0]/openmmqmmm.constants.bohr2ang))
        # print(str(coords[atom][1]/openmmqmmm.constants.bohr2ang))
        # print(str(coords[atom][2]/openmmqmmm.constants.bohr2ang))
        # orcahessfile.write(" "+str(elems[atom])+'    '+str(mass)+"  "+str(coords[atom][0]/openmmqmmm.constants.bohr2ang)+
        #                   " "+str(coords[atom][1]/openmmqmmm.constants.bohr2ang)+" "+str(coords[atom][2]/openmmqmmm.constants.bohr2ang)+"\n")
        orcahessfile.write(" " + el + '    ' + str(mass) + "  " + str(coord[0] / openmmqmmm.constants.bohr2ang) +
                           " " + str(coord[1] / openmmqmmm.constants.bohr2ang) + " " + str(
            coord[2] / openmmqmmm.constants.bohr2ang) + "\n")
    orcahessfile.write("\n")
    orcahessfile.write("\n")
    orcahessfile.close()
    print("")
    print("ORCA-style Hessian written to:", outputname)


# Grab frequencies from ORCA-Hessian file


# Function to grab Hessian from ORCA-Hessian file
def Hessgrab(hessfile):
    hesstake = False
    j = 0
    orcacoldim = 5
    shiftpar = 0
    lastchunk = False
    grabsize = False
    with open(hessfile) as hfile:
        for line in hfile:

            if '$vibrational_frequencies' in line:
                hesstake = False
                continue
            if hesstake == True and len(line.split()) == 1 and grabsize == True:
                grabsize = False
                hessdim = int(line.split()[0])

                hessarray2d = np.zeros((hessdim, hessdim))

            if hesstake == True and lastchunk == True:
                if len(line.split()) == hessdim - shiftpar + 1:
                    for i in range(0, hessdim - shiftpar):
                        hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                    j += 1
            elif hesstake == True and len(line.split()) == 5:
                continue
                # Headerline
            if hesstake == True and len(line.split()) == 6:
                # Hessianline
                for i in range(0, orcacoldim):
                    hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                j += 1
                if j == hessdim:
                    shiftpar += orcacoldim
                    j = 0
                    if hessdim - shiftpar < orcacoldim:
                        lastchunk = True
            if '$hessian' in line:
                hesstake = True
                grabsize = True
        return hessarray2d


# Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Compound method version. Doing both redox states in same job.
# Adds specific basis set on atoms not defined as solute-atoms.


# Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# new_job feature. Doing both redox states in same job.
# Works buts discouraged.


# Create gas ORCA inputfile from elems,coords, input, charge, mult. No pointcharges.
# new_job version. Works but discouraged.


# Create gas ORCA inputfile from elems,coords, input, charge, mult. No pointcharges.
# compoundmethod version.


# Create PC-embedded ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Allows for extraline that could be another '!' line or block-inputline.
def create_orca_input_pc(name, elems, coords, orcasimpleinput, orcablockinput, charge, mult, Grad=False, extraline='',
                         HSmult=None, atomstoflip=None, Hessian=False, extrabasisatoms=None, extrabasis=None,
                         atom_specific_basis_dict=None, extraspecialbasisatoms=None, extraspecialbasis=None,
                         moreadfile=None, propertyblock=None, fragment_indices=None, ROHF_UHF_swap=False,
                         deltaSCFblock=None):
    if extrabasisatoms is None:
        extrabasisatoms = []
    pcfile = name + '.pc'
    with open(name + '.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput + '\n')
        if extraline != '':
            orcafile.write(extraline + '\n')
        if Grad == True:
            orcafile.write('! Engrad' + '\n')
        if Hessian == True:
            orcafile.write('! Freq' + '\n')
        if moreadfile is not None:
            print("MOREAD option active. Will read orbitals from file:", moreadfile)
            orcafile.write('\n! MOREAD' + '\n')
            orcafile.write('%moinp \"{}\"'.format(moreadfile) + '\n')
        orcafile.write('%pointcharges "{}"\n'.format(pcfile))
        orcafile.write(orcablockinput + '\n')
        if atomstoflip is not None:
            atomstoflipstring = ','.join(map(str, atomstoflip))
            orcafile.write('%scf\n')
            orcafile.write('Flipspin {}'.format(atomstoflipstring) + '\n')
            orcafile.write('FinalMs {}'.format((mult - 1) / 2) + '\n')
            orcafile.write('end  \n')
        orcafile.write('\n')
        # DELTASCF
        if deltaSCFblock is not None:
            orcafile.write(deltaSCFblock)
        orcafile.write('\n')
        if atomstoflip is not None:
            orcafile.write('*xyz {} {}\n'.format(charge, HSmult))
        else:
            orcafile.write('*xyz {} {}\n'.format(charge, mult))
        # Writing coordinates. Adding extrabasis keyword for atom if option active
        for i, (el, c) in enumerate(zip(elems, coords)):
            if i in extrabasisatoms:
                orcafile.write('{} {} {} {} newgto \"{}\" end\n'.format(el, c[0], c[1], c[2], extrabasis))
            # Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                print("Writing atom-specific basis for atom:", i)
                # Regular line
                orcafile.write('{} {} {} {} \n'.format(el, c[0], c[1], c[2]))
                for bline in atom_specific_basis_dict[(el, i)]:
                    orcafile.write(str(bline))
            # Adding fragment specification
            elif fragment_indices != None:
                fragmentindex = search_list_of_lists_for_index(i, fragment_indices)
                # To prevent linkatoms:
                if fragmentindex != None:
                    orcafile.write('{} {} {} {} \n'.format(f"{el}({fragmentindex + 1})", c[0], c[1], c[2]))
            else:
                orcafile.write('{} {} {} {} \n'.format(el, c[0], c[1], c[2]))
        orcafile.write('*\n')
        if propertyblock != None:
            orcafile.write(propertyblock)
        # For ROHF job, add newjob and switch to UHF noiter
        if ROHF_UHF_swap:
            newjobline = f"""\n$new_job
{orcasimpleinput.replace("ROHF", "UHF noiter ")}
{orcablockinput}
* xyz {charge} {mult}
"""
            orcafile.write(newjobline)
            for i, (el, c) in enumerate(zip(elems, coords)):
                orcafile.write('{} {} {} {} \n'.format(el, c[0], c[1], c[2]))
            orcafile.write('*\n')


# Create simple ORCA inputfile from elems,coords, input, charge, mult,pointcharges
# Allows for extraline that could be another '!' line or block-inputline.
def create_orca_input_plain(name, elems, coords, orcasimpleinput, orcablockinput, charge, mult, Grad=False,
                            Hessian=False, extraline='',
                            HSmult=None, atomstoflip=None, extrabasis=None, extrabasisatoms=None, moreadfile=None,
                            propertyblock=None,
                            ghostatoms=None, dummyatoms=None, fragment_indices=None, atom_specific_basis_dict=None,
                            ROHF_UHF_swap=False,
                            deltaSCFblock=None):
    if extrabasisatoms == None:
        extrabasisatoms = []
    if ghostatoms == None:
        ghostatoms = []
    if dummyatoms == None:
        dummyatoms = []
    with open(name + '.inp', 'w') as orcafile:
        orcafile.write(orcasimpleinput + '\n')
        if extraline != '':
            orcafile.write(extraline)
        if Grad is True:
            orcafile.write('! Engrad' + '\n')
        if Hessian is True:
            orcafile.write('! Freq' + '\n')
        if moreadfile is not None:
            print("MOREAD option active. Will read orbitals from file:", moreadfile)
            orcafile.write('\n! MOREAD' + '\n')
            orcafile.write('%moinp \"{}\"'.format(moreadfile) + '\n')
        orcafile.write(orcablockinput)
        if atomstoflip is not None:
            if type(atomstoflip) == int:
                atomstoflipstring = str(atomstoflip)
            else:
                atomstoflipstring = ','.join(map(str, atomstoflip))
            orcafile.write('%scf\n')
            orcafile.write('Flipspin {}'.format(atomstoflipstring) + '\n')
            orcafile.write('FinalMs {}'.format((mult - 1) / 2) + '\n')
            orcafile.write('end  \n')
            orcafile.write('\n')
        # DELTASCF
        if deltaSCFblock is not None:
            orcafile.write(deltaSCFblock)
        orcafile.write('\n')
        if atomstoflip is not None:
            orcafile.write('*xyz {} {}\n'.format(charge, HSmult))
        else:
            orcafile.write('*xyz {} {}\n'.format(charge, mult))

        for i, (el, c) in enumerate(zip(elems, coords)):
            # Extra basis on each atom
            if i in extrabasisatoms:
                orcafile.write('{} {} {} {} newgto \"{}\" end\n'.format(el, c[0], c[1], c[2], extrabasis))
            # Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                print("Writing atom-specific basis for atom:", i)
                # Regular line
                orcafile.write('{} {} {} {} \n'.format(el, c[0], c[1], c[2]))
                for bline in atom_specific_basis_dict[(el, i)]:
                    orcafile.write(str(bline))
            # Setting atom to be a ghost atom
            elif i in ghostatoms:
                orcafile.write('{}{} {} {} {} \n'.format(el, ":", c[0], c[1], c[2]))
            elif i in dummyatoms:
                orcafile.write('{} {} {} {} \n'.format("DA", c[0], c[1], c[2]))
            # Adding fragment specification
            elif fragment_indices != None:
                fragmentindex = search_list_of_lists_for_index(i, fragment_indices)
                orcafile.write('{} {} {} {} \n'.format(f"{el}({fragmentindex + 1})", c[0], c[1], c[2]))
            else:
                orcafile.write('{} {} {} {} \n'.format(el, c[0], c[1], c[2]))
        orcafile.write('*\n')
        if propertyblock != None:
            orcafile.write(propertyblock)
        # For ROHF job, add newjob and switch to UHF noiter
        if ROHF_UHF_swap:
            newjobline = f"""\n$new_job
{orcasimpleinput.replace("ROHF", "UHF noiter ")}
{orcablockinput}
* xyz {charge} {mult}
"""
            orcafile.write(newjobline)
            for i, (el, c) in enumerate(zip(elems, coords)):
                orcafile.write('{} {} {} {} \n'.format(el, c[0], c[1], c[2]))
            orcafile.write('*\n')


# Create ORCA pointcharge file based on provided list of elems and coords (MM region elems and coords)
# and list of point charges of MM atoms
def create_orca_pcfile(name, coords, listofcharges):
    with open(name + '.pc', 'w') as pcfile:
        pcfile.write(str(len(listofcharges)) + '\n')
        for p, c in zip(listofcharges, coords):
            line = "{} {} {} {}".format(p, c[0], c[1], c[2])
            pcfile.write(line + '\n')


# Chargemodel select. Creates ORCA-inputline with appropriate keywords
# To be added to ORCA input.
def chargemodel_select(chargemodel):
    extraline = ""
    if chargemodel == 'NPA':
        extraline = '! NPA'
    elif chargemodel == 'CHELPG':
        extraline = '! CHELPG'
    elif chargemodel == 'Hirshfeld':
        extraline = '! Hirshfeld'
    elif chargemodel == 'CM5':
        extraline = '! Hirshfeld'
    elif chargemodel == 'Mulliken':
        pass
    elif chargemodel == 'Loewdin':
        pass
    elif chargemodel == 'DDEC6':
        pass
    elif chargemodel == "IAO":
        extraline = '\n%loc LocMet IAOIBO \n T_CORE -99999999 end'

    return extraline


# Grabbing spin populations
def grabspinpop_ORCA(chargemodel, outputfile):
    grab = False
    coordgrab = False
    spinpops = []
    BS = False  # if broken-symmetry job
    numatoms = int(pygrep('Number of atoms                             ...', outputfile)[-1])
    # if
    if len(pygrep2("WARNING: Broken symmetry calculations", outputfile)):
        BS = True

    if chargemodel == "Mulliken":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab is True:
                    if 'Sum of atomic' in line:
                        grab = False
                    elif '------' not in line:
                        spinpops.append(float(line.split()[-1]))
                if 'MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS' in line:
                    grab = True
    elif chargemodel == "Loewdin":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab is True:
                    if 'Sum of atomic' in line:
                        grab = False
                    elif len(line.replace(' ', '')) < 2:
                        grab = False
                    elif '------' not in line:
                        spinpops.append(float(line.split()[-1]))
                if 'LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS' in line:
                    grab = True
    else:
        print("Unknown chargemodel. Exiting...")
        ashexit()
    # If BS then we have grabbed charges for both high-spin and BS solution
    if BS is True:
        print("Broken-symmetry job detected. Only taking BS-state populations")
        # spinpops=spinpops[int(len(spinpops)/2):]
        if len(spinpops) != numatoms:
            spinpops = spinpops[-numatoms:]
    # if len(spinpops) == 0:
    #    print("Warning: No spinpopulations were found in ORCA outputfile")
    return spinpops


def grabatomcharges_ORCA(chargemodel, outputfile):
    grab = False
    coordgrab = False
    charges = []
    BS = False  # if broken-symmetry job
    column = None

    numatoms = int(pygrep('Number of atoms                             ...', outputfile)[-1])

    # if
    if len(pygrep2("WARNING: Broken symmetry calculations", outputfile)):
        BS = True

    if chargemodel == "NPA" or chargemodel == "NBO":
        print("Warning: NPA/NBO charge-option in ORCA requires setting environment variable NBOEXE:")
        print("e.g. export NBOEXE=/path/to/nbo7.exe")
        with open(outputfile) as ofile:
            for line in ofile:
                if grab == True:
                    if '=======' in line:
                        grab = False
                    elif '------' not in line:
                        charges.append(float(line.split()[2]))
                if 'Atom No    Charge        Core      Valence    Rydberg      Total' in line:
                    grab = True
    elif chargemodel.upper() == "CHELPG":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab == True:
                    if 'Total charge: ' in line:
                        grab = False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-1]))
                if 'CHELPG Charges' in line:
                    grab = True
                    # Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges = []
    elif chargemodel.upper() == "HIRSHFELD":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab == True:
                    if len(line) < 3:
                        grab = False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-2]))
                if '  ATOM     CHARGE      SPIN' in line:
                    grab = True
                    # Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges = []
    elif chargemodel.upper() == "CM5":
        elems = []
        coords = []
        with open(outputfile) as ofile:
            for line in ofile:
                # Getting coordinates as used in CM5 definition
                if coordgrab is True:
                    if '----------------------' not in line:
                        if len(line.split()) < 2:
                            coordgrab = False
                        else:
                            elems.append(line.split()[0])
                            coords_x = float(line.split()[1]);
                            coords_y = float(line.split()[2]);
                            coords_z = float(line.split()[3])
                            coords.append([coords_x, coords_y, coords_z])
                if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
                    coordgrab = True
                if grab == True:
                    if len(line) < 3:
                        grab = False
                    if len(line.split()) == 4:
                        charges.append(float(line.split()[-2]))
                if '  ATOM     CHARGE      SPIN' in line:
                    # Setting charges list to zero in case of multiple charge-tables. Means we grab second table
                    charges = []
                    grab = True
        print("Hirshfeld charges :", charges)
        atomicnumbers = openmmqmmm.modules.module_coords.elemstonuccharges(elems)
        charges = openmmqmmm.functions.functions_elstructure.calc_cm5(atomicnumbers, coords, charges)
        print("CM5 charges :", list(charges))
    elif chargemodel.upper() == "MULLIKEN":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab == True:
                    if 'Sum of atomic' in line:
                        grab = False
                    elif '------' not in line:
                        charges.append(float(line.split()[column]))
                if 'MULLIKEN ATOMIC CHARGES' in line:
                    grab = True
                    if 'SPIN POPULATIONS' in line:
                        column = -2
                    else:
                        column = -1

    elif chargemodel.upper() == "LOEWDIN":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab == True:
                    if 'Sum of atomic' in line:
                        grab = False
                    elif len(line.replace(' ', '')) < 2:
                        grab = False
                    elif '------' not in line:
                        charges.append(float(line.split()[column]))
                if 'LOEWDIN ATOMIC CHARGES' in line:
                    grab = True
                    if 'SPIN POPULATIONS' in line:
                        column = -2
                    else:
                        column = -1
    elif chargemodel.upper() == "IAO":
        with open(outputfile) as ofile:
            for line in ofile:
                if grab == True:
                    if 'Sum of atomic' in line:
                        grab = False
                    elif '------' not in line:
                        if 'Warning' not in line:
                            print("line:", line)
                            charges.append(float(line.split()[-1]))
                if 'IAO PARTIAL CHARGES' in line:
                    grab = True
    else:
        print("Unknown chargemodel. Exiting...")
        ashexit()

    # If BS then we have grabbed charges for both high-spin and BS solution
    if BS is True:
        print("Broken-symmetry job detected. Only taking BS-state populations")
        if len(charges) != numatoms:
            charges = charges[numatoms:]
        print("charges:", charges)
    return charges


# Wrapper around interactive orca_plot
# Todo: add TDDFT difference density, natural orbitals, MDCI spin density?

    # print(p.returncode)


# Grab IPs from an EOM-IP calculation and also largest singles amplitudes. Approximation to Dyson norm.


# Reading stability analysis from output. Returns true if stab-analysis good, otherwise falsee
# If no stability analysis present in output, then also return true


def SCF_FODocc_grab(filename):
    occgrab = False
    occupations = []
    with open(filename) as f:
        for line in f:
            if occgrab is True:
                if '***********' in line:
                    return occupations
                if ' SPIN DOWN' in line:
                    occgrab = False
                    return occupations
                if len(line.split()) == 4:
                    if '  NO   OCC' not in line:
                        occupations.append(float(line.split()[1]))
            if 'SPIN UP ORBITALS' in line or 'ORBITAL ENERGIES' in line:
                occgrab = True
    return occupations


# Grab ICE-WF info from CASSCF job


# Grab ICE-WF CFG info from CI job
def ICE_WF_CFG_CI_size(filename):
    num_after_SD_CFGs = 0
    num_genCFGs = 0
    num_selected_CFGs = 0
    with open(filename) as g:
        for line in g:
            if '# of configurations after S+D' in line:
                num_after_SD_CFGs = int(line.split()[-1])
            if '# of configurations after Selection' in line:
                num_selected_CFGs = int(line.split()[-1])
            if ' # of generator configurations' in line:
                num_genCFGs = int(line.split()[5])
    return num_genCFGs, num_selected_CFGs, num_after_SD_CFGs


# Charge/mult must be in fragments


# Writes the ORCA-style .engrad file that the generated otool_external script
# (see create_ASH_otool below) produces for ORCA's ExtOpt driver.
def print_gradient_in_ORCAformat(energy, gradient, basename, extrabasename="_EXT"):
    numatoms = len(gradient)
    with open(basename + extrabasename + ".engrad", "w") as f:
        f.write("#\n")
        f.write("# Number of atoms\n")
        f.write("#\n")
        f.write(f"         {numatoms}\n")
        f.write("#\n")
        f.write("# The current total energy in E\n")
        f.write("#\n")
        f.write("     {}\n".format(energy))
        f.write("#\n")
        f.write("# The current gradient in Eh/Bohr\n")
        f.write("#\n")
        for g in gradient:
            for gg in g:
                f.write("{}\n".format(gg))
        f.write("#\n")
        f.write("# The atomic numbers and current coordinates in Bohr\n")
        f.write("#\n")


def create_ASH_otool(basename=None, theoryfile=None, scriptlocation=None, charge=None, mult=None):
    import stat
    with open(scriptlocation + "/otool_external", 'w') as otool:
        otool.write("#!/usr/bin/env python3\n")
        otool.write("from openmmqmmm import *\n")
        otool.write("import pickle\n")
        otool.write("import numpy as np\n\n")
        otool.write("frag=Fragment(xyzfile=\"{}.xyz\")\n".format(basename))
        otool.write("\n")
        otool.write("#Unpickling theory object\n")
        otool.write("theory = pickle.load(open(\"{}\", \"rb\" ))\n".format(theoryfile))
        otool.write(
            "result=Singlepoint(theory=theory,fragment=frag,Grad=True, charge={}, mult={})\n".format(charge, mult))
        otool.write("energy = result.energy\n")
        otool.write("gradient = result.gradient\n")
        otool.write("print(gradient)\n")
        otool.write(
            "openmmqmmm.interfaces.interface_ORCA.print_gradient_in_ORCAformat(energy,gradient,\"{}\")\n".format(basename))
    st = os.stat(scriptlocation + "/otool_external")
    os.chmod(scriptlocation + "/otool_external", st.st_mode | stat.S_IEXEC)


# Using ORCA as External Optimizer for ASH
# Will only work for theories that can be pickled.
def ORCA_External_Optimizer(fragment=None, theory=None, orcadir=None, charge=None, mult=None,
                            ORCA_jobkeyword="Opt", ORCA_blockinput="", actatoms=None):
    print_line_with_mainheader("ORCA_External_Optimizer")
    if fragment is None or theory is None:
        print("ORCA_External_Optimizer requires fragment and theory keywords")
        ashexit()

    if charge is None or mult is None:
        print(BC.WARNING, "Warning: Charge/mult was not provided to ORCA_External_Optimizer", BC.END)
        if fragment.charge != None and fragment.mult != None:
            print(BC.WARNING,
                  "Fragment contains charge/mult information: Charge: {} Mult: {} Using this instead".format(
                      fragment.charge, fragment.mult), BC.END)
            print(BC.WARNING, "Make sure this is what you want!", BC.END)
            charge = fragment.charge;
            mult = fragment.mult
        else:
            print(BC.FAIL, "No charge/mult information present in fragment either. Exiting.", BC.END)
            ashexit()

    # Making sure we have a working ORCA location
    print("Checking for ORCA location")
    orcadir = check_ORCA_location(orcadir, modulename="ORCA_External_Optimizer")
    # Making sure ORCA binary works (and is not orca the screenreader)
    check_ORCAbinary(orcadir)
    # Adding orcadir to PATH. Only required if ORCA not in PATH already
    if orcadir != None:
        os.environ["PATH"] += os.pathsep + orcadir

    # Pickle for serializing theory object
    import pickle

    # Serialize theory object for later use
    theoryfilename = "theory.saved"
    pickle.dump(theory, open(theoryfilename, "wb"))

    # Write otool_script once in location that ORCA will launch. This is an ASH E+Grad calculator
    # ORCA will call : otool_external test_EXT.extinp.tmp
    # ASH_otool creates basename_Ext.engrad that ORCA reads
    basename = "ORCAEXTERNAL"
    scriptlocation = "."
    os.environ["PATH"] += os.pathsep + "."
    create_ASH_otool(basename=basename, theoryfile=theoryfilename, scriptlocation=scriptlocation, charge=charge,
                     mult=mult)

    # Create XYZ-file for ORCA-Extopt
    xyzfile = "ASH-xyzfile.xyz"
    fragment.write_xyzfile(xyzfile)

    # Active atoms become inverted constraints
    constraintsblock = ""
    if actatoms is not None:
        print("Activeatoms list was provided. This means that we need to provide constraints to ORCA")
        frozenatoms = listdiff(fragment.allatoms, actatoms)
        print("Freezing the non-active atoms:", frozenatoms)
        cons = []
        for f in frozenatoms:
            cons.append(f"{{C {f} C}}\n")
        consstring = ''.join(cons)
        constraintsblock = f"""%geom Constraints
{consstring}end
end
"""
    # ORCA input file
    with open(basename + ".inp", 'w') as o:
        o.write(f"! ExtOpt {ORCA_jobkeyword}\n")
        o.write("\n")
        o.write(f"{ORCA_blockinput}")
        o.write(f"{constraintsblock}")
        o.write("%method\n")
        o.write(f"ProgExt \"otool_external\"\n")
        # o.write(f"Ext_Params \"\"\n")
        o.write("end\n")
        o.write("*xyzfile {} {} {}\n".format(charge, mult, xyzfile))

    if 'GOAT' in ORCA_jobkeyword.upper():
        print("GOAT keyword found. ")

    # Call ORCA to do Opt/GOAT etc. job
    with open(basename + '.out', 'w') as ofile:
        process = sp.run(['orca', basename + '.inp'], check=True, stdout=ofile, stderr=ofile, universal_newlines=True)

    # Check if ORCA finished
    ORCAfinished, iter = checkORCAfinished(basename + '.out')
    if ORCAfinished is not True:
        print("Something failed about external ORCA job")
        ashexit()
    # Check if optimization completed
    if checkORCAOptfinished(basename + '.out') is not True:
        print("ORCA external job failed. Check outputfile:", basename + '.out')
        ashexit()
    print("ORCA external job finished")

    # Grabbing final geometry to update fragment object
    elems, coords = openmmqmmm.modules.module_coords.read_xyzfile(basename + ".xyz")
    fragment.coords = coords

    # Grabbing final energy
    energylines = pygrep2("FINAL SINGLE POINT ENERGY (From external program)", f"{basename}.out", errors="ignore")
    energy = float(energylines[-1].split()[-1])
    print("Final energy from external ORCA job:", energy)

    return energy


# Simple Wrapper around orca_mapspc


# Simple function to get elems and coordinates from ORCA outputfile
# Should read both single-point and optimization jobs correctly


# Make an ORCA fragment guess


# Find localized orbitals in ORCA outputfile for a given element
# Return orbital indices (to be fed into run_orca_plot)


# Reverse JSON to GBW


# Using orca_2json to create JSON file from ORCA GBW file
# Format options: json, bson, ubjson, msgpack


# Parse ORCA json file
# Good for getting MO-coefficients, MO-energies, basis set, H,S,T matrices, densities etc.


# Read BSON files using independent BSON codec for Python (not MongoDB)
# Msgpack probably better


# Grab ORCA wfn from jsonfile or data-dictionary


# Function to prepare ORCA orbitals for another ORCA calculation
# Mainly for getting natural orbitals


# TODO: fix once ORCA6 bugfix is done
# https://orcaforum.kofo.mpg.de/viewtopic.php?f=11&t=11657&p=47529&hilit=vpot#p47529
# Either use input-file option (vpot.inp) or other


# Function to create FCIDUMP file
# Change header_format from FCIDUMP to MRCC to get MRCC fort.55 file
# TODO: SCF-type beyond RHF


# calculate_natorbs_from_density
# Convenient function to get natural orbitals from any density even if ORCA did create the natural orbitals


# Get natural orbitals of any calculated density of an ORCA calculation
# Convenient when ORCA natural orbital printing is buggy
# NOTE: Not fully tested
