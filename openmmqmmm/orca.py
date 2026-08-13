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


class ORCATheory:
    """Interface to the ORCA quantum chemistry program."""

    def __init__(
        self,
        *,
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

        self.orcadir = find_orca(orcadir)
        if numcores != 1:
            logger.info(
                f"ORCA parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version "
                f"(for the ORCA version) is available in your environment"
            )
            openmmqmmm.parallel.check_openmpi()

        # Bind to core option when calling ORCA: i.e. execute: /path/to/orca file.inp "--bind-to none"
        self.bind_to_core_option = bind_to_core_option
        logger.info("bind_to_core_option: %s", self.bind_to_core_option)

        if " OPT" in orcasimpleinput.upper() or " FREQ" in orcasimpleinput.upper():
            raise InputError(
                f"Error. orcasimpleinput variable can not contain ORCA job-directives like: Opt, Freq, "
                f"Numfreq\nString: {orcasimpleinput.upper()}\norcasimpleinput should only contain information on "
                f"electronic-structure method (e.g. functional), basis set, grid, SCF convergence etc."
            )
        if "!" not in orcasimpleinput:
            raise InputError(
                "Error. orcasimpleinput should contain at least a '!' with method and basis set information"
            )

        # Whether to check ORCA outputfile for errors and warnings or not
        # Generally recommended. Could be disabled to speed up I/O a tiny bit
        self.check_for_errors = check_for_errors
        self.check_for_warnings = check_for_warnings

        self.runcalls = 0

        self.keep_each_run_output = keep_each_run_output
        if save_output_with_label is True and label is None:
            raise InputError("Error: save_output_with_label option requires a label keyword also")
        self.save_output_with_label = save_output_with_label

        self.print_population_analysis = print_population_analysis

        self.label = label

        self.filename = filename

        self.ignore_orca_error = ignore_orca_error

        self.moreadfile = moreadfile
        self.moreadfile_always = moreadfile_always
        self.autostart = autostart
        # Each ORCA calculation will save path to last GBW-file used in case we have switched directories
        # and we want to use last one
        self.path_to_last_gbwfile_used = None  # default None

        self.tddft = tddft
        self.tddft_roots = tddft_roots
        self.follow_root = follow_root

        # NOTE: nprocs is deprecated but kept on for a bit
        if nprocs is None:
            self.numcores = numcores
        else:
            self.numcores = nprocs

        # Property block. Added after coordinates unless None
        self.propertyblock = propertyblock

        self.properties = {}

        if self.autostart is False:
            self.extraline = extraline + "\n! Noautostart\n"
        else:
            self.extraline = extraline

        self.orcasimpleinput = orcasimpleinput
        self.orcablocks = orcablocks

        if first_iteration_input is not None:
            self.first_iteration_input = first_iteration_input
        else:
            self.first_iteration_input = ""

        self.brokensym = brokensym
        self.hs_mult = hs_mult
        if isinstance(atomstoflip, int):
            raise InputError(
                "Error: atomstoflip should be list of integers (e.g. [0] or [2,3,5]), not a single integer."
            )
        if self.brokensym is True and "UKS" not in self.orcasimpleinput and "UHF" not in self.orcasimpleinput:
            logger.warning("UKS/UHF keyword not present in orcasimpleinput for BS job. Adding.")
            self.orcasimpleinput = self.orcasimpleinput + " UKS"
        if atomstoflip is not None:
            self.atomstoflip = atomstoflip
        else:
            self.atomstoflip = []
        self.delta_scf = delta_scf
        self.delta_scf_pmom = delta_scf_pmom
        self.delta_scf_confline = delta_scf_confline
        self.delta_scf_turn_off_automatically = delta_scf_turn_off_automatically
        if self.delta_scf is True and self.delta_scf_confline is None:
            raise InputError("Error: DELTASCF is True but no deltaSCF_confline provided. Exiting")
        if self.delta_scf is True:
            logger.info("DeltaSCF True, turning on population analysis printing")
            self.print_population_analysis = True

        self.basis_per_element = basis_per_element
        if self.basis_per_element is not None:
            logger.info("Basis set dictionary for each element provided: %s", basis_per_element)

        if extrabasisatoms is not None:
            self.extrabasisatoms = extrabasisatoms
            self.extrabasis = extrabasis
        else:
            self.extrabasisatoms = []
            self.extrabasis = ""
        # Within ORCA inputfile, define a basis set for each and every atom. Requires a dictionary with element as key
        # and basis set as value
        self.atom_specific_basis_dict = atom_specific_basis_dict
        self.ecp_dict = ecp_dict  # ECP dict that usually goes with atom_specific dict

        # Used in the case of counterpoise calculations
        self.ghostatoms = []  # Adds ":" in front of element in coordinate block. Have basis functions and grid points
        self.dummyatoms = []  # Adds DA instead of element. No real atom

        self.fragment_indices = fragment_indices

        # self.qmatoms need to be set for Flipspin to work for QM/MM job.
        # Overwritten by QMMMtheory, used in Flip-spin
        self.qmatoms = []

        self.keep_last_output = True

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
        self.rohf_uhf_swap = rohf_uhf_swap

        # Specific CPCM radii. e.g. to use DRACO radii
        if cpcm_radii is not None:
            logger.info("CPCM radii provided: %s", cpcm_radii)
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

    def set_numcores(self, numcores):
        """Set how many cores ORCA is launched with."""
        self.numcores = numcores

    def cleanup(self):
        """Delete the ORCA scratch and output files of the previous run."""
        logger.info("Cleaning up old ORCA files")
        list_files = []
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
        for tmpfile in glob.glob(f"{self.filename}*tmp"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(tmpfile)

    # Do an ORCA-optimization instead of geomeTRIC optimization. Useful for gas-phase chemistry when ORCA-optimizer is
    # better than geomeTRIC
    def opt(self, fragment=None, grad=None, hessian=None, numcores=None, charge=None, mult=None):
        """Optimize the geometry with ORCA's own optimizer rather than geomeTRIC."""
        module_init_time = time.time()
        logger.info("------------RUNNING INTERNAL ORCA OPTIMIZATION-------------")
        # Coords provided to run or else taken from initialization.

        if fragment is None:
            raise InputError("No fragment provided to Opt.")
        logger.info("Fragment provided to Opt")

        current_coords = fragment.coords
        elems = fragment.elems
        charge, mult = check_charge_mult(charge, mult, self.theorytype, fragment, "ORCATheory.Opt", theory=self)

        if charge is None or mult is None:
            raise InputError("Error. charge and mult has not been defined for ORCATheory.Opt method")

        if numcores is None:
            numcores = self.numcores

        # Built locally, not stored: appending to self.extraline would accumulate
        # "! OPT" across repeated opt() calls and leak into later run() single points.
        extraline = "\n".join(filter(None, [self.extraline.strip("\n"), "! OPT"]))

        logger.info(f"Running ORCA with {numcores} cores available")
        logger.info("Object label: %s", self.label)

        logger.info("Creating inputfile: %s", self.filename + ".inp")
        logger.info("ORCA input:")
        logger.info("%s", self.orcasimpleinput)
        logger.info("%s", extraline)
        logger.info("%s", self.orcablocks)
        if self.propertyblock is not None:
            logger.info("%s", self.propertyblock)
        logger.info(f"Charge: {charge}  Mult: {mult}")

        create_orca_input_plain(
            self.filename,
            elems,
            current_coords,
            self.orcasimpleinput,
            self.orcablocks,
            charge,
            mult,
            extraline=extraline,
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
        fragment.print_system(filename="fragment_optimized.frag")
        fragment.write_xyzfile(xyzfilename="Fragment-optimized.xyz")

        _print_internal_coordinate_table(fragment)
        log_time_since(module_init_time, "ORCA Opt-run")
        return self.energy

    def get_dipole_moment(self):
        """Read the dipole moment from the last ORCA output file."""
        dm = grab_dipole_moment(self.filename + ".out")
        logger.info("Dipole moment: %s", dm)
        return dm

    def get_polarizability_tensor(self):
        """Read the static polarizability tensor from the last ORCA output file."""
        logger.debug("Reading polarizability from: %s", self.filename + ".out")
        polarizability, _diag_pz = grab_polarizability_tensor(self.filename + ".out")
        logger.info("polarizability: %s", polarizability)
        return polarizability

    def run(
        self,
        *,
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
        """Run an ORCA calculation and return the energy (and gradient)."""
        module_init_time = time.time()
        self.runcalls += 1
        logger.info("------------RUNNING ORCA INTERFACE-------------")
        logger.info("Object-label: %s", self.label)
        logger.info("Run-label: %s", label)
        if current_coords is None:
            raise InputError("Error:no current_coords")

        if charge is None or mult is None:
            raise InputError("Error. charge and mult has not been defined for ORCATheory.run method")

        # What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                raise InputError("No elems provided")
            qm_elems = elems

        # If QM/MM then atomindices lists like extrabasisatoms, atomstoflip and fragment_indices have to be updated
        if len(self.qmatoms) != 0:
            if self.fragment_indices is not None:
                fragment_indices = []
                for f in self.fragment_indices:
                    temp = [self.qmatoms.index(i) for i in f]
                    fragment_indices.append(temp)
            else:
                fragment_indices = self.fragment_indices
            qmatoms_extrabasis = [self.qmatoms.index(i) for i in self.extrabasisatoms]
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

        self._append_basis_blocks()

        logger.info(f"Running ORCA with {numcores} cores available")

        self._prepare_orbital_guess()

        extraline = self.extraline + "\n" + self.first_iteration_input if self.runcalls == 1 else self.extraline

        logger.info("Creating inputfile: %s", self.filename + ".inp")
        logger.info("ORCA input:")
        logger.info("%s", self.orcasimpleinput)
        logger.info("%s", extraline)
        logger.info("%s", self.orcablocks)
        logger.info(f"Charge: {charge}  Mult: {mult}")
        if self.brokensym is True:
            logger.info(f"Brokensymmetry SpinFlipping on! HSmult: {self.hs_mult}.")

            if self.hs_mult is None:
                raise InputError("Error:HSmult keyword in ORCATheory has not been set. This is required. Exiting.")
            if len(qmatomstoflip) == 0:
                raise InputError("Error: atomstoflip keyword needs to be set. This is required. Exiting.")

            for flipatom, qmflipatom in zip(self.atomstoflip, qmatomstoflip, strict=False):
                logger.info(f"Flipping atom: {flipatom} QMregionindex: {qmflipatom} Element: {qm_elems[qmflipatom]}")
        deltascfblock = None
        if self.delta_scf is True:
            logger.info("DeltaSCF option chosen. Will attempt MOM excited state SCF solution in first run")
            logger.info("DeltaSCF PMOM: %s", self.delta_scf_pmom)
            logger.info("Configuration line: %s", self.delta_scf_confline)
            if mult == 1 and "UKS" not in self.orcasimpleinput and "UHF" not in self.orcasimpleinput:
                logger.warning("Singlet DeltaSCF calculation requested but no UKS/UHF keyword present.")
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

        # Broken-symmetry runs additionally flip the spin of selected atoms and start
        # from the high-spin multiplicity; everything else is common to all four cases.
        brokensym_options = {"hs_mult": self.hs_mult, "atomstoflip": qmatomstoflip} if self.brokensym is True else {}
        if pc is True:
            logger.info("Pointcharge embedding is on!")
            create_orca_pcfile(self.filename, current_mm_coords, mm_charges)
            write_orca_input = create_orca_input_pc
            # Ghost/dummy atoms are a gas-phase option only: they have no MM counterpart.
            embedding_options = {}
        else:
            write_orca_input = create_orca_input_plain
            embedding_options = {"ghostatoms": self.ghostatoms, "dummyatoms": self.dummyatoms}
        write_orca_input(
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
            **brokensym_options,
            **embedding_options,
        )

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

        if self.ignore_orca_error is False:
            ORCAfinished, numiterations = check_orca_finished(outfile)
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

        # Now that we have possibly run a BS-DFT calculation, turning Brokensym off for future calcs (opt, restart,
        # etc.)
        # using this theory object
        if self.brokensym is True:
            logger.info(
                "ORCA Flipspin calculation done. Now turning off brokensym in ORCA object for possible future "
                "calculations"
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
                        "Adding NOSOSCF to orcasimpleinput to avoid future calculations from falling back to "
                        "ground-state"
                    )
                    self.orcasimpleinput = self.orcasimpleinput + " nososcf"
                if "nodamp" not in self.orcasimpleinput:
                    logger.info(
                        "Adding NODAMP to orcasimpleinput to avoid future calculations from falling back to "
                        "ground-state"
                    )
                    self.orcasimpleinput = self.orcasimpleinput + " nodamp"
                if "nolshift" not in self.orcasimpleinput:
                    logger.info(
                        "Adding NOLSHIFT to orcasimpleinput to avoid future calculations from falling back to "
                        "ground-state"
                    )
                    self.orcasimpleinput = self.orcasimpleinput + " nolshift"
            else:
                logger.info("deltaSCF_turn_off_automatically option if False. Will keep DeltaSCF settings")

        # Now that we have possibly run a ORCA job with moreadfile we now turn the moreadfile option off
        #  as we probably want to use the orbitals we created
        if self.moreadfile is not None:
            logger.info("First ORCATheory calculation finished.")
            if not self.moreadfile_always:
                logger.info("Now turning moreadfile option off.")
                self.moreadfile = None

        if self.save_output_with_label is True:
            shutil.copy(self.filename + ".out", self.filename + f"_{self.label}_{charge}_{mult}.out")

        if self.keep_each_run_output is True:
            logger.info("\nkeep_each_run_output is True")
            logger.info(
                "%s", "Copying {} to {}".format(self.filename + ".out", self.filename + f"_run{self.runcalls}" + ".out")
            )
            shutil.copy(self.filename + ".out", self.filename + f"_run{self.runcalls}" + ".out")

        if self.keep_last_output is True:
            shutil.copy(self.filename + ".out", self.filename + "_last.out")

        # Save path to last GBW-file (used if the run changes directories, e.g. goes from NumFreq)
        self.path_to_last_gbwfile_used = f"{os.getcwd()}/{self.filename}.gbw"

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
                logger.warning("No charges or spinpops were found in ORCA output. Continuing")
        if self.ignore_orca_error is False:
            self.energy = grab_orca_final_energy(outfile)
            logger.info("ORCA energy: %s", self.energy)
        else:
            self.energy = grab_orca_final_energy(outfile)

            if self.energy is None:
                logger.info("No energy could be found in ORCA outputfile.")
                logger.info("Setting energy to 0.0 and returning")
                return 0.0
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

        if self.tddft is True:
            transition_energies = tddftgrab(f"{self.filename}.out")
            transition_intensities = tddftintens_grab(f"{self.filename}.out")

            self.properties["TDDFT_transition_energies"] = transition_energies
            self.properties["TDDFT_transition_intensities"] = transition_intensities

        orca_timings = grab_orca_timings(outfile)

        self.grad = np.zeros((len(qm_elems), 3))

        # XDM option: WFX file should have been created.
        if self.xdm:
            dispE, dispgrad = openmmqmmm.elstructure.xdm_run(
                wfxfile=self.filename + ".wfx", a1=self.xdm_a1, a2=self.xdm_a2, functional=self.xdm_func
            )
            logger.info("XDM dispersion energy: %s", dispE)
            self.energy = self.energy + dispE
            logger.info("DFT+XDM energy: %s", self.energy)
            self.grad = self.grad + dispgrad

        if hessian is True:
            logger.info("Reading Hessian from file: %s", self.filename + ".hess")
            self.hessian = grab_hessian(self.filename + ".hess")
            self.ir_intensities = grab_ir_intensities(self.filename + ".hess")

        if grad is True:
            grad = grab_orca_gradient(engradfile)
            self.grad = self.grad + grad
            logger.debug("ORCA gradient: %s", self.grad)

            if pc:
                if "pc_gradient" in orca_timings:
                    logger.info(
                        "%s", "Time calculating QM-Pointcharge gradient: {} seconds".format(orca_timings["pc_gradient"])
                    )
                # Grab pointcharge gradient. i.e. gradient on MM atoms from QM-MM elstat interaction.
                self.pcgrad = grab_orca_pc_gradient(pcgradfile)
                logger.info("------------ENDING ORCA-INTERFACE-------------")
                log_time_since(module_init_time, "ORCA run")
                return self.energy, self.grad, self.pcgrad
            logger.info("------------ENDING ORCA-INTERFACE-------------")
            log_time_since(module_init_time, "ORCA run")
            return self.energy, self.grad

        logger.info("Single-point ORCA energy: %s", self.energy)
        logger.info("------------ENDING ORCA-INTERFACE-------------")
        log_time_since(module_init_time, "ORCA run")
        return self.energy

    def _append_basis_blocks(self):
        """Append per-element basis and ECP blocks to orcablocks, if not already present."""
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

    def _prepare_orbital_guess(self):
        """Put a GBW file where ORCA will find it, or let it guess new orbitals."""
        if self.moreadfile is not None:
            logger.info(f"Moreadfile option active. File path: {self.moreadfile}")
            if os.path.isfile(self.moreadfile) is True:
                logger.info(f"File exists in current directory: {os.getcwd()}")
                return
            logger.info(f"File does not exist in current directory: {os.getcwd()}")
            if os.path.isabs(self.moreadfile) is True:
                raise FileFormatError("Error: Absolute path provided but file does not exists. Exiting")
            logger.info("Checking if file exists in parentdir instead:")
            if os.path.isfile(f"../{self.moreadfile}") is True:
                logger.info("Yes. Copying file to current dir")
                shutil.copy(f"../{self.moreadfile}", f"./{self.moreadfile}")
            return

        logger.info("Moreadfile option not active")
        if os.path.isfile(f"{self.filename}.gbw") is True:
            logger.info(f"A GBW-file with same basename : {self.filename}.gbw is present")
            if self.autostart is False:
                logger.info("Autostart is False. ORCA will ignore any file present")
            else:
                logger.info("Autostart feature is active. ORCA will read GBW-file present.")
            return

        logger.info(f"No {self.filename}.gbw file is present in dir.")
        if self.path_to_last_gbwfile_used is None:
            logger.info(f"Checking if a file {self.filename}.gbw exists in parentdir:")
            if os.path.isfile(f"../{self.filename}.gbw") is True:
                logger.info("Yes. Copying file from parentdir to current dir")
                shutil.copy(f"../{self.filename}.gbw", f"./{self.filename}.gbw")
            else:
                logger.info("Found no file. ORCA will guess new orbitals")
            return

        logger.info(
            f"Found a path ({self.path_to_last_gbwfile_used}) to last GBW-file used by this Theory object. "
            f"Will try to copy this file do current dir"
        )
        try:
            shutil.copy(self.path_to_last_gbwfile_used, f"./{self.filename}.gbw")
        except FileNotFoundError:
            logger.info("File was not found. May have been deleted")
        if self.autostart is False:
            logger.info("Autostart option is False. ORCA will ignore this file")
        else:
            logger.info("Autostart feature is active. ORCA will read GBW-file present.")


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
    for source, directory in (
        ("orcadir argument", orcadir),
        ("OPENMMQMMM_ORCADIR environment variable", os.environ.get("OPENMMQMMM_ORCADIR")),
    ):
        if not directory:
            continue
        resolved = os.path.expanduser(directory)
        if _looks_like_orca_dir(resolved) and _orca_binary_runs(resolved):
            logger.info(f"Using ORCA installation: {resolved} (from {source})")
            return resolved
        if required:
            raise ExternalProgramError(
                f"The {source} points at {resolved}, which is not a working ORCA installation "
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
            "Found no working ORCA installation.\nPass orcadir= , set the OPENMMQMMM_ORCADIR environment variable, or "
            "put the orca binary in PATH"
        )
    return None


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
            raise ExternalProgramError(f"ORCA run failed - check the output file: {basename}.out") from e


# Lines matching "warning" that carry no information: banner headings and notices ORCA
# emits on every run of a given kind. Matched against the start of the line.
_BENIGN_WARNING_PREFIXES = (
    "                       Please study these wa",
    "                                        WARNINGS",
    "Warning: in a DFT calculation",
    "WARNING: Old DensityContainer",
    "WARNING: your system is open-shell",
)

# Lines matching "error" that are ordinary output: convergence tables report a DIIS error
# and a truncation error per iteration, and TD-DFT announces finishing without one.
_BENIGN_ERROR_PREFIXES = (
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
)


def _report_matching_lines(filename, needles, benign_prefixes, headline):
    with open(filename, errors="ignore") as f:
        matches = [
            line
            for line in f
            if any(needle in line.casefold() for needle in needles) and not line.startswith(benign_prefixes)
        ]
    if matches:
        logger.info(headline)
        logger.info("%s", "".join(matches))


def grab_orca_warnings(filename):
    _report_matching_lines(
        filename, ("warning",), _BENIGN_WARNING_PREFIXES, "Found warning messages in ORCA outputfile:"
    )


def grab_orca_errors(filename):
    _report_matching_lines(
        filename,
        ("error", "aborting"),
        _BENIGN_ERROR_PREFIXES,
        "Found possible error messages in ORCA outputfile:",
    )


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


def grab_orca_final_energy(file, errors="ignore"):
    Energy = None
    with open(file, errors=errors) as f:
        for line in f:
            if "FINAL SINGLE POINT ENERGY" in line:
                if "Wavefunction not fully converged!" in line:
                    raise ExternalProgramError("ORCA WF not fully converged!\nNot using energy. Modify ORCA settings")
                # Changing: sometimes ORCA adds info to the right of energy
                Energy = float(line.split()[5]) if "(MM)" in line else float(line.split()[4])
    if Energy is None:
        logger.error("Found no energy in file: %s", file)
        logger.error("Something went wrong with ORCA run. Check ORCA outputfile: %s", file)
        logger.info("------------ENDING ORCA-INTERFACE-------------")
        return None
    return Energy


# ORCA timing lines, as "line label" -> "key in the returned dictionary". The labels
# are matched against the start of the stripped line; the column widths and the number
# of dots separating label from value vary between ORCA versions and between sections
# of the same output, so neither may be relied on.
ORCA_TIMING_LABELS = {
    "Calculating one electron integrals": "one_elec_integrals",
    "GTO integral calculation": "time_gtointegrals",
    "One electron gradient": "one_elec_gradient",
    "Point charge gradient": "pc_gradient",
    "RI-J Coulomb gradient": "rij_coulomb_gradient",
    "SCF Gradient evaluation": "time_scfgrad",
    "SCF iterations": "time_scfiterations",
    "Sum of individual times": "total_time",
    "XC gradient": "xc_gradient",
}


def _seconds_from_timing_line(line):
    fields = line.split()
    for index, field in enumerate(fields):
        if field.startswith("sec") and index:
            try:
                return float(fields[index - 1].strip("()"))
            except ValueError:
                return None
    return None


def grab_orca_timings(file):
    timings = {}  # in seconds
    try:
        with open(file, errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                for label, key in ORCA_TIMING_LABELS.items():
                    if stripped.startswith(label):
                        seconds = _seconds_from_timing_line(stripped)
                        if seconds is not None:
                            timings[key] = seconds
                        break
    except OSError:
        pass
    return timings


def grab_orca_gradient(engradfile):
    grab = False
    numatomsgrab = False
    row = 0
    col = 0
    with open(engradfile) as gradfile:
        for line in gradfile:
            if numatomsgrab and "#" not in line:
                numatoms = int(line.split()[0])
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


def grab_orca_pc_gradient(pcgradfile):
    with open(pcgradfile) as pgradfile:
        for count, line in enumerate(pgradfile):
            if count == 0:
                numatoms = int(line.split()[0])
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
                grab = True
    return pz_tensor, diag_pz_tensor


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


def write_orca_hessfile(hessian, coords, elems, masses, outputname):
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

        # Either full system lists were passed or partial-system lists
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
        # ORCA terminates its own .hess files this way; tools that read the file
        # (including grab_hessian) use the section markers to bound each block.
        orcahessfile.write("$end\n")
    logger.info("")
    logger.info("ORCA-style Hessian written to: %s", outputname)


def grab_hessian(hessfile):
    hesstake = False
    j = 0
    orcacoldim = 5
    shiftpar = 0
    lastchunk = False
    grabsize = False
    with open(hessfile) as hfile:
        for line in hfile:
            # Any following section ends the Hessian block: $vibrational_frequencies
            # in ORCA's own files, $atoms in the ones write_orca_hessfile produces.
            if hesstake and line.startswith("$"):
                break
            if hesstake and len(line.split()) == 1 and grabsize:
                grabsize = False
                hessdim = int(line.split()[0])

                hessarray2d = np.zeros((hessdim, hessdim))

            if hesstake and lastchunk:
                if len(line.split()) == hessdim - shiftpar + 1:
                    for i in range(hessdim - shiftpar):
                        hessarray2d[j, i + shiftpar] = line.split()[i + 1]
                    j += 1
                    if j == hessdim:
                        # Matrix complete; stop before trailing sections are misread as rows
                        hesstake = False
            elif hesstake and len(line.split()) == 5:
                continue
            if hesstake and len(line.split()) == 6:
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


def _write_input_block(orcafile, text):
    if text:
        orcafile.write(text.rstrip("\n") + "\n")


def _create_orca_input(
    name,
    elems,
    coords,
    orcasimpleinput,
    orcablockinput,
    charge,
    mult,
    *,
    pcfile=None,
    grad=False,
    hessian=False,
    extraline="",
    hs_mult=None,
    atomstoflip=None,
    extrabasis=None,
    extrabasisatoms=None,
    atom_specific_basis_dict=None,
    moreadfile=None,
    propertyblock=None,
    ghostatoms=None,
    dummyatoms=None,
    fragment_indices=None,
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
        _write_input_block(orcafile, orcasimpleinput)
        _write_input_block(orcafile, extraline)
        if grad:
            orcafile.write("! Engrad" + "\n")
        if hessian:
            orcafile.write("! Freq" + "\n")
        if moreadfile is not None:
            logger.info("MOREAD option active. Will read orbitals from file: %s", moreadfile)
            orcafile.write("\n! MOREAD" + "\n")
            orcafile.write(f'%moinp "{moreadfile}"' + "\n")
        if pcfile is not None:
            orcafile.write(f'%pointcharges "{pcfile}"\n')
        _write_input_block(orcafile, orcablockinput)
        if atomstoflip is not None:
            atomstoflipstring = str(atomstoflip) if isinstance(atomstoflip, int) else ",".join(map(str, atomstoflip))
            orcafile.write("%scf\n")
            orcafile.write(f"Flipspin {atomstoflipstring}" + "\n")
            orcafile.write(f"FinalMs {(mult - 1) / 2}" + "\n")
            orcafile.write("end  \n")
        orcafile.write("\n")
        if delta_scf_block is not None:
            orcafile.write(delta_scf_block)
        orcafile.write("\n")
        if atomstoflip is not None:
            orcafile.write(f"*xyz {charge} {hs_mult}\n")
        else:
            orcafile.write(f"*xyz {charge} {mult}\n")
        for i, (el, c) in enumerate(zip(elems, coords, strict=False)):
            if i in extrabasisatoms:
                orcafile.write(f'{el} {c[0]} {c[1]} {c[2]} newgto "{extrabasis}" end\n')
            # Atom-specific basis-dict option (new basis set definition for each atom)
            elif atom_specific_basis_dict is not None:
                logger.info("Writing atom-specific basis for atom: %s", i)
                orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
                orcafile.writelines(str(bline) for bline in atom_specific_basis_dict[(el, i)])
            # Setting atom to be a ghost atom
            elif i in ghostatoms:
                orcafile.write("{}{} {} {} {} \n".format(el, ":", c[0], c[1], c[2]))
            elif i in dummyatoms:
                orcafile.write("{} {} {} {} \n".format("DA", c[0], c[1], c[2]))
            # Adding fragment specification. Atoms belonging to no fragment (link
            # atoms, most commonly) are written without a fragment tag.
            elif fragment_indices is not None:
                fragmentindex = search_list_of_lists_for_index(i, fragment_indices)
                if fragmentindex is not None:
                    orcafile.write("{} {} {} {} \n".format(f"{el}({fragmentindex + 1})", c[0], c[1], c[2]))
                else:
                    orcafile.write(f"{el} {c[0]} {c[1]} {c[2]} \n")
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


def create_orca_input_pc(name, elems, coords, orcasimpleinput, orcablockinput, charge, mult, **kwargs):
    _create_orca_input(
        name, elems, coords, orcasimpleinput, orcablockinput, charge, mult, pcfile=name + ".pc", **kwargs
    )


def create_orca_input_plain(name, elems, coords, orcasimpleinput, orcablockinput, charge, mult, **kwargs):
    _create_orca_input(name, elems, coords, orcasimpleinput, orcablockinput, charge, mult, pcfile=None, **kwargs)


def create_orca_pcfile(name, coords, listofcharges):
    with open(name + ".pc", "w") as pcfile:
        pcfile.write(str(len(listofcharges)) + "\n")
        for p, c in zip(listofcharges, coords, strict=False):
            line = f"{p} {c[0]} {c[1]} {c[2]}"
            pcfile.write(line + "\n")


# ORCA prints every population analysis as a table: a heading, a rule, one row per atom,
# then a terminator. The models differ only in the heading, the column the number sits in,
# and how the table ends -- so they are described here rather than written out nine times.
#
# column is the index into the row's whitespace-separated fields.
# stop is checked before the row is parsed; a row is skipped when it holds no data.
_CHARGE_TABLES = {
    "NPA": {
        "start": "Atom No    Charge        Core      Valence    Rydberg      Total",
        "stop": lambda line: "=======" in line,
        "column": 2,
    },
    "NBO": {
        "start": "Atom No    Charge        Core      Valence    Rydberg      Total",
        "stop": lambda line: "=======" in line,
        "column": 2,
    },
    "CHELPG": {
        "start": "CHELPG Charges",
        "stop": lambda line: "Total charge: " in line,
        "column": -1,
        "row": lambda fields: len(fields) == 4,
    },
    "HIRSHFELD": {
        "start": "  ATOM     CHARGE      SPIN",
        "stop": lambda line: len(line) < 3,
        "column": -2,
        "row": lambda fields: len(fields) == 4,
    },
    "CM5": {
        "start": "  ATOM     CHARGE      SPIN",
        "stop": lambda line: len(line) < 3,
        "column": -2,
        "row": lambda fields: len(fields) == 4,
    },
    "MULLIKEN": {
        "start": "MULLIKEN ATOMIC CHARGES",
        "stop": lambda line: "Sum of atomic" in line,
        # The heading is also the prefix of "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS",
        # where the charge is the second-to-last column and the last one is the spin.
        "column": lambda heading: -2 if "SPIN POPULATIONS" in heading else -1,
    },
    "LOEWDIN": {
        "start": "LOEWDIN ATOMIC CHARGES",
        "stop": lambda line: "Sum of atomic" in line or len(line.replace(" ", "")) < 2,
        "column": lambda heading: -2 if "SPIN POPULATIONS" in heading else -1,
    },
    "IAO": {
        "start": "IAO PARTIAL CHARGES",
        "stop": lambda line: "Sum of atomic" in line,
        "skip": ("------", "Warning"),
    },
}

_SPIN_POPULATION_TABLES = {
    "MULLIKEN": {
        "start": "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS",
        "stop": lambda line: "Sum of atomic" in line,
    },
    "LOEWDIN": {
        "start": "LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS",
        "stop": lambda line: "Sum of atomic" in line or len(line.replace(" ", "")) < 2,
    },
}


def _scan_orca_table(outputfile, spec):
    start = spec["start"]
    stop = spec["stop"]
    column_spec = spec.get("column", -1)
    row_is_data = spec.get("row")
    skip = spec.get("skip", ("------",))

    values = []
    column = None
    grabbing = False
    with open(outputfile) as f:
        for line in f:
            if grabbing:
                if stop(line):
                    grabbing = False
                elif not any(marker in line for marker in skip):
                    fields = line.split()
                    if row_is_data is None or row_is_data(fields):
                        values.append(float(fields[column]))
            if start in line:
                # A second table of the same kind supersedes the first, and the column is
                # re-read from this heading: the two Mulliken tables differ in width.
                values = []
                grabbing = True
                column = column_spec(line) if callable(column_spec) else column_spec
    return values


def _trim_to_broken_symmetry_solution(values, outputfile, kind):
    if not pygrep2("WARNING: Broken symmetry calculations", outputfile):
        return values
    numatoms = int(pygrep("Number of atoms                             ...", outputfile)[-1])
    if len(values) == numatoms:
        return values
    logger.info("Broken-symmetry job detected. Only taking BS-state %s", kind)
    return values[-numatoms:]


def grab_orca_spin_populations(chargemodel, outputfile):
    spec = _SPIN_POPULATION_TABLES.get(chargemodel.upper())
    if spec is None:
        raise FileFormatError(
            f"Unknown chargemodel '{chargemodel}' for spin populations. "
            f"Expected one of: {', '.join(sorted(_SPIN_POPULATION_TABLES))}."
        )
    spinpops = _scan_orca_table(outputfile, spec)
    return _trim_to_broken_symmetry_solution(spinpops, outputfile, "populations")


def grab_orca_atom_charges(chargemodel, outputfile):
    model = chargemodel.upper()
    spec = _CHARGE_TABLES.get(model)
    if spec is None:
        raise FileFormatError(
            f"Unknown chargemodel '{chargemodel}'. Expected one of: {', '.join(sorted(_CHARGE_TABLES))}."
        )
    if model in ("NPA", "NBO"):
        logger.warning("NPA/NBO charge-option in ORCA requires setting environment variable NBOEXE:")
        logger.info("e.g. export NBOEXE=/path/to/nbo7.exe")

    charges = _scan_orca_table(outputfile, spec)

    if model == "CM5":
        # CM5 is a correction applied to the Hirshfeld charges, using the geometry ORCA
        # used for them.
        logger.info("Hirshfeld charges : %s", charges)
        elems, coords = _grab_orca_cartesian_coordinates(outputfile)
        atomicnumbers = openmmqmmm.coords.elemstonuccharges(elems)
        charges = list(openmmqmmm.elstructure.calc_cm5(atomicnumbers, coords, charges))
        logger.info("CM5 charges : %s", charges)

    return _trim_to_broken_symmetry_solution(charges, outputfile, "charges")


def _grab_orca_cartesian_coordinates(outputfile):
    elems = []
    coords = []
    grabbing = False
    with open(outputfile) as f:
        for line in f:
            if grabbing and "----------------------" not in line:
                fields = line.split()
                if len(fields) < 2:
                    grabbing = False
                else:
                    elems.append(fields[0])
                    coords.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                elems = []
                coords = []
                grabbing = True
    return elems, coords


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
) -> float:
    """Optimize a geometry using ORCA's optimizer while openmmqmmm provides energies+gradients."""
    logger.info(main_header("ORCA_External_Optimizer"))
    if fragment is None or theory is None:
        raise InputError("ORCA_External_Optimizer requires fragment and theory keywords")

    if charge is None or mult is None:
        logger.warning("Charge/mult was not provided to ORCA_External_Optimizer")
        if fragment.charge is not None and fragment.mult is not None:
            logger.warning(
                f"Fragment contains charge/mult information: Charge: {fragment.charge} Mult: {fragment.mult} Using "
                f"this instead"
            )
            logger.warning("Make sure this is what you want!")
            charge = fragment.charge
            mult = fragment.mult
        else:
            raise InputError("No charge/mult information present in fragment either. Exiting.")

    orcadir = find_orca(orcadir)
    # Prepend orcadir to PATH so ORCA's helper binaries resolve to this installation
    # (appending would let an unrelated `orca` earlier in PATH shadow the real one)
    os.environ["PATH"] = orcadir + os.pathsep + os.environ["PATH"]

    import pickle

    theoryfilename = "theory.saved"
    with open(theoryfilename, "wb") as theoryfh:
        pickle.dump(theory, theoryfh)

    # Write otool_script once in location that ORCA will launch. This is an energy+gradient calculator script
    # ORCA will call : otool_external test_EXT.extinp.tmp
    # The tool writes basename_Ext.engrad which ORCA reads
    basename = "ORCAEXTERNAL"
    scriptlocation = "."
    # ORCA >= 6 locates the external tool via EXTOPTEXE (it does not search PATH)
    os.environ["EXTOPTEXE"] = os.path.abspath(os.path.join(scriptlocation, "otool_external"))
    write_otool_script(
        basename=basename, theoryfile=theoryfilename, scriptlocation=scriptlocation, charge=charge, mult=mult
    )

    xyzfile = "orca_external.xyz"
    fragment.write_xyzfile(xyzfile)

    # Active atoms become inverted constraints
    constraintsblock = ""
    if actatoms is not None:
        logger.info("Activeatoms list was provided. This means that we need to provide constraints to ORCA")
        frozenatoms = listdiff(fragment.allatoms, actatoms)
        logger.info("Freezing the non-active atoms: %s", frozenatoms)
        consstring = "".join(f"{{C {f} C}}\n" for f in frozenatoms)
        constraintsblock = f"""%geom Constraints
{consstring}end
end
"""
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

    with open(basename + ".out", "w") as ofile:
        sp.run([os.path.join(orcadir, "orca"), basename + ".inp"], check=True, stdout=ofile, stderr=ofile, text=True)

    ORCAfinished, _iter = check_orca_finished(basename + ".out")
    if ORCAfinished is not True:
        raise ExternalProgramError("Something failed about external ORCA job")
    if check_orca_opt_finished(basename + ".out") is not True:
        raise ExternalProgramError("ORCA external job failed. Check outputfile: {}".format(basename + ".out"))
    logger.info("ORCA external job finished")

    _elems, coords = openmmqmmm.coords.read_xyzfile(basename + ".xyz")
    fragment.coords = coords

    energylines = pygrep2("FINAL SINGLE POINT ENERGY (From external program)", f"{basename}.out", errors="ignore")
    energy = float(energylines[-1].split()[-1])
    logger.info("Final energy from external ORCA job: %s", energy)

    return energy


# Convenient when ORCA natural orbital printing is buggy
# NOTE: Not fully tested
