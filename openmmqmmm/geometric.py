"""Geometry optimization through the geomeTRIC library (minimization, TS, constraints, active regions)."""

import contextlib
import logging
import os
import shutil
import time

import numpy as np

import openmmqmmm.constants
from openmmqmmm.coords import (
    _print_internal_coordinate_table,
    check_charge_mult,
    fullindex_to_actindex,
    print_coords_for_atoms,
    write_coords_all,
    write_xyz_for_atoms,
    write_xyzfile,
)
from openmmqmmm.coords_pbc import (
    align_to_standard_orientation,
    cell_vectors_to_params,
    cell_volume,
    write_cif_file,
    write_poscar_file,
    write_xsf_file,
)
from openmmqmmm.exceptions import (
    FileFormatError,
    InputError,
    MissingDependencyError,
    OpenMMQMMMError,
)
from openmmqmmm.freq import approximate_full_hessian_from_smaller, read_hessian, write_hessian
from openmmqmmm.numgrad import NumGrad
from openmmqmmm.openmm import OpenMMTheory
from openmmqmmm.qmmm import QMMMTheory
from openmmqmmm.results import Results
from openmmqmmm.utils import log_time_since, main_header, pygrep2, sub_header

logger = logging.getLogger(__name__)

##################################################
# NEW Interface to geomeTRIC Optimization Library
##################################################


# Wrapper function around GeometricOptimizer
def optimize_geometry(
    theory=None,
    fragment=None,
    charge=None,
    mult=None,
    coordsystem="tric",
    force_coordsystem=False,
    frozenatoms=None,
    constraints=None,
    constraintsinputfile=None,
    irc=False,
    rigid=False,
    enforce_constraints=None,
    constrainvalue=False,
    maxiter=250,
    active_region=False,
    actatoms=None,
    num_grad=False,
    convergence_setting=None,
    conv_criteria=None,
    print_atoms_list=None,
    ts_opt=False,
    hessian=None,
    partial_hessian_atoms=None,
    modelhessian=None,
    subfrctor=1,
    mm_pdb_traj_write=False,
    result_write_to_disk=True,
    force_no_pbc=False,
    pbc_format_option="CIF",
) -> "Results":
    """Wrapper function around the GeometricOptimizer class."""
    timeA = time.time()

    # EARLY EXIT
    if theory is None or fragment is None:
        raise InputError("geomeTRICOptimizer requires theory and fragment objects provided. Exiting.")
    # NOTE: Class does not take fragment and theory
    optimizer = GeometricOptimizer(
        theory=theory,
        charge=charge,
        mult=mult,
        coordsystem=coordsystem,
        frozenatoms=frozenatoms,
        maxiter=maxiter,
        active_region=active_region,
        actatoms=actatoms,
        ts_opt=ts_opt,
        hessian=hessian,
        partial_hessian_atoms=partial_hessian_atoms,
        modelhessian=modelhessian,
        constraintsinputfile=constraintsinputfile,
        irc=irc,
        rigid=rigid,
        enforce_constraints=enforce_constraints,
        convergence_setting=convergence_setting,
        conv_criteria=conv_criteria,
        print_atoms_list=print_atoms_list,
        subfrctor=subfrctor,
        mm_pdb_traj_write=mm_pdb_traj_write,
        force_coordsystem=force_coordsystem,
        result_write_to_disk=result_write_to_disk,
        force_no_pbc=force_no_pbc,
        pbc_format_option=pbc_format_option,
    )

    # If num_grad then we wrap the theory object into the NumGrad class
    if num_grad:
        logger.info("NumGrad flag detected. Wrapping theory object into NumGrad class")
        logger.info("This enables numerical-gradient calculation for theory")
        theory = NumGrad(theory=theory)

    # Providing theory and fragment to run method. Also constraints
    result = optimizer.run(
        theory=theory,
        fragment=fragment,
        charge=charge,
        mult=mult,
        constraints=constraints,
        constrainvalue=constrainvalue,
    )
    log_time_since(timeA, "geomeTRIC")

    return result


# Class for optimization.
class GeometricOptimizer:
    """Geometry optimizer wrapping the geomeTRIC library.

    Supports minimizations and TS optimizations, constraints, frozen atoms and
    active-region optimizations of large systems. Usually invoked through
    optimize_geometry.
    """

    def __init__(
        self,
        theory=None,
        charge=None,
        mult=None,
        coordsystem="tric",
        frozenatoms=None,
        maxiter=250,
        active_region=False,
        actatoms=None,
        convergence_setting=None,
        conv_criteria=None,
        ts_opt=False,
        hessian=None,
        constraintsinputfile=None,
        irc=False,
        rigid=False,
        enforce_constraints=None,
        print_atoms_list=None,
        partial_hessian_atoms=None,
        modelhessian=None,
        subfrctor=1,
        mm_pdb_traj_write=False,
        force_coordsystem=False,
        result_write_to_disk=True,
        force_no_pbc=False,
        pbc_format_option="CIF",
    ):
        import time

        self.time_init = time.time()
        logger.info(main_header("geomeTRICOptimizer initialization"))
        logger.info("Creating optimizer object")
        ###############################
        # Going through user options
        ###############################

        if actatoms is not None:
            logger.info("List of active atoms provided. Setting ActiveRegion to True")
            active_region = True
        if actatoms is None:
            actatoms = []
        if frozenatoms is None:
            frozenatoms = []

        if active_region is True and coordsystem.lower() == "tric":
            logger.warning(
                "Warning: ActiveRegion is set but the coordsystem is TRIC. The HDLC coordinate system is usually much "
                "more robust for large systems than TRIC."
            )
            logger.info("")
            if force_coordsystem is True:
                logger.info("force_coordsystem is True.")
                logger.info("Sticking with coordsystem TRIC")
            else:
                logger.info("force_coordsystem is False.")
                logger.warning(
                    "Warning: Now switching to HDLC to avoid likely robustness problems with TRIC. To avoid this "
                    "behaviour (and force use of TRIC) you can use set the Boolean force_coordsystem to True."
                )
                coordsystem = "hdlc"

        # Defining some attributes
        self.maxiter = maxiter
        self.actatoms = actatoms
        self.frozenatoms = frozenatoms
        self.coordsystem = coordsystem
        self.print_atoms_list = print_atoms_list
        self.active_region = active_region
        self.ts_opt = ts_opt
        self.subfrctor = subfrctor

        # IRC
        self.irc = irc
        # Rigid opt
        self.rigid = rigid
        # Enforce constraints option
        self.enforce_constraints = enforce_constraints

        # For MM or QM/MM whether to write PDB-trajectory or not
        self.mm_pdb_traj_write = mm_pdb_traj_write
        # Hessian stuff
        self.hessian = hessian
        self.modelhessian = modelhessian
        self.partial_hessian_atoms = partial_hessian_atoms

        # Constraints by default set to None
        self.constraints = None
        # Optional user-constraintsfile in geometric syntax
        self.constraintsinputfile = constraintsinputfile
        ######################

        self.result_write_to_disk = result_write_to_disk

        # Setup convergence criteria (sets self.conv_criteria)
        self.convergence_criteria(convergence_setting, conv_criteria)

        # PBC
        if getattr(theory, "periodic", False):
            logger.info("Detected periodicity in Theory object")
            logger.info("Activating periodic routines ")
            self.pbc_active = True
            self.pbc_format_option = pbc_format_option
            logger.info("Switching coordsystem to hdlc")
            self.coordsystem = "hdlc"
            logger.info("Final PBC coordinate file written in format: %s", self.pbc_format_option)

            if force_no_pbc is True:
                logger.warning("Option force_noPBC set to True. Turning off PBC")
                self.pbc_active = False
        else:
            logger.info("Theory is not periodic")
            self.pbc_active = False

        ######################
        # SOME PRINTING of settings
        ######################
        logger.info("Coordinate system:  %s", self.coordsystem)
        logger.info("Max iterations:  %s", self.maxiter)
        logger.info("Frozen atoms: %s", self.frozenatoms)
        logger.info("Active Region: %s", self.active_region)
        if self.active_region is True:
            logger.info("Number of active atoms: %s", len(self.actatoms))
        logger.info("TS Optimization: %s", self.ts_opt)
        logger.info("Hessian Option: %s", self.hessian)
        logger.info("Convergence criteria: %s", self.conv_criteria)

    # Requires info on theory and fragment
    def print_atoms_output_setting(self, theory, fragment):
        # What atoms to print in outputfile in each opt-step. Example choice: QM-region only
        # If not specified then active-region or all-atoms
        """Decide which atoms are printed in each optimization step's output.

        Defaults to the QM region for QM/MM, the active region if one is set, and otherwise
        all atoms.

        Args:
            theory: the theory being optimized, used to find the QM region.
            fragment: the fragment being optimized.
        """
        if self.print_atoms_list is None:
            # Print-atoms list not specified. What to do:
            if self.active_region is True:
                # If QM/MM object then QM-region:
                if isinstance(theory, QMMMTheory):
                    logger.info("Theory class: QMMMTheory")
                    logger.info(
                        "Will by default print only QM-region in output (use print_atoms_list option to change)"
                    )
                    self.print_atoms_list = theory.qmatoms
                else:
                    # Print actatoms since using Active Region (can be too much)
                    self.print_atoms_list = self.actatoms
            else:
                # No act-region. Print all atoms
                self.print_atoms_list = fragment.allatoms

    def convergence_criteria(self, convergence_setting, userconv):
        ########################################
        # Dealing with convergence criteria
        ########################################
        """Resolve the geomeTRIC convergence thresholds to use.

        Args:
            convergence_setting: named preset, e.g. "ORCA", "Chemshell", "ORCA_TIGHT", "GAU".
            userconv: dict of individual thresholds overriding the preset.
        """
        if userconv is not None:
            logger.info("User-defined convergence criteria:")
            # Setting defaults first
            self.conv_criteria = {
                "convergence_energy": 5e-6,
                "convergence_grms": 1e-4,
                "convergence_gmax": 3.0e-4,
                "convergence_drms": 2.0e-3,
                "convergence_dmax": 4.0e-3,
                "convergence_cmax": 1.0e-2,
            }
            # Then overriding with user selection
            for conv_key in userconv:
                self.conv_criteria[conv_key] = userconv[conv_key]

        elif convergence_setting is None and userconv is None:
            logger.info("No convergence settings by user. Using default criteria (same as ORCA)")
            self.conv_criteria = {
                "convergence_energy": 5e-6,
                "convergence_grms": 1e-4,
                "convergence_gmax": 3.0e-4,
                "convergence_drms": 2.0e-3,
                "convergence_dmax": 4.0e-3,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "ORCA":
            self.conv_criteria = {
                "convergence_energy": 5e-6,
                "convergence_grms": 1e-4,
                "convergence_gmax": 3.0e-4,
                "convergence_drms": 2.0e-3,
                "convergence_dmax": 4.0e-3,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "Chemshell":
            self.conv_criteria = {
                "convergence_energy": 1e-6,
                "convergence_grms": 3e-4,
                "convergence_gmax": 4.5e-4,
                "convergence_drms": 1.2e-3,
                "convergence_dmax": 1.8e-3,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "ORCA_TIGHT":
            self.conv_criteria = {
                "convergence_energy": 1e-6,
                "convergence_grms": 3e-5,
                "convergence_gmax": 1.0e-4,
                "convergence_drms": 6.0e-4,
                "convergence_dmax": 1.0e-3,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "GAU":
            self.conv_criteria = {
                "convergence_energy": 1e-6,
                "convergence_grms": 3e-4,
                "convergence_gmax": 4.5e-4,
                "convergence_drms": 1.2e-3,
                "convergence_dmax": 1.8e-3,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "GAU_TIGHT":
            self.conv_criteria = {
                "convergence_energy": 1e-6,
                "convergence_grms": 1e-5,
                "convergence_gmax": 1.5e-5,
                "convergence_drms": 4.0e-5,
                "convergence_dmax": 6e-5,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "GAU_VERYTIGHT":
            self.conv_criteria = {
                "convergence_energy": 1e-6,
                "convergence_grms": 1e-6,
                "convergence_gmax": 2e-6,
                "convergence_drms": 4.0e-6,
                "convergence_dmax": 6e-6,
                "convergence_cmax": 1.0e-2,
            }
        elif convergence_setting == "SuperLoose":
            self.conv_criteria = {
                "convergence_energy": 1e-1,
                "convergence_grms": 1e-1,
                "convergence_gmax": 1e-1,
                "convergence_drms": 1e-1,
                "convergence_dmax": 1e-1,
                "convergence_cmax": 1.0e-2,
            }
        else:
            raise InputError("Unknown convergence setting. Exiting...")

    # Parse the constraints into bond, angle, dihedral
    def define_constraints(self, constraints):
        """Translate the user constraints dict into geomeTRIC's constraint lists.

        Args:
            constraints: dict keyed by constraint type ("bond", "angle", "dihedral",
                "frozenatoms", "x", "y", "z", ...) with atom-index lists as values.

        Returns:
            The per-type constraint lists in the order write_constraintsfile expects.
        """
        logger.info("Inside define_constraints")
        logger.info("Constraints: %s", constraints)
        ########################################
        # CONSTRAINTS
        ########################################
        # For QM/MM we need to convert full-system atoms into active region atoms
        if self.active_region and constraints is not None:
            logger.info("Constraints set. Active region true")
            logger.info("User-defined constraints (fullsystem-indices): %s", constraints)
            constraints = constraints_indices_convert(constraints, self.actatoms)
            logger.info("Converting constraints indices to active-region indices")
            logger.info("Constraints (actregion-indices): %s", constraints)

        # Getting individual constraints from constraints dict
        if constraints is not None:
            bondconstraints = constraints.get("bond")
            angleconstraints = constraints.get("angle")
            if "dihedral" in constraints:
                dihedralconstraints = constraints["dihedral"]
            elif "torsion" in constraints:
                dihedralconstraints = constraints["torsion"]
            else:
                dihedralconstraints = None
            xyzconstraints = constraints.get("xyz")
            xconstraints = constraints.get("x")
            yconstraints = constraints.get("y")
            zconstraints = constraints.get("z")
            xyconstraints = constraints.get("xy")
            xzconstraints = constraints.get("xz")
            yzconstraints = constraints.get("yz")
        else:
            bondconstraints = None
            angleconstraints = None
            dihedralconstraints = None
            xyzconstraints = None
            xconstraints = None
            yconstraints = None
            zconstraints = None
            xyconstraints = None
            xzconstraints = None
            yzconstraints = None

        return (
            bondconstraints,
            angleconstraints,
            dihedralconstraints,
            xyzconstraints,
            xconstraints,
            yconstraints,
            zconstraints,
            xyconstraints,
            xzconstraints,
            yzconstraints,
        )

    def write_constraintsfile(
        self,
        frozenatoms,
        bondconstraints,
        constrainvalue,
        angleconstraints,
        dihedralconstraints,
        xconstraints,
        yconstraints,
        zconstraints,
        xyconstraints,
        xzconstraints,
        yzconstraints,
    ):
        """Write the geomeTRIC constraints.txt file.

        Args:
            frozenatoms: atom indices held fixed in all three Cartesian directions.
            bondconstraints: [i, j] pairs whose distance is constrained.
            constrainvalue: whether the constraint lists carry target values as a final element.
            angleconstraints: [i, j, k] triples whose angle is constrained.
            dihedralconstraints: [i, j, k, l] quadruples whose dihedral is constrained.
            xconstraints: atom indices frozen in x.
            yconstraints: atom indices frozen in y.
            zconstraints: atom indices frozen in z.
            xyconstraints: atom indices frozen in x and y.
            xzconstraints: atom indices frozen in x and z.
            yzconstraints: atom indices frozen in y and z.
        """
        logger.info("Inside write_constraintsfile")

        # Delete possible old constraintsfile
        with contextlib.suppress(FileNotFoundError):
            os.remove("constraints.txt")
        ########################################
        # CONSTRAINTS
        ########################################
        # Write constraints to constraints.txt file
        # Frozen atom option. Only for small systems. Not QM/MM etc.
        self.constraintsfile = None
        if len(frozenatoms) > 0:
            logger.info("Writing frozen atom constraints")
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                for frozat in frozenatoms:
                    # Changing from zero-indexing (openmmqmmm) to 1-indexing (geomeTRIC)
                    frozenatomindex = frozat + 1
                    confile.write(f"xyz {frozenatomindex}\n")
        # Bond constraints
        if bondconstraints is not None:
            logger.info("Writing bond constraints %s", bondconstraints)
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                if constrainvalue is True:
                    confile.write("$set\n")
                else:
                    confile.write("$freeze\n")

                for bondpair in bondconstraints:
                    # Changing from zero-indexing (openmmqmmm) to 1-indexing (geomeTRIC)
                    if constrainvalue is True:
                        # First 2 are indices, last is value
                        confile.write(f"distance {bondpair[0] + 1} {bondpair[1] + 1} {bondpair[2]}\n")
                    else:
                        confile.write(f"distance {bondpair[0] + 1} {bondpair[1] + 1}\n")
        # Angle constraints
        if angleconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                if constrainvalue is True:
                    confile.write("$set\n")
                else:
                    confile.write("$freeze\n")
                for angleentry in angleconstraints:
                    # Changing from zero-indexing (openmmqmmm) to 1-indexing (geomeTRIC)
                    if constrainvalue is True:
                        confile.write(
                            f"angle {angleentry[0] + 1} {angleentry[1] + 1} {angleentry[2] + 1} {angleentry[3]}\n"
                        )
                    else:
                        confile.write(f"angle {angleentry[0] + 1} {angleentry[1] + 1} {angleentry[2] + 1}\n")
        if dihedralconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                if constrainvalue is True:
                    confile.write("$set\n")
                else:
                    confile.write("$freeze\n")
                for dihedralentry in dihedralconstraints:
                    # Changing from zero-indexing (openmmqmmm) to 1-indexing (geomeTRIC)
                    if constrainvalue is True:
                        confile.write(
                            f"dihedral {dihedralentry[0] + 1} {dihedralentry[1] + 1} {dihedralentry[2] + 1} "
                            f"{dihedralentry[3] + 1} {dihedralentry[4]}\n"
                        )
                    else:
                        confile.write(
                            f"dihedral {dihedralentry[0] + 1} {dihedralentry[1] + 1} {dihedralentry[2] + 1} "
                            f"{dihedralentry[3] + 1}\n"
                        )
        if xconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                confile.writelines(f"x {xentry + 1}\n" for xentry in xconstraints)
        if yconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                confile.writelines(f"y {yentry + 1}\n" for yentry in yconstraints)
        if zconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                confile.writelines(f"z {zentry + 1}\n" for zentry in zconstraints)
        if xyconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                confile.writelines(f"xy {xyentry + 1}\n" for xyentry in xyconstraints)
        if xzconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                confile.writelines(f"xz {xzentry + 1}\n" for xzentry in xzconstraints)
        if yzconstraints is not None:
            self.constraintsfile = "constraints.txt"
            with open("constraints.txt", "a") as confile:
                confile.write("$freeze\n")
                confile.writelines(f"yz {yzentry + 1}\n" for yzentry in yzconstraints)

    def cleanup(self):
        # Clean-up before we begin
        """Delete the optimizer's scratch files, including any constraints.txt, before a run."""
        tmpfiles = [
            "geometric_OPTtraj.log",
            "geometric_OPTtraj.xyz",
            "geometric_OPTtraj_Full.xyz",
            "geometric_OPTtraj_QMregion.xyz",
            "optimization_energies.log",
            "constraints.txt",
            "initialxyzfiletric.xyz",
            "geometric_OPTtraj.tmp",
            "dummyprefix.tmp",
            "dummyprefix.log",
            "fragment_optimized.frag",
            "Fragment-optimized.xyz",
            "Fragment-optimized_Active.xyz",
            "geometric_OPTtraj-PDB.pdb",
        ]
        for tmpfile in tmpfiles:
            try:
                shutil.rmtree(tmpfile)
            except FileNotFoundError:
                pass
            except NotADirectoryError:
                os.remove(tmpfile)
            else:
                pass

    def hessian_option(self, fragment, actatoms, theory, charge, mult, modelhessian):
        # If actatoms is empty list then we must be using all atoms so defining this
        """Provide the starting Hessian geomeTRIC was asked for.

        Computes a numerical or model Hessian as required and writes it where geomeTRIC
        expects to find it.

        Args:
            fragment: the fragment being optimized.
            actatoms: active-region atom indices; empty means all atoms.
            theory: theory used for a numerical Hessian.
            charge: total charge.
            mult: spin multiplicity.
            modelhessian: model-Hessian name, e.g. "Almloef", "Lindh" or "Schlegel".
        """
        atomsused = fragment.allatoms if len(actatoms) == 0 else actatoms

        if isinstance(self.hessian, np.ndarray):
            logger.info("Hessian option provided is a Numpy array.")

            # Sanity check. Check that the Hessian provided is compatible with actatoms
            logger.info("Checking that Hessian is compatible with active atoms")
            if self.hessian.shape[0] != 3 * len(atomsused):
                raise InputError(
                    "{}\n{}".format(
                        f"Error: Hessian shape is {self.hessian.shape}  which is incompatible with the  number of "
                        f"active atoms present ({len(atomsused)})",
                        f"Hessian should have dimension of 3*N x 3*N where N is the number of active-atoms of the "
                        f"system (should be : {3 * len(atomsused)} x {3 * len(atomsused)})",
                    )
                )

            logger.info("Writing Hessian array to disk.")

            hessianfile = "Hessian_np"
            write_hessian(self.hessian, hessfile=hessianfile)
            self.hessian = "file:" + hessianfile
            logger.info("Hessian option to be used by geometric: %s", self.hessian)

        elif isinstance(self.hessian, str):
            logger.info("Hessian option provided is a string")
            if self.hessian == "xtb":
                raise InputError(
                    "Error: hessian='xtb' is not available in this ORCA+OpenMM build. Use '1point', '2point', "
                    "'partial' or a Hessian file instead."
                )
            # NumFreq 1 and 2-point Hessians
            elif self.hessian == "1point":
                logger.info("Requested Hessian from Numfreq 1-point approximation (running in serial)")
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory, fragment=fragment, npoint=1, runmode="serial", numcores=theory.numcores
                )
                hessianfile = "Hessian_from_theory"
                shutil.copyfile("Numfreq_dir/Hessian", hessianfile)
                self.hessian = "file:" + str(hessianfile)
            elif self.hessian == "2point":
                logger.info("Requested Hessian from Numfreq 2-point approximation (running in serial)")
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory, fragment=fragment, npoint=2, runmode="serial", numcores=theory.numcores
                )
                hessianfile = "Hessian_from_theory"
                shutil.copyfile("Numfreq_dir/Hessian", hessianfile)
                self.hessian = "file:" + str(hessianfile)
            elif self.hessian == "partial":
                logger.info("Partial Hessian option requested")

                if self.partial_hessian_atoms is None:
                    raise InputError(
                        "hessian='partial' option requires setting the partial_hessian_atoms option. Exiting."
                    )

                logger.info("Now doing partial Hessian calculation using atoms: %s", self.partial_hessian_atoms)
                # Note: hardcoding runmode='serial' for now
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory,
                    fragment=fragment,
                    npoint=1,
                    hessatoms=self.partial_hessian_atoms,
                    runmode="serial",
                    numcores=1,
                )
                # Combine partial exact Hessian with model Hessian(Almloef, Lindh, Schlegel or unit)
                # Large Hessian is the actatoms Hessian if actatoms provided

                combined_hessian = approximate_full_hessian_from_smaller(
                    fragment,
                    result_freq.hessian,
                    self.partial_hessian_atoms,
                    large_atomindices=actatoms,
                    rest_hessian=modelhessian,
                )

                # Write combined Hessian to disk
                hessianfile = "Hessian_from_partial"
                write_hessian(combined_hessian, hessfile=hessianfile)
                self.hessian = "file:" + hessianfile
            elif self.hessian == "partial2":
                logger.info("Partial Numpoint=2 Hessian option requested")

                if self.partial_hessian_atoms is None:
                    raise InputError(
                        "hessian='partial' option requires setting the partial_hessian_atoms option. Exiting."
                    )

                logger.info("Now doing partial Hessian calculation using atoms: %s", self.partial_hessian_atoms)
                # Note: hardcoding runmode='serial' for now
                result_freq = openmmqmmm.numerical_frequencies(
                    theory=theory,
                    fragment=fragment,
                    npoint=2,
                    hessatoms=self.partial_hessian_atoms,
                    runmode="serial",
                    numcores=1,
                )
                # Combine partial exact Hessian with model Hessian(Almloef, Lindh, Schlegel or unit)
                # Large Hessian is the actatoms Hessian if actatoms provided

                combined_hessian = approximate_full_hessian_from_smaller(
                    fragment,
                    result_freq.hessian,
                    self.partial_hessian_atoms,
                    large_atomindices=actatoms,
                    rest_hessian=modelhessian,
                )

                # Write combined Hessian to disk
                hessianfile = "Hessian_from_partial"
                write_hessian(combined_hessian, hessfile=hessianfile)
                self.hessian = "file:" + hessianfile
            elif "file:" in self.hessian:
                hessianfile = self.hessian.replace("file:", "")

            # Allow first and each options still
            if self.hessian not in ["first", "each"]:
                logger.info("Checking that defined Hessian is compatible with active-region")
                hessian_read = read_hessian(hessianfile)
                logger.info("actatoms: %s", actatoms)
                if hessian_read.shape[0] != 3 * len(atomsused):
                    raise InputError(
                        "{}\n{}".format(
                            f"Error: Hessian shape is {hessian_read.shape}  which is incompatible with the  number of "
                            f"active atoms present ({len(atomsused)})",
                            f"Hessian should have dimension of 3*N x 3*N where N is the number of active-atoms of the "
                            f"system (should be : {3 * len(atomsused)} x {3 * len(atomsused)})",
                        )
                    )

        elif self.hessian is None:
            logger.info("No Hessian option provided.")
        else:
            raise InputError("Unknown Hessian option")

    # If using Active region then we write only those coordinates to disk (initialxyzfiletric)
    def setup_active_region_geometry(self, fragment):
        """Build the reduced geometry and topology for an active-region optimization.

        Only the active atoms enter the optimizer's coordinate system; the rest are frozen
        and reinstated afterwards.

        Args:
            fragment: the full-system fragment.
        """
        if len(self.actatoms) == 0:
            raise InputError("Error: List of active atoms (actatoms) provided is empty. This is not allowed.")
        # Sorting list, otherwise trouble
        self.actatoms.sort()
        logger.info("Active Region option Active. Passing only active-region coordinates to geomeTRIC.")
        logger.info("Active atoms list: %s", self.actatoms)
        logger.info("Number of active atoms: %s", len(self.actatoms))

        # Check that the actatoms list does not contain atom indices higher than the number of atoms
        largest_atom_index = max(self.actatoms)
        if largest_atom_index >= fragment.numatoms:
            raise InputError(
                "{}\nThis does not make sense. Please provide a correct actatoms list. Exiting.".format(
                    f"Found active-atom index ({largest_atom_index}) that is larger or equal (>=) than the number of "
                    f"atoms of system ({fragment.numatoms})!"
                )
            )
        # Get active region coordinates and elements
        actcoords, actelems = fragment.get_coords_for_atoms(self.actatoms)

        # Writing act-region coords (only) of fragment to disk as XYZ file and reading into geomeTRIC
        write_xyzfile(actelems, actcoords, "initialxyzfiletric")

    # Running geomeTRIC object
    def run(self, theory=None, fragment=None, charge=None, mult=None, constraints=None, constrainvalue=False):
        """Optimize a geometry with geomeTRIC.

        Updates the fragment's coordinates in place and writes results_optimizer.json.

        Args:
            theory: theory providing energies and gradients.
            fragment: fragment to optimize.
            charge: total charge; defaults to the fragment's.
            mult: spin multiplicity; defaults to the fragment's.
            constraints: constraints dict, see define_constraints.
            constrainvalue: whether the constraint entries carry target values.

        Returns:
            The optimized energy in hartree.
        """
        logger.info("")
        logger.info(sub_header("Running geomeTRIC object"))
        logger.info(
            f"\nDoing geometry optimization on fragment. Formula: {fragment.prettyformula} Label: {fragment.label} "
        )
        # Cleanup of temp-files before we begin
        self.cleanup()  # NOTE: This deletes constraintsfile

        # EARLY EXITS:
        # Check charge/mult
        charge, mult = check_charge_mult(charge, mult, theory.theorytype, fragment, "geomeTRICOptimizer", theory=theory)
        fragment.charge = charge
        fragment.mult = mult

        # Printlevel of fragment

        #################
        # CONSTRAINTS
        #################
        # If constraints not directly provided to run method, then we look at self.constraints and then
        # fragment.constraints
        if constraints is None:
            logger.info("No constraints provided to run method.")
            logger.info("Testing if constraints present in optimizer object")
            if self.constraints is not None:
                logger.info("Found constraints in optimizer object")
                constraints = self.constraints
                constrainvalue = self.constrainvalue
            else:
                logger.info("No constraints in optimizer object.")
                logger.info("Now testing if constraints in fragment object ")
                if fragment.constraints is not None:
                    # Option used by Surface-scan relaxed parallel
                    logger.info("Found constraints in fragment object")
                    constraints = fragment.constraints
                    constrainvalue = True  # Assuming to be the case.
                else:
                    logger.info("No constraints in fragment object.")
        else:
            logger.info("Constraints provided to run method.")
        logger.info("\nConstraints:  %s", constraints)
        logger.info("constrainvalue:  %s", constrainvalue)
        # Getting specific constraints and writing to file
        (
            bondconstraints,
            angleconstraints,
            dihedralconstraints,
            xyzconstraints,
            xconstraints,
            yconstraints,
            zconstraints,
            xyconstraints,
            xzconstraints,
            yzconstraints,
        ) = self.define_constraints(constraints)
        if xyzconstraints is not None:
            logger.info("xyzconstraints found. Adding to frozenatoms")
            self.frozenatoms = self.frozenatoms + xyzconstraints
        self.write_constraintsfile(
            self.frozenatoms,
            bondconstraints,
            constrainvalue,
            angleconstraints,
            dihedralconstraints,
            xconstraints,
            yconstraints,
            zconstraints,
            xyconstraints,
            xzconstraints,
            yzconstraints,
        )
        if self.constraintsinputfile is not None:
            logger.info("constraintsinputfile provided: %s", self.constraintsinputfile)
            if os.path.isfile(self.constraintsinputfile) is False:
                raise FileFormatError(f"Error:File {self.constraintsinputfile} does not exist")
            self.constraintsfile = self.constraintsinputfile
        #################

        # Check if atom and do Singlepoint instead if so
        if fragment.numatoms == 1:
            logger.info("System contains 1 atom, optimization makes no sense.")
            logger.info("Doing single-point energy calculation instead")
            result = openmmqmmm.single_point(fragment=fragment, theory=theory, charge=charge, mult=mult)
            return result

        # ActiveRegion option where geomeTRIC only sees the QM part that is being optimized
        if self.active_region is True:
            self.setup_active_region_geometry(fragment)
        # Whole system
        else:
            # Write coordinates from fragment to disk as XYZ-file and reading into geomeTRIC
            fragment.write_xyzfile("initialxyzfiletric.xyz")

        # Determine geometry-printout in each iteration. Requires knowledge on theory and fragment
        self.print_atoms_output_setting(theory, fragment)
        # Hessian option
        self.hessian_option(fragment, self.actatoms, theory, charge, mult, self.modelhessian)

        ######################
        # CALLING LIBRARY
        ######################
        try:
            import geometric
        except Exception as e:
            logger.info("")
            raise MissingDependencyError(
                f"Problem importing geomeTRIC module!\nEither install geomeTRIC using pip:\n conda install geometric\n "
                f"or \n pip install geometric\n or manually from Github (https://github.com/leeping/geomeTRIC)\nActual "
                f"error message: {e}"
            ) from e
        # bondorders
        # generally unused, except PBC
        self.bothre = 0.0

        # Read geometry from XYZ-file into geomeTRIC Molecule object
        if self.pbc_active is True:
            logger.info("For PBC we activate constraints")
            self.bothre = 0.5
        mol_geometric_frag = geometric.molecule.Molecule("initialxyzfiletric.xyz")

        # Defining GeometricEngine engine object containing geometry and theory. ActiveRegion boolean passed.
        # Also now passing list of atoms to print in each step.
        engine = GeometricEngine(
            mol_geometric_frag,
            theory,
            active_region=self.active_region,
            actatoms=self.actatoms,
            print_atoms_list=self.print_atoms_list,
            mm_pdb_traj_write=self.mm_pdb_traj_write,
            charge=charge,
            mult=mult,
            conv_criteria=self.conv_criteria,
            fragment=fragment,
            maxiter=self.maxiter,
            pbc_active=self.pbc_active,
        )
        # Defining args object, containing engine object
        logger.debug("Constraints file: %s", self.constraintsfile)
        final_geometric_args = GeometricArgs(
            engine,
            self.constraintsfile,
            coordsys=self.coordsystem,
            maxiter=self.maxiter,
            conv_criteria=self.conv_criteria,
            transition=self.ts_opt,
            hessian=self.hessian,
            subfrctor=self.subfrctor,
            verbose=0,
            irc=self.irc,
            rigid=self.rigid,
            enforce_constraints=self.enforce_constraints,
            bothre=self.bothre,
        )

        logger.info("Convergence criteria: %s", self.conv_criteria)
        logger.info("Hessian option: %s", self.hessian)
        logger.info("Coordinate system: %s", self.coordsystem)

        if self.ts_opt:
            logger.info("Starting saddlepoint optimization")
        else:
            logger.info("Starting optimization")

        ###################################
        # RUNNING
        ###################################
        log_time_since(self.time_init, "Time spent before run_optimizer")
        geometric.optimize.run_optimizer(**vars(final_geometric_args))
        time.sleep(1)

        ###################################
        logger.info("")
        logger.info(f"geomeTRIC Geometry optimization converged in {engine.iteration_count + 1} steps!")
        logger.info("")

        # QM/MM: Doing final energy evaluation if Truncated PC option was on
        if isinstance(theory, QMMMTheory):
            if theory.truncated_pc is True:
                logger.info(
                    "Truncated PC approximation was active. Doing final energy calculation with full PC environment"
                )
                theory.truncated_pc = False
                finalenergy, _finalgrad = theory.run(
                    current_coords=engine.full_current_coords,
                    elems=fragment.elems,
                    grad=True,
                    charge=charge,
                    mult=mult,
                )
            else:
                finalenergy = engine.energy
        else:
            # Updating energy and coordinates of fragment before ending
            finalenergy = engine.energy

        logger.info("Final optimized energy: %s", finalenergy)

        # Replacing coordinates in fragment
        fragment.replace_coords(fragment.elems, engine.full_current_coords, conn=False)
        # Writing out fragment file and XYZ file
        fragment.print_system(filename="fragment_optimized.frag")
        fragment.write_xyzfile(xyzfilename="Fragment-optimized.xyz")
        fragment.set_energy(finalenergy)

        if self.active_region is not True:
            logger.info("Final geometry")
            fragment.print_coords()
        logger.info("")

        # PBC
        if self.pbc_active:
            logger.info("PBC True. Writing final optimized geometry in PBC-format")
            logger.info("PBC_format_option: %s", self.pbc_format_option)
            if self.pbc_format_option.upper() == "CIF":
                convert_to_pbcfile = write_cif_file
                file_ext = "cif"
            elif self.pbc_format_option.upper() == "XSF":
                convert_to_pbcfile = write_xsf_file
                file_ext = "xsf"
            elif self.pbc_format_option.upper() == "POSCAR":
                convert_to_pbcfile = write_poscar_file
                file_ext = "POSCAR"
            convert_to_pbcfile(
                fragment.coords,
                fragment.elems,
                cellvectors=theory.periodic_cell_vectors,
                filename=f"Fragment-optimized.{file_ext}",
            )
            logger.info(f"Final cell vectors (Å):{theory.periodic_cell_vectors}")
            logger.info(f"Final cell parameters: ({cell_vectors_to_params(theory.periodic_cell_vectors)})")
            logger.info(f"Final cell volume (Å):{cell_volume(theory.periodic_cell_vectors)}")
        # Active region XYZ-file
        if self.active_region is True:
            write_xyz_for_atoms(fragment.coords, fragment.elems, self.actatoms, "Fragment-optimized_Active")
        # QM-region XYZ-file
        if isinstance(theory, QMMMTheory):
            write_xyz_for_atoms(fragment.coords, fragment.elems, theory.qmatoms, "Fragment-optimized_QMregion")

        # Printing internal coordinate table
        if len(self.print_atoms_list) < 50:
            _print_internal_coordinate_table(fragment, actatoms=self.print_atoms_list)
        logger.info("")

        # Now returning final Results object
        # Note: could include the geometry in object but can be very large causing printing head-aches on screen,
        # ignoring for now since the geometry is in the Fragment object anyway
        result = Results(label="Optimizer", energy=finalenergy)
        if self.result_write_to_disk is True:
            result.write_to_disk(filename="results_optimizer.json")
        return result


class GeometricArgs:
    """Argument container passed to geometric.optimize.run_optimizer.

    Attribute names mirror the geomeTRIC keyword arguments exactly (including
    logIni) - do not rename them.
    """

    def __init__(
        self,
        eng,
        constraintsfile,
        coordsys,
        maxiter,
        conv_criteria,
        transition,
        hessian,
        subfrctor,
        verbose,
        irc,
        rigid,
        enforce_constraints,
        bothre,
    ):
        self.coordsys = coordsys
        self.maxiter = maxiter
        self.transition = transition
        self.hessian = hessian
        self.subfrctor = subfrctor
        self.verbose = verbose
        self.irc = irc
        self.rigid = rigid
        self.bothre = bothre
        if self.rigid is True:
            logger.info("Rigid optimization enabled.")
            logger.info("Activating revised constraint algorithm")
            self.conmethod = 1
        # For constraints:
        if enforce_constraints is not None:
            logger.info("enforce_constraints value passed: %s", enforce_constraints)
            self.enforce = enforce_constraints

        # Setting these to be part of kwargs that geometric reads
        self.convergence_energy = conv_criteria["convergence_energy"]
        self.convergence_grms = conv_criteria["convergence_grms"]
        self.convergence_gmax = conv_criteria["convergence_gmax"]
        self.convergence_drms = conv_criteria["convergence_drms"]
        self.convergence_dmax = conv_criteria["convergence_dmax"]
        self.convergence_cmax = conv_criteria["convergence_cmax"]
        self.prefix = "geometric_OPTtraj"
        self.input = "dummyinputname"
        self.constraints = constraintsfile
        # Created log.ini file here. Missing from pip installation for some reason?
        # Storing log.ini in openmmqmmm dir
        path = openmmqmmm.constants.PACKAGE_DIR
        self.logIni = path + "/log.ini"
        self.customengine = eng


# Engine class used to communicate with geomeTRIC
class GeometricEngine:
    """Custom geomeTRIC engine that evaluates energies/gradients with a theory object.

    Method names (calc, load_guess_files, save_guess_files, detect_dft,
    calc_bondorder, clearCalcs) and the attribute M are the geomeTRIC engine
    protocol - do not rename them.
    """

    def __init__(
        self,
        geometric_molf,
        theory,
        active_region=False,
        actatoms=None,
        print_atoms_list=None,
        charge=None,
        mult=None,
        conv_criteria=None,
        fragment=None,
        mm_pdb_traj_write=False,
        maxiter=None,
        pbc_active=False,
    ):
        # MM_PDB_traj_write on/off. Can be pretty big files
        self.mm_pdb_traj_write = mm_pdb_traj_write
        # Defining M attribute of engine object as geomeTRIC Molecule object
        self.M = geometric_molf
        # Defining theory from argument
        self.theory = theory
        self.active_region = active_region
        # Defining current_coords for full system (not only act region)
        self.full_current_coords = []
        # E+G count
        self.EG_count = 0
        # Proper iteration count
        self.iteration_count = 0

        # Maxiter
        self.maxiter = maxiter
        # Defining initial E
        self.energy = 0
        # Active atoms
        self.actatoms = actatoms
        # Print-list atoms (set above)
        self.print_atoms_list = print_atoms_list
        self.charge = charge
        self.mult = mult
        self.conv_criteria = conv_criteria
        self.fragment = fragment

        # Setting BO matrix to be None
        self.BOmatrix = None
        # PBC

        self.pbc_active = pbc_active
        if self.pbc_active is True:
            # Real elements
            self.elems_phys = self.fragment.elems
            # Align to standard orientation
            aligned_atom_coords, aligned_vectors = align_to_standard_orientation(
                self.fragment.coords, theory.periodic_cell_vectors
            )
            self.fragment.coords = aligned_atom_coords
            self.theory.update_cell(aligned_vectors)

            # Reference
            self.H_ref = aligned_vectors.copy()
            self.H_ref_inv = np.linalg.inv(self.H_ref)

            # Modifying self.M to have aligned coords and 4 dummyatoms
            self.M.xyzs = [np.concatenate((aligned_atom_coords, [[0.0, 0.0, 0.0]], aligned_vectors), axis=0)]
            self.M.elem = [*self.M.elem, "F", "F", "F", "F"]

    def load_guess_files(self, dirname):
        logger.info("geometric called load_guess_files option for GeometricEngine.")
        logger.info("This option is currently not supported here. Continuing.")

    def save_guess_files(self, dirname):
        logger.info("geometric called save_guess_files option option for GeometricEngine.")
        logger.info("This option is currently not supported here. Continuing.")

    # Optimizer may call this to see if the engine class is doing DFT with grid to print warning
    def detect_dft(self):
        logger.info("geometric called detect_dft option option for GeometricEngine.")
        return True

    # geometric checks if calc_bondorder method is implemented for the custom engine. Disabled until we implement this
    def calc_bondorder(self, coords, dirname):
        logger.info("geometric called calc_bondorder option option for GeometricEngine.")
        if self.BOmatrix is not None:
            return self.BOmatrix
        else:
            logger.info("no BOmatrix found")
            if self.pbc_active:
                logger.info("PBC and BOmatrix handling")
                # Bond orders
                self.BOmatrix = np.zeros((len(self.M.elem), len(self.M.elem)), dtype=int)
                # bond orders based on fragment connectivity
                self.fragment.calc_connectivity()
                from openmmqmmm.coords import get_connected_atoms_dict

                conndict = get_connected_atoms_dict(self.fragment.coords, self.fragment.elems, 1.0, 0.1)
                logger.info("conndict: %s", conndict)
                for i, conn in conndict.items():
                    for c in conn:
                        self.BOmatrix[i, c] = self.BOmatrix[c, i] = 1.0

                # Connecting origin and lattice atoms
                n_orig = len(self.elems_phys)
                self.BOmatrix[n_orig, n_orig + 1] = self.BOmatrix[n_orig + 1, n_orig] = 1
                self.BOmatrix[n_orig, n_orig + 2] = self.BOmatrix[n_orig + 2, n_orig] = 1
                self.BOmatrix[n_orig, n_orig + 3] = self.BOmatrix[n_orig + 3, n_orig] = 1

                return self.BOmatrix
            else:
                logger.info("No BO option implemented")
                return None

            return None

    # TODO: geometric will regularly do ClearCalcs in an optimization
    def clearCalcs(self):  # noqa: N802 - geomeTRIC engine API, do not rename
        logger.info("geometric called clearCalcs option for GeometricEngine.")
        logger.info("This option is currently not supported here. Continuing.")

    # Writing out trajectory file for full system in case of ActiveRegion. Note: Actregion coordinates are done done by
    # GeomeTRIC
    def write_trajectory_full(self):
        logger.info("Writing trajectory for Full system to file: geometric_OPTtraj_Full.xyz")
        with open("geometric_OPTtraj_Full.xyz", "a") as trajfile:
            trajfile.write(str(self.fragment.numatoms) + "\n")
            trajfile.write(f"Iteration {self.iteration_count} Energy {self.energy} \n")
            trajfile.writelines(
                el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n"
                for el, cor in zip(self.fragment.elems, self.full_current_coords, strict=False)
            )

    # QM/MM: Writing out trajectory file for QM-region if QM/MM.
    def write_trajectory_qmregion(self):
        logger.info("Writing trajectory for QM-region to file: geometric_OPTtraj_QMregion.xyz")
        with open("geometric_OPTtraj_QMregion.xyz", "a") as trajfile:
            trajfile.write(str(len(self.theory.qmatoms)) + "\n")
            trajfile.write(f"Iteration {self.iteration_count} Energy {self.energy} \n")
            qm_coords, qm_elems = self.fragment.get_coords_for_atoms(self.theory.qmatoms)
            trajfile.writelines(
                el + "  " + str(cor[0]) + " " + str(cor[1]) + " " + str(cor[2]) + "\n"
                for el, cor in zip(qm_elems, qm_coords, strict=False)
            )

    def write_energy_logfile(self):
        # QM/MM: Writing out logfile containing QM-energy, MM-energy, QM/MM-energy
        logger.info("Writing logfile with energies: optimization_energies.log")
        with open("optimization_energies.log", "a") as trajfile:
            if self.iteration_count == 0:
                trajfile.write("Iteration QM-energy       (Eh) MM-Energy (Eh)  QM/MM-Energy (Eh)\n")
            trajfile.write(
                f"{self.iteration_count}         {self.theory.QMenergy} {self.theory.MMenergy} "
                f"{self.theory.QM_MM_energy}\n"
            )

    def write_pdbtrajectory(self):
        logger.info("Writing PDB-trajectory to file: geometric_OPTtraj-PDB.pdb")
        pdbtrajectoryfile = "geometric_OPTtraj-PDB.pdb"
        # Get OpenMM positions
        # STILL problem with PBC
        state = self.theory.mm_theory.simulation.context.getState(
            getEnergy=False, getPositions=True, getForces=False, enforcePeriodicBox=True
        )
        newpos = state.getPositions()
        with open(pdbtrajectoryfile, "a") as pdbfh:
            self.theory.mm_theory.openmm.app.PDBFile.writeFile(self.theory.mm_theory.topology, newpos, file=pdbfh)

    # Defining calculator.
    # Read_data and copydir not used (dummy variables)
    def calc(self, coords, tmp, read_data=None, copydir=None):
        logger.info("")
        if self.iteration_count == self.maxiter:
            raise OpenMMQMMMError(
                f"Geometry optimization stopped: maxiter ({self.maxiter}) reached without convergence"
            )

        # Note: tmp and read_data not used. Needed for geomeTRIC version compatibility
        logger.info("Convergence criteria: %s", self.conv_criteria)

        logger.info("")
        # Updating coords in object
        # Need to combine with rest of full-system coords
        time.time()
        self.M.xyzs[0] = coords.reshape(-1, 3) * openmmqmmm.constants.bohr2ang
        currcoords = self.M.xyzs[0]

        # Call method to use
        if self.active_region is True:
            egdict = self.actregion_calc(currcoords)
        elif self.pbc_active is True:
            logger.info("Doing PBC opt-step")
            egdict = self.pbc_calc(currcoords)
        else:
            egdict = self.regular_calc(currcoords)

        return egdict

    def actregion_calc(self, currcoords):
        # Special act-region (for QM/MM) since GeomeTRIC does not handle huge system and constraints
        if self.active_region is True:
            # Defining full_coords as original coords temporarily
            full_coords = self.fragment.coords

            # Replacing act-region coordinates in full_coords with coords from currcoords
            for act_i, curr_i in zip(self.actatoms, currcoords, strict=False):
                full_coords[act_i] = curr_i
            time.time()
            self.full_current_coords = full_coords

            # Write out fragment with updated coordinates for the purpose of doing restart
            self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
            self.fragment.print_system(filename="fragment_currentgeo.frag")
            self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
            time.time()

            # PRINTING TO OUTPUT SPECIFIC GEOMETRY IN EACH GEOMETRIC ITERATION (now: self.print_atoms_list)
            logger.info(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
            logger.info("-------------------------------------------------")

            # print_atoms_list
            # Previously act: print_coords_for_atoms(self.full_current_coords, fragment.elems, self.actatoms)
            print_coords_for_atoms(self.full_current_coords, self.fragment.elems, self.print_atoms_list)
            time.time()
            logger.info("Note: Only print_atoms_list region printed above")
            # Request Engrad calc for full system

            E, grad = self.theory.run(
                current_coords=self.full_current_coords,
                elems=self.fragment.elems,
                charge=self.charge,
                mult=self.mult,
                grad=True,
            )
            time.time()

            if logger.isEnabledFor(logging.DEBUG):
                write_coords_all(
                    grad,
                    self.fragment.elems,
                    indices=self.fragment.allatoms,
                    file="Grad",
                    description="Grad (au/Bohr):",
                )
            # Trim Full gradient down to only act-atoms gradient
            Grad_act = np.array([grad[i] for i in self.actatoms])
            if logger.isEnabledFor(logging.DEBUG):
                act_elems = [self.fragment.elems[i] for i in self.actatoms]
                write_coords_all(
                    Grad_act,
                    act_elems,
                    indices=list(range(len(self.actatoms))),
                    file="Grad_act",
                    description="Grad_act (au/Bohr):",
                )
            time.time()
            self.energy = E

            logger.info("Writing trajectory for Active Region to file: geometric_OPTtraj.xyz")

            # Now writing trajectory for full system
            self.write_trajectory_full()

            # Case QM/MM:
            if isinstance(self.theory, QMMMTheory):
                # Writing trajectory for QM-region only
                self.write_trajectory_qmregion()
                # Writing logfile with QM,MM and QM/MM energies
                self.write_energy_logfile()

                # Case MMtheory is OpenMM: Write out PDB-trajectory via OpenMM
                if isinstance(self.theory.mm_theory, OpenMMTheory) and self.mm_pdb_traj_write is True:
                    self.write_pdbtrajectory()

            time.time()

            # Read last line of geometric_OPTtraj.log to get step
            step_lines = pygrep2("Step ", "geometric_OPTtraj.log", print_output=False, errors=None)
            if len(step_lines) > 0:
                iteration = int(step_lines[-1].split("Step", 1)[1].split(":", 1)[0].strip())
                self.iteration_count = int(iteration)
            self.EG_count += 1

            return {"energy": E, "gradient": Grad_act.flatten()}

    # Basic calc: no actregion, no PBC
    def regular_calc(self, currcoords):
        self.full_current_coords = currcoords
        self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
        # PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
        self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
        logger.info(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
        logger.info("---------------------------------------------------")
        print_coords_for_atoms(currcoords, self.fragment.elems, self.print_atoms_list)
        logger.info("")
        logger.info("Note: printed only print_atoms_list (this is not necessarily all atoms) ")
        E, grad = self.theory.run(
            current_coords=currcoords, elems=self.M.elem, charge=self.charge, mult=self.mult, grad=True
        )
        # Read last line of geometric_OPTtraj.log to get step
        step_lines = pygrep2("Step ", "geometric_OPTtraj.log", print_output=False, errors=None)
        if len(step_lines) > 0:
            iteration = int(step_lines[-1].split("Step", 1)[1].split(":", 1)[0].strip())
            self.iteration_count = int(iteration)
        self.EG_count += 1
        self.energy = E
        return {"energy": E, "gradient": grad.flatten()}

    def pbc_calc(self, currcoords):
        # Split  coords into atomic and lattic
        R_geo = currcoords[:-4]
        origin = currcoords[-4]
        H_geo = currcoords[-3:] - origin

        # --- Enforce Standard Orientation in each step ---
        logger.info("Enforcing orientation")
        # 1. Ensure the Origin dummy atom stays at exactly 0,0,0
        origin[:] = 0.0
        # 2. Force H_geo to be strictly upper-triangular
        # Vector A: Only Ax is allowed (Ay and Az are zero)
        H_geo[0, 1] = 0.0  # ay = 0
        H_geo[0, 2] = 0.0  # az = 0
        # Vector B: Only Bx and By are allowed (Bz is zero)
        H_geo[1, 2] = 0.0  # bz = 0
        # -----------------------------------------------------
        s = np.dot(R_geo - origin, self.H_ref_inv)
        R_phys = np.dot(s, H_geo) + origin
        # Update cell parameters in theory
        self.theory.update_cell(H_geo)

        self.full_current_coords = R_phys
        self.fragment.replace_coords(self.fragment.elems, self.full_current_coords, conn=False)
        # PRINTING ACTIVE GEOMETRY IN EACH GEOMETRIC ITERATION
        self.fragment.write_xyzfile(xyzfilename="Fragment-currentgeo.xyz")
        logger.info(f"Current geometry (Å) in step {self.iteration_count} (print_atoms_list region)")
        logger.info("---------------------------------------------------")
        print_coords_for_atoms(R_phys, self.elems_phys, self.print_atoms_list)
        logger.info("")
        logger.info("Note: printed only print_atoms_list (this is not necessarily all atoms) ")
        logger.info(f"Current cell vectors (Å):{H_geo}")
        logger.info(f"Current cell volume (Å):{cell_volume(H_geo)}")

        # E + G from theory
        E, grad_phys = self.theory.run(
            current_coords=R_phys, elems=self.elems_phys, charge=self.charge, mult=self.mult, grad=True
        )
        self.EG_count += 1
        self.energy = E

        # Read last line of geometric_OPTtraj.log to get step
        step_lines = pygrep2("Step ", "geometric_OPTtraj.log", print_output=False, errors=None)
        if len(step_lines) > 0:
            iteration = int(step_lines[-1].split("Step", 1)[1].split(":", 1)[0].strip())
            self.iteration_count = int(iteration)

        # Transformation
        # M is the transformation matrix: R_phys = R_geo @ M
        M = np.dot(self.H_ref_inv, H_geo)
        grad_Rgeo = np.dot(grad_phys, M.T)

        # Convection, implicit lattice gradient

        # Lattice gradient and masking
        # Total lattice gradient: current theory cell-gradient + convection
        grad_latt_total = self.theory.get_cell_gradient()
        # Standard orientation mask:
        # This zeros out: a_y, a_z, and b_z
        mask = np.array(
            [
                [1, 0, 0],  # dE/dax (ay, az frozen)
                [1, 1, 0],  # dE/dbx, dE/dby (bz frozen)
                [1, 1, 1],  # dE/dcx, dE/dcy, dE/dcz (all free)
            ]
        )
        grad_latt_masked = grad_latt_total * mask
        # Making sure origin is zero
        grad_origin = np.zeros((1, 3))
        # Final modified gradient to pass to geomeTRIC
        mod_gradient = np.concatenate(
            [
                grad_Rgeo,  # (N, 3)
                grad_origin,  # (1, 3)
                grad_latt_masked,  # (3, 3)
            ],
            axis=0,
        )

        return {"energy": E, "gradient": mod_gradient.flatten()}


# Function Convert constraints indices to actatom indices
def constraints_indices_convert(con, actatoms):
    try:
        bondcons = con["bond"]
    except KeyError:
        bondcons = []
    try:
        anglecons = con["angle"]
    except KeyError:
        anglecons = []
    try:
        dihedralcons = con["dihedral"]
    except KeyError:
        dihedralcons = []
    # Looping over constraints-class (bond,angle-dihedral)
    # list-item:
    for bc in bondcons:
        bc[0] = fullindex_to_actindex(bc[0], actatoms)
        bc[1] = fullindex_to_actindex(bc[1], actatoms)
    for ac in anglecons:
        ac[0] = fullindex_to_actindex(ac[0], actatoms)
        ac[1] = fullindex_to_actindex(ac[1], actatoms)
        ac[2] = fullindex_to_actindex(ac[2], actatoms)
    for dc in dihedralcons:
        dc[0] = fullindex_to_actindex(dc[0], actatoms)
        dc[1] = fullindex_to_actindex(dc[1], actatoms)
        dc[2] = fullindex_to_actindex(dc[2], actatoms)
        dc[3] = fullindex_to_actindex(dc[3], actatoms)
    return con
