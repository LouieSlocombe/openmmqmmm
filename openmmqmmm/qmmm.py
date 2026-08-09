import copy
import logging
import math
import time

import numpy as np

import openmmqmmm.coords
from openmmqmmm.coords import CONNECTIVITY_SCALE, CONNECTIVITY_TOL, Fragment
from openmmqmmm.exceptions import (
    InputError,
    InternalError,
)
from openmmqmmm.utils import log_time_since, main_header, writelisttofile

logger = logging.getLogger(__name__)

# QM/MM theory object.
# Required at init: qm_theory and qmatoms and fragment


class QMMMTheory:
    def __init__(
        self,
        qm_theory=None,
        qmatoms=None,
        fragment=None,
        mm_theory=None,
        charges=None,
        embedding="elstat",
        numcores=1,
        excludeboundaryatomlist=None,
        unusualboundary=False,
        openmm_externalforce=False,
        truncated_pc=False,
        truncated_pc_radius=55,
        truncated_pc_recalc_iter=50,
        qm_charge=None,
        qm_mult=None,
        chargeboundary_method="shift",
        exit_after_customexternalforce_update=False,
        dipole_correction=True,
        linkatom_method="simple",
        linkatom_simple_distance=None,
        linkatom_forceproj_method="adv",
        linkatom_ratio=0.723,
        linkatom_type="H",
        update_qm_region_charges=False,
    ):

        module_init_time = time.time()
        time.time()
        logger.info(main_header("QM/MM Theory"))

        # Check for necessary keywords
        if qm_theory is None or qmatoms is None:
            raise InputError("Error: QMMMTheory requires defining: qm_theory, qmatoms, fragment")
        # If fragment object has not been defined
        if fragment is None:
            raise InputError("fragment= keyword has not been defined for QM/MM. Exiting")

        # Defining charge/mult of QM-region
        self.qm_charge = qm_charge
        self.qm_mult = qm_mult

        # Indicate that this is a hybrid QM/MM type theory
        self.theorytype = "QM/MM"
        self.theorynamelabel = "QMMMTheory"

        # External force energy. Zero except when using openmm_externalforce
        self.extforce_energy = 0.0
        # Subtractive corrections that might be defined later on
        # Added due to pbcmm-elstat
        self.subtractive_correction_E = 0.0
        self.subtractive_correction_G = np.zeros((len(fragment.coords), 3))

        # update_QMregion_charges
        # After each QM-region calculation, the charges of the QM-region may have been calculated
        # These charges can be used to update the charges of the whole system. Only used for mechanical embedding
        self.update_qm_region_charges = update_qm_region_charges

        # Linkatoms False by default. Later checked.
        self.linkatoms = False

        # Linkatom method strategy to determine linkatom position or QM-L distance
        self.linkatom_type = linkatom_type  # Usually 'H'
        self.linkatom_method = linkatom_method  # Options: 'simple' or 'ratio'
        self.linkatom_simple_distance = linkatom_simple_distance  # For method simple, Default 1.09 Angstrom
        # For method ratio. see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
        self.linkatom_ratio = linkatom_ratio
        # Linkatom projection method Options: 'adv', 'lever', 'chain', 'none'
        self.linkatom_forceproj_method = linkatom_forceproj_method
        if self.linkatom_forceproj_method is None:
            linkatom_forceproj_method = "none"

        # Counter for how often QMMMTheory.run is called
        self.runcalls = 0

        # Whether we are using OpenMM custom external forces or not
        # NOTE: affects runmode
        self.openmm_externalforce = openmm_externalforce

        self.exit_after_customexternalforce_update = exit_after_customexternalforce_update

        # Theory level definitions
        self.qm_theory = qm_theory
        self.qm_theory_name = self.qm_theory.__class__.__name__
        self.mm_theory = mm_theory
        self.mm_theory_name = self.mm_theory.__class__.__name__
        if self.mm_theory_name == "str":
            self.mm_theory_name = "None"

        logger.info("QM-theory: %s", self.qm_theory_name)
        logger.info("MM-theory: %s", self.mm_theory_name)

        self.fragment = fragment
        self.coords = fragment.coords
        self.elems = fragment.elems
        self.connectivity = fragment.connectivity

        self.excludeboundaryatomlist = excludeboundaryatomlist
        self.unusualboundary = unusualboundary

        # Region definitions
        self.allatoms = list(range(len(self.elems)))
        logger.info("All atoms in fragment: %s", len(self.allatoms))
        self.num_allatoms = len(self.allatoms)

        # Sorting qmatoms list making sure only unique values are taken
        self.qmatoms = sorted(set(qmatoms))

        # All-atom Bool-array for whether atom-index is a QM-atom index or not
        # Used by make_QM_PC_gradient
        self.xatom_mask = np.isin(self.allatoms, self.qmatoms)
        self.sum_xatom_mask = np.sum(self.xatom_mask)

        if len(self.qmatoms) == 0:
            raise InputError("Error: List of qmatoms provided is empty. This is not allowed.")
        self.mmatoms = np.setdiff1d(self.allatoms, self.qmatoms)

        logger.info(f"QM region ({len(self.qmatoms)} atoms): {self.qmatoms}")
        logger.info(f"MM region ({len(self.mmatoms)} atoms)")

        # Setting QM/MM qmatoms in QMtheory also (used for Spin-flipping currently)
        self.qm_theory.qmatoms = self.qmatoms

        # Setting numcores of object.
        # This will be when calling QMtheory and probably MMtheory

        # numcores-setting in QMMMTheory takes precedent
        if numcores != 1:
            self.numcores = numcores
        # If QMtheory numcores was set (and QMMMTHeory not)
        elif self.qm_theory.numcores != 1:
            self.numcores = self.qm_theory.numcores
        # Default 1 proc
        else:
            self.numcores = 1
        logger.info(f"QM/MM object selected to use {self.numcores} cores")

        # Embedding type: mechanical, electrostatic etc.
        self.embedding = embedding
        # Charge-boundary method
        self.chargeboundary_method = chargeboundary_method  # Options: 'shift', 'rcd'

        if (
            self.embedding.lower() == "elstat"
            or self.embedding.lower() == "electrostatic"
            or self.embedding.lower() == "electronic"
        ):
            self.embedding = "elstat"
            self.pc = True
        elif (
            self.embedding.lower() == "pbcmm-elstat"
            or self.embedding.lower() == "pbcmm-electrostatic"
            or self.embedding.lower() == "pbcmm-electronic"
        ):
            raise InputError("embedding='pbcmm-elstat' is not supported in this distribution")
        elif self.embedding.lower() == "mechanical" or self.embedding.lower() == "mech":
            self.embedding = "mech"
            self.pc = False
        elif self.embedding.lower() == "polembed_drude" or self.embedding.lower() == "drude":
            self.embedding = "polembed_drude"
            self.pc = True
        else:
            raise InputError(
                "Unknown embedding. Valid options are: elstat (synonyms: electrostatic, electronic), mech (synonym: mechanical)"
            )
        logger.info("Embedding: %s", self.embedding)
        # Whether to do dipole correction or not
        # Note: For regular electrostatic embedding this should be True
        # Turn off for charge-shifting
        self.dipole_correction = dipole_correction

        # Whether MM-shifted performed or not. Will be set to True by self.ShiftMMCharges
        self.chargeshifting_done = False

        # if atomcharges are not passed to QMMMTheory object, get them from MMtheory (that should have defined then)
        if charges is None:
            logger.info("No atomcharges list passed to QMMMTheory object")
            self.charges = []
            if self.mm_theory_name == "OpenMMTheory":
                logger.info("Getting system charges from OpenMM object")
                self.charges = mm_theory.charges
            else:
                raise InputError(
                    "QMMMTheory requires either a charges list or an OpenMMTheory mm_theory providing charges"
                )
        else:
            logger.info("Reading in charges")
            if len(charges) != len(fragment.atomlist):
                raise InputError("Number of charges not matching number of fragment atoms. Exiting.")
            self.charges = charges

            # Update charges in mm_theory if defined
            if self.mm_theory is not None:
                self.mm_theory.update_charges(self.fragment.allatoms, self.charges)

        if len(self.charges) == 0:
            raise InputError("No charges present in QM/MM object. Exiting...")

        # Flag to check whether QMCharges have been zeroed in self.charges_qmregionzeroed list
        self.QMChargesZeroed = False

        # CHARGES DEFINED FOR OBJECT:
        # Self.charges are original charges that are defined above (on input or from OpenMM)
        # self.charges_qmregionzeroed is self.charges but with 0-value for QM-atoms
        # self.pointcharges are pointcharges that the QM-code will see (dipole-charges, no zero-valued charges etc)
        # Length of self.charges: system size
        # Length of self.charges_qmregionzeroed: system size
        # Length of self.pointcharges: unknown. does not contain zero-valued charges (e.g. QM-atoms etc.), contains dipole-charges

        # self.charges_qmregionzeroed will have QM-charges zeroed (but not removed)
        self.charges_qmregionzeroed = []

        # Self.pointcharges are pointcharges that the QM-program will see (but not the MM program)
        # They have QM-atoms zeroed, zero-charges removed, dipole-charges added etc.
        # Defined later
        self.pointcharges = []

        # Truncated PC-region option
        self.truncated_pc = truncated_pc
        self.truncated_pc_radius = truncated_pc_radius
        self.truncated_pc_calls = 0
        self.truncated_pc_recalc_flag = False
        self.truncated_pc_recalc_iter = truncated_pc_recalc_iter

        if self.truncated_pc is True:
            logger.info("Truncated PC approximation in QM/MM is active.")
            logger.info("TruncPCRadius: %s", self.truncated_pc_radius)
            logger.info("TruncPC Recalculation iteration: %s", self.truncated_pc_recalc_iter)

        # If MM THEORY (not just pointcharges)
        if mm_theory is not None:
            # Sanity check. Same number of atoms in fragment and MM object ?
            if fragment.numatoms != mm_theory.numatoms:
                raise InputError(
                    "{}\nThis does not make sense. Check coordinates and forcefield files. Exiting...".format(
                        f"Number of atoms in fragment ({fragment.numatoms}) and MMtheory object differ ({mm_theory.numatoms})"
                    )
                )

            # Update: Tolerance modification to make sure we definitely catch connected atoms and get QM-MM boundary right.
            # Scale=1.0 and tol=0.1 fails for S-C bond in rubredoxin from a classical MD run
            # Bumping up a bit here.
            # 21 Sep 2023. bumping from +0.1 to +0.2. C-C bond in lysine failed
            conn_scale = CONNECTIVITY_SCALE
            conn_tolerance = CONNECTIVITY_TOL + 0.2

            # If a QM-MM boundary issue aborts the run then printing QM-coordinates is useful
            logger.info("QM-region coordinates (before linkatoms):")
            openmmqmmm.coords.print_coords_for_atoms(self.coords, self.elems, self.qmatoms, labels=self.qmatoms)
            logger.info("")
            self.boundaryatoms = openmmqmmm.coords.get_boundary_atoms(
                self.qmatoms,
                self.coords,
                self.elems,
                conn_scale,
                conn_tolerance,
                excludeboundaryatomlist=self.excludeboundaryatomlist,
                unusualboundary=self.unusualboundary,
            )
            if len(self.boundaryatoms) > 0:
                logger.info("Found covalent QM-MM boundary. Linkatoms option set to True")
                logger.info("Boundaryatoms (QM:MM pairs): %s", self.boundaryatoms)
                logger.info(
                    f"Note: used connectivity settings, scale={conn_scale} and tol={conn_tolerance} to determine boundary."
                )
                self.linkatoms = True
                logger.info("Linkatom_forceprojection_method: %s", self.linkatom_forceproj_method)
                # Get MM boundary information. Stored as self.MMboundarydict
                self.get_mm_boundary(conn_scale, conn_tolerance)
            else:
                logger.info("No covalent QM-MM boundary. Linkatoms and dipole_correction options set to False")
                self.linkatoms = False
                self.dipole_correction = False

            if self.mm_theory_name == "OpenMMTheory":
                # Removing possible QM atom constraints in OpenMMTheory
                # Will only apply when running OpenMM_Opt or OpenMM_MD
                self.mm_theory.remove_constraints_for_atoms(self.qmatoms)

                # Remove bonded interactions in MM part. Only in OpenMM. Assuming they were never defined in NonbondedTheory
                # Applies to both elstat and mech embedding.
                logger.info("Removing bonded terms for QM-region in MMtheory")
                self.mm_theory.modify_bonded_forces(self.qmatoms)

                # Adding exceptions for nonbonded QM atoms. Will ignore QM-QM Coulomb and QM-QM LJ interactions.
                # Applies to both elstat and mech embedding.
                # NOTE: For QM-MM elstat interactions Coulomb charges are zeroed below (update_charges and delete_exceptions)
                logger.info("Removing nonbonded terms for QM-region in MMtheory (QM-QM interactions)")
                self.mm_theory.addexceptions(self.qmatoms)

            ########################
            # CHANGE CHARGES
            ########################
            # Keeping self.charges as originally defined.
            # Setting QM charges to 0 since electrostatic embedding
            # and Charge-shift QM-MM boundary

            # Zero QM charges for electrostatic embedding
            if self.embedding.lower() == "elstat":
                logger.info("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
                self.zero_qm_charges()  # Modifies self.charges_qmregionzeroed
                # Updating charges in MM object.
                self.mm_theory.update_charges(self.qmatoms, [0.0 for i in self.qmatoms])
            elif self.embedding.lower() == "polembed_drude":
                raise InputError(
                    "Polembed Drude embedding enabled.\nThis means that QM-atoms will be zeroed for QM-MM interactions calculated by QM program\nBut MM program will have charged defined for QM-region\nNot implemented yet. Exiting"
                )
                self.zero_qm_charges()  # Modifies self.charges_qmregionzeroed
                # Also removing QM-MM Coulomb interaction exceptions in OpenMM
                if self.mm_theory_name == "OpenMMTheory":
                    # Deleting Coulomb exception interactions involving QM and MM atoms
                    self.mm_theory.delete_exceptions(self.qmatoms)
            elif self.embedding.lower() == "pbcmm-elstat":
                logger.info("PBC Electrostatic embedding enabled.")
                logger.info("This means that QM-atoms will be zeroed for QM-MM interactions calculated by QM program")
                logger.info("But MM program will have charged defined for QM-region")
                raise InputError("embedding='pbcmm-elstat' is not supported in this distribution")

                # TODO: Exceptions
                # Note: possible to set QM-charges to something specific: Mulliken, ESP

            # Printing charges: all or only QM
            if logger.isEnabledFor(logging.DEBUG):
                for i in self.allatoms:
                    if i in self.qmatoms:
                        if self.embedding.lower() == "elstat":
                            logger.info(f"QM atom {i} ({self.elems[i]}) charge: {self.charges_qmregionzeroed[i]}")
                        else:
                            logger.info(f"QM atom {i} ({self.elems[i]}) charge: {self.charges[i]}")
                    else:
                        logger.debug(f"MM atom {i} ({self.elems[i]}) charge: {self.charges_qmregionzeroed[i]}")
            logger.info("")
        else:
            # Case: No actual MM theory but we still want to zero charges for QM elstat embedding calculation
            # TODO: Remove option for no MM theory or keep this ??
            if self.embedding.lower() == "elstat":
                self.zero_qm_charges()  # Modifies self.charges_qmregionzeroed
            self.linkatoms = False
            self.dipole_correction = False
        log_time_since(module_init_time, "QM/MM object creation")

    # From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_mm_boundary(self, scale, tol):
        timeA = time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        # Creating dictionary for each MM1 atom and its connected atoms: MM2-4
        self.MMboundarydict = {}
        for QM1atom, MM1atom in self.boundaryatoms.items():
            if isinstance(MM1atom, list):
                for mat in MM1atom:
                    connatoms = openmmqmmm.coords.get_connected_atoms(self.coords, self.elems, scale, tol, mat)
                    # Deleting QM-atom from connatoms list
                    connatoms.remove(QM1atom)
                    self.MMboundarydict[mat] = connatoms
            # OLD: should never apply anymore, we always have a list
            # TODO: delete
            else:
                connatoms = openmmqmmm.coords.get_connected_atoms(self.coords, self.elems, scale, tol, MM1atom)
                # Deleting QM-atom from connatoms list
                connatoms.remove(QM1atom)
                self.MMboundarydict[MM1atom] = connatoms

        # Used by ShiftMMCharges
        self.MMboundary_indices = list(self.MMboundarydict.keys())
        self.MMboundary_counts = np.array([len(self.MMboundarydict[i]) for i in self.MMboundary_indices])

        logger.info("")
        logger.info("MM boundary (MM1:MMx pairs): %s", self.MMboundarydict)
        log_time_since(timeA, "get_MMboundary")

    # Set QMcharges to Zero and shift charges at boundary
    # TODO: Add both L2 scheme (delete whole charge-group of M1) and charge-shifting scheme (shift charges to Mx atoms and add dipoles for each Mx atom)

    def zero_qm_charges(self):
        timeA = time.time()
        logger.info("Setting QM charges to Zero")
        # Looping over charges and setting QM atoms to zero
        # 1. Copy charges to charges_qmregionzeroed
        self.charges_qmregionzeroed = copy.copy(self.charges)
        # 2. change charge for QM-atom
        for i, _c in enumerate(self.charges_qmregionzeroed):
            # Setting QMatom charge to 0
            if i in self.qmatoms:
                self.charges_qmregionzeroed[i] = 0.0
        # 3. Flag that this has been done
        self.QMChargesZeroed = True
        log_time_since(timeA, "ZeroQMCharges")

    def rcd_shifting_prep(self, charges_qmregionzeroed):
        timeA = time.time()
        logger.info("Shifting MM charges at QM/MM boundary by RCD.")
        # Convert lists to NumPy arrays for faster computations
        pointcharges = np.array(charges_qmregionzeroed)
        self.charges = np.array(self.charges)
        # Extract charges for MM boundary atoms
        MM1_charges = self.charges[self.MMboundary_indices]
        # Set charges of MM boundary atoms to 0
        pointcharges[self.MMboundary_indices] = 0.0
        # Calculate charge fractions to distribute
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        # Only keep pointcharges for PC region
        pointcharges = [pointcharges[x] for x in self.mmatoms]

        # Distribute charge fractions to neighboring MM atoms
        RCD_additional_charges = []
        for _MM1index, MM2indices, fract in zip(
            self.MMboundarydict.keys(), self.MMboundarydict.values(), MM1charge_fract, strict=False
        ):
            newfract = fract * 2  # q0*2
            # Looping over MM2 atoms
            for i in MM2indices:
                # RC/RCD: Instead of adding the M1 charge to the M2 atoms we create new RC/RCD sites
                pointcharges = np.append(pointcharges, newfract)
                RCD_additional_charges.append(newfract)
                # RCD: Reduce the MM2 charge by q0
                pointcharges[i] -= fract
        self.chargeshifting_done = True

        log_time_since(timeA, "RCD_shifting_prep")
        return pointcharges, RCD_additional_charges

    def rcd_shifting_update(self, used_mmcoords, fullcoords):
        timeA = time.time()
        logger.info("Adding updated RCD charges at QM/MM boundary by RCD.")

        # Distribute charge fractions to neighboring MM atoms
        for MM1index, MM2indices in zip(self.MMboundarydict.keys(), self.MMboundarydict.values(), strict=False):
            # Looping over MM2 atoms
            for i in MM2indices:
                # Add new RCD sites to pointchargecoords and pointcharges
                newsite = (fullcoords[i] + fullcoords[MM1index]) / 2
                pointchargecoords = np.append(used_mmcoords, [newsite], axis=0)

        log_time_since(timeA, "RCD_shifting_update")
        return pointchargecoords

    def shift_mm_charges(self):
        if self.chargeshifting_done is False:
            self._shift_mm_charges_impl()
        else:
            logger.info("Charge shifting already done. Using previous charges")

    def _shift_mm_charges_impl(self):
        timeA = time.time()
        logger.info("new. Shifting MM charges at QM-MM boundary.")

        # Convert lists to NumPy arrays for faster computations
        log_time_since(timeA, "x0")
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        self.charges = np.array(self.charges)

        log_time_since(timeA, "x1")
        # Extract charges for MM boundary atoms
        MM1_charges = self.charges[self.MMboundary_indices]
        # Set charges of MM boundary atoms to 0
        self.pointcharges[self.MMboundary_indices] = 0.0

        # Calculate charge fractions to distribute
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        # Distribute charge fractions to neighboring MM atoms
        for indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract, strict=False):
            self.pointcharges[[indices]] += fract

        self.chargeshifting_done = True
        log_time_since(timeA, "ShiftMMCharges-new2")
        return

    # Create dipole charge (twice) for each MM2 atom that gets fraction of MM1 charge
    def get_dipole_charge(self, delq, direction, mm1index, mm2index, current_coords):
        # oldMM_distance = openmmqmmm.coords.distance_between_atoms(fragment=self.fragment,
        #                                                               atoms=[mm1index, mm2index])
        # Coordinates and distance
        mm1coords = np.array(current_coords[mm1index])
        mm2coords = np.array(current_coords[mm2index])
        MM_distance = openmmqmmm.coords.distance(mm1coords, mm2coords)  # Distance between MM1 and MM2

        SHIFT = 0.15

        # Normalize vector
        def vnorm(p1):
            r = math.sqrt((p1[0] * p1[0]) + (p1[1] * p1[1]) + (p1[2] * p1[2]))
            v1 = np.array([p1[0] / r, p1[1] / r, p1[2] / r])
            return v1

        diffvector = mm2coords - mm1coords
        normdiffvector = vnorm(diffvector)

        # Dipole
        d = delq * 2.5
        # Charge (abs value)
        q0 = 0.5 * d / SHIFT
        # Actual shift
        shift = direction * SHIFT * (MM_distance / 2.5)
        # Position
        pos = mm2coords + np.array(shift * normdiffvector)
        # Returning charge with sign based on direction and position
        # Return coords as regular list
        return -q0 * direction, list(pos)

    def set_dipole_charges(self, current_coords):
        checkpoint = time.time()
        logger.info("Adding extra charges to preserve dipole moment for charge-shifting")
        logger.info("MMboundarydict: %s", self.MMboundarydict)
        # Adding 2 dipole pointcharges for each MM2 atom
        self.dipole_charges = []
        self.dipole_coords = []

        for MM1, MMx in self.MMboundarydict.items():
            # Getting original MM1 charge (before set to 0)
            MM1charge = self.charges[MM1]
            MM1charge_fract = MM1charge / len(MMx)

            for MM in MMx:
                q_d1, pos_d1 = self.get_dipole_charge(MM1charge_fract, 1, MM1, MM, current_coords)
                q_d2, pos_d2 = self.get_dipole_charge(MM1charge_fract, -1, MM1, MM, current_coords)
                self.dipole_charges.append(q_d1)
                self.dipole_charges.append(q_d2)
                self.dipole_coords.append(pos_d1)
                self.dipole_coords.append(pos_d2)
        log_time_since(checkpoint, "SetDipoleCharges")

    # Reasonably efficient version (this dominates QM/MM gradient prepare)
    # def make_QM_PC_gradient_old(self):

    # Faster version. Also, uses precalculated mask.
    def make_qm_pc_gradient(self):
        self.QM_PC_gradient[self.xatom_mask] = self.QMgradient_wo_linkatoms
        self.QM_PC_gradient[~self.xatom_mask] = self.PCgradient[: self.num_allatoms - self.sum_xatom_mask]
        return

    # TruncatedPCfunction control flow for pointcharge field passed to QM program
    def truncated_pc_function(self, used_qmcoords):
        self.truncated_pc_calls += 1
        logger.info("TruncatedPC approximation!")
        if self.truncated_pc_calls == 1 or self.truncated_pc_calls % self.truncated_pc_recalc_iter == 0:
            self.truncated_pc_recalc_flag = True
            logger.info(
                f"This is QM/MM run no. {self.truncated_pc_calls}.  Will calculate Full-Trunc correction in this step"
            )
            # Origin coords point is center of QM-region
            origincoords = openmmqmmm.coords.get_centroid(used_qmcoords)
            # Determine the indices associated with the truncated PC field once
            self.determine_truncated_pc_indices(origincoords)
            logger.info(f"Truncated PC-region size: {len(self.truncated_PC_region_indices)} charges")
            # Saving full PCs and coords for 1st iteration
            # NOTE: Here using self.pointcharges_original (set by runprep)
            # since self.pointcharges may be truncated-version from last iter
            self.pointcharges_full = copy.copy(self.pointcharges_original)
            self.pointchargecoords_full = copy.copy(self.pointchargecoords)

            # Determining truncated PC-field
            self.pointcharges = [self.pointcharges_full[i] for i in self.truncated_PC_region_indices]
            self.pointchargecoords = np.take(self.pointchargecoords_full, self.truncated_PC_region_indices, axis=0)
        else:
            self.truncated_pc_recalc_flag = False
            logger.info(
                f"This is QM/MM run no. {self.truncated_pc_calls}. Using approximate truncated PC field: {len(self.truncated_PC_region_indices)} charges"
            )
            # NOTE: Here taking 1st-iter full PCs (values have not changed during opt/md)
            self.pointcharges = [self.pointcharges_full[i] for i in self.truncated_PC_region_indices]
            # NOTE: Here taking from CURRENT full pointchargecoords (not old full from step 1) since coords have changed
            self.pointchargecoords = np.take(self.pointchargecoords, self.truncated_PC_region_indices, axis=0)

    # Determine truncated PC field indices based on initial coordinates
    # Coordinates and charges for each Opt cycle defined later.
    def determine_truncated_pc_indices(self, origincoords):
        region_indices = []
        for index, allc in enumerate(self.pointchargecoords):
            dist = openmmqmmm.coords.distance(origincoords, allc)
            if dist < self.truncated_pc_radius:
                region_indices.append(index)
        # Only unique and sorting:
        self.truncated_PC_region_indices = np.unique(region_indices).tolist()
        # Removing dipole charges also (end of list)

    # New more efficient version
    def calculate_trunc_pc_gradient_correction(
        self, QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc
    ):
        # QM part
        qm_difference = (
            QMgradient_full[: len(QMgradient_full) - self.num_linkatoms]
            - QMgradient_trunc[: len(QMgradient_full) - self.num_linkatoms]
        )
        self.original_QMcorrection_gradient = qm_difference
        # PC part
        truncated_indices = np.array(self.truncated_PC_region_indices)
        pc_difference = np.zeros((len(PCgradient_full), 3))
        pc_difference[truncated_indices] = PCgradient_full[truncated_indices] - PCgradient_trunc
        pc_difference[~np.isin(np.arange(len(PCgradient_full)), truncated_indices)] = PCgradient_full[
            ~np.isin(np.arange(len(PCgradient_full)), truncated_indices)
        ]
        self.original_PCcorrection_gradient = pc_difference
        return

    def truncated_pc_gradient_update(self, QMgradient_wo_linkatoms, PCgradient):
        newQMgradient_wo_linkatoms = QMgradient_wo_linkatoms + self.original_QMcorrection_gradient

        new_full_PC_gradient = np.copy(self.original_PCcorrection_gradient)
        new_full_PC_gradient[self.truncated_PC_region_indices] += PCgradient

        return newQMgradient_wo_linkatoms, new_full_PC_gradient

    def set_numcores(self, numcores):
        logger.info(f"Setting new numcores {numcores}for QMtheory and MMtheory")
        self.qm_theory.set_numcores(numcores)
        self.mm_theory.set_numcores(numcores)

    # Method to grab dipole moment from outputfile (assumes run has been executed)
    def get_dipole_moment(self):
        logger.info("Grabbing dipole moment from QM-part of QM/MM theory.")
        dipole = None
        try:
            dipole = self.qm_theory.get_dipole_moment()
        except AttributeError:
            logger.info("Error: Could not grab dipole moment from QM-part of QM/MM theory.")
        return dipole

    # Method to polarizability from outputfile (assumes run has been executed)
    def get_polarizability_tensor(self):
        logger.info("Grabbing polarizability from QM-part of QM/MM theory.")
        polarizability = None
        try:
            polarizability = self.qm_theory.get_polarizability_tensor()
        except AttributeError:
            logger.info("Error: Could not grab polarizability from QM-part of QM/MM theory.")
        return polarizability

    # General run
    def run(
        self,
        current_coords=None,
        elems=None,
        grad=False,
        numcores=1,
        exit_after_customexternalforce_update=False,
        label=None,
        charge=None,
        mult=None,
        current_mm_coords=None,
        mm_charges=None,
        qm_elems=None,
        pc=None,
    ):

        logger.warning("------------RUNNING QM/MM MODULE-------------")
        logger.info("QM Module: %s", self.qm_theory_name)
        logger.info("MM Module: %s", self.mm_theory_name)

        # exit_after_customexternalforce_update can be enabled both at runtime and by initialization
        if self.exit_after_customexternalforce_update is True:
            exit_after_customexternalforce_update = self.exit_after_customexternalforce_update

        # OPTION: QM-region charge/mult from QMMMTheory definition
        # If qm_charge/qm_mult defined then we use. Otherwise charge/mult may have been defined by jobtype-function and passed on via run
        if self.qm_charge is not None:
            logger.info("Charge provided from QMMMTheory object:  %s", self.qm_charge)
            charge = self.qm_charge
        if self.qm_mult is not None:
            logger.info("Mult provided from QMMMTheory object:  %s", self.qm_mult)
            mult = self.qm_mult

        # Checking if charge and mult has been provided. Exit if not.
        if charge is None or mult is None:
            raise InputError("Error. charge and mult has not been defined for QMMMTheory.run method")

        logger.info(f"QM-region Charge: {charge} Mult: {mult}")

        if self.embedding.lower() == "mech":
            return self.mech_run(
                current_coords=current_coords,
                elems=elems,
                grad=grad,
                numcores=numcores,
                exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label,
                charge=charge,
                mult=mult,
            )
        elif self.embedding.lower() == "elstat":
            return self.elstat_run(
                current_coords=current_coords,
                elems=elems,
                grad=grad,
                numcores=numcores,
                exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label,
                charge=charge,
                mult=mult,
            )
        elif self.embedding.lower() == "pbcmm-elstat":
            # Things should be the same except QM-charges have not been zeroed in MM-program
            # MM-program thus double-counts (SR QM-QM and SR QM-MM) and we need subtractive corrections
            return self.elstat_run(
                current_coords=current_coords,
                elems=elems,
                grad=grad,
                numcores=numcores,
                exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label,
                charge=charge,
                mult=mult,
            )
        elif self.embedding.lower() == "polembed_drude":
            return self.elstat_run(
                current_coords=current_coords,
                elems=elems,
                grad=grad,
                numcores=numcores,
                exit_after_customexternalforce_update=exit_after_customexternalforce_update,
                label=label,
                charge=charge,
                mult=mult,
            )
        else:
            raise InputError("Unknown embedding. Exiting")

    # Mechanical embedding run
    def mech_run(
        self,
        current_coords=None,
        elems=None,
        grad=False,
        numcores=1,
        exit_after_customexternalforce_update=False,
        label=None,
        charge=None,
        mult=None,
    ):
        module_init_time = time.time()
        CheckpointTime = time.time()
        logger.info("Embedding: Mechanical")

        #############################################
        # If this is first run then do QM/MM runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            logger.info("First QMMMTheory run. Running runprep")
            self.runprep(current_coords)
            # This creates self.current_qmelems,
            # self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords

        # Updating runcalls
        self.runcalls += 1

        #########################################################################################
        # General QM-code energy+gradient call.
        #########################################################################################

        # Split current_coords into MM-part and QM-part efficiently.
        _used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            # Add linkatom coordinates to QM-coordinates
            used_qmcoords = np.append(used_qmcoords, np.array(linkatoms_coords), axis=0)

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        logger.info(f"Running QM/MM object with {numcores} cores available")

        ################
        # QMTheory.run
        ################
        log_time_since(module_init_time, "before-QMstep")
        CheckpointTime = time.time()
        if self.qm_theory_name == "None" or self.qm_theory_name == "ZeroTheory":
            logger.info("No QMtheory. Skipping QM calc")
            QMenergy = 0.0
            self.linkatoms = False
            QMgradient = np.zeros((len(used_qmcoords), 3))
        else:
            # Calling QM theory, providing current QM and MM coordinates.
            if grad is True:
                QMenergy, QMgradient = self.qm_theory.run(
                    current_coords=used_qmcoords,
                    qm_elems=self.current_qmelems,
                    grad=True,
                    pc=False,
                    numcores=numcores,
                    charge=charge,
                    mult=mult,
                )
            else:
                QMenergy = self.qm_theory.run(
                    current_coords=used_qmcoords,
                    qm_elems=self.current_qmelems,
                    grad=False,
                    pc=False,
                    numcores=numcores,
                    charge=charge,
                    mult=mult,
                )

        log_time_since(CheckpointTime, "QM step")
        CheckpointTime = time.time()

        ############################
        # Update QM-region charges
        ############################

        if self.update_qm_region_charges:
            logger.info("update_QMregion_charges is True")
            logger.info("Will try to find charges attribute in QM-object")
            try:
                newqmcharges = self.qm_theory.charges
            except AttributeError:
                raise InputError(
                    "Found no charges attribute on the QM-theory object - update_QMregion_charges can not be used"
                ) from None
            # Removing linkatoms
            if self.num_linkatoms > 0:
                newqmcharges = newqmcharges[0 : -self.num_linkatoms]
            for i, index in enumerate(self.qmatoms):
                self.charges[index] = newqmcharges[i]
            logger.info("Updating charges of QM-region in MMTheory object")
            self.mm_theory.update_charges(self.qmatoms, list(newqmcharges))
        logger.info("Defined charges of QM-region:")
        for i in self.qmatoms:
            logger.info(f"QM atom {i} has charge : {self.charges[i]}")

        ##################################################################################
        # QM/MM gradient: Initializing and then adding QM gradient, linkatom gradient
        ##################################################################################

        self.QMenergy = QMenergy

        # Initializing QM/MM gradient
        self.QM_MM_gradient = np.zeros((len(current_coords), 3))
        if grad:
            Grad_prep_CheckpointTime = time.time()
            # Defining QMgradient without linkatoms if present
            if self.linkatoms is True:
                self.QMgradient = QMgradient
                self.QMgradient_wo_linkatoms = QMgradient[0 : -self.num_linkatoms]  # remove linkatoms
            else:
                self.QMgradient = QMgradient
                self.QMgradient_wo_linkatoms = QMgradient

            # Adding QM gradient (without linkatoms) to QM_MM_gradient
            self.QM_MM_gradient[self.qmatoms] += self.QMgradient_wo_linkatoms

            # LINKATOM FORCE PROJECTION
            # Add contribution to QM1 and MM1 contribution???
            if self.linkatoms is True:
                CheckpointTime = time.time()

                for pair in sorted(self.linkatoms_dict.keys()):
                    # Grabbing linkatom data
                    linkatomindex = self.linkatom_indices.pop(0)
                    Lgrad = self.QMgradient[linkatomindex]
                    Lcoord = self.linkatoms_dict[pair]
                    # Grabbing QMatom info
                    fullatomindex_qm = pair[0]
                    qmatomindex = fullindex_to_qmindex(fullatomindex_qm, self.qmatoms)
                    Qcoord = used_qmcoords[qmatomindex]
                    # Grabbing MMatom info
                    fullatomindex_mm = pair[1]
                    Mcoord = current_coords[fullatomindex_mm]
                    # Getting gradient contribution to QM1 and MM1 atoms from linkatom
                    if self.linkatom_forceproj_method == "adv":
                        QM1grad_contrib, MM1grad_contrib = linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method == "lever":
                        QM1grad_contrib, MM1grad_contrib = linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method == "chain":
                        QM1grad_contrib, MM1grad_contrib = linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method.lower() == "none" or self.linkatom_forceproj_method is None:
                        QM1grad_contrib = np.zeros(3)
                        MM1grad_contrib = np.zeros(3)
                    else:
                        raise InputError("Unknown linkatom_forceproj_method. Exiting")
                    # Updating full QM_MM_gradient
                    self.QM_MM_gradient[fullatomindex_qm] += QM1grad_contrib
                    self.QM_MM_gradient[fullatomindex_mm] += MM1grad_contrib

            # Defining QM_PC_gradient for simplicity (used by OpenMM_MD)
            self.QM_PC_gradient = self.QM_MM_gradient

            log_time_since(CheckpointTime, "linkatomgrad prepare")
            log_time_since(Grad_prep_CheckpointTime, "QM/MM gradient prepare")
            CheckpointTime = time.time()
        else:
            # No Grad
            self.QMenergy = QMenergy

        ################
        # MM THEORY
        ################
        if self.mm_theory_name == "OpenMMTheory":
            logger.info("Using OpenMM theory as part of QM/MM.")
            if grad:
                CheckpointTime = time.time()
                # Provide self.QM_MM_gradient to OpenMMTheory
                if self.openmm_externalforce is True:
                    logger.info("OpenMM externalforce is True")
                    # Calculate energy associated with external force so that we can subtract it later
                    scaled_current_coords = current_coords * 1.88972612546
                    self.extforce_energy = 3 * np.mean(np.sum(self.QM_MM_gradient * scaled_current_coords, axis=0))
                    logger.info(f"Extforce energy: {self.extforce_energy}")
                    log_time_since(CheckpointTime, "extforce prepare")
                    # NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update is True:
                        logger.info("OpenMM custom external force updated. Exit requested")
                        # This is used if OpenMM MD is handling forces and dynamics
                        return self.QMenergy, self.QM_MM_gradient

                self.MMenergy, self.MMgradient = self.mm_theory.run(
                    current_coords=current_coords, qmatoms=self.qmatoms, grad=True
                )
            else:
                logger.info("QM/MM Grad is false")
                self.MMenergy = self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMenergy = 0
        log_time_since(CheckpointTime, "MM step")
        CheckpointTime = time.time()

        if grad:
            # Now assemble full QM/MM gradient by adding MM gradient
            if len(self.QM_MM_gradient) != len(self.MMgradient):
                raise InternalError("QM/MM gradient and MM gradient size mismatch")
            self.QM_MM_gradient = self.QM_MM_gradient + self.MMgradient

        # Final QM/MM Energy
        self.QM_MM_energy = self.QMenergy + self.MMenergy - self.subtractive_correction_E

        # Final QM/MM Gradient
        # Possible subtractive correction
        self.QM_MM_gradient -= self.subtractive_correction_G

        logger.info("")
        logger.info("%s", "{:<20} {:>20.12f}".format("QM energy: ", self.QMenergy))
        logger.info("%s", "{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
        logger.info("%s", "{:<20} {:>20.12f}".format("Subtractive correction energy: ", self.subtractive_correction_E))
        logger.info("%s", "{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_energy))
        logger.info("")

        # FINAL QM/MM GRADIENT ASSEMBLY and return
        if grad is True:
            if logger.isEnabledFor(logging.DEBUG):
                # Writing QM gradient only
                openmmqmmm.coords.write_coords_all(
                    self.QMgradient_wo_linkatoms,
                    self.qmelems,
                    indices=self.qmatoms,
                    file=f"QMgradient-without-linkatoms_{label}",
                    description=f"QM gradient w/o linkatoms {label} (au/Bohr):",
                )
                # Writing QM+Linkatoms gradient
                openmmqmmm.coords.write_coords_all(
                    self.MMgradient,
                    self.elems,
                    indices=self.allatoms,
                    file=f"MMgradient_{label}",
                    description=f"MM gradient {label} (au/Bohr):",
                )
                # Writing full QM/MM gradient
                openmmqmmm.coords.write_coords_all(
                    self.QM_MM_gradient,
                    self.elems,
                    indices=self.allatoms,
                    file=f"QM_MMgradient_{label}",
                    description=f"QM/MM gradient {label} (au/Bohr):",
                )
            logger.warning("------------ENDING QM/MM MODULE-------------")
            log_time_since(module_init_time, "QM/MM mech run")
            return self.QM_MM_energy, self.QM_MM_gradient
        else:
            log_time_since(module_init_time, "QM/MM mech run")
            return self.QM_MM_energy

    def create_linkatoms(self, current_coords):
        checkpoint = time.time()
        # Get linkatom coordinates
        self.linkatoms_dict = openmmqmmm.coords.get_linkatom_positions(
            self.boundaryatoms,
            self.qmatoms,
            current_coords,
            self.elems,
            linkatom_method=self.linkatom_method,
            linkatom_type=self.linkatom_type,
            linkatom_simple_distance=self.linkatom_simple_distance,
            linkatom_ratio=self.linkatom_ratio,
        )
        logger.debug("linkatoms_dict: %s", self.linkatoms_dict)
        logger.info("Adding linkatom positions to QM coords")
        self.linkatom_indices = [len(self.qmatoms) + i for i in range(len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        log_time_since(checkpoint, "create_linkatoms")
        return linkatoms_coords

    # Run-preparation (for both electrostatic and mechanical)
    # Things that only have to be done in the first QM/MM run
    def runprep(self, current_coords):
        logger.info("Inside QMMMTheory runprep")
        init_time_runprep = time.time()
        time.time()

        # Set basic element lists
        self.qmelems = [self.elems[i] for i in self.qmatoms]
        self.mmelems = [self.elems[i] for i in self.mmatoms]

        # LINKATOMS (both mech and elstat)
        check_before_linkatoms = time.time()
        if self.linkatoms is True:
            self.create_linkatoms(current_coords)
            self.current_qmelems = self.qmelems + [self.linkatom_type] * self.num_linkatoms
            logger.info("Number of MM atoms: %s", len(self.mmatoms))
            logger.info(f"There are {self.num_linkatoms} linkatoms")
            # Do possible Charge-shifting. MM1 charge distributed to MM2 atoms
            if self.embedding.lower() == "elstat":
                logger.info("Doing charge-shifting...")

                # CHARGEBOUNDARY METHOD
                if self.chargeboundary_method == "shift":
                    logger.info("Chargeboundary method is:  shift  ")
                    self.shift_mm_charges()  # Creates self.pointcharges
                    # Defining pointcharges as only containing MM atoms
                    self.pointcharges = [self.pointcharges[i] for i in self.mmatoms]

                    if self.dipole_correction is True:
                        logger.info("Dipole correction is on. Adding dipole charges")
                        self.set_dipole_charges(current_coords)  # Creates self.dipole_charges and self.dipole_coords

                        # Adding dipole charge coords to MM coords (given to QM code) and defining pointchargecoords
                        logger.info(f"Adding {len(self.dipole_charges)} dipole charges to PC environment")

                        # Adding dipole charges to MM charges list (given to QM code)
                        self.pointcharges = list(self.pointcharges) + list(self.dipole_charges)
                        logger.info("Number of pointcharges after dipole addition:  %s", len(self.pointcharges))
                        log_time_since(check_before_linkatoms, "Linkatom-dipolecorrection")
                    else:
                        logger.info("Dipole correction is off. Not adding any dipole charges")
                        logger.info("Number of pointcharges:  %s", len(self.pointcharges))
                # RCD
                elif self.chargeboundary_method == "rcd":
                    logger.info("Chargeboundary method is:  rcd  ")
                    self.pointcharges, _RCD_additional_charges = self.rcd_shifting_prep(self.charges_qmregionzeroed)
                else:
                    raise InputError("Unknown chargeboundary_method. Exiting")

                logger.info("Number of pointcharges defined for whole system:  %s", len(self.pointcharges))
                logger.info("Number of pointcharges defined for MM region:  %s", len(self.pointcharges))

        # CASE: No Linkatoms
        else:
            self.num_linkatoms = 0
            # If no linkatoms then use original self.qmelems
            self.current_qmelems = self.qmelems
            # If no linkatoms then self.pointcharges are just original charges with QM-region zeroed
            if self.embedding.lower() == "elstat" or self.embedding.lower() == "polembed_drude":
                self.pointcharges = [self.charges_qmregionzeroed[i] for i in self.mmatoms]

        # NOTE: Now we have updated MM-coordinates (if doing linkatoms, with dipolecharges etc) and updated mm-charges (more, due to dipolecharges if linkatoms)
        # We also have MMcharges that have been set to zero due to QM/MM
        # We do not delete charges but set to zero
        # If no qmatoms then do MM-only
        if len(self.qmatoms) == 0:
            logger.info("No qmatoms list provided. Setting QMtheory to None")
            self.qm_theory_name = "None"
            self.QMenergy = 0.0

        # For truncatedPC option.
        self.pointcharges_original = copy.copy(self.pointcharges)

        # Initialize QM_PC_gradient for efficiency
        if self.embedding.lower() == "elstat" or self.embedding.lower() == "polembed_drude":
            self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))

        log_time_since(init_time_runprep, "runprep")

    # Electrostatic embedding run
    def elstat_run(
        self,
        current_coords=None,
        elems=None,
        grad=False,
        numcores=1,
        exit_after_customexternalforce_update=False,
        label=None,
        charge=None,
        mult=None,
    ):
        module_init_time = time.time()
        CheckpointTime = time.time()

        logger.info("Embedding: Electrostatic")

        #############################################
        # If this is first run then do QM/MM runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            logger.info("First QMMMTheory run. Running runprep")
            self.runprep(current_coords)
            # This creates self.pointcharges and self.current_qmelems
            # self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords

        # Updating runcalls
        self.runcalls += 1

        #########################################################################################
        # General QM-code energy+gradient call.
        #########################################################################################

        # Split current_coords into MM-part and QM-part efficiently.
        used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            # Add linkatom coordinates to QM-coordinates
            used_qmcoords = np.append(used_qmcoords, np.array(linkatoms_coords), axis=0)

        # Update self.pointchargecoords based on new current_coords
        if self.chargeboundary_method == "shift" and self.dipole_correction is True:
            self.set_dipole_charges(current_coords)  # Note: running again
            self.pointchargecoords = np.append(used_mmcoords, np.array(self.dipole_coords), axis=0)
        elif self.chargeboundary_method == "rcd":
            # Appends RCD chargepositions to MM-coords
            self.pointchargecoords = self.rcd_shifting_update(used_mmcoords, current_coords)
        else:
            self.pointchargecoords = used_mmcoords

        # TRUNCATED PC Option: Speeding up QM/MM jobs of large systems by passing only a truncated PC field to the QM-code most of the time
        # Speeds up QM-pointcharge gradient that otherwise dominates
        # TODO: TruncatedPC is inactive
        if self.truncated_pc is True:
            self.truncated_pc_function(used_qmcoords)

            # Modifies self.pointcharges and self.pointchargecoords

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        logger.info("Number of pointcharges (to QM program): %s", len(self.pointcharges))
        logger.info("Number of charge coordinates: %s", len(self.pointchargecoords))
        logger.info(f"Running QM/MM object with {numcores} cores available")
        ################
        # QMTheory.run
        ################
        log_time_since(module_init_time, "before-QMstep")
        CheckpointTime = time.time()
        if self.qm_theory_name == "None" or self.qm_theory_name == "ZeroTheory":
            logger.info("No QMtheory. Skipping QM calc")
            QMenergy = 0.0
            self.linkatoms = False
            PCgradient = np.array([0.0, 0.0, 0.0])
            QMgradient = np.array([0.0, 0.0, 0.0])
        else:
            # TODO: Add check whether QM-code supports both pointcharges and pointcharge-gradient?

            # Calling QM theory, providing current QM and MM coordinates.
            if grad is True:
                if self.pc is True:
                    QMenergy, QMgradient, PCgradient = self.qm_theory.run(
                        current_coords=used_qmcoords,
                        current_mm_coords=self.pointchargecoords,
                        mm_charges=self.pointcharges,
                        qm_elems=self.current_qmelems,
                        charge=charge,
                        mult=mult,
                        grad=True,
                        pc=True,
                        numcores=numcores,
                    )
                else:
                    QMenergy, QMgradient = self.qm_theory.run(
                        current_coords=used_qmcoords,
                        current_mm_coords=self.pointchargecoords,
                        mm_charges=self.pointcharges,
                        qm_elems=self.current_qmelems,
                        grad=True,
                        pc=False,
                        numcores=numcores,
                        charge=charge,
                        mult=mult,
                    )
            else:
                QMenergy = self.qm_theory.run(
                    current_coords=used_qmcoords,
                    current_mm_coords=self.pointchargecoords,
                    mm_charges=self.pointcharges,
                    qm_elems=self.current_qmelems,
                    grad=False,
                    pc=self.pc,
                    numcores=numcores,
                    charge=charge,
                    mult=mult,
                )

        log_time_since(CheckpointTime, "QM step")
        CheckpointTime = time.time()

        # Final QM/MM gradient. Combine QM gradient, MM gradient, PC-gradient (elstat MM gradient from QM code).
        # Do linkatom force projections in the end
        # UPDATE: Do MM step in the end now so that we have options for OpenMM extern force
        if grad is True:
            Grad_prep_CheckpointTime = time.time()
            # Defining QMgradient without linkatoms if present
            if self.linkatoms:
                self.QMgradient = QMgradient
                QMgradient_wo_linkatoms = QMgradient[0 : -self.num_linkatoms]  # remove linkatoms
            else:
                self.QMgradient = QMgradient
                QMgradient_wo_linkatoms = QMgradient

            # TRUNCATED PC Option:
            if self.truncated_pc is True:
                # DONE ONCE: CALCULATE FULL PC GRADIENT TO DETERMINE CORRECTION
                if self.truncated_pc_recalc_flag is True:
                    CheckpointTime = time.time()
                    truncfullCheckpointTime = time.time()

                    # We have calculated truncated QM and PC gradient
                    QMgradient_trunc = QMgradient
                    PCgradient_trunc = PCgradient

                    logger.info("Now calculating full QM and PC gradient")
                    logger.info("Number of PCs provided to QM-program: %s", len(self.pointcharges_full))
                    QMenergy_full, QMgradient_full, PCgradient_full = self.qm_theory.run(
                        current_coords=used_qmcoords,
                        current_mm_coords=self.pointchargecoords_full,
                        mm_charges=self.pointcharges_full,
                        qm_elems=self.current_qmelems,
                        charge=charge,
                        mult=mult,
                        grad=True,
                        pc=True,
                        numcores=numcores,
                    )
                    log_time_since(CheckpointTime, "trunc-pc full calculation")
                    CheckpointTime = time.time()

                    # TruncPC correction to QM energy
                    self.truncPC_E_correction = QMenergy_full - QMenergy
                    logger.info(f"Truncated PC energy correction: {self.truncPC_E_correction} Eh")
                    self.QMenergy = QMenergy + self.truncPC_E_correction
                    # Now determine the correction once and for all
                    CheckpointTime = time.time()
                    self.calculate_trunc_pc_gradient_correction(
                        QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc
                    )
                    log_time_since(CheckpointTime, "calculate_truncPC_gradient_correction")
                    CheckpointTime = time.time()

                    # Now defining final QMgradient and PCgradient
                    self.QMgradient_wo_linkatoms, self.PCgradient = self.truncated_pc_gradient_update(
                        QMgradient_wo_linkatoms, PCgradient
                    )
                    log_time_since(CheckpointTime, "truncPC_gradient update ")
                    log_time_since(truncfullCheckpointTime, "trunc-full-step pcgrad update")

                else:
                    CheckpointTime = time.time()
                    # TruncPC correction to QM energy
                    self.QMenergy = QMenergy + self.truncPC_E_correction
                    self.QMgradient_wo_linkatoms, self.PCgradient = self.truncated_pc_gradient_update(
                        QMgradient_wo_linkatoms, PCgradient
                    )
                    log_time_since(CheckpointTime, "trunc pcgrad update")
            else:
                self.QMenergy = QMenergy
                # No TruncPC approximation active. No change to original QM and PCgradient from QMcode
                self.QMgradient_wo_linkatoms = QMgradient_wo_linkatoms
                if self.embedding.lower() == "elstat" or self.embedding.lower() == "polembed_drude":
                    self.PCgradient = PCgradient

            # Populatee QM_PC gradient (has full system size)
            CheckpointTime = time.time()
            self.make_qm_pc_gradient()  # populates self.QM_PC_gradient
            log_time_since(CheckpointTime, "QMpcgrad prepare")
            # LINKATOM FORCE PROJECTION
            if self.linkatoms is True:
                CheckpointTime = time.time()
                for pair in sorted(self.linkatoms_dict.keys()):
                    # Grabbing linkatom data
                    linkatomindex = self.linkatom_indices.pop(0)
                    Lgrad = self.QMgradient[linkatomindex]
                    Lcoord = self.linkatoms_dict[pair]
                    # Grabbing QMatom info
                    fullatomindex_qm = pair[0]
                    qmatomindex = fullindex_to_qmindex(fullatomindex_qm, self.qmatoms)
                    Qcoord = used_qmcoords[qmatomindex]
                    # Grabbing MMatom info
                    fullatomindex_mm = pair[1]
                    Mcoord = current_coords[fullatomindex_mm]

                    # Getting gradient contribution to QM1 and MM1 atoms from linkatom
                    if self.linkatom_forceproj_method == "adv":
                        QM1grad_contrib, MM1grad_contrib = linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method == "lever":
                        QM1grad_contrib, MM1grad_contrib = linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method == "chain":
                        QM1grad_contrib, MM1grad_contrib = linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad)
                    elif self.linkatom_forceproj_method.lower() == "none" or self.linkatom_forceproj_method is None:
                        QM1grad_contrib = np.zeros(3)
                        MM1grad_contrib = np.zeros(3)
                    else:
                        raise InputError("Unknown linkatom_forceproj_method. Exiting")

                    self.QM_PC_gradient[fullatomindex_qm] += QM1grad_contrib
                    self.QM_PC_gradient[fullatomindex_mm] += MM1grad_contrib

            log_time_since(CheckpointTime, "linkatomgrad prepare")
            log_time_since(Grad_prep_CheckpointTime, "QM/MM gradient prepare")
            CheckpointTime = time.time()
        else:
            # No Grad
            self.QMenergy = QMenergy

        # MM THEORY
        if self.mm_theory_name == "OpenMMTheory":
            logger.info("Using OpenMM theory as part of QM/MM.")
            if self.QMChargesZeroed:
                logger.info(f"Using MM on full system. Charges for QM region {self.qmatoms} have been set to zero ")
            else:
                raise InternalError("QMCharges have not been zeroed")
            # Todo: Need to make sure OpenMM skips QM-QM Lj interaction => Exclude
            # Todo: Need to have OpenMM skip frozen region interaction for speed  => => Exclude
            if grad is True:
                CheckpointTime = time.time()
                # Provide QM_PC_gradient to OpenMMTheory

                if self.openmm_externalforce is True:
                    logger.info("OpenMM externalforce is True")
                    # Calculate energy associated with external force so that we can subtract it later
                    scaled_current_coords = current_coords * 1.88972612546
                    self.extforce_energy = 3 * np.mean(np.sum(self.QM_PC_gradient * scaled_current_coords, axis=0))
                    logger.info(f"Extforce energy: {self.extforce_energy}")
                    log_time_since(CheckpointTime, "extforce prepare")
                    # NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update is True:
                        logger.info("OpenMM custom external force updated. Exit requested")
                        # This is used if OpenMM MD is handling forces and dynamics
                        return self.QMenergy - self.extforce_energy, self.QM_PC_gradient

                self.MMenergy, self.MMgradient = self.mm_theory.run(
                    current_coords=current_coords, qmatoms=self.qmatoms, grad=True
                )
            else:
                logger.info("QM/MM Grad is false")
                self.MMenergy = self.mm_theory.run(current_coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMenergy = 0
        log_time_since(CheckpointTime, "MM step")
        CheckpointTime = time.time()

        # Final QM/MM Energy. Possible correction for OpenMM external force term
        self.QM_MM_energy = self.QMenergy + self.MMenergy - self.extforce_energy - self.subtractive_correction_E
        logger.info("")
        if self.embedding.lower() == "elstat":
            logger.info(
                "Note: You are using electrostatic embedding. This means that the QM-energy is actually the polarized QM-energy"
            )
            logger.info("Note: MM energy also contains the QM-MM Lennard-Jones interaction\n")
        energywarning = ""
        if self.truncated_pc is True:
            # if self.TruncatedPCflag is True:
            logger.info(
                "Warning: Truncated PC approximation is active. This means that QM and QM/MM energies are approximate."
            )
            energywarning = "(approximate)"

        logger.info("%s", "{:<20} {:>20.12f} {}".format("QM energy: ", self.QMenergy, energywarning))
        logger.info("%s", "{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
        logger.info("%s", "{:<20} {:>20.12f} {}".format("QM/MM energy: ", self.QM_MM_energy, energywarning))
        logger.info("")

        # FINAL QM/MM GRADIENT ASSEMBLY
        if grad is True:
            # If OpenMM external force method then QM/MM gradient is already complete
            # NOTE: Not possible anymore
            if self.openmm_externalforce is True:
                pass
            # Otherwise combine
            else:
                # Now assemble full QM/MM gradient
                if len(self.QM_PC_gradient) != len(self.MMgradient):
                    raise InternalError("QM-PC gradient and MM gradient size mismatch")
                self.QM_MM_gradient = self.QM_PC_gradient + self.MMgradient - self.subtractive_correction_G

            if logger.isEnabledFor(logging.DEBUG):
                openmmqmmm.coords.write_coords_all(
                    self.QMgradient_wo_linkatoms,
                    self.qmelems,
                    indices=self.qmatoms,
                    file=f"QMgradient-without-linkatoms_{label}",
                    description=f"QM gradient w/o linkatoms {label} (au/Bohr):",
                )
                # Writing QM+Linkatoms gradient
                openmmqmmm.coords.write_coords_all(
                    self.QMgradient,
                    self.qmelems + ["L" for i in range(self.num_linkatoms)],
                    indices=self.qmatoms + [0 for i in range(self.num_linkatoms)],
                    file=f"QMgradient-with-linkatoms_{label}",
                    description=f"QM gradient with linkatoms {label} (au/Bohr):",
                )
                openmmqmmm.coords.write_coords_all(
                    self.PCgradient,
                    self.mmelems,
                    indices=self.mmatoms,
                    file=f"PCgradient_{label}",
                    description=f"PC gradient {label} (au/Bohr):",
                )
                openmmqmmm.coords.write_coords_all(
                    self.QM_PC_gradient,
                    self.elems,
                    indices=self.allatoms,
                    file=f"QM+PCgradient_{label}",
                    description=f"QM+PC gradient {label} (au/Bohr):",
                )
                openmmqmmm.coords.write_coords_all(
                    self.MMgradient,
                    self.elems,
                    indices=self.allatoms,
                    file=f"MMgradient_{label}",
                    description=f"MM gradient {label} (au/Bohr):",
                )
                openmmqmmm.coords.write_coords_all(
                    self.QM_MM_gradient,
                    self.elems,
                    indices=self.allatoms,
                    file=f"QM_MMgradient_{label}",
                    description=f"QM/MM gradient {label} (au/Bohr):",
                )
            logger.warning("------------ENDING QM/MM MODULE-------------")
            log_time_since(module_init_time, "QM/MM run")
            return self.QM_MM_energy, self.QM_MM_gradient
        else:
            log_time_since(module_init_time, "QM/MM run")
            return self.QM_MM_energy


# Micro-iterative QM/MM Optimization
# NOTE: Not ready
# Wrapper around QM/MM run and geometric optimizer for performing microiterative QM/MM opt
# I think this is easiest
# Thiel: https://pubs.acs.org/doi/10.1021/ct600346p
# Look into new: https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.6b00547

# 3. QM/MM single-point with new charges?
# 3b. Or do geometric job until a certain threshold and then do MM again??


# frozen-density micro-iterative QM/MM


def fullindex_to_qmindex(fullindex, qmatoms):
    qmindex = qmatoms.index(fullindex)
    return qmindex


# Grab resid column from PDB-file and return list of resids
# NOTE: New resid-indices are used to avoid problem of PDB-file having
# repeating sequences of resids, additional chains or segments
def grab_resids_from_pdbfile(pdbfile):
    resids = []  # New list of resid indices, starting from 0
    actual_resids = []  # Actual resid values from PDB-file, used to check if resid has changed
    indexcount = 0  # This will be used to define residues
    with open(pdbfile) as f:
        for line in f:
            if "ATOM" in line or "HETATM" in line:
                # Based on: https://cupnet.net/pdb-format/
                resid_part = int(line[22:26].replace(" ", ""))
                # Very first atom and first residue
                if len(resids) == 0 or resid_part == actual_resids[-1]:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                # Resid changed, meaning new residue
                else:
                    indexcount += 1
                    # New residue
                    resids.append(indexcount)
                    actual_resids.append(resid_part)

    return resids


# Grab resid column from PSF-file and return list of resids
# NOTE: New resid-indices are used to avoid problem of PSF-file having
# repeating sequences of resids, additional chains or segments
def grab_resids_from_psffile(psffile):
    resids = []  # New list of resid indices, starting from 0
    actual_resids = []  # Actual resid values from PSF-file, used to check if resid has changed
    indexcount = 0  # This will be used to define residues
    with open(psffile) as f:
        for line in f:
            if "REMARKS" in line:
                continue
            if len(line.split()) > 8:
                line.split()[3]
                resid_part = int(line.split()[2])
                # Very first atom and first residue
                if len(resids) == 0 or resid_part == actual_resids[-1]:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                # Resid changed, meaning new residue
                else:
                    indexcount += 1
                    # New residue
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
    return resids


# Read atomic charges present in PSF-file. assuming Xplor format
def read_charges_from_psf(file):
    charges = []
    grab = False
    with open(file) as f:
        for line in f:
            if len(line.split()) == 9 and "REMARKS" not in line:
                grab = True
            if len(line.split()) < 8:
                grab = False
            if "NBOND" in line:
                return charges
            if grab is True:
                charge = float(line.split()[6])
                charges.append(charge)
    return charges


# Define active region based on radius from an origin atom.
# Requires fragment (for coordinates) and residue information from either:
# 1. resids list inside OpenMMTheory object
# 2. residues taken from PDB-file
# 3. residues taken from PSF-file


def define_active_region(pdbfile=None, mmtheory=None, psffile=None, fragment=None, radius=None, originatom=None):
    logger.info(main_header("ActregionDefine"))

    # Checking if proper information has been provided
    if radius is None or originatom is None:
        raise InputError("actregiondefine requires radius and originatom keyword arguments")
    if pdbfile is None and fragment is None:
        raise InputError("actregiondefine requires either fragment or pdbfile arguments (for coordinates)")
    if pdbfile is None and mmtheory is None and psffile is None:
        raise InputError(
            "actregiondefine requires either pdbfile, psffile or mmtheory arguments (for residue topology information)"
        )

    # Creating fragment from pdbfile
    if fragment is None:
        logger.info("No fragment provided. Creating fragment from PDBfile")
        fragment = Fragment(pdbfile=pdbfile)

    logger.info("Radius: %s", radius)
    logger.info(f"Origin atom: {originatom} ({fragment.elems[originatom]})")
    logger.info(f"Will find all atoms within {radius} Å from atom: {originatom} ({fragment.elems[originatom]})")
    logger.info("Will select all whole residues within region and export list")
    if mmtheory is not None:
        if not mmtheory.resids:
            raise InputError("mmtheory.resids list is empty! Something wrong with OpenMMTheory setup. Exiting")
        # Defining list of residue from OpenMMTheory object
        resids = mmtheory.resids
    elif psffile is not None:
        logger.info("PSF-file provided. Using residue information")
        resids = grab_resids_from_psffile(psffile)
    else:
        logger.info("PDB-file provided. Using residue information")
        # No mmtheory but PDB file should have been provided
        # Defining resids list from PDB-file
        # NOTE: Call grab_resids_from_pdbfile
        resids = grab_resids_from_pdbfile(pdbfile)

    origincoords = fragment.coords[originatom]
    logger.info("Origin-atom coordinates: %s", origincoords)
    act_indices = []
    for index, allc in enumerate(fragment.coords):
        dist = openmmqmmm.coords.distance(origincoords, allc)
        if dist < radius:
            # Get residue ID for this atom index
            resid_value = resids[index]
            # Get all residue members (atom indices)
            resid_members = [i for i, x in enumerate(resids) if x == resid_value]
            # Adding to act_indices list unless already added
            for k in resid_members:
                if k not in act_indices:
                    act_indices.append(k)

    # Only unique and sorting:
    logger.info("act_indices: %s", act_indices)
    act_indices = np.unique(act_indices).tolist()

    # Print indices to output
    # Print indices to disk as file
    writelisttofile(act_indices, "active_atoms")
    # Print information on how to use
    logger.info("Active region size: %s", len(act_indices))
    logger.info("Active-region indices written to file: active_atoms")
    logger.info(
        'The active_atoms list  can be read-into Python script like this:	 actatoms = read_intlist_from_file("active_atoms")'
    )
    # Print XYZ file with active region shown
    openmmqmmm.coords.write_xyz_for_atoms(fragment.coords, fragment.elems, act_indices, "ActiveRegion")
    logger.info("Wrote Active region XYZfile: ActiveRegion.xyz  (inspect with visualization program)")
    return act_indices


# This projects the linkatom force onto the respective QM atom and MM atom
def linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad):
    # QM1-L and QM1-MM1 distances
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord) * openmmqmmm.constants.ang2bohr
    MQdistance = openmmqmmm.coords.distance(Mcoord, Qcoord) * openmmqmmm.constants.ang2bohr
    # Coords in Bohr
    Mcoord = Mcoord * openmmqmmm.constants.ang2bohr
    Qcoord = Qcoord * openmmqmmm.constants.ang2bohr
    # B and C: a 3x3 arrays
    B = np.zeros([3, 3])
    C = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            B[i, j] = (
                -1
                * QLdistance
                * (Mcoord[i] - Qcoord[i])
                * (Mcoord[j] - Qcoord[j])
                / (MQdistance * MQdistance * MQdistance)
            )
    for i in range(3):
        B[i, i] = B[i, i] + QLdistance / MQdistance
    for i in range(3):
        for j in range(3):
            C[i, j] = -1 * B[i, j]
    for i in range(3):
        C[i, i] = C[i, i] + 1.0

    # Multiplying C matrix with Linkatom gradient
    g_x = float(C[0, 0] * Lgrad[0] + C[0, 1] * Lgrad[1] + C[0, 2] * Lgrad[2])
    g_y = float(C[1, 0] * Lgrad[0] + C[1, 1] * Lgrad[1] + C[1, 2] * Lgrad[2])
    g_z = float(C[2, 0] * Lgrad[0] + C[2, 1] * Lgrad[1] + C[2, 2] * Lgrad[2])

    # Multiplying B matrix with Linkatom gradient
    gg_x = float(B[0, 0] * Lgrad[0] + B[0, 1] * Lgrad[1] + B[0, 2] * Lgrad[2])
    gg_y = float(B[1, 0] * Lgrad[0] + B[1, 1] * Lgrad[1] + B[1, 2] * Lgrad[2])
    gg_z = float(B[2, 0] * Lgrad[0] + B[2, 1] * Lgrad[1] + B[2, 2] * Lgrad[2])

    # Return QM1_gradient and MM1_gradient contribution (to be added)
    return [g_x, g_y, g_z], [gg_x, gg_y, gg_z]


# Should be what ORCA uses
def linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad):
    # QM1-L and QM1-MM1 distances
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord)
    MQdistance = openmmqmmm.coords.distance(Mcoord, Qcoord)
    # scaling factor
    scal = QLdistance / MQdistance
    gradMM = Lgrad * scal
    gradQM = Lgrad * (1.0 - scal)
    return gradQM, gradMM


# simplistic, unused
def linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad):
    # QM1-L and QM1-MM1 distances
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord) * openmmqmmm.constants.ang2bohr
    vec = (Mcoord - Qcoord) * openmmqmmm.constants.ang2bohr
    R2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
    oneR = 1.0 / math.sqrt(R2)
    lnk_dis_oneR = QLdistance * oneR
    vec = vec * oneR
    dotprod = Lgrad[0] * (-1) * vec[0] + Lgrad[1] * (-1) * vec[1] + Lgrad[2] * (-1) * vec[2]
    forcemod = np.zeros(3)
    forcemod[0] = lnk_dis_oneR * (-1 * Lgrad[0] - (dotprod * vec[0]))
    forcemod[1] = lnk_dis_oneR * (-1 * Lgrad[1] - (dotprod * vec[1]))
    forcemod[2] = lnk_dis_oneR * (-1 * Lgrad[2] - (dotprod * vec[2]))
    # Returning forcemod as QM1 and MM1 contributions
    # subtract from QM1,  add to MM1
    return -1 * forcemod, forcemod


# Convenient function to calculate and decompose the QM/MM energy of a system and QMMMTheory object
def compute_decomposed_qm_mm_energy(fragment=None, theory=None):
    logger.info(main_header("Decomposed QM/MM Energy Calculation"))

    if isinstance(theory, QMMMTheory) is False:
        raise InputError("Please provide a QMMMTheory object as theory.")
    # if mm_theory is None:
    if theory.qm_charge is None or theory.qm_mult is None:
        raise InputError("Please define qm_charge and qm_mult attributes in the QMMMtheory object")

    # Single-point energy calculation of QM/MM object
    result = openmmqmmm.single_point(theory=theory, fragment=fragment)

    # Grabbing the basic terms (the ones always calculated)
    E_QM_MM_tot = result.energy
    E_QM_pol = result.qm_energy
    E_MM_mod = result.mm_energy

    # Extra calculation to decompose E_MM_mod into pure E_MM and QM-MM vdw terms
    # Updating MM theory: etting LJ part of QM-sites to zero. and recalculating MM part
    theory.mm_theory.update_lj_epsilons(theory.qmatoms, [0.0 for i in theory.qmatoms])
    result_MM_mod2 = openmmqmmm.single_point(theory=theory.mm_theory, fragment=fragment, charge=0, mult=1)
    # Taking the difference in MM energies: will be the QM-MM Lennard-Jones contribution
    E_QM_MM_vdw = E_MM_mod - result_MM_mod2.energy

    # QM-MM bonded (covalent) term
    logger.info("WARNING: QM-MM bonded term not implemented yet. Setting to zero.")
    logger.info("This means that the MM term still contains the QM-MM bonded contribution")
    E_QM_MM_bond = 0.0

    E_MM_pure = result_MM_mod2.energy

    ######################################
    # Extra calculation to decompose E_QM_pol into pure E_QM and elstatc energy
    # Defining a mechanical QM/MM object for the purpose of getting the pure QM-energy (no polarization)
    QM_MM_mech = QMMMTheory(
        fragment=fragment,
        qm_theory=theory.qm_theory,
        mm_theory=theory.mm_theory,
        qmatoms=theory.qmatoms,
        embedding="mech",
        qm_charge=theory.qm_charge,
        qm_mult=theory.qm_mult,
        unusualboundary=theory.unusualboundary,
        excludeboundaryatomlist=theory.excludeboundaryatomlist,
    )

    # Single-point energy calculation of mechanical QM/MM object. Taking only QM-energy
    result_mech = openmmqmmm.single_point(theory=QM_MM_mech, fragment=fragment)
    E_QM_pure = result_mech.qm_energy
    E_QM_MM_elstat = E_QM_pol - E_QM_pure

    # Defining the total coupling term
    E_coupling = E_QM_MM_elstat + E_QM_MM_vdw + E_QM_MM_bond

    # Sanity check
    if E_QM_MM_tot - (E_QM_pol + E_MM_mod) >= 1e-6:
        raise InternalError("QM/MM energy decomposition inconsistency (E_QM_pol + E_MM_mod)")
    if E_QM_MM_tot - (E_QM_pure + E_MM_pure + E_coupling) >= 1e-6:
        raise InternalError("QM/MM energy decomposition inconsistency (pure + coupling terms)")

    logger.info("")
    logger.info("%s", "=" * 70)
    logger.info("The standard QM/MM energy terms always printed:")
    logger.info("%s", "-" * 70)
    logger.info("E_QM/MM (Total QM/MM energy): %s", E_QM_MM_tot)
    logger.info("E_QM^pol (polarized QM-energy): %s", E_QM_pol)
    logger.info("E_MM^mod (MM-energy with QM-MM vdw contribution) %s", E_MM_mod)
    logger.info("%s", "-" * 70)
    logger.info("The decomposed terms:")
    logger.info("%s", "-" * 70)
    logger.info("E_QM/MM (Total QM/MM energy): %s", E_QM_MM_tot)
    logger.info("E_QM (The pure QM energy) %s", E_QM_pure)
    logger.info("E_MM (The pure MM energy) %s", E_MM_pure)
    logger.info("E_coupling (QM-MM total coupling energy) %s", E_coupling)
    logger.info("E_QM-MM_elstat (QM-MM elstat coupling energy) %s", E_QM_MM_elstat)
    logger.info("E_QM-MM_vdw (the QM-MM vdw coupling energy) %s", E_QM_MM_vdw)
    logger.info("E_QM_MM_bond (the QM-MM covalent coupling energy) %s", E_QM_MM_bond)
    logger.info("%s", "=" * 70)
    logger.info("")
