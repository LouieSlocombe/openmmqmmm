import copy
import logging
import math
import time

import numpy as np

import openmmqmmm.constants
import openmmqmmm.coords
from openmmqmmm.coords import CONNECTIVITY_SCALE, CONNECTIVITY_TOL, Fragment
from openmmqmmm.exceptions import (
    InputError,
    InternalError,
)
from openmmqmmm.utils import log_time_since, main_header, writelisttofile

logger = logging.getLogger(__name__)

# Required at init: qm_theory and qmatoms and fragment


class QMMMTheory:
    """Electrostatically embedded QM/MM combining a QM theory with an MM theory."""

    def __init__(
        self,
        *,
        qm_theory=None,
        qmatoms=None,
        fragment=None,
        mm_theory=None,
        charges=None,
        embedding="elstat",
        numcores=1,
        label="QM/MM",
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
        logger.info(main_header("QM/MM Theory"))

        if qm_theory is None or qmatoms is None:
            raise InputError("Error: QMMMTheory requires defining: qm_theory, qmatoms, fragment")
        if fragment is None:
            raise InputError("fragment= keyword has not been defined for QM/MM. Exiting")

        self.qm_charge = qm_charge
        self.qm_mult = qm_mult

        self.theorytype = "QM/MM"
        self.theorynamelabel = "QMMMTheory"
        self.label = label

        # External force energy. Zero except when using openmm_externalforce
        self.extforce_energy = 0.0
        # Subtractive corrections that might be defined later on
        # Added due to pbcmm-elstat
        self.subtractive_correction_E = 0.0
        self.subtractive_correction_G = np.zeros((len(fragment.coords), 3))

        # After each QM-region calculation, the charges of the QM-region may have been calculated
        # These charges can be used to update the charges of the whole system. Only used for mechanical embedding
        self.update_qm_region_charges = update_qm_region_charges

        self.linkatoms = False

        self.linkatom_type = linkatom_type  # Usually 'H'
        self.linkatom_method = linkatom_method  # Options: 'simple' or 'ratio'
        self.linkatom_simple_distance = linkatom_simple_distance  # For method simple, Default 1.09 Angstrom
        # For method ratio. see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
        self.linkatom_ratio = linkatom_ratio
        # Linkatom projection method Options: 'adv', 'lever', 'chain', 'none'
        self.linkatom_forceproj_method = linkatom_forceproj_method
        if self.linkatom_forceproj_method is None:
            linkatom_forceproj_method = "none"

        self.runcalls = 0
        self.qm_charge_consistency_logged = False

        # NOTE: affects runmode
        self.openmm_externalforce = openmm_externalforce

        self.exit_after_customexternalforce_update = exit_after_customexternalforce_update

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

        self.allatoms = list(range(len(self.elems)))
        logger.info("All atoms in fragment: %s", len(self.allatoms))
        self.num_allatoms = len(self.allatoms)

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

        # numcores-setting in QMMMTheory takes precedent
        if numcores != 1:
            self.numcores = numcores
        elif self.qm_theory.numcores != 1:
            self.numcores = self.qm_theory.numcores
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
                "Unknown embedding. Valid options are: elstat (synonyms: electrostatic, electronic), mech (synonym: "
                "mechanical)"
            )
        logger.info("Embedding: %s", self.embedding)
        # Whether to do dipole correction or not
        # Note: For regular electrostatic embedding this should be True
        # Turn off for charge-shifting
        self.dipole_correction = dipole_correction

        # Whether MM-shifted performed or not. Will be set to True by self.ShiftMMCharges
        self.chargeshifting_done = False

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

            if self.mm_theory is not None:
                self.mm_theory.update_charges(self.fragment.allatoms, self.charges)

        if len(self.charges) == 0:
            raise InputError("No charges present in QM/MM object. Exiting...")

        self.QMChargesZeroed = False

        # CHARGES DEFINED FOR OBJECT:
        # Self.charges are original charges that are defined above (on input or from OpenMM)
        # self.charges_qmregionzeroed is self.charges but with 0-value for QM-atoms
        # self.pointcharges are pointcharges that the QM-code will see (dipole-charges, no zero-valued charges etc)
        # Length of self.charges: system size
        # Length of self.charges_qmregionzeroed: system size
        # Length of self.pointcharges: unknown. does not contain zero-valued charges (e.g. QM-atoms etc.), contains
        # dipole-charges

        self.charges_qmregionzeroed = []

        self.pointcharges = []

        self.truncated_pc = truncated_pc
        self.truncated_pc_radius = truncated_pc_radius
        self.truncated_pc_calls = 0
        self.truncated_pc_recalc_flag = False
        self.truncated_pc_recalc_iter = truncated_pc_recalc_iter

        if self.truncated_pc is True:
            logger.info("Truncated PC approximation in QM/MM is active.")
            logger.info("TruncPCRadius: %s", self.truncated_pc_radius)
            logger.info("TruncPC Recalculation iteration: %s", self.truncated_pc_recalc_iter)

        if mm_theory is None:
            # No MM theory, but the QM charges still have to be zeroed for elstat embedding
            if self.embedding.lower() == "elstat":
                self.zero_qm_charges()
            self.linkatoms = False
            self.dipole_correction = False
        else:
            self._setup_mm_theory(fragment)
        log_time_since(module_init_time, "QM/MM object creation")

    # From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_mm_boundary(self, scale, tol):
        """Find the QM-MM covalent boundary and the MM atoms bonded across it."""
        timeA = time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        self.MMboundarydict = {}
        for QM1atom, MM1atom in self.boundaryatoms.items():
            if isinstance(MM1atom, list):
                for mat in MM1atom:
                    connatoms = openmmqmmm.coords.get_connected_atoms(self.coords, self.elems, scale, tol, mat)
                    connatoms.remove(QM1atom)
                    self.MMboundarydict[mat] = connatoms
            # OLD: should never apply anymore, we always have a list
            else:
                connatoms = openmmqmmm.coords.get_connected_atoms(self.coords, self.elems, scale, tol, MM1atom)
                connatoms.remove(QM1atom)
                self.MMboundarydict[MM1atom] = connatoms

        # Used by ShiftMMCharges
        self.MMboundary_indices = list(self.MMboundarydict.keys())
        self.MMboundary_counts = np.array([len(self.MMboundarydict[i]) for i in self.MMboundary_indices])

        logger.info("")
        logger.info("MM boundary (MM1:MMx pairs): %s", self.MMboundarydict)
        log_time_since(timeA, "get_MMboundary")

    def zero_qm_charges(self):
        """Set the MM charges of the QM-region atoms to zero for electrostatic embedding."""
        timeA = time.time()
        logger.info("Setting QM charges to Zero")
        self.charges_qmregionzeroed = copy.copy(self.charges)
        for i, _c in enumerate(self.charges_qmregionzeroed):
            if i in self.qmatoms:
                self.charges_qmregionzeroed[i] = 0.0
        self.QMChargesZeroed = True
        log_time_since(timeA, "ZeroQMCharges")

    def rcd_shifting_prep(self, charges_qmregionzeroed):
        """Set up redistributed-charge-and-dipole (RCD) charge shifting."""
        timeA = time.time()
        logger.info("Shifting MM charges at QM/MM boundary by RCD.")
        pointcharges = np.array(charges_qmregionzeroed)
        self.charges = np.array(self.charges)
        MM1_charges = self.charges[self.MMboundary_indices]
        pointcharges[self.MMboundary_indices] = 0.0
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        pointcharges = [pointcharges[x] for x in self.mmatoms]

        RCD_additional_charges = []
        for _MM1index, MM2indices, fract in zip(
            self.MMboundarydict.keys(), self.MMboundarydict.values(), MM1charge_fract, strict=False
        ):
            newfract = fract * 2  # q0*2
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
        """Rebuild the RCD point-charge positions for the current geometry."""
        timeA = time.time()
        logger.info("Adding updated RCD charges at QM/MM boundary by RCD.")

        # One RCD site per MM2 atom, in the same order as rcd_shifting_prep created the
        # matching extra charges, so that charges and coordinates stay index-aligned.
        # New RCD site sits midway between each MM1 atom and each of its MM2 atoms
        newsites = [
            (fullcoords[i] + fullcoords[MM1index]) / 2
            for MM1index, MM2indices in self.MMboundarydict.items()
            for i in MM2indices
        ]

        pointchargecoords = np.append(used_mmcoords, np.array(newsites), axis=0) if newsites else used_mmcoords

        log_time_since(timeA, "RCD_shifting_update")
        return pointchargecoords

    def shift_mm_charges(self):
        """Shift the MM1 boundary charges onto their MM2 neighbours."""
        if self.chargeshifting_done is False:
            self._shift_mm_charges_impl()
        else:
            logger.info("Charge shifting already done. Using previous charges")

    def _shift_mm_charges_impl(self):
        timeA = time.time()
        logger.info("new. Shifting MM charges at QM-MM boundary.")

        log_time_since(timeA, "x0")
        self.pointcharges = np.array(self.charges_qmregionzeroed)
        self.charges = np.array(self.charges)

        log_time_since(timeA, "x1")
        MM1_charges = self.charges[self.MMboundary_indices]
        self.pointcharges[self.MMboundary_indices] = 0.0

        MM1charge_fract = MM1_charges / self.MMboundary_counts

        for indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract, strict=False):
            self.pointcharges[[indices]] += fract

        self.chargeshifting_done = True
        log_time_since(timeA, "ShiftMMCharges-new2")

    def get_dipole_charge(self, delq, direction, mm1index, mm2index, current_coords):
        """Return the two charges and positions of a dipole placed on an MM1-MM2 bond."""
        mm1coords = np.array(current_coords[mm1index])
        mm2coords = np.array(current_coords[mm2index])
        MM_distance = openmmqmmm.coords.distance(mm1coords, mm2coords)  # Distance between MM1 and MM2

        SHIFT = 0.15

        def vnorm(p1):
            r = math.sqrt((p1[0] * p1[0]) + (p1[1] * p1[1]) + (p1[2] * p1[2]))
            return np.array([p1[0] / r, p1[1] / r, p1[2] / r])

        diffvector = mm2coords - mm1coords
        normdiffvector = vnorm(diffvector)

        d = delq * 2.5
        q0 = 0.5 * d / SHIFT
        shift = direction * SHIFT * (MM_distance / 2.5)
        pos = mm2coords + np.array(shift * normdiffvector)
        return -q0 * direction, list(pos)

    def set_dipole_charges(self, current_coords):
        """Rebuild the dipole-correction point charges for the current geometry."""
        checkpoint = time.time()
        logger.info("Adding extra charges to preserve dipole moment for charge-shifting")
        logger.info("MMboundarydict: %s", self.MMboundarydict)
        self.dipole_charges = []
        self.dipole_coords = []

        for MM1, MMx in self.MMboundarydict.items():
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

    # Uses a precalculated mask; this dominates QM/MM gradient prepare.
    def make_qm_pc_gradient(self):
        """Assemble the full-system gradient from the QM and point-charge gradients."""
        self.QM_PC_gradient[self.xatom_mask] = self.QMgradient_wo_linkatoms
        self.QM_PC_gradient[~self.xatom_mask] = self.PCgradient[: self.num_allatoms - self.sum_xatom_mask]

    def truncated_pc_function(self, used_qmcoords):
        """Reduce the point-charge field to the atoms near the QM region."""
        self.truncated_pc_calls += 1
        logger.info("TruncatedPC approximation!")
        if self.truncated_pc_calls == 1 or self.truncated_pc_calls % self.truncated_pc_recalc_iter == 0:
            self.truncated_pc_recalc_flag = True
            logger.info(
                f"This is QM/MM run no. {self.truncated_pc_calls}.  Will calculate Full-Trunc correction in this step"
            )
            origincoords = openmmqmmm.coords.get_centroid(used_qmcoords)
            self.determine_truncated_pc_indices(origincoords)
            logger.info(f"Truncated PC-region size: {len(self.truncated_PC_region_indices)} charges")
            # Saving full PCs and coords for 1st iteration
            # NOTE: Here using self.pointcharges_original (set by runprep)
            # since self.pointcharges may be truncated-version from last iter
            self.pointcharges_full = copy.copy(self.pointcharges_original)
            self.pointchargecoords_full = copy.copy(self.pointchargecoords)

            self.pointcharges = [self.pointcharges_full[i] for i in self.truncated_PC_region_indices]
            self.pointchargecoords = np.take(self.pointchargecoords_full, self.truncated_PC_region_indices, axis=0)
        else:
            self.truncated_pc_recalc_flag = False
            logger.info(
                f"This is QM/MM run no. {self.truncated_pc_calls}. Using approximate truncated PC field: "
                f"{len(self.truncated_PC_region_indices)} charges"
            )
            # NOTE: Here taking 1st-iter full PCs (values have not changed during opt/md)
            self.pointcharges = [self.pointcharges_full[i] for i in self.truncated_PC_region_indices]
            # NOTE: Here taking from CURRENT full pointchargecoords (not old full from step 1) since coords have changed
            self.pointchargecoords = np.take(self.pointchargecoords, self.truncated_PC_region_indices, axis=0)

    # Coordinates and charges for each Opt cycle defined later.
    def determine_truncated_pc_indices(self, origincoords):
        """Select the point charges within truncated_pc_radius of the QM region."""
        region_indices = []
        for index, allc in enumerate(self.pointchargecoords):
            dist = openmmqmmm.coords.distance(origincoords, allc)
            if dist < self.truncated_pc_radius:
                region_indices.append(index)
        self.truncated_PC_region_indices = np.unique(region_indices).tolist()

    def calculate_trunc_pc_gradient_correction(
        self, QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc
    ):
        """Compute the QM and point-charge gradient corrections for PC truncation and store them."""
        qm_difference = (
            QMgradient_full[: len(QMgradient_full) - self.num_linkatoms]
            - QMgradient_trunc[: len(QMgradient_full) - self.num_linkatoms]
        )
        self.original_QMcorrection_gradient = qm_difference
        truncated_indices = np.array(self.truncated_PC_region_indices)
        pc_difference = np.zeros((len(PCgradient_full), 3))
        pc_difference[truncated_indices] = PCgradient_full[truncated_indices] - PCgradient_trunc
        pc_difference[~np.isin(np.arange(len(PCgradient_full)), truncated_indices)] = PCgradient_full[
            ~np.isin(np.arange(len(PCgradient_full)), truncated_indices)
        ]
        self.original_PCcorrection_gradient = pc_difference

    def truncated_pc_gradient_update(self, QMgradient_wo_linkatoms, PCgradient):
        """Apply the stored truncation correction to this step's gradients."""
        newQMgradient_wo_linkatoms = QMgradient_wo_linkatoms + self.original_QMcorrection_gradient

        new_full_PC_gradient = np.copy(self.original_PCcorrection_gradient)
        new_full_PC_gradient[self.truncated_PC_region_indices] += PCgradient

        return newQMgradient_wo_linkatoms, new_full_PC_gradient

    def set_numcores(self, numcores):
        """Set the core count used by both the QM and MM theories."""
        logger.info(f"Setting new numcores {numcores}for QMtheory and MMtheory")
        self.qm_theory.set_numcores(numcores)
        self.mm_theory.set_numcores(numcores)

    def get_dipole_moment(self):
        """Return the QM theory's dipole moment, or None if it does not provide one."""
        logger.info("Grabbing dipole moment from QM-part of QM/MM theory.")
        dipole = None
        try:
            dipole = self.qm_theory.get_dipole_moment()
        except AttributeError:
            logger.error("Could not grab dipole moment from QM-part of QM/MM theory.")
        return dipole

    def get_polarizability_tensor(self):
        """Return the QM theory's polarizability tensor, or None if it does not provide one."""
        logger.info("Grabbing polarizability from QM-part of QM/MM theory.")
        polarizability = None
        try:
            polarizability = self.qm_theory.get_polarizability_tensor()
        except AttributeError:
            logger.error("Could not grab polarizability from QM-part of QM/MM theory.")
        return polarizability

    def resolve_qm_charge_mult(self, *, charge=None, mult=None) -> tuple[int, int]:
        """Resolve the charge and multiplicity of the QM region."""
        return (
            self._resolve_qm_electronic_state("charge", charge),
            self._resolve_qm_electronic_state("mult", mult),
        )

    def _resolve_qm_electronic_state(self, name, supplied_value):
        theory_value = getattr(self, f"qm_{name}")
        if theory_value is not None:
            # Equality rather than presence: one job resolves 2-3 times on its way down to run()
            if supplied_value is not None and supplied_value != theory_value:
                raise InputError(f"{name}={supplied_value} conflicts with QMMMTheory.qm_{name}={theory_value}")
            return theory_value

        if supplied_value is not None:
            return supplied_value

        # self.fragment is the whole system, so its net charge is the QM region's only when the
        # two regions coincide. Checked after the None-test so that it is never reached for a
        # fragment carrying no charge at all.
        fragment_value = getattr(self.fragment, name, None)
        if fragment_value is not None:
            if set(self.qmatoms) == set(self.allatoms):
                logger.info(f"QM region is the whole system. Using fragment {name}={fragment_value}")
                return fragment_value
            raise InputError(
                f"Fragment {name}={fragment_value} describes all {self.num_allatoms} atoms, not the "
                f"{len(self.qmatoms)}-atom QM region. Set QMMMTheory(qm_{name}=...) or pass {name}= to the job"
            )

        raise InputError(f"QM-region {name} is undefined. Set QMMMTheory(qm_{name}=...) or pass {name}= to the job")

    def _log_qm_charge_consistency(self, charge):
        mm_sum = float(np.sum(np.asarray(self.charges)[self.qmatoms]))
        deviation = mm_sum - round(mm_sum)
        if self.linkatoms:
            # Link atoms and charge shifting redistribute charge across the boundary by design,
            # so the MM sum over qmatoms is not expected to reproduce the QM-region charge.
            logger.info(
                f"Sum of MM charges over the QM region: {mm_sum:.4f} (QM-region charge: {charge}). Link atoms "
                f"present, so a difference is expected."
            )
            return
        if abs(deviation) > 0.01:
            logger.warning(
                f"Sum of MM charges over the QM region is {mm_sum:.4f}, not an integer. The QM region may cut "
                f"through a molecule without a covalent boundary being detected."
            )
        elif round(mm_sum) != charge:
            logger.warning(
                f"QM-region charge is {charge} but the MM charges it replaces sum to {mm_sum:.4f}. One of the two "
                f"is wrong."
            )

    def _setup_mm_theory(self, fragment):
        """Find the QM-MM boundary, strip the QM region out of the MM force field and zero its charges."""
        if fragment.numatoms != self.mm_theory.numatoms:
            raise InputError(
                f"Number of atoms in fragment ({fragment.numatoms}) and MMtheory object differ "
                f"({self.mm_theory.numatoms})\nThis does not make sense. Check coordinates and forcefield files."
            )

        # Tolerance is bumped so that connected atoms are definitely caught and the QM-MM
        # boundary comes out right: scale=1.0/tol=0.1 missed the S-C bond in rubredoxin from
        # a classical MD run, and +0.1 missed a lysine C-C bond (21 Sep 2023).
        conn_scale = CONNECTIVITY_SCALE
        conn_tolerance = CONNECTIVITY_TOL + 0.2

        # If a QM-MM boundary issue aborts the run then printing QM-coordinates is useful
        logger.info("QM-region coordinates (before linkatoms):")
        openmmqmmm.coords.print_coords_for_atoms(self.coords, self.elems, self.qmatoms, labels=self.qmatoms)
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
            logger.info(
                f"Found covalent QM-MM boundary. Linkatoms option set to True\n"
                f"Boundaryatoms (QM:MM pairs): {self.boundaryatoms}\n"
                f"Note: used connectivity settings, scale={conn_scale} and tol={conn_tolerance} to determine boundary."
            )
            self.linkatoms = True
            logger.info("Linkatom_forceprojection_method: %s", self.linkatom_forceproj_method)
            self.get_mm_boundary(conn_scale, conn_tolerance)
        else:
            logger.info("No covalent QM-MM boundary. Linkatoms and dipole_correction options set to False")
            self.linkatoms = False
            self.dipole_correction = False

        if self.mm_theory_name == "OpenMMTheory":
            # Only applies when running OpenMM_Opt or OpenMM_MD, and to both embeddings.
            # NonbondedTheory never defines these terms in the first place.
            self.mm_theory.remove_constraints_for_atoms(self.qmatoms)
            logger.info("Removing bonded terms for QM-region in MMtheory")
            self.mm_theory.modify_bonded_forces(self.qmatoms)
            # Exceptions make OpenMM ignore QM-QM Coulomb and LJ. QM-MM elstat Coulomb
            # charges are zeroed separately below.
            logger.info("Removing nonbonded terms for QM-region in MMtheory (QM-QM interactions)")
            self.mm_theory.addexceptions(self.qmatoms)

        embedding = self.embedding.lower()
        if embedding == "elstat":
            logger.info("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
            self.zero_qm_charges()
            self.mm_theory.update_charges(self.qmatoms, [0.0 for _ in self.qmatoms])
        elif embedding == "polembed_drude":
            # Would zero the QM charges and then delete the QM-MM Coulomb exceptions
            # in OpenMM via mm_theory.delete_exceptions(self.qmatoms).
            raise InputError(
                "Polembed Drude embedding enabled.\nThis means that QM-atoms will be zeroed for QM-MM interactions "
                "calculated by QM program\nBut MM program will have charged defined for QM-region\nNot implemented "
                "yet. Exiting"
            )
        elif embedding == "pbcmm-elstat":
            raise InputError("embedding='pbcmm-elstat' is not supported in this distribution")

        self._log_region_charges()

    def _log_region_charges(self):
        """Log the per-atom charge each region carries, at DEBUG."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        qm_charges = self.charges_qmregionzeroed if self.embedding.lower() == "elstat" else self.charges
        for i in self.allatoms:
            charges = qm_charges if i in self.qmatoms else self.charges_qmregionzeroed
            region = "QM" if i in self.qmatoms else "MM"
            logger.debug(f"{region} atom {i} ({self.elems[i]}) charge: {charges[i]}")

    def run(
        self,
        *,
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
        """Run a QM/MM energy (and gradient) calculation."""
        logger.info("------------RUNNING QM/MM MODULE-------------")
        logger.info("QM Module: %s", self.qm_theory_name)
        logger.info("MM Module: %s", self.mm_theory_name)

        # exit_after_customexternalforce_update can be enabled both at runtime and by initialization
        if self.exit_after_customexternalforce_update is True:
            exit_after_customexternalforce_update = self.exit_after_customexternalforce_update

        charge, mult = self.resolve_qm_charge_mult(charge=charge, mult=mult)
        logger.info(f"QM-region Charge: {charge} Mult: {mult}")

        if not self.qm_charge_consistency_logged:
            self.qm_charge_consistency_logged = True
            self._log_qm_charge_consistency(charge)

        # pbcmm-elstat differs only in that the QM charges have not been zeroed in the MM
        # program, so it double-counts short-range QM-QM and QM-MM and elstat_run applies
        # subtractive corrections. polembed_drude likewise runs the electrostatic path.
        runner = {
            "mech": self.mech_run,
            "elstat": self.elstat_run,
            "pbcmm-elstat": self.elstat_run,
            "polembed_drude": self.elstat_run,
        }.get(self.embedding.lower())
        if runner is None:
            raise InputError(f"Unknown embedding '{self.embedding}'. Expected one of mech, elstat, pbcmm-elstat.")

        return runner(
            current_coords=current_coords,
            elems=elems,
            grad=grad,
            numcores=numcores,
            exit_after_customexternalforce_update=exit_after_customexternalforce_update,
            label=label,
            charge=charge,
            mult=mult,
        )

    def run_openmm_python_force(self, *, current_coords, elems, charge, mult):
        """Return the physical external energy and gradient for an OpenMM ``PythonForce``.

        The older ``CustomExternalForce`` MD path represents a frozen gradient with a
        coordinate-linear potential.  Electrostatic embedding therefore subtracts that
        artificial potential from its reported energy.  ``PythonForce`` directly owns the
        QM potential and must instead receive the uncorrected QM energy.
        """
        result = self.run(
            current_coords=current_coords,
            elems=elems,
            grad=True,
            exit_after_customexternalforce_update=True,
            charge=charge,
            mult=mult,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise InternalError("QM/MM force evaluation must return an (energy, gradient) pair.")
        _legacy_external_energy, gradient = result
        return self.QMenergy, gradient

    def _prepare_run(self, current_coords, embedding_label):
        logger.info("Embedding: %s", embedding_label)

        # Only do once to avoid cost in each step
        if self.runcalls == 0:
            logger.info("First QMMMTheory run. Running runprep")
            # Creates self.current_qmelems, the linkatom bookkeeping
            # (self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms,
            # self.linkatoms_coords) and, for elstat embedding, self.pointcharges
            self.runprep(current_coords)

        self.runcalls += 1

        used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms,
            # self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            used_qmcoords = np.append(used_qmcoords, np.array(linkatoms_coords), axis=0)

        return used_mmcoords, used_qmcoords

    def _add_linkatom_force_projection(self, gradient, used_qmcoords, current_coords):
        for pair in sorted(self.linkatoms_dict.keys()):
            linkatomindex = self.linkatom_indices.pop(0)
            Lgrad = self.QMgradient[linkatomindex]
            Lcoord = self.linkatoms_dict[pair]
            fullatomindex_qm = pair[0]
            qmatomindex = fullindex_to_qmindex(fullatomindex_qm, self.qmatoms)
            Qcoord = used_qmcoords[qmatomindex]
            fullatomindex_mm = pair[1]
            Mcoord = current_coords[fullatomindex_mm]

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

            gradient[fullatomindex_qm] += QM1grad_contrib
            gradient[fullatomindex_mm] += MM1grad_contrib

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
        """Run mechanical embedding: QM and MM energies added with no electrostatic coupling."""
        module_init_time = time.time()
        CheckpointTime = time.time()
        _used_mmcoords, used_qmcoords = self._prepare_run(current_coords, "Mechanical")

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        logger.info(f"Running QM/MM object with {numcores} cores available")

        log_time_since(module_init_time, "before-QMstep")
        CheckpointTime = time.time()
        if self.qm_theory_name in {"None", "ZeroTheory"}:
            logger.info("No QMtheory. Skipping QM calc")
            QMenergy = 0.0
            self.linkatoms = False
            QMgradient = np.zeros((len(used_qmcoords), 3))
        elif grad is True:
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

        if self.update_qm_region_charges:
            logger.info("update_QMregion_charges is True")
            logger.info("Will try to find charges attribute in QM-object")
            try:
                newqmcharges = self.qm_theory.charges
            except AttributeError:
                raise InputError(
                    "Found no charges attribute on the QM-theory object - update_QMregion_charges can not be used"
                ) from None
            if self.num_linkatoms > 0:
                newqmcharges = newqmcharges[0 : -self.num_linkatoms]
            for i, index in enumerate(self.qmatoms):
                self.charges[index] = newqmcharges[i]
            logger.info("Updating charges of QM-region in MMTheory object")
            self.mm_theory.update_charges(self.qmatoms, list(newqmcharges))
        logger.info("Defined charges of QM-region:")
        for i in self.qmatoms:
            logger.info(f"QM atom {i} has charge : {self.charges[i]}")

        self.QMenergy = QMenergy

        self.QM_MM_gradient = np.zeros((len(current_coords), 3))
        if grad:
            prep_start = time.time()
            self.QMgradient_wo_linkatoms = self._qm_gradient_without_linkatoms(QMgradient)
            self.QM_MM_gradient[self.qmatoms] += self.QMgradient_wo_linkatoms

            if self.linkatoms is True:
                checkpoint = time.time()
                self._add_linkatom_force_projection(self.QM_MM_gradient, used_qmcoords, current_coords)
                log_time_since(checkpoint, "linkatomgrad prepare")

            # Defining QM_PC_gradient for simplicity (used by OpenMM_MD)
            self.QM_PC_gradient = self.QM_MM_gradient
            log_time_since(prep_start, "QM/MM gradient prepare")
        else:
            self.QMenergy = QMenergy

        if self.mm_theory_name == "OpenMMTheory":
            logger.info("Using OpenMM theory as part of QM/MM.")
            if grad:
                CheckpointTime = time.time()
                if self.openmm_externalforce is True:
                    logger.info("OpenMM externalforce is True")
                    # Calculate energy associated with external force so that we can subtract it later
                    scaled_current_coords = current_coords * openmmqmmm.constants.ANG_TO_BOHR
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
            if len(self.QM_MM_gradient) != len(self.MMgradient):
                raise InternalError("QM/MM gradient and MM gradient size mismatch")
            self.QM_MM_gradient = self.QM_MM_gradient + self.MMgradient

        self.QM_MM_energy = self.QMenergy + self.MMenergy - self.subtractive_correction_E

        self.QM_MM_gradient -= self.subtractive_correction_G

        logger.info("")
        logger.info("%s", "{:<20} {:>20.12f}".format("QM energy: ", self.QMenergy))
        logger.info("%s", "{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
        logger.info("%s", "{:<20} {:>20.12f}".format("Subtractive correction energy: ", self.subtractive_correction_E))
        logger.info("%s", "{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_energy))
        logger.info("")

        if grad is True:
            if logger.isEnabledFor(logging.DEBUG):
                openmmqmmm.coords.write_coords_all(
                    self.QMgradient_wo_linkatoms,
                    self.qmelems,
                    indices=self.qmatoms,
                    file=f"QMgradient-without-linkatoms_{label}",
                    description=f"QM gradient w/o linkatoms {label} (au/Bohr):",
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
            logger.info("------------ENDING QM/MM MODULE-------------")
            log_time_since(module_init_time, "QM/MM mech run")
            return self.QM_MM_energy, self.QM_MM_gradient
        log_time_since(module_init_time, "QM/MM mech run")
        return self.QM_MM_energy

    def create_linkatoms(self, current_coords):
        """Place hydrogen link atoms along each QM1-MM1 bond."""
        checkpoint = time.time()
        self.linkatoms_dict = openmmqmmm.coords.get_linkatom_positions(
            self.boundaryatoms,
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

    def runprep(self, current_coords):
        """Do the one-off setup the first run needs: link atoms, boundary and charges."""
        logger.info("Inside QMMMTheory runprep")
        init_time_runprep = time.time()

        self.qmelems = [self.elems[i] for i in self.qmatoms]
        self.mmelems = [self.elems[i] for i in self.mmatoms]

        check_before_linkatoms = time.time()
        if self.linkatoms is True:
            self.create_linkatoms(current_coords)
            self.current_qmelems = self.qmelems + [self.linkatom_type] * self.num_linkatoms
            logger.info("Number of MM atoms: %s", len(self.mmatoms))
            logger.info(f"There are {self.num_linkatoms} linkatoms")
            if self.embedding.lower() == "elstat":
                logger.info("Doing charge-shifting...")

                if self.chargeboundary_method == "shift":
                    logger.info("Chargeboundary method is:  shift  ")
                    self.shift_mm_charges()  # Creates self.pointcharges
                    self.pointcharges = [self.pointcharges[i] for i in self.mmatoms]

                    if self.dipole_correction is True:
                        logger.info("Dipole correction is on. Adding dipole charges")
                        self.set_dipole_charges(current_coords)  # Creates self.dipole_charges and self.dipole_coords

                        logger.info(f"Adding {len(self.dipole_charges)} dipole charges to PC environment")

                        self.pointcharges = list(self.pointcharges) + list(self.dipole_charges)
                        logger.info("Number of pointcharges after dipole addition:  %s", len(self.pointcharges))
                        log_time_since(check_before_linkatoms, "Linkatom-dipolecorrection")
                    else:
                        logger.info("Dipole correction is off. Not adding any dipole charges")
                        logger.info("Number of pointcharges:  %s", len(self.pointcharges))
                elif self.chargeboundary_method == "rcd":
                    logger.info("Chargeboundary method is:  rcd  ")
                    self.pointcharges, _RCD_additional_charges = self.rcd_shifting_prep(self.charges_qmregionzeroed)
                else:
                    raise InputError("Unknown chargeboundary_method. Exiting")

                logger.info("Number of pointcharges defined for whole system:  %s", len(self.pointcharges))
                logger.info("Number of pointcharges defined for MM region:  %s", len(self.pointcharges))
        else:
            self.num_linkatoms = 0
            self.current_qmelems = self.qmelems
            if self.embedding.lower() == "elstat" or self.embedding.lower() == "polembed_drude":
                self.pointcharges = [self.charges_qmregionzeroed[i] for i in self.mmatoms]

        # NOTE: Now we have updated MM-coordinates (if doing linkatoms, with dipolecharges etc) and updated mm-charges
        # (more, due to dipolecharges if linkatoms)
        # We also have MMcharges that have been set to zero due to QM/MM
        # We do not delete charges but set to zero
        if len(self.qmatoms) == 0:
            logger.info("No qmatoms list provided. Setting QMtheory to None")
            self.qm_theory_name = "None"
            self.QMenergy = 0.0

        # For truncatedPC option.
        self.pointcharges_original = copy.copy(self.pointcharges)

        if self.embedding.lower() == "elstat" or self.embedding.lower() == "polembed_drude":
            self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))

        log_time_since(init_time_runprep, "runprep")

    def _qm_gradient_without_linkatoms(self, QMgradient):
        """Drop the link-atom rows, which the QM program appends after the real QM atoms."""
        self.QMgradient = QMgradient
        return QMgradient[0 : -self.num_linkatoms] if self.linkatoms else QMgradient

    def _prepare_qm_pc_gradient(
        self, *, QMenergy, QMgradient, PCgradient, used_qmcoords, current_coords, charge, mult, numcores
    ):
        """Turn this step's QM and point-charge gradients into self.QM_PC_gradient."""
        prep_start = time.time()
        QMgradient_wo_linkatoms = self._qm_gradient_without_linkatoms(QMgradient)

        if self.truncated_pc is not True:
            self.QMenergy = QMenergy
            # No TruncPC approximation active. No change to original QM and PCgradient from QMcode
            self.QMgradient_wo_linkatoms = QMgradient_wo_linkatoms
            if self.embedding.lower() in {"elstat", "polembed_drude"}:
                self.PCgradient = PCgradient
        elif self.truncated_pc_recalc_flag is True:
            self._recalculate_truncated_pc_correction(
                QMenergy=QMenergy,
                QMgradient=QMgradient,
                PCgradient=PCgradient,
                QMgradient_wo_linkatoms=QMgradient_wo_linkatoms,
                used_qmcoords=used_qmcoords,
                charge=charge,
                mult=mult,
                numcores=numcores,
            )
        else:
            checkpoint = time.time()
            self.QMenergy = QMenergy + self.truncPC_E_correction
            self.QMgradient_wo_linkatoms, self.PCgradient = self.truncated_pc_gradient_update(
                QMgradient_wo_linkatoms, PCgradient
            )
            log_time_since(checkpoint, "trunc pcgrad update")

        checkpoint = time.time()
        self.make_qm_pc_gradient()  # populates self.QM_PC_gradient
        log_time_since(checkpoint, "QMpcgrad prepare")
        if self.linkatoms is True:
            checkpoint = time.time()
            self._add_linkatom_force_projection(self.QM_PC_gradient, used_qmcoords, current_coords)
            log_time_since(checkpoint, "linkatomgrad prepare")
        log_time_since(prep_start, "QM/MM gradient prepare")

    def _recalculate_truncated_pc_correction(
        self, *, QMenergy, QMgradient, PCgradient, QMgradient_wo_linkatoms, used_qmcoords, charge, mult, numcores
    ):
        """Run the full point-charge field once to calibrate the truncated-PC energy and gradient."""
        full_start = time.time()
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
        log_time_since(full_start, "trunc-pc full calculation")

        self.truncPC_E_correction = QMenergy_full - QMenergy
        logger.info(f"Truncated PC energy correction: {self.truncPC_E_correction} Eh")
        self.QMenergy = QMenergy + self.truncPC_E_correction

        checkpoint = time.time()
        self.calculate_trunc_pc_gradient_correction(QMgradient_full, PCgradient_full, QMgradient, PCgradient)
        log_time_since(checkpoint, "calculate_truncPC_gradient_correction")

        checkpoint = time.time()
        self.QMgradient_wo_linkatoms, self.PCgradient = self.truncated_pc_gradient_update(
            QMgradient_wo_linkatoms, PCgradient
        )
        log_time_since(checkpoint, "truncPC_gradient update ")
        log_time_since(full_start, "trunc-full-step pcgrad update")

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
        """Run electrostatic embedding: the QM region sees the MM charges as point charges."""
        module_init_time = time.time()
        CheckpointTime = time.time()

        used_mmcoords, used_qmcoords = self._prepare_run(current_coords, "Electrostatic")

        if self.chargeboundary_method == "shift" and self.dipole_correction is True:
            self.set_dipole_charges(current_coords)  # Note: running again
            self.pointchargecoords = np.append(used_mmcoords, np.array(self.dipole_coords), axis=0)
        elif self.chargeboundary_method == "rcd":
            # Appends RCD chargepositions to MM-coords
            self.pointchargecoords = self.rcd_shifting_update(used_mmcoords, current_coords)
        else:
            self.pointchargecoords = used_mmcoords

        # TRUNCATED PC Option: Speeding up QM/MM jobs of large systems by passing only a truncated PC field to the
        # QM-code most of the time
        # Speeds up QM-pointcharge gradient that otherwise dominates
        if self.truncated_pc is True:
            self.truncated_pc_function(used_qmcoords)

            # Modifies self.pointcharges and self.pointchargecoords

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        logger.info("Number of pointcharges (to QM program): %s", len(self.pointcharges))
        logger.info("Number of charge coordinates: %s", len(self.pointchargecoords))
        # The QM code pairs charges with coordinates positionally: a mismatch here is silently
        # wrong physics rather than an error, so check it before handing the field over.
        if len(self.pointcharges) != len(self.pointchargecoords):
            raise InternalError(
                f"Point-charge field is inconsistent: {len(self.pointcharges)} charges but "
                f"{len(self.pointchargecoords)} coordinates (chargeboundary_method={self.chargeboundary_method}, "
                f"dipole_correction={self.dipole_correction})"
            )
        logger.info(f"Running QM/MM object with {numcores} cores available")
        log_time_since(module_init_time, "before-QMstep")
        CheckpointTime = time.time()
        if self.qm_theory_name in {"None", "ZeroTheory"}:
            logger.info("No QMtheory. Skipping QM calc")
            QMenergy = 0.0
            self.linkatoms = False
            # Per-atom zero gradients, matching the shapes a real QM code would return
            # (mech_run does the same). A flat (3,) array breaks make_qm_pc_gradient.
            PCgradient = np.zeros((len(self.pointchargecoords), 3))
            QMgradient = np.zeros((len(used_qmcoords), 3))
        elif grad is True:
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
            self._prepare_qm_pc_gradient(
                QMenergy=QMenergy,
                QMgradient=QMgradient,
                PCgradient=PCgradient,
                used_qmcoords=used_qmcoords,
                current_coords=current_coords,
                charge=charge,
                mult=mult,
                numcores=numcores,
            )
        else:
            self.QMenergy = QMenergy

        if self.mm_theory_name == "OpenMMTheory":
            logger.info("Using OpenMM theory as part of QM/MM.")
            if self.QMChargesZeroed:
                logger.info(f"Using MM on full system. Charges for QM region {self.qmatoms} have been set to zero ")
            else:
                raise InternalError("QMCharges have not been zeroed")
            if grad is True:
                CheckpointTime = time.time()

                if self.openmm_externalforce is True:
                    logger.info("OpenMM externalforce is True")
                    # Calculate energy associated with external force so that we can subtract it later
                    scaled_current_coords = current_coords * openmmqmmm.constants.ANG_TO_BOHR
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
                "Note: You are using electrostatic embedding. This means that the QM-energy is actually the polarized "
                "QM-energy"
            )
            logger.info("Note: MM energy also contains the QM-MM Lennard-Jones interaction\n")
        energywarning = ""
        if self.truncated_pc is True:
            logger.warning(
                "Warning: Truncated PC approximation is active. This means that QM and QM/MM energies are approximate."
            )
            energywarning = "(approximate)"

        logger.info("%s", "{:<20} {:>20.12f} {}".format("QM energy: ", self.QMenergy, energywarning))
        logger.info("%s", "{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
        logger.info("%s", "{:<20} {:>20.12f} {}".format("QM/MM energy: ", self.QM_MM_energy, energywarning))
        logger.info("")

        if grad is True:
            # If OpenMM external force method then QM/MM gradient is already complete
            # NOTE: Not possible anymore
            if self.openmm_externalforce is True:
                pass
            else:
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
            logger.info("------------ENDING QM/MM MODULE-------------")
            log_time_since(module_init_time, "QM/MM run")
            return self.QM_MM_energy, self.QM_MM_gradient
        log_time_since(module_init_time, "QM/MM run")
        return self.QM_MM_energy


def fullindex_to_qmindex(fullindex, qmatoms):
    return qmatoms.index(fullindex)


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
                if len(resids) == 0 or resid_part == actual_resids[-1]:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                else:
                    indexcount += 1
                    resids.append(indexcount)
                    actual_resids.append(resid_part)

    return resids


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
                resid_part = int(line.split()[2])
                if len(resids) == 0 or resid_part == actual_resids[-1]:
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
                else:
                    indexcount += 1
                    resids.append(indexcount)
                    actual_resids.append(resid_part)
    return resids


# Read atomic charges present in PSF-file. assuming Xplor format
def read_charges_from_psf(file) -> list[float]:
    """Read atom charges from a CHARMM PSF file."""
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


# Requires fragment (for coordinates) and residue information from either:
# 1. resids list inside OpenMMTheory object
# 2. residues taken from PDB-file
# 3. residues taken from PSF-file


def define_active_region(
    pdbfile=None, mmtheory=None, psffile=None, fragment=None, radius=None, originatom=None
) -> list[int]:
    """Define an active region as all whole residues within a distance of a central atom."""
    logger.info(main_header("ActregionDefine"))

    if radius is None or originatom is None:
        raise InputError("actregiondefine requires radius and originatom keyword arguments")
    if pdbfile is None and fragment is None:
        raise InputError("actregiondefine requires either fragment or pdbfile arguments (for coordinates)")
    if pdbfile is None and mmtheory is None and psffile is None:
        raise InputError(
            "actregiondefine requires either pdbfile, psffile or mmtheory arguments (for residue topology information)"
        )

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
        resids = mmtheory.resids
    elif psffile is not None:
        logger.info("PSF-file provided. Using residue information")
        resids = grab_resids_from_psffile(psffile)
    else:
        logger.info("PDB-file provided. Using residue information")
        resids = grab_resids_from_pdbfile(pdbfile)

    origincoords = fragment.coords[originatom]
    logger.info("Origin-atom coordinates: %s", origincoords)
    act_indices = []
    for index, allc in enumerate(fragment.coords):
        dist = openmmqmmm.coords.distance(origincoords, allc)
        if dist < radius:
            resid_value = resids[index]
            resid_members = [i for i, x in enumerate(resids) if x == resid_value]
            for k in resid_members:
                if k not in act_indices:
                    act_indices.append(k)

    logger.info("act_indices: %s", act_indices)
    act_indices = np.unique(act_indices).tolist()

    writelisttofile(act_indices, "active_atoms")
    logger.info("Active region size: %s", len(act_indices))
    logger.info("Active-region indices written to file: active_atoms")
    logger.info(
        "The active_atoms list  can be read-into Python script like this:	 actatoms = "
        'read_intlist_from_file("active_atoms")'
    )
    openmmqmmm.coords.write_xyz_for_atoms(fragment.coords, fragment.elems, act_indices, "ActiveRegion")
    logger.info("Wrote Active region XYZfile: ActiveRegion.xyz  (inspect with visualization program)")
    return act_indices


# This projects the linkatom force onto the respective QM atom and MM atom
def linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad):
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord) * openmmqmmm.constants.ANG_TO_BOHR
    MQdistance = openmmqmmm.coords.distance(Mcoord, Qcoord) * openmmqmmm.constants.ANG_TO_BOHR
    # Coords in Bohr
    Mcoord = Mcoord * openmmqmmm.constants.ANG_TO_BOHR
    Qcoord = Qcoord * openmmqmmm.constants.ANG_TO_BOHR
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

    g_x = float(C[0, 0] * Lgrad[0] + C[0, 1] * Lgrad[1] + C[0, 2] * Lgrad[2])
    g_y = float(C[1, 0] * Lgrad[0] + C[1, 1] * Lgrad[1] + C[1, 2] * Lgrad[2])
    g_z = float(C[2, 0] * Lgrad[0] + C[2, 1] * Lgrad[1] + C[2, 2] * Lgrad[2])

    gg_x = float(B[0, 0] * Lgrad[0] + B[0, 1] * Lgrad[1] + B[0, 2] * Lgrad[2])
    gg_y = float(B[1, 0] * Lgrad[0] + B[1, 1] * Lgrad[1] + B[1, 2] * Lgrad[2])
    gg_z = float(B[2, 0] * Lgrad[0] + B[2, 1] * Lgrad[1] + B[2, 2] * Lgrad[2])

    return [g_x, g_y, g_z], [gg_x, gg_y, gg_z]


# Should be what ORCA uses
def linkatom_force_lever(Qcoord, Mcoord, Lcoord, Lgrad):
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord)
    MQdistance = openmmqmmm.coords.distance(Mcoord, Qcoord)
    scal = QLdistance / MQdistance
    gradMM = Lgrad * scal
    gradQM = Lgrad * (1.0 - scal)
    return gradQM, gradMM


# Simplistic; selected with linkatom_forceproj_method="chain"
def linkatom_force_chainrule(Qcoord, Mcoord, Lcoord, Lgrad):
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord) * openmmqmmm.constants.ANG_TO_BOHR
    vec = (Mcoord - Qcoord) * openmmqmmm.constants.ANG_TO_BOHR
    R2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
    oneR = 1.0 / math.sqrt(R2)
    lnk_dis_oneR = QLdistance * oneR
    vec = vec * oneR
    dotprod = Lgrad[0] * (-1) * vec[0] + Lgrad[1] * (-1) * vec[1] + Lgrad[2] * (-1) * vec[2]
    forcemod = np.zeros(3)
    forcemod[0] = lnk_dis_oneR * (-1 * Lgrad[0] - (dotprod * vec[0]))
    forcemod[1] = lnk_dis_oneR * (-1 * Lgrad[1] - (dotprod * vec[1]))
    forcemod[2] = lnk_dis_oneR * (-1 * Lgrad[2] - (dotprod * vec[2]))
    # subtract from QM1,  add to MM1
    return -1 * forcemod, forcemod


def compute_decomposed_qm_mm_energy(fragment=None, theory=None) -> None:
    """Decompose a QM/MM single-point energy into QM, MM and coupling terms."""
    logger.info(main_header("Decomposed QM/MM Energy Calculation"))

    if isinstance(theory, QMMMTheory) is False:
        raise InputError("Please provide a QMMMTheory object as theory.")
    if theory.qm_charge is None or theory.qm_mult is None:
        raise InputError("Please define qm_charge and qm_mult attributes in the QMMMtheory object")

    result = openmmqmmm.single_point(theory=theory, fragment=fragment)

    E_QM_MM_tot = result.energy
    E_QM_pol = result.qm_energy
    E_MM_mod = result.mm_energy

    theory.mm_theory.update_lj_epsilons(theory.qmatoms, [0.0 for i in theory.qmatoms])
    result_MM_mod2 = openmmqmmm.single_point(theory=theory.mm_theory, fragment=fragment, charge=0, mult=1)
    E_QM_MM_vdw = E_MM_mod - result_MM_mod2.energy

    logger.warning("QM-MM bonded term not implemented yet. Setting to zero.")
    logger.info("This means that the MM term still contains the QM-MM bonded contribution")
    E_QM_MM_bond = 0.0

    E_MM_pure = result_MM_mod2.energy

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

    result_mech = openmmqmmm.single_point(theory=QM_MM_mech, fragment=fragment)
    E_QM_pure = result_mech.qm_energy
    E_QM_MM_elstat = E_QM_pol - E_QM_pure

    E_coupling = E_QM_MM_elstat + E_QM_MM_vdw + E_QM_MM_bond

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
