from __future__ import annotations

import contextlib
import copy
import logging
import math
import time
from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
import openmm.unit

import openmmqmmm.constants
import openmmqmmm.coords
from openmmqmmm.coords import CONNECTIVITY_SCALE, CONNECTIVITY_TOL, Fragment
from openmmqmmm.exceptions import (
    InputError,
    InternalError,
)
from openmmqmmm.periodic_embedding import PeriodicQMGeometry
from openmmqmmm.utils import log_time_since, main_header, write_list_to_file

logger = logging.getLogger(__name__)

_DIPOLE_SHIFT_ANGSTROM = 0.15
_DIPOLE_REFERENCE_DISTANCE_ANGSTROM = 2.5
_DIPOLE_POSITION_SCALE = _DIPOLE_SHIFT_ANGSTROM / _DIPOLE_REFERENCE_DISTANCE_ANGSTROM

# Required at init: qm_theory and qmatoms and fragment


class QMMMTheory:
    """Electrostatically embedded QM/MM combining a QM theory with an MM theory."""

    def __init__(
        self,
        *,
        qm_theory: Any | None = None,
        qmatoms: Sequence[int] | None = None,
        fragment: Fragment | None = None,
        mm_theory: Any | None = None,
        charges: Sequence[float] | None = None,
        embedding: str = "elstat",
        numcores: int = 1,
        label: str = "QM/MM",
        excludeboundaryatomlist: Sequence[int] | None = None,
        unusualboundary: bool = False,
        openmm_externalforce: bool = False,
        truncated_pc: bool = False,
        truncated_pc_radius: float = 55,
        truncated_pc_recalc_iter: int = 50,
        qm_charge: int | None = None,
        qm_mult: int | None = None,
        chargeboundary_method: str = "shift",
        exit_after_customexternalforce_update: bool = False,
        dipole_correction: bool = True,
        linkatom_method: str = "simple",
        linkatom_simple_distance: float | None = None,
        linkatom_forceproj_method: str | None = "adv",
        linkatom_ratio: float = 0.723,
        linkatom_type: str = "H",
        update_qm_region_charges: bool = False,
    ) -> None:
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
        # Historical projection names are aliases for the exact derivative of
        # the selected placement rule. Only 'none' deliberately omits projection.
        if linkatom_forceproj_method is None:
            self.linkatom_forceproj_method = "none"
        elif isinstance(linkatom_forceproj_method, str):
            self.linkatom_forceproj_method = linkatom_forceproj_method.casefold()
        else:
            raise InputError("linkatom_forceproj_method must be one of: adv, lever, chain, none")
        if self.linkatom_forceproj_method not in {"adv", "lever", "chain", "none"}:
            raise InputError("linkatom_forceproj_method must be one of: adv, lever, chain, none")

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

        raw_qmatoms = list(qmatoms)
        if len(raw_qmatoms) == 0:
            raise InputError("List of qmatoms provided is empty. This is not allowed.")
        invalid_qmatoms = [
            atom
            for atom in raw_qmatoms
            if isinstance(atom, (bool, np.bool_)) or not isinstance(atom, (int, np.integer))
        ]
        if invalid_qmatoms:
            raise InputError(f"QM atom indices must be integers; got {invalid_qmatoms!r}")

        self.qmatoms = sorted({int(atom) for atom in raw_qmatoms})
        out_of_range_qmatoms = [atom for atom in self.qmatoms if atom < 0 or atom >= self.num_allatoms]
        if out_of_range_qmatoms:
            raise InputError(
                f"QM atom indices must be between 0 and {self.num_allatoms - 1}; got {out_of_range_qmatoms}"
            )

        # All-atom Bool-array for whether atom-index is a QM-atom index or not
        # Used by make_QM_PC_gradient
        self.xatom_mask = np.isin(self.allatoms, self.qmatoms)

        self.mmatoms = np.setdiff1d(self.allatoms, self.qmatoms)

        self._periodic_geometry: PeriodicQMGeometry | None = None
        self._current_periodic_box_vectors: np.ndarray | None = None
        if self.mm_theory_name == "OpenMMTheory" and getattr(self.mm_theory, "periodic", False):
            image_links = []
            for index in range(self.mm_theory.system.getNumParticles()):
                if self.mm_theory.system.isVirtualSite(index):
                    site = self.mm_theory.system.getVirtualSite(index)
                    image_links.extend((index, site.getParticle(parent)) for parent in range(site.getNumParticles()))
            self._periodic_geometry = PeriodicQMGeometry(
                self.num_allatoms,
                ((first.index, second.index) for first, second in self.mm_theory.topology.bonds()),
                self.qmatoms,
                image_links=image_links,
            )
            self.coords = self._image_periodic_coords(np.asarray(fragment.coords))

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
        if self._periodic_geometry is not None and self.embedding == "elstat":
            logger.warning(
                "Periodic QM/MM uses a finite, molecule-imaged point-charge cluster. "
                "The QM Hamiltonian has no periodic/Ewald electrostatics; check convergence with cell size."
            )
        if self.update_qm_region_charges and (self.embedding != "mech" or self.mm_theory is None):
            raise InputError(
                "update_qm_region_charges requires mechanical embedding and an MM theory whose charges can be updated"
            )
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
                # Keep the original force-field charges as QMMM-owned state. OpenMM's
                # charge list is mutated below when the QM-region charges are zeroed.
                self.charges = list(mm_theory.charges)
            else:
                raise InputError(
                    "QMMMTheory requires either a charges list or an OpenMMTheory mm_theory providing charges"
                )
        else:
            logger.info("Reading in charges")
            if len(charges) != len(fragment.atomlist):
                raise InputError("Number of charges not matching number of fragment atoms. Exiting.")
            self.charges = list(charges)

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
        # Each entry describes how one virtual point-charge coordinate depends on
        # its real host atoms: ((atom_index, derivative_weight), ...). Entries use
        # the same order as the virtual coordinates appended to pointchargecoords.
        self._virtual_site_gradient_mappings: list[tuple[tuple[int, float], ...]] = []

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

    def get_mm_boundary(self, scale: float, tol: float) -> None:
        """Find the QM-MM covalent boundary and the MM atoms bonded across it."""
        timeA = time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        self.MMboundarydict = {}
        qm_atom_set = set(self.qmatoms)
        for MM1atom in self.boundaryatoms.values():
            # Boundary values are lists in current callers; retain support for the
            # historical scalar representation while normalizing the loop here.
            mm1_atoms = MM1atom if isinstance(MM1atom, list) else [MM1atom]
            for mat in mm1_atoms:
                if mat in self.MMboundarydict:
                    continue
                periodic_geometry = getattr(self, "_periodic_geometry", None)
                connatoms = (
                    periodic_geometry.neighbors[mat]
                    if periodic_geometry is not None
                    else openmmqmmm.coords.get_connected_atoms(self.coords, self.elems, scale, tol, mat)
                )
                self.MMboundarydict[mat] = [atom for atom in connatoms if atom not in qm_atom_set]

        empty_boundaries = [mm1 for mm1, mm_neighbors in self.MMboundarydict.items() if not mm_neighbors]
        if self.embedding == "elstat" and empty_boundaries:
            raise InputError(
                "Electrostatic QM/MM charge shifting cannot redistribute charge from MM boundary atom(s) "
                f"{empty_boundaries}: they have no MM-side neighbours beyond the QM-MM bond. "
                "Expand the QM region or use mechanical embedding."
            )

        # Used by ShiftMMCharges
        self.MMboundary_indices = list(self.MMboundarydict.keys())
        self.MMboundary_counts = np.array([len(self.MMboundarydict[i]) for i in self.MMboundary_indices])

        logger.info("MM boundary (MM1:MMx pairs): %s", self.MMboundarydict)
        log_time_since(timeA, "get_MMboundary")

    def zero_qm_charges(self) -> None:
        """Set the MM charges of the QM-region atoms to zero for electrostatic embedding."""
        timeA = time.time()
        logger.info("Setting QM charges to Zero")
        self.charges_qmregionzeroed = copy.copy(self.charges)
        for i, _c in enumerate(self.charges_qmregionzeroed):
            if i in self.qmatoms:
                self.charges_qmregionzeroed[i] = 0.0
        self.QMChargesZeroed = True
        log_time_since(timeA, "ZeroQMCharges")

    def rcd_shifting_prep(self, charges_qmregionzeroed: Sequence[float]) -> tuple[np.ndarray, list[float]]:
        """Set up redistributed-charge-and-dipole (RCD) charge shifting."""
        timeA = time.time()
        logger.info("Shifting MM charges at QM/MM boundary by RCD.")
        full_pointcharges = np.asarray(charges_qmregionzeroed, dtype=float).copy()
        MM1_charges = np.asarray(self.charges, dtype=float)[self.MMboundary_indices]
        full_pointcharges[self.MMboundary_indices] = 0.0
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        RCD_additional_charges = []
        for MM2indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract, strict=True):
            for i in MM2indices:
                # RC/RCD: Instead of adding the M1 charge to the M2 atoms we create new RC/RCD sites
                RCD_additional_charges.append(float(fract * 2))
                # RCD: Reduce the MM2 charge by q0
                full_pointcharges[i] -= fract

        # Coordinates are ordered as the real MM atoms followed by the additional RCD sites;
        # keep the charge array in exactly the same order. Full-system indices must only be
        # used before this compaction to the MM region.
        pointcharges = np.concatenate(
            (full_pointcharges[self.mmatoms], np.asarray(RCD_additional_charges, dtype=float))
        )
        self.chargeshifting_done = True

        log_time_since(timeA, "RCD_shifting_prep")
        return pointcharges, RCD_additional_charges

    def rcd_shifting_update(self, used_mmcoords: np.ndarray, fullcoords: np.ndarray) -> np.ndarray:
        """Rebuild the RCD point-charge positions for the current geometry."""
        timeA = time.time()
        logger.info("Adding updated RCD charges at QM/MM boundary by RCD.")

        # One RCD site per MM2 atom, in the same order as rcd_shifting_prep created the
        # matching extra charges, so that charges and coordinates stay index-aligned.
        # New RCD site sits midway between each MM1 atom and each of its MM2 atoms
        newsites = []
        self._virtual_site_gradient_mappings = []
        for MM1index, MM2indices in self.MMboundarydict.items():
            for MM2index in MM2indices:
                newsites.append((fullcoords[MM2index] + fullcoords[MM1index]) / 2)
                self._virtual_site_gradient_mappings.append(((MM1index, 0.5), (MM2index, 0.5)))

        pointchargecoords = np.append(used_mmcoords, np.array(newsites), axis=0) if newsites else used_mmcoords

        log_time_since(timeA, "RCD_shifting_update")
        return pointchargecoords

    def shift_mm_charges(self) -> None:
        """Shift the MM1 boundary charges onto their MM2 neighbours."""
        if self.chargeshifting_done is False:
            self._shift_mm_charges_impl()
        else:
            logger.info("Charge shifting already done. Using previous charges")

    def _shift_mm_charges_impl(self) -> None:
        timeA = time.time()
        logger.info("new. Shifting MM charges at QM-MM boundary.")

        log_time_since(timeA, "x0")
        self.pointcharges = np.asarray(self.charges_qmregionzeroed, dtype=float).copy()
        original_charges = np.asarray(self.charges, dtype=float)

        log_time_since(timeA, "x1")
        MM1_charges = original_charges[self.MMboundary_indices]
        self.pointcharges[self.MMboundary_indices] = 0.0

        MM1charge_fract = MM1_charges / self.MMboundary_counts

        for indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract, strict=False):
            self.pointcharges[[indices]] += fract

        self.chargeshifting_done = True
        log_time_since(timeA, "ShiftMMCharges-new2")

    def get_dipole_charge(
        self,
        delq: float,
        direction: int,
        mm1index: int,
        mm2index: int,
        current_coords: np.ndarray,
    ) -> tuple[float, list[float]]:
        """Return the two charges and positions of a dipole placed on an MM1-MM2 bond."""
        mm1coords = np.array(current_coords[mm1index])
        mm2coords = np.array(current_coords[mm2index])
        MM_distance = openmmqmmm.coords.distance(mm1coords, mm2coords)  # Distance between MM1 and MM2

        def vnorm(p1: np.ndarray) -> np.ndarray:
            r = math.sqrt((p1[0] * p1[0]) + (p1[1] * p1[1]) + (p1[2] * p1[2]))
            return np.array([p1[0] / r, p1[1] / r, p1[2] / r])

        diffvector = mm2coords - mm1coords
        normdiffvector = vnorm(diffvector)

        d = delq * _DIPOLE_REFERENCE_DISTANCE_ANGSTROM
        q0 = 0.5 * d / _DIPOLE_SHIFT_ANGSTROM
        shift = direction * _DIPOLE_SHIFT_ANGSTROM * (MM_distance / _DIPOLE_REFERENCE_DISTANCE_ANGSTROM)
        pos = mm2coords + np.array(shift * normdiffvector)
        return -q0 * direction, list(pos)

    def set_dipole_charges(self, current_coords: np.ndarray) -> None:
        """Rebuild the dipole-correction point charges for the current geometry."""
        checkpoint = time.time()
        logger.info("Adding extra charges to preserve dipole moment for charge-shifting")
        logger.info("MMboundarydict: %s", self.MMboundarydict)
        self.dipole_charges = []
        self.dipole_coords = []
        self._virtual_site_gradient_mappings = []

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
                self._virtual_site_gradient_mappings.append(
                    ((MM1, -_DIPOLE_POSITION_SCALE), (MM, 1.0 + _DIPOLE_POSITION_SCALE))
                )
                self._virtual_site_gradient_mappings.append(
                    ((MM1, _DIPOLE_POSITION_SCALE), (MM, 1.0 - _DIPOLE_POSITION_SCALE))
                )
        log_time_since(checkpoint, "SetDipoleCharges")

    # Uses a precalculated mask; this dominates QM/MM gradient prepare.
    def make_qm_pc_gradient(self) -> None:
        """Assemble the full-system gradient from the QM and point-charge gradients."""
        pc_gradient = np.asarray(self.PCgradient)
        num_real_mm_atoms = len(self.mmatoms)
        expected_pc_rows = num_real_mm_atoms + len(self._virtual_site_gradient_mappings)
        if pc_gradient.shape != (expected_pc_rows, 3):
            raise InternalError(
                "Point-charge gradient has shape "
                f"{pc_gradient.shape}; expected ({expected_pc_rows}, 3) for {num_real_mm_atoms} real MM atoms "
                f"and {len(self._virtual_site_gradient_mappings)} virtual sites"
            )

        self.QM_PC_gradient[self.xatom_mask] = self.QMgradient_wo_linkatoms
        self.QM_PC_gradient[~self.xatom_mask] = pc_gradient[:num_real_mm_atoms]

        virtual_gradients = pc_gradient[num_real_mm_atoms:]
        for site_gradient, host_mappings in zip(virtual_gradients, self._virtual_site_gradient_mappings, strict=True):
            for atom_index, derivative_weight in host_mappings:
                self.QM_PC_gradient[atom_index] += derivative_weight * site_gradient

    def truncated_pc_function(self, used_qmcoords: np.ndarray, *, require_gradients: bool) -> None:
        """Reduce the point-charge field to the atoms near the QM region."""
        self.truncated_pc_calls += 1
        logger.info("TruncatedPC approximation!")
        energy_correction_missing = not hasattr(self, "truncPC_E_correction")
        gradient_correction_missing = require_gradients and not all(
            hasattr(self, name) for name in ("original_QMcorrection_gradient", "original_PCcorrection_gradient")
        )
        scheduled_recalculation = (
            self.truncated_pc_calls == 1 or self.truncated_pc_calls % self.truncated_pc_recalc_iter == 0
        )
        if scheduled_recalculation or energy_correction_missing or gradient_correction_missing:
            self.truncated_pc_recalc_flag = True
            logger.debug(
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
    def determine_truncated_pc_indices(self, origincoords: Sequence[float] | np.ndarray) -> None:
        """Select the point charges within truncated_pc_radius of the QM region."""
        region_indices = []
        for index, allc in enumerate(self.pointchargecoords):
            dist = openmmqmmm.coords.distance(origincoords, allc)
            if dist < self.truncated_pc_radius:
                region_indices.append(index)
        self.truncated_PC_region_indices = np.unique(region_indices).tolist()

    def calculate_trunc_pc_gradient_correction(
        self,
        QMgradient_full: np.ndarray,
        PCgradient_full: np.ndarray,
        QMgradient_trunc: np.ndarray,
        PCgradient_trunc: np.ndarray,
    ) -> None:
        """Compute the QM and point-charge gradient corrections for PC truncation and store them."""
        # Correct link-atom rows as well: they are projected onto the real
        # boundary atoms only after the full QM gradient has been restored.
        self.original_QMcorrection_gradient = QMgradient_full - QMgradient_trunc
        truncated_indices = np.asarray(self.truncated_PC_region_indices, dtype=int)
        pc_difference = np.zeros((len(PCgradient_full), 3))
        pc_difference[truncated_indices] = PCgradient_full[truncated_indices] - PCgradient_trunc
        pc_difference[~np.isin(np.arange(len(PCgradient_full)), truncated_indices)] = PCgradient_full[
            ~np.isin(np.arange(len(PCgradient_full)), truncated_indices)
        ]
        self.original_PCcorrection_gradient = pc_difference

    def truncated_pc_gradient_update(
        self, QMgradient: np.ndarray, PCgradient: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the stored truncation correction to this step's gradients."""
        newQMgradient = QMgradient + self.original_QMcorrection_gradient

        new_full_PC_gradient = np.copy(self.original_PCcorrection_gradient)
        new_full_PC_gradient[self.truncated_PC_region_indices] += PCgradient

        return newQMgradient, new_full_PC_gradient

    def set_numcores(self, numcores: int) -> None:
        """Set the core count used by both the QM and MM theories."""
        logger.debug("Setting %s cores for the QM and MM theories", numcores)
        self.numcores = numcores
        self.qm_theory.set_numcores(numcores)
        if self.mm_theory is not None:
            self.mm_theory.set_numcores(numcores)

    def _delegate_to_qm_theory(self, name: str, description: str) -> Any:
        """Call the QM theory's own accessor of that name, or return None when it has none."""
        logger.debug("Getting %s from QM part of QM/MM theory", description)
        getter = getattr(self.qm_theory, name, None)
        if getter is None:
            logger.debug("QM theory does not provide a %s", description)
            return None
        return getter()

    def get_dipole_moment(self) -> Sequence[float] | np.ndarray | None:
        """Return the QM theory's dipole moment, or None if it does not provide one."""
        return self._delegate_to_qm_theory("get_dipole_moment", "dipole moment")

    def get_polarizability_tensor(self) -> np.ndarray | None:
        """Return the QM theory's polarizability tensor, or None if it does not provide one."""
        return self._delegate_to_qm_theory("get_polarizability_tensor", "polarizability tensor")

    def resolve_qm_charge_mult(self, *, charge: int | None = None, mult: int | None = None) -> tuple[int, int]:
        """Resolve the charge and multiplicity of the QM region."""
        return (
            self._resolve_qm_electronic_state("charge", charge),
            self._resolve_qm_electronic_state("mult", mult),
        )

    def _resolve_qm_electronic_state(self, name: str, supplied_value: int | None) -> int:
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

    def _log_qm_charge_consistency(self, charge: int) -> None:
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

    def _setup_mm_theory(self, fragment: Fragment) -> None:
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
        if self._periodic_geometry is not None:
            # The force-field topology remains authoritative across box faces;
            # a distance search on arbitrarily wrapped input misses these bonds.
            self.boundaryatoms = self._periodic_boundary_atoms()
        else:
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
            logger.debug("No covalent QM-MM boundary. Linkatoms and dipole_correction options set to False")
            self.linkatoms = False
            self.dipole_correction = False

        if self.mm_theory_name == "OpenMMTheory":
            # Only applies when running OpenMM_Opt or OpenMM_MD, and to both embeddings.
            # NonbondedTheory never defines these terms in the first place.
            self.mm_theory.remove_constraints_for_atoms(self.qmatoms)
            logger.debug("Removing bonded terms for QM-region in MMtheory")
            self.mm_theory.modify_bonded_forces(self.qmatoms)
            # Exceptions make OpenMM ignore QM-QM Coulomb and LJ. QM-MM elstat Coulomb
            # charges are zeroed separately below.
            logger.debug("Removing nonbonded terms for QM-region in MMtheory (QM-QM interactions)")
            self.mm_theory.addexceptions(self.qmatoms)

        embedding = self.embedding.lower()
        if embedding == "elstat":
            logger.info("Charges of QM atoms set to 0 (since Electrostatic Embedding):")
            self.zero_qm_charges()
            self.mm_theory.update_charges(self.qmatoms, [0.0 for _ in self.qmatoms])
            if self.mm_theory_name == "OpenMMTheory":
                self.mm_theory.delete_exceptions(self.qmatoms)
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

    def _periodic_boundary_atoms(self) -> dict[int, list[int]]:
        excluded = set(() if self.excludeboundaryatomlist is None else self.excludeboundaryatomlist)
        qm_atoms = set(self.qmatoms)
        boundary = {}
        for atom in self.qmatoms:
            if atom in excluded:
                continue
            neighbors = [other for other in self._periodic_geometry.neighbors[atom] if other not in qm_atoms]
            if not neighbors:
                continue
            if not self.unusualboundary and any(self.elems[index] != "C" for index in [atom, *neighbors]):
                raise InputError(
                    f"QM-MM boundary at atom {atom} is not a C-C bond; use unusualboundary=True to accept this cut"
                )
            boundary[atom] = neighbors
        return boundary

    def _image_periodic_coords(
        self, current_coords: np.ndarray, periodic_box_vectors: np.ndarray | None = None
    ) -> np.ndarray:
        if self._periodic_geometry is None:
            if periodic_box_vectors is not None:
                raise InputError("periodic_box_vectors requires a periodic OpenMM QM/MM system")
            self._current_periodic_box_vectors = None
            return np.asarray(current_coords)
        if periodic_box_vectors is None:
            periodic_box_vectors = openmm.unit.Quantity(
                self.mm_theory.system.getDefaultPeriodicBoxVectors()
            ).value_in_unit(openmm.unit.angstrom)
        self._current_periodic_box_vectors = np.asarray(periodic_box_vectors, dtype=float)
        return self._periodic_geometry.image(current_coords, self._current_periodic_box_vectors)

    def _log_region_charges(self) -> None:
        """Log the per-atom charge each region carries, at DEBUG."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        charges = self.charges_qmregionzeroed if self.embedding.lower() == "elstat" else self.charges
        for i in self.allatoms:
            region = "QM" if i in self.qmatoms else "MM"
            logger.debug("%s atom %s (%s) charge: %s", region, i, self.elems[i], charges[i])

    def run(
        self,
        *,
        current_coords: np.ndarray | None = None,
        elems: Sequence[str] | None = None,
        grad: bool = False,
        numcores: int = 1,
        exit_after_customexternalforce_update: bool = False,
        label: str | None = None,
        charge: int | None = None,
        mult: int | None = None,
        current_mm_coords: np.ndarray | None = None,
        mm_charges: Sequence[float] | None = None,
        qm_elems: Sequence[str] | None = None,
        pc: bool | None = None,
        periodic_box_vectors: np.ndarray | None = None,
    ) -> float | tuple[float, np.ndarray]:
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
            current_coords=self._image_periodic_coords(current_coords, periodic_box_vectors),
            elems=elems,
            grad=grad,
            numcores=numcores,
            exit_after_customexternalforce_update=exit_after_customexternalforce_update,
            label=label,
            charge=charge,
            mult=mult,
        )

    def run_openmm_python_force(
        self,
        *,
        current_coords: np.ndarray,
        elems: Sequence[str],
        charge: int,
        mult: int,
        periodic_box_vectors: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray]:
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
            periodic_box_vectors=periodic_box_vectors,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise InternalError("QM/MM force evaluation must return an (energy, gradient) pair.")
        _legacy_external_energy, gradient = result
        return self.QMenergy, gradient

    def _prepare_run(self, current_coords: np.ndarray, embedding_label: str) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Embedding: %s", embedding_label)

        # Only do once to avoid cost in each step
        if self.runcalls == 0:
            logger.debug("First QMMMTheory run. Running runprep")
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

    def _add_linkatom_force_projection(
        self, gradient: np.ndarray, used_qmcoords: np.ndarray, current_coords: np.ndarray
    ) -> None:
        """Apply the exact placement Jacobian, retaining historical method aliases."""
        for pair, linkatomindex in zip(sorted(self.linkatoms_dict), self.linkatom_indices, strict=True):
            Lgrad = self.QMgradient[linkatomindex]
            Lcoord = self.linkatoms_dict[pair]
            fullatomindex_qm = pair[0]
            qmatomindex = _fullindex_to_qmindex(fullatomindex_qm, self.qmatoms)
            Qcoord = used_qmcoords[qmatomindex]
            fullatomindex_mm = pair[1]
            Mcoord = current_coords[fullatomindex_mm]

            if self.linkatom_forceproj_method == "none":
                QM1grad_contrib = np.zeros(3)
                MM1grad_contrib = np.zeros(3)
            elif self.linkatom_method == "ratio":
                # L = (1-r)Q + rM, so its Jacobians are (1-r)I and rI.
                # Use the signed input ratio, rather than recovering its
                # magnitude from a distance ratio.
                QM1grad_contrib = (1.0 - self.linkatom_ratio) * Lgrad
                MM1grad_contrib = self.linkatom_ratio * Lgrad
            elif self.linkatom_method == "simple":
                QM1grad_contrib, MM1grad_contrib = _linkatom_force_adv(Qcoord, Mcoord, Lcoord, Lgrad)
            else:
                raise InputError("Unknown linkatom_method. Exiting")

            gradient[fullatomindex_qm] += QM1grad_contrib
            gradient[fullatomindex_mm] += MM1grad_contrib

    def _resolve_numcores(self, numcores: int) -> int:
        """Fall back to the theory's own core count when run() was not given one."""
        return self.numcores if numcores == 1 else numcores

    def _compute_extforce_energy(self, current_coords: np.ndarray, gradient: np.ndarray, checkpoint: float) -> None:
        """Energy of the OpenMM external force, subtracted from the QM/MM energy later."""
        logger.info("OpenMM externalforce is True")
        scaled_current_coords = current_coords * openmmqmmm.constants.ANG_TO_BOHR
        self.extforce_energy = 3 * np.mean(np.sum(gradient * scaled_current_coords, axis=0))
        logger.info(f"Extforce energy: {self.extforce_energy}")
        log_time_since(checkpoint, "extforce prepare")

    def _run_mm_theory(self, current_coords: np.ndarray, *, grad: bool) -> None:
        """Run the MM theory over the full system, or zero its contribution when there is none."""
        if self.mm_theory_name != "OpenMMTheory":
            self.MMenergy = 0.0
            if grad:
                self.MMgradient = np.zeros((len(current_coords), 3))
        elif grad:
            self.MMenergy, self.MMgradient = self.mm_theory.run(
                current_coords=current_coords,
                qmatoms=self.qmatoms,
                grad=True,
                periodic_box_vectors=self._current_periodic_box_vectors,
            )
        else:
            logger.info("QM/MM Grad is false")
            self.MMenergy = self.mm_theory.run(
                current_coords=current_coords,
                qmatoms=self.qmatoms,
                periodic_box_vectors=self._current_periodic_box_vectors,
            )

    def _write_gradient_debug_files(
        self,
        label: str | None,
        entries: Sequence[tuple[np.ndarray, Sequence[str], Sequence[int], str, str]],
    ) -> None:
        """Write each named gradient to its own file."""
        for gradient, elems, indices, stem, description in entries:
            openmmqmmm.coords.write_coords_all(
                gradient,
                elems,
                indices=indices,
                file=f"{stem}_{label}",
                description=f"{description} {label} (au/Bohr):",
            )

    def mech_run(
        self,
        current_coords: np.ndarray | None = None,
        elems: Sequence[str] | None = None,
        grad: bool = False,
        numcores: int = 1,
        exit_after_customexternalforce_update: bool = False,
        label: str | None = None,
        charge: int | None = None,
        mult: int | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """Run mechanical embedding: QM and MM energies added with no electrostatic coupling."""
        module_init_time = time.time()
        CheckpointTime = time.time()
        _used_mmcoords, used_qmcoords = self._prepare_run(current_coords, "Mechanical")

        numcores = self._resolve_numcores(numcores)

        logger.debug("Running QM/MM with %s cores available", numcores)

        log_time_since(module_init_time, "before-QMstep")
        CheckpointTime = time.time()
        if self.qm_theory_name in {"None", "ZeroTheory"}:
            logger.debug("No QMtheory. Skipping QM calc")
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Defined charges of QM region:")
            for i in self.qmatoms:
                logger.debug("QM atom %s has charge: %s", i, self.charges[i])

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
                    self._compute_extforce_energy(current_coords, self.QM_MM_gradient, CheckpointTime)
                    # NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update is True:
                        logger.info("OpenMM custom external force updated. Exit requested")
                        # This is used if OpenMM MD is handling forces and dynamics
                        return self.QMenergy, self.QM_MM_gradient
        self._run_mm_theory(current_coords, grad=grad)
        log_time_since(CheckpointTime, "MM step")
        CheckpointTime = time.time()

        if grad:
            if len(self.QM_MM_gradient) != len(self.MMgradient):
                raise InternalError("QM/MM gradient and MM gradient size mismatch")
            self.QM_MM_gradient = self.QM_MM_gradient + self.MMgradient

        self.QM_MM_energy = self.QMenergy + self.MMenergy - self.subtractive_correction_E

        self.QM_MM_gradient -= self.subtractive_correction_G

        logger.info("%s", "{:<20} {:>20.12f}".format("QM energy: ", self.QMenergy))
        logger.info("%s", "{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
        logger.info("%s", "{:<20} {:>20.12f}".format("Subtractive correction energy: ", self.subtractive_correction_E))
        logger.info("%s", "{:<20} {:>20.12f}".format("QM/MM energy: ", self.QM_MM_energy))

        if grad is True:
            if logger.isEnabledFor(logging.DEBUG):
                self._write_gradient_debug_files(
                    label,
                    [
                        (
                            self.QMgradient_wo_linkatoms,
                            self.qmelems,
                            self.qmatoms,
                            "QMgradient-without-linkatoms",
                            "QM gradient w/o linkatoms",
                        ),
                        (self.MMgradient, self.elems, self.allatoms, "MMgradient", "MM gradient"),
                        (self.QM_MM_gradient, self.elems, self.allatoms, "QM_MMgradient", "QM/MM gradient"),
                    ],
                )
            logger.info("------------ENDING QM/MM MODULE-------------")
            log_time_since(module_init_time, "QM/MM mech run")
            return self.QM_MM_energy, self.QM_MM_gradient
        log_time_since(module_init_time, "QM/MM mech run")
        return self.QM_MM_energy

    def create_linkatoms(self, current_coords: np.ndarray) -> list[np.ndarray | list[float]]:
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
        logger.debug("Adding linkatom positions to QM coords")
        self.linkatom_indices = [len(self.qmatoms) + i for i in range(len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        log_time_since(checkpoint, "create_linkatoms")
        return linkatoms_coords

    def runprep(self, current_coords: np.ndarray) -> None:
        """Do the one-off setup the first run needs: link atoms, boundary and charges."""
        logger.debug("Preparing QM/MM run")
        init_time_runprep = time.time()

        self.qmelems = [self.elems[i] for i in self.qmatoms]
        self.mmelems = [self.elems[i] for i in self.mmatoms]

        check_before_linkatoms = time.time()
        if self.linkatoms is True:
            self.create_linkatoms(current_coords)
            self.current_qmelems = self.qmelems + [self.linkatom_type] * self.num_linkatoms
            logger.info("Number of MM atoms: %s", len(self.mmatoms))
            logger.debug("There are %s link atoms", self.num_linkatoms)
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
            logger.debug("No qmatoms list provided. Setting QMtheory to None")
            self.qm_theory_name = "None"
            self.QMenergy = 0.0

        # For truncatedPC option.
        self.pointcharges_original = copy.copy(self.pointcharges)

        if self.embedding.lower() == "elstat" or self.embedding.lower() == "polembed_drude":
            self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))

        log_time_since(init_time_runprep, "runprep")

    def _qm_gradient_without_linkatoms(self, QMgradient: np.ndarray) -> np.ndarray:
        """Drop the link-atom rows, which the QM program appends after the real QM atoms."""
        self.QMgradient = QMgradient
        return QMgradient[0 : -self.num_linkatoms] if self.linkatoms else QMgradient

    def _prepare_qm_pc_gradient(
        self,
        *,
        QMenergy: float,
        QMgradient: np.ndarray,
        PCgradient: np.ndarray,
        used_qmcoords: np.ndarray,
        current_coords: np.ndarray,
        charge: int,
        mult: int,
        numcores: int,
    ) -> None:
        """Turn this step's QM and point-charge gradients into self.QM_PC_gradient."""
        prep_start = time.time()

        if self.truncated_pc is not True:
            self.QMenergy = QMenergy
            # No TruncPC approximation active. No change to original QM and PCgradient from QMcode
            self.QMgradient = QMgradient
            if self.embedding.lower() in {"elstat", "polembed_drude"}:
                self.PCgradient = PCgradient
        elif self.truncated_pc_recalc_flag is True:
            self._recalculate_truncated_pc_correction(
                QMenergy=QMenergy,
                QMgradient=QMgradient,
                PCgradient=PCgradient,
                used_qmcoords=used_qmcoords,
                charge=charge,
                mult=mult,
                numcores=numcores,
            )
        else:
            checkpoint = time.time()
            self.QMenergy = QMenergy + self.truncPC_E_correction
            self.QMgradient, self.PCgradient = self.truncated_pc_gradient_update(QMgradient, PCgradient)
            log_time_since(checkpoint, "trunc pcgrad update")

        self.QMgradient_wo_linkatoms = self._qm_gradient_without_linkatoms(self.QMgradient)
        checkpoint = time.time()
        self.make_qm_pc_gradient()  # populates self.QM_PC_gradient
        log_time_since(checkpoint, "QMpcgrad prepare")
        if self.linkatoms is True:
            checkpoint = time.time()
            self._add_linkatom_force_projection(self.QM_PC_gradient, used_qmcoords, current_coords)
            log_time_since(checkpoint, "linkatomgrad prepare")
        log_time_since(prep_start, "QM/MM gradient prepare")

    def _recalculate_truncated_pc_correction(
        self,
        *,
        QMenergy: float,
        QMgradient: np.ndarray,
        PCgradient: np.ndarray,
        used_qmcoords: np.ndarray,
        charge: int,
        mult: int,
        numcores: int,
    ) -> None:
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
        self.QMgradient, self.PCgradient = self.truncated_pc_gradient_update(QMgradient, PCgradient)
        log_time_since(checkpoint, "truncPC_gradient update ")
        log_time_since(full_start, "trunc-full-step pcgrad update")

    def _prepare_truncated_pc_energy(
        self,
        *,
        QMenergy: float,
        used_qmcoords: np.ndarray,
        charge: int,
        mult: int,
        numcores: int,
    ) -> None:
        """Apply or refresh the full-field correction for an energy-only call."""
        if self.truncated_pc_recalc_flag:
            if self.qm_theory_name in {"None", "ZeroTheory"}:
                full_energy = QMenergy
            else:
                full_energy = self.qm_theory.run(
                    current_coords=used_qmcoords,
                    current_mm_coords=self.pointchargecoords_full,
                    mm_charges=self.pointcharges_full,
                    qm_elems=self.current_qmelems,
                    charge=charge,
                    mult=mult,
                    grad=False,
                    pc=True,
                    numcores=numcores,
                )
            self.truncPC_E_correction = full_energy - QMenergy
            # A scheduled energy-only recalibration has no current full-field
            # gradients. Do not let a later gradient call reuse an older geometry's
            # correction merely because those attributes still exist.
            for name in ("original_QMcorrection_gradient", "original_PCcorrection_gradient"):
                with contextlib.suppress(AttributeError):
                    delattr(self, name)
        self.QMenergy = QMenergy + self.truncPC_E_correction

    def elstat_run(
        self,
        current_coords: np.ndarray | None = None,
        elems: Sequence[str] | None = None,
        grad: bool = False,
        numcores: int = 1,
        exit_after_customexternalforce_update: bool = False,
        label: str | None = None,
        charge: int | None = None,
        mult: int | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """Run electrostatic embedding: the QM region sees the MM charges as point charges."""
        module_init_time = time.time()
        CheckpointTime = time.time()

        used_mmcoords, used_qmcoords = self._prepare_run(current_coords, "Electrostatic")

        if self.linkatoms and self.chargeboundary_method == "shift" and self.dipole_correction is True:
            self.set_dipole_charges(current_coords)  # Note: running again
            self.pointchargecoords = np.append(used_mmcoords, np.array(self.dipole_coords), axis=0)
        elif self.linkatoms and self.chargeboundary_method == "rcd":
            # Appends RCD chargepositions to MM-coords
            self.pointchargecoords = self.rcd_shifting_update(used_mmcoords, current_coords)
        else:
            self._virtual_site_gradient_mappings = []
            self.pointchargecoords = used_mmcoords

        # TRUNCATED PC Option: Speeding up QM/MM jobs of large systems by passing only a truncated PC field to the
        # QM-code most of the time
        # Speeds up QM-pointcharge gradient that otherwise dominates
        if self.truncated_pc is True:
            self.truncated_pc_function(used_qmcoords, require_gradients=grad)

            # Modifies self.pointcharges and self.pointchargecoords

        numcores = self._resolve_numcores(numcores)

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
        logger.debug("Running QM/MM with %s cores available", numcores)
        log_time_since(module_init_time, "before-QMstep")
        CheckpointTime = time.time()
        if self.qm_theory_name in {"None", "ZeroTheory"}:
            logger.debug("No QMtheory. Skipping QM calc")
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
        elif self.truncated_pc:
            self._prepare_truncated_pc_energy(
                QMenergy=QMenergy,
                used_qmcoords=used_qmcoords,
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
                    self._compute_extforce_energy(current_coords, self.QM_PC_gradient, CheckpointTime)
                    # NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update is True:
                        logger.info("OpenMM custom external force updated. Exit requested")
                        # This is used if OpenMM MD is handling forces and dynamics
                        return self.QMenergy - self.extforce_energy, self.QM_PC_gradient
        self._run_mm_theory(current_coords, grad=grad)
        log_time_since(CheckpointTime, "MM step")
        CheckpointTime = time.time()

        # Final QM/MM Energy. Possible correction for OpenMM external force term
        self.QM_MM_energy = self.QMenergy + self.MMenergy - self.extforce_energy - self.subtractive_correction_E
        if self.embedding.lower() == "elstat":
            logger.info(
                "Note: You are using electrostatic embedding. This means that the QM-energy is actually the polarized "
                "QM-energy"
            )
            logger.info("Note: MM energy also contains the QM-MM Lennard-Jones interaction\n")
        energywarning = ""
        if self.truncated_pc is True:
            logger.warning("Truncated PC approximation is active. QM and QM/MM energies are approximate.")
            energywarning = "(approximate)"

        logger.info("%s", "{:<20} {:>20.12f} {}".format("QM energy: ", self.QMenergy, energywarning))
        logger.info("%s", "{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
        logger.info("%s", "{:<20} {:>20.12f} {}".format("QM/MM energy: ", self.QM_MM_energy, energywarning))

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
                self._write_gradient_debug_files(
                    label,
                    [
                        (
                            self.QMgradient_wo_linkatoms,
                            self.qmelems,
                            self.qmatoms,
                            "QMgradient-without-linkatoms",
                            "QM gradient w/o linkatoms",
                        ),
                        (
                            self.QMgradient,
                            self.qmelems + ["L"] * self.num_linkatoms,
                            self.qmatoms + [0] * self.num_linkatoms,
                            "QMgradient-with-linkatoms",
                            "QM gradient with linkatoms",
                        ),
                        (self.PCgradient, self.mmelems, self.mmatoms, "PCgradient", "PC gradient"),
                        (self.QM_PC_gradient, self.elems, self.allatoms, "QM+PCgradient", "QM+PC gradient"),
                        (self.MMgradient, self.elems, self.allatoms, "MMgradient", "MM gradient"),
                        (self.QM_MM_gradient, self.elems, self.allatoms, "QM_MMgradient", "QM/MM gradient"),
                    ],
                )
            logger.info("------------ENDING QM/MM MODULE-------------")
            log_time_since(module_init_time, "QM/MM run")
            return self.QM_MM_energy, self.QM_MM_gradient
        log_time_since(module_init_time, "QM/MM run")
        return self.QM_MM_energy


def _fullindex_to_qmindex(fullindex: int, qmatoms: Sequence[int]) -> int:
    return qmatoms.index(fullindex)


# NOTE: New resid-indices are used to avoid problem of PDB-file having
# repeating sequences of resids, additional chains or segments
def _grab_resids_from_pdbfile(pdbfile: str | PathLike[str]) -> list[int]:
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
def _grab_resids_from_psffile(psffile: str | PathLike[str]) -> list[int]:
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
def read_charges_from_psf(file: str | PathLike[str]) -> list[float]:
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
    pdbfile: str | PathLike[str] | None = None,
    mmtheory: Any | None = None,
    psffile: str | PathLike[str] | None = None,
    fragment: Fragment | None = None,
    radius: float | None = None,
    originatom: int | None = None,
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
        logger.debug("No fragment provided. Creating fragment from PDBfile")
        fragment = Fragment(pdbfile=pdbfile)

    logger.info("Radius: %s", radius)
    logger.info(f"Origin atom: {originatom} ({fragment.elems[originatom]})")
    logger.debug("Finding all atoms within %s Å of atom %s (%s)", radius, originatom, fragment.elems[originatom])
    logger.debug("Will select all whole residues within region and export list")
    if mmtheory is not None:
        if not mmtheory.resids:
            raise InputError("mmtheory.resids list is empty! Something wrong with OpenMMTheory setup. Exiting")
        resids = mmtheory.resids
    elif psffile is not None:
        logger.info("PSF-file provided. Using residue information")
        resids = _grab_resids_from_psffile(psffile)
    else:
        logger.info("PDB-file provided. Using residue information")
        resids = _grab_resids_from_pdbfile(pdbfile)

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

    write_list_to_file(act_indices, "active_atoms")
    logger.info("Active region size: %s", len(act_indices))
    logger.info("Active-region indices written to file: active_atoms")
    logger.info(
        "The active_atoms list  can be read-into Python script like this:	 actatoms = "
        'read_intlist_from_file("active_atoms")'
    )
    openmmqmmm.coords.write_xyz_for_atoms(fragment.coords, fragment.elems, act_indices, "ActiveRegion")
    logger.info("Wrote Active region XYZfile: ActiveRegion.xyz  (inspect with visualization program)")
    return act_indices


def _linkatom_force_adv(
    Qcoord: np.ndarray,
    Mcoord: np.ndarray,
    Lcoord: Sequence[float] | np.ndarray,
    Lgrad: np.ndarray,
) -> tuple[list[float], list[float]]:
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
def _linkatom_force_lever(
    Qcoord: np.ndarray,
    Mcoord: np.ndarray,
    Lcoord: Sequence[float] | np.ndarray,
    Lgrad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    QLdistance = openmmqmmm.coords.distance(Qcoord, Lcoord)
    MQdistance = openmmqmmm.coords.distance(Mcoord, Qcoord)
    scal = QLdistance / MQdistance
    gradMM = Lgrad * scal
    gradQM = Lgrad * (1.0 - scal)
    return gradQM, gradMM


def _linkatom_force_chainrule(
    Qcoord: np.ndarray,
    Mcoord: np.ndarray,
    Lcoord: Sequence[float] | np.ndarray,
    Lgrad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate L = Q + d(M-Q)/|M-Q| for a fixed link distance d."""
    separation = np.asarray(Mcoord) - np.asarray(Qcoord)
    distance = np.linalg.norm(separation)
    direction = separation / distance
    ratio = np.linalg.norm(np.asarray(Lcoord) - Qcoord) / distance
    # B = d/R (I - uu^T), dL/dM = B, and dL/dQ = I-B.
    gradMM = ratio * (Lgrad - direction * np.dot(direction, Lgrad))
    return Lgrad - gradMM, gradMM


def compute_decomposed_qm_mm_energy(fragment: Fragment | None = None, theory: QMMMTheory | None = None) -> None:
    """Decompose a QM/MM single-point energy into QM, MM and coupling terms."""
    logger.info(main_header("Decomposed QM/MM Energy Calculation"))

    if isinstance(theory, QMMMTheory) is False:
        raise InputError("Please provide a QMMMTheory object as theory.")
    if theory.qm_charge is None or theory.qm_mult is None:
        raise InputError("Please define qm_charge and qm_mult attributes in the QMMMtheory object")
    if theory.mm_theory is None or not callable(getattr(theory.mm_theory, "qmmm_lj_energy", None)):
        raise InputError("QM/MM energy decomposition requires an MM theory supporting isolated Lennard-Jones energies")
    if theory.embedding != "elstat":
        raise InputError("QM/MM energy decomposition currently requires electrostatic embedding")
    if theory.openmm_externalforce:
        raise InputError("Use a standalone QM/MM theory for energy decomposition, before attaching an MD force")
    if fragment is None:
        fragment = theory.fragment

    # Inspect/evaluate cloned LJ forces, including exceptions and CHARMM custom
    # terms. The live System is never altered, even if a later QM call fails.
    E_QM_MM_vdw = theory.mm_theory.qmmm_lj_energy(theory.qmatoms, theory._image_periodic_coords(fragment.coords))

    result = openmmqmmm.single_point(theory=theory, fragment=fragment)

    E_QM_MM_tot = result.energy
    E_QM_pol = result.qm_energy
    E_MM_mod = result.mm_energy

    logger.warning(
        "QM-MM bonded decomposition is not implemented; reporting it as zero, so the MM term still contains "
        "that contribution"
    )
    E_QM_MM_bond = 0.0
    E_MM_pure = E_MM_mod - E_QM_MM_vdw

    # Preserve every cap setting and the already established boundary. Calling
    # QMMMTheory.__init__ again on the shared MM System would strip forces twice.
    QM_MM_mech = copy.copy(theory)
    QM_MM_mech.embedding = "mech"
    QM_MM_mech.pc = False
    QM_MM_mech.runcalls = 0
    QM_MM_mech.truncated_pc = False
    QM_MM_mech.update_qm_region_charges = False
    result_mech = openmmqmmm.single_point(theory=QM_MM_mech, fragment=fragment)
    E_QM_pure = result_mech.qm_energy
    E_QM_MM_elstat = E_QM_pol - E_QM_pure

    E_coupling = E_QM_MM_elstat + E_QM_MM_vdw + E_QM_MM_bond

    standard_sum = E_QM_pol + E_MM_mod
    if not np.isclose(E_QM_MM_tot, standard_sum, atol=1e-6, rtol=0.0):
        raise InternalError(
            f"QM/MM energy decomposition inconsistency (E_QM_pol + E_MM_mod): residual={E_QM_MM_tot - standard_sum}"
        )
    decomposed_sum = E_QM_pure + E_MM_pure + E_coupling
    if not np.isclose(E_QM_MM_tot, decomposed_sum, atol=1e-6, rtol=0.0):
        raise InternalError(
            f"QM/MM energy decomposition inconsistency (pure + coupling terms): residual={E_QM_MM_tot - decomposed_sum}"
        )

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
