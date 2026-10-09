from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

import openmmqmmm
import openmmqmmm.constants
from openmmqmmm.coords import print_coords_all
from openmmqmmm.exceptions import InputError
from openmmqmmm.utils import require_int_in_range, require_positive_finite

logger = logging.getLogger(__name__)

# ORCA's step of 0.005 Bohr, in Angstrom: displacements are given in Angstrom here.
DEFAULT_DISPLACEMENT = 0.005 * openmmqmmm.constants.BOHR_TO_ANG

type Displacement = tuple[int, int, Literal["+", "-"]]
type RunMode = Literal["serial", "parallel"]


def _validate_numcores(numcores: int) -> int:
    return require_int_in_range(numcores, f"NumGrad numcores must be a positive integer, not {numcores!r}")


def _displacement_label(displacement: Displacement) -> str:
    """Return the one key spelling for a displacement, shared by both runmodes and by the job labels."""
    atom_index, coord_index, direction = displacement
    return f"{atom_index}_{coord_index}_{direction}"


def _validate_geometry(current_coords: np.ndarray | None, elems: Sequence[str] | None) -> tuple[np.ndarray, list[str]]:
    if current_coords is None:
        raise InputError("NumGrad requires current_coords")
    try:
        coords = np.array(current_coords, dtype=float, copy=True)
    except (TypeError, ValueError):
        raise InputError("NumGrad current_coords must be a numeric N x 3 array") from None
    if coords.ndim != 2 or coords.shape[0] == 0 or coords.shape[1] != 3:
        raise InputError(f"NumGrad current_coords must have shape (N, 3) with N > 0, not {coords.shape}")
    if not np.all(np.isfinite(coords)):
        raise InputError("NumGrad current_coords must contain only finite values")
    if elems is None:
        raise InputError("NumGrad requires one element symbol per coordinate")
    element_list = list(elems)
    if len(element_list) != len(coords):
        raise InputError(
            f"NumGrad received {len(element_list)} elements for {len(coords)} coordinate rows; the lengths must match"
        )
    return coords, element_list


class NumGrad:
    """Wrapper theory computing gradients numerically by finite differences of energies."""

    def __init__(
        self,
        theory: Any,
        npoint: int = 2,
        displacement: float = DEFAULT_DISPLACEMENT,
        runmode: RunMode = "serial",
        numcores: int = 1,
    ) -> None:
        logger.debug("Creating NumGrad wrapper object")
        if not callable(getattr(theory, "run", None)):
            raise InputError("NumGrad requires a wrapped theory with a callable run method")
        npoint = require_int_in_range(
            npoint, f"NumGrad npoint must be 1 (forward difference) or 2 (central difference), not {npoint}", maximum=2
        )
        if runmode not in ("serial", "parallel"):
            raise InputError(f"NumGrad runmode must be 'serial' or 'parallel', not {runmode!r}")
        displacement = require_positive_finite(
            displacement, f"NumGrad displacement must be a positive finite number, not {displacement!r}"
        )
        self.theory = theory
        self.theorytype = "QM"
        self.theorynamelabel = "NumGrad"
        self.displacement = displacement
        self.npoint = int(npoint)
        self.runmode = runmode
        self.numcores = _validate_numcores(numcores)

    def set_numcores(self, numcores: int) -> None:
        """Set the number of cores used for parallel displacement runs."""
        self.numcores = _validate_numcores(numcores)

    def cleanup(self) -> None:
        """Do nothing: NumGrad has no scratch files and does not clean up the wrapped theory's."""
        logger.debug("NumGrad cleanup: nothing to remove")

    def run(
        self,
        *,
        current_coords: np.ndarray | None = None,
        current_mm_coords: np.ndarray | None = None,
        mm_charges: Sequence[float] | np.ndarray | None = None,
        qm_elems: Sequence[str] | None = None,
        elems: Sequence[str] | None = None,
        grad: bool = False,
        hessian: bool = False,
        pc: bool = False,
        numcores: int | None = None,
        restart: bool = False,
        label: str | float | tuple[object, ...] | None = None,
        charge: int | None = None,
        mult: int | None = None,
    ) -> float | tuple[float, np.ndarray]:
        """Compute the wrapped theory's energy and, with grad=True, its finite-difference gradient."""
        logger.info(f"------------RUNNING {self.theorynamelabel} WRAPPER -------------")

        element_source = elems if elems is not None else qm_elems
        coords, element_list = _validate_geometry(current_coords, element_source)
        if hessian:
            raise InputError("NumGrad does not compute Hessians; request a gradient instead")
        if grad and pc:
            raise InputError(
                "NumGrad does not support point-charge gradients; use a theory with analytic QM and point-charge "
                "gradients for electrostatic embedding"
            )

        def run_energy(geometry: np.ndarray) -> float:
            # Keep the ordinary wrapper contract small: custom theories that worked
            # before NumGrad gained QM/MM support should not have to accept unrelated
            # point-charge keywords whose values are all absent.
            arguments: dict[str, Any] = {
                "current_coords": geometry,
                "elems": element_list,
                "grad": False,
                "label": label,
                "charge": charge,
                "mult": mult,
            }
            if qm_elems is not None:
                arguments["qm_elems"] = element_list
            if current_mm_coords is not None:
                arguments["current_mm_coords"] = current_mm_coords
            if mm_charges is not None:
                arguments["mm_charges"] = mm_charges
            if pc:
                arguments["pc"] = True
            if numcores is not None:
                arguments["numcores"] = numcores
            return self.theory.run(**arguments)

        if not grad:
            self.energy = run_energy(coords)
            self.gradient = None
            return self.energy

        numatoms = len(coords)
        displacement_bohr = self.displacement * openmmqmmm.constants.ANG_TO_BOHR

        list_of_displaced_geos, list_of_displacements = _create_displaced_geometries(
            coords, element_list, self.displacement, self.npoint
        )
        if self.runmode == "serial":
            logger.info("Numgrad: runmode is serial")
            logger.debug("Running original geometry first")
            orig_energy = run_energy(coords)
            dispdict = {}
            logger.debug("Will now loop over %s displacements", len(list_of_displacements))

            for i, (dispgeo, disp) in enumerate(
                zip(list_of_displaced_geos, list_of_displacements, strict=True), start=1
            ):
                logger.debug(
                    f"Running displacement {i} / {len(list_of_displaced_geos)}. Displacing Atom:{disp[0]} "
                    f"Coord:{disp[1]} Direction:{disp[2]}"
                )
                dispdict[_displacement_label(disp)] = run_energy(dispgeo)
        elif self.runmode == "parallel":
            logger.info("Numgrad: runmode is parallel")
            effective_numcores = _validate_numcores(self.numcores if numcores is None else numcores)
            all_disp_fragments = [
                openmmqmmm.Fragment(coords=coords, elems=element_list, label="orig", charge=charge, mult=mult),
                *(
                    openmmqmmm.Fragment(
                        coords=dispgeo,
                        elems=element_list,
                        label=_displacement_label(disp),
                        charge=charge,
                        mult=mult,
                    )
                    for dispgeo, disp in zip(list_of_displaced_geos, list_of_displacements, strict=True)
                ),
            ]
            result = openmmqmmm.parallel.job_parallel(
                fragments=all_disp_fragments,
                theories=[self.theory],
                numcores=effective_numcores,
                allow_theory_parallelization=True,
                grad=False,
                copytheory=True,
            )
            dispdict = result.energies_dict
            orig_energy = dispdict["orig"]

        gradient = np.zeros((numatoms, 3))
        for atindex in range(numatoms):
            for u in range(3):
                posval = dispdict[_displacement_label((atindex, u, "+"))]
                if self.npoint == 2:
                    gradient[atindex, u] = (posval - dispdict[_displacement_label((atindex, u, "-"))]) / (
                        2 * displacement_bohr
                    )
                else:
                    gradient[atindex, u] = (posval - orig_energy) / displacement_bohr

        self.energy = orig_energy
        self.gradient = gradient

        return self.energy, self.gradient


def _create_displaced_geometries(
    current_coords: np.ndarray,
    elems: Sequence[str],
    displacement: float,
    npoint: int,
) -> tuple[list[np.ndarray], list[Displacement]]:
    displacement_bohr = displacement * openmmqmmm.constants.ANG_TO_BOHR
    logger.info(f"Displacement: {displacement:5.4f} Å ({displacement_bohr:5.4f} Bohr)")
    logger.debug("Starting geometry:")
    logger.info("\nPrinting original geometry...")
    print_coords_all(current_coords, elems)

    geometries, displacements = _displaced_geometries(current_coords, range(len(current_coords)), displacement, npoint)
    logger.debug("List of displacements: %s", displacements)
    return geometries, displacements


def _displaced_geometries(
    coords: np.ndarray,
    atoms: Sequence[int],
    step: float,
    npoint: int,
) -> tuple[list[np.ndarray], list[Displacement]]:
    """Create independent Å geometries in atom/axis/positive-negative order."""
    reference_coords = np.array(coords, dtype=float, copy=True)
    geometries = []
    displacements = []
    directions = ((1, "+"),) if npoint == 1 else ((1, "+"), (-1, "-"))
    for atom_index in atoms:
        for coord_index in range(3):
            for sign, direction in directions:
                geometry = reference_coords.copy()
                geometry[atom_index, coord_index] += sign * step
                geometries.append(geometry)
                displacements.append((atom_index, coord_index, direction))
    return geometries, displacements
