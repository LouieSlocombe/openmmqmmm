import logging

import numpy as np

import openmmqmmm
from openmmqmmm.coords import print_coords_all
from openmmqmmm.exceptions import InputError

logger = logging.getLogger(__name__)


class NumGrad:
    """Wrapper theory computing gradients numerically by finite differences of energies."""

    def __init__(self, theory, npoint=2, displacement=0.00264589, runmode="serial", numcores=1):
        logger.info("Creating NumGrad wrapper object")
        # Only the 1- and 2-point stencils are implemented. Without this check any other
        # value skips gradient assembly entirely and returns a zero gradient, which an
        # optimizer happily reads as a converged structure.
        if npoint not in (1, 2):
            raise InputError(f"NumGrad npoint must be 1 (forward difference) or 2 (central difference), not {npoint}")
        self.theory = theory
        self.theorytype = "QM"
        self.theorynamelabel = "NumGrad"
        self.displacement = displacement
        self.npoint = npoint
        self.runmode = runmode
        self.numcores = numcores

    def set_numcores(self, numcores):
        """Set the number of cores used for parallel displacement runs."""
        self.numcores = numcores

    def cleanup(self):
        """No-op: NumGrad leaves cleanup of scratch files to the wrapped theory."""
        logger.info("Cleanup method called but not yet implemented for Numgrad")

    def run(
        self,
        current_coords=None,
        current_mm_coords=None,
        mm_charges=None,
        qm_elems=None,
        elems=None,
        grad=False,
        hessian=False,
        pc=False,
        numcores=None,
        restart=False,
        label=None,
        charge=None,
        mult=None,
    ):
        """Compute the energy and a finite-difference gradient of the wrapped theory."""
        logger.info(f"------------RUNNING {self.theorynamelabel} WRAPPER -------------")

        numatoms = len(current_coords)
        displacement_bohr = self.displacement * 1.88972612546

        list_of_displaced_geos, list_of_displacements, all_disp_fragments = creating_displaced_geos(
            current_coords, elems, self.displacement, self.npoint, charge, mult
        )
        if self.runmode == "serial":
            logger.info("Numgrad: runmode is serial")
            logger.info("Running original geometry first")
            orig_energy = self.theory.run(
                current_coords=current_coords, elems=elems, grad=False, label=label, charge=charge, mult=mult
            )
            dispdict = {}
            logger.info(f"Will now loop over {len(list_of_displacements)} displacements")

            for i, dispgeo in enumerate(list_of_displaced_geos):
                disp = list_of_displacements[i]
                logger.info(
                    f"Running displacement {i + 1} / {len(list_of_displaced_geos)}. Displacing Atom:{disp[0]} "
                    f"Coord:{disp[1]} Direction:{disp[2]}"
                )
                energy = self.theory.run(
                    current_coords=dispgeo, elems=elems, grad=False, label=label, charge=charge, mult=mult
                )
                dispdict[(disp)] = energy
        elif self.runmode == "parallel":
            logger.info("Numgrad: runmode is parallel")
            origfrag = openmmqmmm.Fragment(coords=current_coords, elems=elems, label="orig", charge=charge, mult=mult)
            all_disp_fragments = [origfrag, *all_disp_fragments]
            result = openmmqmmm.parallel.job_parallel(
                fragments=all_disp_fragments,
                theories=[self.theory],
                numcores=self.numcores,
                allow_theory_parallelization=True,
                grad=False,
                copytheory=True,
            )
            logger.info("result: %s", result)
            dispdict = result.energies_dict
            orig_energy = dispdict["orig"]

        gradient = np.zeros((numatoms, 3))
        if self.npoint == 2:
            for atindex in range(numatoms):
                for u in [0, 1, 2]:
                    if self.runmode == "parallel":
                        posval = dispdict[f"{atindex}_{u}_+"]
                        negval = dispdict[f"{atindex}_{u}_-"]
                    else:
                        posval = dispdict[(atindex, u, "+")]
                        negval = dispdict[(atindex, u, "-")]
                    grad_component = (posval - negval) / (2 * displacement_bohr)
                    gradient[atindex, u] = grad_component
        elif self.npoint == 1:
            for atindex in range(numatoms):
                for u in [0, 1, 2]:
                    posval = dispdict[f"{atindex}_{u}_+"] if self.runmode == "parallel" else dispdict[atindex, u, "+"]
                    grad_component = (posval - orig_energy) / displacement_bohr
                    gradient[atindex, u] = grad_component

        self.energy = orig_energy
        self.gradient = gradient

        # Match the theory-object contract: energy alone unless a gradient was asked for
        if grad:
            return self.energy, self.gradient
        return self.energy


def creating_displaced_geos(current_coords, elems, displacement, npoint, charge, mult):
    displacement_bohr = displacement * 1.88972612546
    logger.info(f"Displacement: {displacement:5.4f} Å ({displacement_bohr:5.4f} Bohr)")
    logger.info("Starting geometry:")
    logger.info("")
    logger.info("Printing original geometry...")
    print_coords_all(current_coords, elems)
    logger.info("")

    # Only displacing atom if in hessatoms list. i.e. possible partial Hessian
    list_of_displaced_geos = []
    list_of_displacements = []
    for atom_index in range(len(current_coords)):
        for coord_index in range(3):
            val = current_coords[atom_index, coord_index]
            current_coords[atom_index, coord_index] = val + displacement
            y = current_coords.copy()
            list_of_displaced_geos.append(y)
            list_of_displacements.append((atom_index, coord_index, "+"))
            if npoint == 2:
                current_coords[atom_index, coord_index] = val - displacement
                y = current_coords.copy()
                list_of_displaced_geos.append(y)
                list_of_displacements.append((atom_index, coord_index, "-"))
            current_coords[atom_index, coord_index] = val

    if npoint == 1:
        list_of_displaced_geos.append(current_coords)
        list_of_displacements.append("Originalgeo")

    logger.debug("List of displacements: %s", list_of_displacements)

    # Also calclabels, currently used by runmode serial only
    all_disp_fragments = []
    for dispgeo, disp in zip(list_of_displaced_geos, list_of_displacements, strict=False):
        # "Originalgeo" for the reference geometry, atom_axis_direction for a displacement
        stringlabel = "Originalgeo" if disp == "Originalgeo" else f"{disp[0]}_{disp[1]}_{disp[2]}"
        frag = openmmqmmm.Fragment(coords=dispgeo, elems=elems, label=stringlabel, charge=charge, mult=mult)
        all_disp_fragments.append(frag)

    return list_of_displaced_geos, list_of_displacements, all_disp_fragments
