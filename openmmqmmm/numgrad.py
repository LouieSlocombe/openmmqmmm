import logging

import numpy as np

import openmmqmmm
from openmmqmmm.coords import print_coords_all

logger = logging.getLogger(__name__)

# Basic Theory class


# Numerical gradient class
class NumGradclass:
    def __init__(self, theory, npoint=2, displacement=0.00264589, runmode="serial", numcores=1):
        logger.info("Creating NumGrad wrapper object")
        self.theory = theory
        self.theorytype = "QM"
        self.theorynamelabel = "NumGrad"
        self.displacement = displacement
        self.npoint = npoint
        self.runmode = runmode
        self.numcores = numcores

    def set_numcores(self, numcores):
        self.numcores = numcores

    def cleanup(self):
        logger.info("Cleanup method called but not yet implemented for Numgrad")

    def run(
        self,
        current_coords=None,
        current_MM_coords=None,
        MMcharges=None,
        qm_elems=None,
        elems=None,
        Grad=False,
        Hessian=False,
        PC=False,
        numcores=None,
        restart=False,
        label=None,
        charge=None,
        mult=None,
    ):

        logger.info(f"------------RUNNING {self.theorynamelabel} WRAPPER -------------")

        numatoms = len(current_coords)
        displacement_bohr = self.displacement * 1.88972612546

        list_of_displaced_geos, list_of_displacements, all_disp_fragments = creating_displaced_geos(
            current_coords, elems, self.displacement, self.npoint, charge, mult
        )
        if self.runmode == "serial":
            logger.info("Numgrad: runmode is serial")
            logger.info("Running original geometry first")
            # Energy for original geometry.
            orig_energy = self.theory.run(
                current_coords=current_coords, elems=elems, Grad=False, label=label, charge=charge, mult=mult
            )
            #
            dispdict = {}
            logger.info(f"Will now loop over {len(list_of_displacements)} displacements")

            # Looping over displacements
            for i, dispgeo in enumerate(list_of_displaced_geos):
                disp = list_of_displacements[i]
                logger.info(
                    f"Running displacement {i + 1} / {len(list_of_displaced_geos)}. Displacing Atom:{disp[0]} Coord:{disp[1]} Direction:{disp[2]}"
                )
                energy = self.theory.run(
                    current_coords=dispgeo, elems=elems, Grad=False, label=label, charge=charge, mult=mult
                )
                dispdict[(disp)] = energy

        elif self.runmode == "parallel":
            logger.info("Numgrad: runmode is parallel")
            origfrag = openmmqmmm.Fragment(coords=current_coords, elems=elems, label="orig", charge=charge, mult=mult)
            all_disp_fragments = [origfrag, *all_disp_fragments]
            result = openmmqmmm.parallel.Job_parallel(
                fragments=all_disp_fragments,
                theories=[self.theory],
                numcores=self.numcores,
                allow_theory_parallelization=True,
                Grad=False,
                copytheory=True,
            )
            logger.info("result: %s", result)
            dispdict = result.energies_dict
            orig_energy = dispdict["orig"]

        # Assemble gradient
        gradient = np.zeros((numatoms, 3))
        # 2-point
        if self.npoint == 2:
            for atindex in range(numatoms):
                # Looping over x,yz
                for u in [0, 1, 2]:
                    # Pos and neg directions
                    if self.runmode == "parallel":
                        # '0_0_+'
                        posval = dispdict[f"{atindex}_{u}_+"]
                        negval = dispdict[f"{atindex}_{u}_-"]
                    else:
                        posval = dispdict[(atindex, u, "+")]
                        negval = dispdict[(atindex, u, "-")]
                    grad_component = (posval - negval) / (2 * displacement_bohr)
                    gradient[atindex, u] = grad_component
        # 1-point
        elif self.npoint == 1:
            for atindex in range(numatoms):
                # Looping over x,yz
                for u in [0, 1, 2]:
                    # Pos direction only
                    posval = dispdict[f"{atindex}_{u}_+"] if self.runmode == "parallel" else dispdict[atindex, u, "+"]
                    grad_component = (posval - orig_energy) / displacement_bohr
                    gradient[atindex, u] = grad_component

        self.energy = orig_energy
        self.gradient = gradient

        return self.energy, self.gradient


def creating_displaced_geos(current_coords, elems, displacement, npoint, charge, mult):
    displacement_bohr = displacement * 1.88972612546
    logger.info(f"Displacement: {displacement:5.4f} Å ({displacement_bohr:5.4f} Bohr)")
    logger.info("Starting geometry:")
    logger.info("")
    logger.info("Printing original geometry...")
    print_coords_all(current_coords, elems)
    logger.info("")

    # Looping over each atom and each coordinate to create displaced geometries
    # Only displacing atom if in hessatoms list. i.e. possible partial Hessian
    list_of_displaced_geos = []
    list_of_displacements = []
    for atom_index in range(len(current_coords)):
        for coord_index in range(3):
            val = current_coords[atom_index, coord_index]
            # Displacing in + direction
            current_coords[atom_index, coord_index] = val + displacement
            y = current_coords.copy()
            list_of_displaced_geos.append(y)
            list_of_displacements.append((atom_index, coord_index, "+"))
            if npoint == 2:
                # Displacing  - direction
                current_coords[atom_index, coord_index] = val - displacement
                y = current_coords.copy()
                list_of_displaced_geos.append(y)
                list_of_displacements.append((atom_index, coord_index, "-"))
            # Displacing back
            current_coords[atom_index, coord_index] = val

    # Original geo added here if onepoint
    if npoint == 1:
        list_of_displaced_geos.append(current_coords)
        list_of_displacements.append("Originalgeo")

    logger.debug("List of displacements: %s", list_of_displacements)

    # Creating ASH fragments
    # Creating displacement labels as strings and adding to fragment
    # Also calclabels, currently used by runmode serial only
    all_disp_fragments = []
    for dispgeo, disp in zip(list_of_displaced_geos, list_of_displacements, strict=False):
        # Original geo
        if disp == "Originalgeo":
            stringlabel = "Originalgeo"
        # Displacements
        else:
            disp[0]
            if disp[1] == 0 or disp[1] == 1 or disp[1] == 2:
                pass
            disp[2]
            stringlabel = f"{disp[0]}_{disp[1]}_{disp[2]}"
        # Create fragment
        frag = openmmqmmm.Fragment(coords=dispgeo, elems=elems, label=stringlabel, charge=charge, mult=mult)
        all_disp_fragments.append(frag)

    return list_of_displaced_geos, list_of_displacements, all_disp_fragments
