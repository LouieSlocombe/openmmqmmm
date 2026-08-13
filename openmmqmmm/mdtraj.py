import logging
import os

import numpy as np

from openmmqmmm.exceptions import (
    MissingDependencyError,
)

logger = logging.getLogger(__name__)


def mdtraj_load():
    logger.info("Importing mdtraj (https://www.mdtraj.org)")
    try:
        import mdtraj
    except ImportError:
        raise MissingDependencyError(
            "Problem importing mdtraj. Try: 'pip install mdtraj' or 'conda install -c conda-forge mdtraj'"
        ) from None
    return mdtraj


def mdtraj_rmsf(
    trajectory, pdbtopology, print_largest_values=True, threshold=0.005, largest_values=10, parallel=True
) -> list[int]:
    """Compute per-atom root-mean-square fluctuations of a trajectory via mdtraj."""
    logger.info("Inside MDtraj_RMSF")
    # Import mdtraj library
    mdtraj = mdtraj_load()

    # Load trajectory
    logger.info("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    firstframe = traj[0]
    rmsflist = mdtraj.rmsf(traj, reference=None, frame=0, atom_indices=None, parallel=parallel)

    if print_largest_values is True:
        logger.info(f"Will print RMSF largest_values={largest_values}")
        large_rmsf_indices = rmsflist.argsort()[::-1][:largest_values]
    else:
        logger.info(f"Will print atom RMSF values larger than threshold={threshold}")
        large_rmsf_indices = np.where(rmsflist > threshold)[0]
    if len(large_rmsf_indices) > 0:
        logger.info("Printing atoms with high root-mean-square fluctuations:")
        logger.info("Index    Residue-atom           Coordinates                              RMSF")
        for i in large_rmsf_indices:
            atom_string = str(firstframe.topology.atom(i))
            rmsfvalue = rmsflist[i]
            logger.info(
                f"{i:>6} {atom_string:<14} {firstframe.xyz[0][i][0]:>12.6f} {firstframe.xyz[0][i][1]:>12.6f} "
                f"{firstframe.xyz[0][i][2]:>12.6f}      {rmsfvalue:>12.6f}"
            )
    return large_rmsf_indices


# anchor_molecules. Use if automatic guess fails
def mdtraj_image_trajectory(
    trajectory, pdbtopology, traj_format="DCD", unitcell_lengths=None, unitcell_angles=None, solute_anchor=None
) -> str:
    # Trajectory basename
    """Re-image (wrap) a periodic trajectory so molecules stay whole, via mdtraj."""
    traj_basename = os.path.splitext(trajectory)[0]
    # PDB-file basename
    pdb_basename = os.path.splitext(pdbtopology)[0]

    # Import mdtraj library
    mdtraj = mdtraj_load()

    # Load trajectory
    logger.info("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)

    numframes = len(traj._time)
    logger.info(f"Found {numframes} frames in trajectory.")
    logger.info("PBC information in trajectory:")
    # If PBC information is missing from traj file (OpenMM: Charmmfiles, Amberfiles option etc) then provide this info
    if unitcell_lengths is not None:
        logger.info("unitcell_lengths info provided by user.")
        unitcell_lengths_nm = [i / 10 for i in unitcell_lengths]
        traj.unitcell_lengths = np.array(unitcell_lengths_nm * numframes).reshape(numframes, 3)
        traj.unitcell_angles = np.array(unitcell_angles * numframes).reshape(numframes, 3)

    # Also load the pdbfile as a trajectory-snapshot (in addition to being topology)
    pdbsnap = mdtraj.load(pdbtopology, top=pdbtopology)
    # Manual anchor if needed
    # NOTE: not sure how well this works but it's something
    if solute_anchor is True:
        anchors = [set(traj.topology.residue(0).atoms)]
        logger.info("anchors: %s", anchors)
        # Re-imaging trajectory
        imaged = traj.image_molecules(anchor_molecules=anchors)
        # Reimaging PDB
        pdbsnap_imaged = pdbsnap.image_molecules(anchor_molecules=anchors)
    else:
        imaged = traj.image_molecules()
        pdbsnap_imaged = pdbsnap.image_molecules()
    # Save trajectory in format
    if traj_format == "DCD":
        imaged.save(traj_basename + "_imaged.dcd")
        logger.info("Saved reimaged trajectory: %s", traj_basename + "_imaged.dcd")
    elif traj_format == "PDB":
        imaged.save(traj_basename + "_imaged.pdb")
        logger.info("Saved reimaged trajectory: %s", traj_basename + "_imaged.pdb")
    else:
        logger.info("Unknown trajectory format.")
    # Save PDB-snapshot
    pdbsnap_imaged.save(pdb_basename + "_imaged.pdb")
    logger.info("Saved reimaged PDB-file: %s", pdb_basename + "_imaged.pdb")
    # Return last frame as coords or fragment ?
    # Last frame coordinates as Angstrom
    return imaged[-1]._xyz[-1] * 10
