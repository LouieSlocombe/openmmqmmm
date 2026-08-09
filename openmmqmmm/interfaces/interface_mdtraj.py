import os

import numpy as np

from openmmqmmm.exceptions import (
    InputError,
    MissingDependencyError,
)
from openmmqmmm.modules.module_coords import Fragment, write_xyzfile


def MDtraj_import():
    print("Importing mdtraj (https://www.mdtraj.org)")
    try:
        import mdtraj
    except ImportError:
        raise MissingDependencyError(
            "Problem importing mdtraj. Try: 'pip install mdtraj' or 'conda install -c conda-forge mdtraj'"
        ) from None
    return mdtraj


def MDtraj_RMSF(trajectory, pdbtopology, print_largest_values=True, threshold=0.005, largest_values=10, parallel=True):
    print("Inside MDtraj_RMSF")
    # Import mdtraj library
    mdtraj = MDtraj_import()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    firstframe = traj[0]
    rmsflist = mdtraj.rmsf(traj, reference=None, frame=0, atom_indices=None, parallel=parallel)

    if print_largest_values is True:
        print(f"Will print RMSF largest_values={largest_values}")
        large_rmsf_indices = rmsflist.argsort()[::-1][:largest_values]
    else:
        print(f"Will print atom RMSF values larger than threshold={threshold}")
        large_rmsf_indices = np.where(rmsflist > threshold)[0]
    if len(large_rmsf_indices) > 0:
        print("Printing atoms with high root-mean-square fluctuations:")
        print("Index    Residue-atom           Coordinates                              RMSF")
        for i in large_rmsf_indices:
            atom_string = str(firstframe.topology.atom(i))
            rmsfvalue = rmsflist[i]
            print(
                f"{i:>6} {atom_string:<14} {firstframe.xyz[0][i][0]:>12.6f} {firstframe.xyz[0][i][1]:>12.6f} {firstframe.xyz[0][i][2]:>12.6f}      {rmsfvalue:>12.6f}"
            )
    return large_rmsf_indices


def MDtraj_RMSD(trajectory, pdbtopology, atom_indices=None, parallel=True):
    print("Inside MDtraj_RMSD")
    # Import mdtraj library
    mdtraj = MDtraj_import()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)

    # RMSD
    rmsds = mdtraj.rmsd(traj, traj, 0, atom_indices=atom_indices, parallel=parallel)

    return rmsds


# anchor_molecules. Use if automatic guess fails
def MDtraj_imagetraj(
    trajectory, pdbtopology, traj_format="DCD", unitcell_lengths=None, unitcell_angles=None, solute_anchor=None
):
    # Trajectory basename
    traj_basename = os.path.splitext(trajectory)[0]
    # PDB-file basename
    pdb_basename = os.path.splitext(pdbtopology)[0]

    # Import mdtraj library
    mdtraj = MDtraj_import()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)

    numframes = len(traj._time)
    print(f"Found {numframes} frames in trajectory.")
    print("PBC information in trajectory:")
    # If PBC information is missing from traj file (OpenMM: Charmmfiles, Amberfiles option etc) then provide this info
    if unitcell_lengths is not None:
        print("unitcell_lengths info provided by user.")
        unitcell_lengths_nm = [i / 10 for i in unitcell_lengths]
        traj.unitcell_lengths = np.array(unitcell_lengths_nm * numframes).reshape(numframes, 3)
        traj.unitcell_angles = np.array(unitcell_angles * numframes).reshape(numframes, 3)

    # Also load the pdbfile as a trajectory-snapshot (in addition to being topology)
    pdbsnap = mdtraj.load(pdbtopology, top=pdbtopology)
    # Manual anchor if needed
    # NOTE: not sure how well this works but it's something
    if solute_anchor is True:
        anchors = [set(traj.topology.residue(0).atoms)]
        print("anchors:", anchors)
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
        print("Saved reimaged trajectory:", traj_basename + "_imaged.dcd")
    elif traj_format == "PDB":
        imaged.save(traj_basename + "_imaged.pdb")
        print("Saved reimaged trajectory:", traj_basename + "_imaged.pdb")
    else:
        print("Unknown trajectory format.")
    # Save PDB-snapshot
    pdbsnap_imaged.save(pdb_basename + "_imaged.pdb")
    print("Saved reimaged PDB-file:", pdb_basename + "_imaged.pdb")
    # Return last frame as coords or ASH fragment ?
    # Last frame coordinates as Angstrom
    lastframe = imaged[-1]._xyz[-1] * 10

    return lastframe


# Slicing trajectory. Mostly to grab specific snapshot
# TODO: allow option to grab by ps? Requires information about timestep and traj-frequency
def MDtraj_slice(trajectory, pdbtopology, traj_format="PDB", frames=None):
    # Trajectory basename
    traj_basename = os.path.basename(os.path.splitext(trajectory)[0])
    print("traj_basename:", traj_basename)
    # os.path.basename(

    # Import mdtraj library
    mdtraj = MDtraj_import()

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    print(f"This trajectory contains {traj.n_frames} frames")

    print("User frame selection:", frames)
    if frames is None:
        raise InputError(
            "Error: frames keyword needs to be set. Should usually be a list of two integers.\nE.g. frames=[0,1] to grab first frame or frames=[0,3] to grab first 3 frames\nAlso possible to do: frames='first' or frames='last' to grab first or last"
        )
    elif frames == "first":
        frames = [0, 1]
    elif frames == "last":
        frames = [traj.n_frames - 1, traj.n_frames]
    elif frames == "all":
        frames = [0, traj.n_frames]
    elif len(frames) != 2:
        raise InputError(
            "Error: frames keyword needs to be a list of two integers.\nE.g. frames=[0,1] to grab first frame or frames=[0,3] to grab first 3 frames\nAlso possible to do: frames='first' or frames='last' to grab first or last"
        )

    # Slicing trajectory
    print("Slicing trajectory using frame selection:", frames)
    tslice = traj[frames[0] : frames[1]]
    print(f"Trajectory slice contains {tslice.n_frames} frames")
    if tslice.n_frames == 0:
        raise InputError(
            "{}\nExiting".format(
                f"0 frames found when slicing. You probably should do: frames=[{frames[0]},{frames[1] + 1}] instead"
            )
        )

    # Save trajectory in format
    print(
        f"Writing sliced trajectory to file in format {format} (you can change this by format keyword to be 'DCD', 'XYZ' or 'PDB') "
    )
    if traj_format == "DCD":
        tslice.save(traj_basename + f"_frame{frames[0]}_{frames[1]}.dcd")
        print("Saved sliced trajectory:", traj_basename + f"_frame{frames[0]}_{frames[1]}.dcd")
        return traj_basename + f"_frame{frames[0]}_{frames[1]}.dcd"
    elif traj_format == "PDB":
        tslice.save(traj_basename + f"_frame{frames[0]}_{frames[1]}.pdb")
        print("Saved sliced trajectory:", traj_basename + f"_frame{frames[0]}_{frames[1]}.pdb")
        return traj_basename + f"_frame{frames[0]}_{frames[1]}.pdb"
    elif traj_format == "XYZ":
        # Looping over selection and writing XYZ since mdtraj does not give proper elements
        print(
            "Warning: the MDtraj_slice XYZ-writing requires guessing element names based on atomnames in the topology PDB-file."
        )
        print("This is not always successful (might require manual change of the atomnames in PDB-file)")
        dummyfrag = Fragment(pdbfile=pdbtopology, printlevel=0)
        elems = dummyfrag.elems
        for _i, t in enumerate(tslice):
            coords = t._xyz[0] * 10
            write_xyzfile(
                elems,
                coords,
                traj_basename + f"_frame{frames[0]}_{frames[1]}",
                printlevel=1,
                writemode="a",
                title="title",
            )
        return traj_basename + f"_frame{frames[0]}_{frames[1]}.xyz"
    else:
        print("Unknown trajectory format.")
    return


# Function to get internal coordinates from trajectory fast
# Give trajectory file
def MDtraj_coord_analyze(trajectory, pdbtopology=None, periodic=True, indices=None):
    print("Inside MDtraj_coord_analyze")
    if indices is None:
        raise InputError("indices needs to be set")
    print("Trajectory:", trajectory)
    print("Topology:", pdbtopology)
    print("Atom indices:", indices)
    # Import mdtraj library
    mdtraj = MDtraj_import()

    if pdbtopology is None:
        print("A topology is required but was not provided")
        print("Checking if trajectory.pdb file (created by the MD run) is available:")
        if not os.path.isfile("trajectory.pdb"):
            raise InputError("A topology file is required (no trajectory.pdb found either)")
        pdbtopology = "trajectory.pdb"

    # Load trajectory
    print("Loading trajectory using mdtraj.")
    traj = mdtraj.load(trajectory, top=pdbtopology)
    print(f"This trajectory contains {traj.n_frames} frames")
    if len(indices) == 4:
        print("4 atom indices given. This must be a dihedral angle.  Returning dihedral in radians")
        output = mdtraj.compute_dihedrals(traj, [indices], periodic=periodic, opt=True)
        unit_label = "radians"
    elif len(indices) == 3:
        print("3 atom indices given. This must be an angle. Returning angle in radians")
        output = mdtraj.compute_angles(traj, [indices], periodic=periodic, opt=True)
        unit_label = "radians"
    elif len(indices) == 2:
        print("2 atom indices given. This must be a distance.  Returning angle in Angstrom")
        output = mdtraj.compute_distances(traj, [indices], periodic=periodic, opt=True)
        output = 10 * output
        unit_label = "Angstrom"
    else:
        raise InputError(f"something wrong with indices supplied: {indices}")
    print(f"List of coordinates ({len(output)}) for each frame:", output)

    ave = np.mean(output)
    stdev = np.std(output)
    print(f"Mean: {ave} {unit_label}")
    print(f"Standard deviation: {stdev} {unit_label}")

    return output
