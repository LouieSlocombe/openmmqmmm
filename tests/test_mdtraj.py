"""MDTraj helpers: periodic re-imaging and per-atom fluctuation ranking."""

import mdtraj
import numpy as np
import openmm
import openmm.app
import pytest

from openmmqmmm.exceptions import InputError
from openmmqmmm.mdtraj import mdtraj_image_trajectory, mdtraj_rmsf
from openmmqmmm.openmm.md import diff_wrap_box_coords

BOX_NM = 1.0
SPLIT_WATER_O = 6
SPLIT_WATER_H = 7
MOVING_ATOM = 8


def _methanol_plus_waters():
    """One methanol and nine waters: mdtraj's anchor heuristic needs ten molecules with one larger than the rest."""
    topology = openmm.app.Topology()
    chain = topology.addChain()
    element = {symbol: openmm.app.Element.getBySymbol(symbol) for symbol in "CHO"}

    residue = topology.addResidue("MOH", chain)
    methanol = [
        topology.addAtom(name, element[symbol], residue)
        for name, symbol in [("C", "C"), ("H1", "H"), ("H2", "H"), ("H3", "H"), ("O", "O"), ("HO", "H")]
    ]
    for index in (1, 2, 3, 4):
        topology.addBond(methanol[0], methanol[index])
    topology.addBond(methanol[4], methanol[5])
    positions = [[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.6, 0.5], [0.5, 0.5, 0.6], [0.4, 0.5, 0.5], [0.4, 0.6, 0.5]]

    # The first water straddles the x boundary: O at 0.98 nm, one H wrapped to 0.02 nm.
    oxygens = [[0.98, 0.2, 0.2]] + [[0.1 + 0.1 * k, 0.8, 0.8] for k in range(8)]
    for k, oxygen in enumerate(oxygens):
        residue = topology.addResidue("HOH", chain)
        water = [topology.addAtom(name, element[name[0]], residue) for name in ("O", "H1", "H2")]
        topology.addBond(water[0], water[1])
        topology.addBond(water[0], water[2])
        offset = 0.04 if k == 0 else 0.05
        positions += [oxygen, [oxygen[0] + offset, oxygen[1], oxygen[2]], [oxygen[0], oxygen[1] + offset, oxygen[2]]]
    positions = np.array(positions)
    positions[SPLIT_WATER_H, 0] %= BOX_NM
    topology.setPeriodicBoxVectors(np.diag([BOX_NM] * 3) * openmm.unit.nanometer)
    return topology, positions


@pytest.fixture
def periodic_trajectory():
    """Write top.pdb and a three-frame traj.dcd whose split water stays split and whose atom 8 drifts."""
    topology, positions = _methanol_plus_waters()
    with open("top.pdb", "w") as pdb:
        openmm.app.PDBFile.writeFile(topology, positions * openmm.unit.nanometer, pdb)
    mdtraj_topology = mdtraj.load("top.pdb").topology
    frames = np.array([positions + np.array([0.01 * frame, 0, 0]) for frame in range(3)])
    frames[:, SPLIT_WATER_H, 0] %= BOX_NM
    frames[1:, MOVING_ATOM, 1] += [0.05, 0.1]
    trajectory = mdtraj.Trajectory(
        frames, mdtraj_topology, unitcell_lengths=np.full((3, 3), BOX_NM), unitcell_angles=np.full((3, 3), 90.0)
    )
    trajectory.save("traj.dcd")
    return mdtraj_topology, frames


def _oh_distance_nm(xyz):
    return np.linalg.norm(xyz[:, SPLIT_WATER_O] - xyz[:, SPLIT_WATER_H], axis=1)


def test_unknown_trajectory_format_is_rejected_before_loading():
    with pytest.raises(InputError, match="XTC"):
        mdtraj_image_trajectory("missing.dcd", "missing.pdb", traj_format="XTC")


def test_imaging_rejoins_a_molecule_split_across_the_box_and_returns_the_last_frame(periodic_trajectory):
    _, frames = periodic_trajectory
    assert _oh_distance_nm(frames) == pytest.approx(BOX_NM - 0.04, abs=1e-6)

    last_frame_angstrom = mdtraj_image_trajectory("traj.dcd", "top.pdb")

    imaged = mdtraj.load("traj_imaged.dcd", top="top.pdb")
    assert imaged.n_frames == 3
    assert _oh_distance_nm(imaged.xyz) == pytest.approx(0.04, abs=1e-6)
    assert last_frame_angstrom.shape == frames[0].shape
    np.testing.assert_allclose(last_frame_angstrom, imaged.xyz[-1] * 10, atol=1e-5)
    snapshot = mdtraj.load("top_imaged.pdb")
    assert _oh_distance_nm(snapshot.xyz) == pytest.approx(0.04, abs=1e-3)


def test_imaging_can_write_pdb_and_anchor_on_the_first_residue(periodic_trajectory):
    mdtraj_image_trajectory("traj.dcd", "top.pdb", traj_format="PDB", solute_anchor=True)

    imaged = mdtraj.load("traj_imaged.pdb")
    assert imaged.n_frames == 3
    assert _oh_distance_nm(imaged.xyz) == pytest.approx(0.04, abs=1e-3)


def test_user_unit_cell_is_applied_to_a_trajectory_written_without_one(periodic_trajectory):
    mdtraj_topology, frames = periodic_trajectory
    mdtraj.Trajectory(frames, mdtraj_topology).save("nobox.dcd")

    last_frame_angstrom = mdtraj_image_trajectory(
        "nobox.dcd", "top.pdb", unitcell_lengths=[10 * BOX_NM] * 3, unitcell_angles=[90, 90, 90]
    )

    imaged = mdtraj.load("nobox_imaged.dcd", top="top.pdb")
    np.testing.assert_allclose(imaged.unitcell_lengths, BOX_NM)
    assert _oh_distance_nm(imaged.xyz) == pytest.approx(0.04, abs=1e-6)
    np.testing.assert_allclose(last_frame_angstrom, imaged.xyz[-1] * 10, atol=1e-5)


def test_rmsf_ranks_the_drifting_atom_first_and_thresholds_select_it_alone(periodic_trajectory):
    largest = mdtraj_rmsf("traj.dcd", "top.pdb", print_largest_values=True, largest_values=2)
    assert len(largest) == 2
    assert largest[0] == MOVING_ATOM

    above_threshold = mdtraj_rmsf("traj.dcd", "top.pdb", print_largest_values=False, threshold=0.03)
    assert above_threshold.tolist() == [MOVING_ATOM]


def test_diff_wrap_box_coords_images_around_the_anchor_molecule(periodic_trajectory):
    mdtraj_topology, frames = periodic_trajectory
    wrapped_angstrom = diff_wrap_box_coords(frames[0], np.diag([BOX_NM] * 3), mdtraj_topology, anchoratoms=range(6))

    assert wrapped_angstrom.shape == frames[0].shape
    assert np.linalg.norm(wrapped_angstrom[SPLIT_WATER_O] - wrapped_angstrom[SPLIT_WATER_H]) == pytest.approx(
        0.4, abs=1e-5
    )
    # image_molecules recentres the anchor in the box: a rigid shift of the methanol, nothing more.
    shifts = wrapped_angstrom[:6] - frames[0][:6] * 10
    np.testing.assert_allclose(shifts - shifts[0], 0, atol=1e-5)
