"""Deterministic finite-cluster diagnostics; this is not a periodic QM solver.

Run ``python examples/periodic_embedding_validation.py --output baseline.json``
from an installed checkout. All package-generated scratch files stay temporary.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import logging
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import ase
import numpy as np
import openmm
from openmm import unit

import openmmqmmm
from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory, constants
from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider, add_rpmd_python_force

QM_CHARGES = np.array([0.6, -0.25])
QM_COORDS = np.array([[-0.65, -0.2, 0.0], [0.65, 0.2, 0.0]])
MM_CHARGE = 0.15
BOND_LENGTH = 0.8
SPACING = 8.0
FORCE_FACTOR = constants.HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
ENERGY_FACTOR = constants.HARTREE_TO_KJ_PER_MOL


class CoulombQM:
    """Fixed unequal QM probe charges coupled only to the finite MM cluster."""

    numcores = 1
    theorytype = "QM"

    def run(self, *, current_coords: Any, current_mm_coords: Any, mm_charges: Any, grad: bool = False, **_: Any) -> Any:
        delta = np.asarray(current_coords)[:, None, :] - np.asarray(current_mm_coords)[None, :, :]
        delta *= constants.ANG_TO_BOHR
        radii = np.linalg.norm(delta, axis=-1)
        products = QM_CHARGES[:, None] * np.asarray(mm_charges)[None, :]
        energy = float(np.sum(products / radii))
        pairs = -products[:, :, None] * delta / radii[:, :, None] ** 3
        return (energy, pairs.sum(axis=1), -pairs.sum(axis=0)) if grad else energy


def _molecule(center: Any, direction: Any) -> np.ndarray:
    direction = np.asarray(direction, dtype=float)
    half_bond = 0.5 * BOND_LENGTH * direction / np.linalg.norm(direction)
    return np.asarray(center) + np.array([-half_bond, half_bond])


def _make_theory(coords: np.ndarray, edge: float) -> QMMMTheory:
    fragment = Fragment(elems=["He"] * len(coords), coords=coords, conncalc=False)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    mm.periodic = True
    mm.system.setDefaultPeriodicBoxVectors(*(np.eye(3) * edge * 0.1))
    charges = [0.0, 0.0] + [MM_CHARGE, -MM_CHARGE] * ((len(coords) - 2) // 2)
    nonbonded = mm.nonbonded_force
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(0.7)
    nonbonded.setEwaldErrorTolerance(1e-6)
    nonbonded.setUseDispersionCorrection(False)
    for atom, charge in enumerate(charges):
        nonbonded.setParticleParameters(atom, charge, 0.1, 0.0)
    atoms = list(mm.topology.atoms())
    bonds = openmm.HarmonicBondForce()
    bonds.setUsesPeriodicBoundaryConditions(True)
    for first in range(2, len(coords), 2):
        mm.topology.addBond(atoms[first], atoms[first + 1])
        bonds.addBond(first, first + 1, BOND_LENGTH * 0.1, 1000.0)
        nonbonded.addException(first, first + 1, 0.0, 0.1, 0.0)
    mm.system.addForce(bonds)
    for index in reversed(range(mm.system.getNumForces())):
        if isinstance(mm.system.getForce(index), openmm.CMMotionRemover):
            mm.system.removeForce(index)
    mm.charges = charges
    return QMMMTheory(
        fragment=fragment,
        qm_theory=CoulombQM(),
        mm_theory=mm,
        qmatoms=[0, 1],
        qm_charge=0,
        qm_mult=1,
    )


def _paired_states(coords: np.ndarray) -> dict[str, np.ndarray]:
    other = coords.copy()
    other[1] += [0.16, -0.09, 0.07]
    other[2:4] += [0.08, 0.03, -0.04]
    return {"A": coords, "B": other}


def _sample(theory: QMMMTheory, coords: np.ndarray, probes: list[int]) -> dict[str, Any]:
    energy, gradient = theory.run(current_coords=coords, grad=True)
    return {
        "energy_kj_mol": {
            "qm": float(theory.QMenergy * ENERGY_FACTOR),
            "mm": float(theory.MMenergy * ENERGY_FACTOR),
            "total": float(energy * ENERGY_FACTOR),
        },
        "probe_forces_kj_mol_nm": {
            "qm_embedding": (-theory.QM_PC_gradient[probes] * FORCE_FACTOR).tolist(),
            "mm": (-theory.MMgradient[probes] * FORCE_FACTOR).tolist(),
            "total": (-gradient[probes] * FORCE_FACTOR).tolist(),
        },
    }


def _scan(cases: list[tuple[float, np.ndarray]], probes: list[int]) -> list[dict[str, Any]]:
    rows = []
    for edge, coords in cases:
        theory = _make_theory(coords, edge)
        samples = {name: _sample(theory, state, probes) for name, state in _paired_states(coords).items()}
        rows.append(
            {
                "edge_angstrom": edge,
                "molecules": (len(coords) - 2) // 2,
                "atoms": len(coords),
                "states": samples,
                "relative_energy_B_minus_A_kj_mol": {
                    component: samples["B"]["energy_kj_mol"][component] - samples["A"]["energy_kj_mol"][component]
                    for component in ("qm", "mm", "total")
                },
            }
        )
        # Reference Contexts retain considerable PME memory; release between cells.
        del theory
    reference = rows[-1]
    for row in rows:
        row["relative_energy_error_kj_mol"] = {
            key: value - reference["relative_energy_B_minus_A_kj_mol"][key]
            for key, value in row["relative_energy_B_minus_A_kj_mol"].items()
        }
        errors = {}
        for component in ("qm_embedding", "mm", "total"):
            delta = np.array([row["states"][state]["probe_forces_kj_mol_nm"][component] for state in ("A", "B")])
            delta -= np.array([reference["states"][state]["probe_forces_kj_mol_nm"][component] for state in ("A", "B")])
            errors[component] = {}
            for region, atom_slice in (("qm_atoms", slice(0, 2)), ("mm_atoms", slice(2, None))):
                norms = np.linalg.norm(delta[:, atom_slice], axis=-1)
                errors[component][region] = {
                    "rms_vector_kj_mol_nm": float(np.sqrt(np.mean(norms**2))),
                    "max_vector_kj_mol_nm": float(np.max(norms)),
                }
        row["force_errors_vs_largest"] = errors
    return rows


def _extent_coords(shell: int) -> np.ndarray:
    cells = [cell for cell in itertools.product(range(-shell, shell + 1), repeat=3) if cell != (0, 0, 0)]
    cells.sort(key=lambda cell: (max(map(abs, cell)), cell))
    molecules = []
    for i, j, k in cells:
        center = SPACING * np.array([i, j, k]) + [0.9, 0.5, -0.4]
        direction = [0.6 + np.cos(i + 2 * j), 0.5 + np.sin(j + 3 * k), 0.3 + np.cos(2 * i - k)]
        molecules.append(_molecule(center, direction))
    return np.vstack([QM_COORDS, *molecules])


def _nve(case: str, timestep_fs: float, duration_fs: float = 12.0) -> dict[str, Any]:
    center = [14.97 if case == "switch" else 8.0, 1.1, 0.4]
    coords = np.vstack([QM_COORDS, _molecule(center, [0.25, 0.4, 0.1])])
    theory = _make_theory(coords, 30.0)
    theory.openmm_externalforce = True
    provider = RPMDQMMMForceProvider(theory, ["He"] * 4, 0, 1, periodic=True)
    add_rpmd_python_force(theory.mm_theory.system, provider, periodic=True)
    # Explicit velocity Verlet stores positions and velocities at the same time,
    # including the initial physical velocities. OpenMM computes the reported KE.
    integrator = openmm.CustomIntegrator(timestep_fs * 0.001)
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.addComputePerDof("x", "x+dt*v")
    integrator.addComputePerDof("v", "v+0.5*dt*f/m")
    integrator.setKineticEnergyExpression("0.5*m*v*v")
    context = openmm.Context(theory.mm_theory.system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(coords * 0.1)
    velocities = np.array([[-0.02, 0.015, 0], [0.03, -0.01, 0.012], [1.0, 0.03, -0.02], [1.0, 0.03, -0.02]])
    context.setVelocities(velocities)
    box = np.eye(3) * 30.0
    trace, events = [], []
    previous_branch = None
    final_coords = coords.copy()
    steps = round(duration_fs / timestep_fs)
    for step in range(steps + 1):
        if step:
            integrator.step(1)
        state = context.getState(getEnergy=True, getPositions=True)
        current = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        imaged = theory._periodic_geometry.image(current, box)
        branch = np.rint(((imaged[2] - imaged[0]) - (current[2] - current[0])) / 30.0).astype(int)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        total = potential + kinetic
        if previous_branch is not None and np.any(branch != previous_branch):
            prior_imaged = imaged.copy()
            prior_imaged[2:4] += (previous_branch - branch) @ box
            old_qm = theory.qm_theory.run(
                current_coords=prior_imaged[:2],
                current_mm_coords=prior_imaged[2:],
                mm_charges=[MM_CHARGE, -MM_CHARGE],
            )
            new_qm = theory.qm_theory.run(
                current_coords=imaged[:2],
                current_mm_coords=imaged[2:],
                mm_charges=[MM_CHARGE, -MM_CHARGE],
            )
            events.append(
                {
                    "time_fs": step * timestep_fs,
                    "from_image": previous_branch.tolist(),
                    "to_image": branch.tolist(),
                    "step_total_energy_change_kj_mol": total - trace[-1]["total_kj_mol"],
                    "same_geometry_qm_branch_jump_kj_mol": (new_qm - old_qm) * ENERGY_FACTOR,
                }
            )
        trace.append(
            {
                "time_fs": step * timestep_fs,
                "potential_kj_mol": potential,
                "kinetic_kj_mol": kinetic,
                "total_kj_mol": total,
                "mm_molecule_image": branch.tolist(),
            }
        )
        previous_branch = branch
        final_coords = current
    energies = np.array([row["total_kj_mol"] for row in trace])
    drift = energies - energies[0]
    result = {
        "case": case,
        "timestep_fs": timestep_fs,
        "duration_fs": duration_fs,
        "steps": steps,
        "initial_coords_angstrom": coords.tolist(),
        "initial_velocities_nm_ps": velocities.tolist(),
        "atom_masses_dalton": [
            theory.mm_theory.system.getParticleMass(atom).value_in_unit(unit.dalton) for atom in range(4)
        ],
        "final_displacements_angstrom": (final_coords - coords).tolist(),
        "net_energy_change_kj_mol": float(drift[-1]),
        "net_drift_kj_mol_ps": float(drift[-1] / (duration_fs * 0.001)),
        "max_abs_energy_deviation_kj_mol": float(np.max(np.abs(drift))),
        "image_switch_count": len(events),
        "events": events,
        "trace": trace,
    }
    del context, integrator
    return result


def run_validation() -> dict[str, Any]:
    """Return reproducible JSON data without leaving package scratch files behind."""
    # Some package setup paths print; stdout remains valid JSON for the CLI.
    with (
        TemporaryDirectory(prefix="finite-periodic-validation-") as scratch,
        contextlib.chdir(scratch),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        logger = logging.getLogger("openmmqmmm")
        previous_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            fixed = np.vstack(
                [
                    QM_COORDS,
                    _molecule([5, 1, 1], [1, 0.3, -0.2]),
                    _molecule([11.7, -2.3, 1], [0.2, 1, 0.4]),
                    _molecule([-13.5, 3, -2], [0.4, -0.3, 1]),
                ]
            )
            box_probes = list(range(len(fixed)))
            extent_probes = list(range(len(_extent_coords(1))))
            box_scan = _scan([(edge, fixed) for edge in (24.0, 30.0, 40.0, 60.0)], box_probes)
            extent_scan = _scan(
                [((2 * shell + 1) * SPACING, _extent_coords(shell)) for shell in (1, 2, 3)], extent_probes
            )
            nve = [_nve(case, timestep) for case in ("smooth", "switch") for timestep in (0.5, 0.25)]
        finally:
            logger.setLevel(previous_level)
    package_dir = Path(openmmqmmm.__file__).parent
    source_files = [
        Path(__file__),
        package_dir / "qmmm.py",
        package_dir / "periodic_embedding.py",
        package_dir / "openmm" / "rpmd_force.py",
        package_dir / "openmm" / "theory.py",
    ]
    return {
        "schema_version": 1,
        "metadata": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ase": ase.__version__,
            "openmm": openmm.__version__,
            "platform": "Reference",
            "seed": None,
            "openmmqmmm": openmmqmmm.__version__,
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
            "scope": "Synthetic fixed-charge Coulomb surrogate, not an electronic-structure or bulk convergence claim.",
            "qm_model": "E=sum(q_QM*q_MM/r) in atomic units; no QM internal potential, polarization or periodic sum.",
            "qm_probe_charges_e": QM_CHARGES.tolist(),
            "qm_coords_A_angstrom": QM_COORDS.tolist(),
            "mm_molecule_charges_e": [MM_CHARGE, -MM_CHARGE],
            "mm_bond_length_angstrom": BOND_LENGTH,
            "mm_bond_k_kj_mol_nm2": 1000.0,
            "mm_intramolecule_electrostatics": "excluded",
            "mm_electrostatics": "PME",
            "mm_cutoff_angstrom": 7.0,
            "mm_ewald_error_tolerance": 1e-6,
            "lennard_jones_epsilon_kj_mol": 0.0,
            "atom_element": "He",
            "paired_state_B": {
                "atom_1_displacement_angstrom": [0.16, -0.09, 0.07],
                "atoms_2_3_translation_angstrom": [0.08, 0.03, -0.04],
            },
            "relative_energy": "B minus A; errors relative to largest case within each scan.",
            "force_errors": "RMS and maximum Euclidean force-vector error over probe atoms and both states A/B.",
            "qm_embedding_force": "QM energy derivative on both QM and MM atoms; total adds native MM force.",
            "nve_integrator": "Explicit velocity Verlet CustomIntegrator; same-time OpenMM getState kinetic energy.",
            "nve_controls": "All QM/MM atoms move; fixed box; no thermostat, barostat, constraints or COM removal.",
            "nve_event_jump": "QM energy difference at the same event geometry using new versus old molecular image.",
        },
        "box_scan": {
            "protocol": "Fixed contents and Cartesian A/B states; box dilation changes density and selected images.",
            "probe_atom_indices": box_probes,
            "coords_A_angstrom": fixed.tolist(),
            "rows": box_scan,
        },
        "extent_scan": {
            "protocol": "Nested neutral whole molecules; fixed local geometry and underlying lattice density.",
            "lattice_spacing_angstrom": SPACING,
            "lattice_density_molecules_per_angstrom3": 1 / SPACING**3,
            "cavity": "Origin lattice site omitted for QM region; every cell contains one vacant lattice site.",
            "shells": [1, 2, 3],
            "probe_atom_indices": extent_probes,
            "probe_coords_A_angstrom": _extent_coords(1).tolist(),
            "rows": extent_scan,
        },
        "nve": nve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this file (default: stdout).")
    args = parser.parse_args()
    payload = json.dumps(run_validation(), indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)


if __name__ == "__main__":
    main()
