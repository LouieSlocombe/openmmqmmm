"""Compare periodic imaging and complete fresh Python QM/MM callback timings.

Run with the development environment; no external QM executable is needed. The
baseline is loaded from the recorded Git revision, while all other callback code
comes from the working checkout. Setup, OpenMM State creation, and validation are
outside the timed regions. This measures neither native MM forces nor MD steps.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_REVISION = "d599059eae5d4c7f56acc8758fbe4e1936838daf"
sys.path.insert(0, str(REPO_ROOT))


def git_output(*args):
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True).strip()


def load_baseline(revision):
    source = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{revision}:openmmqmmm/periodic_embedding.py"], text=True
    )
    namespace = {"__name__": "periodic_imaging_benchmark_baseline"}
    exec(compile(source, f"{revision}:periodic_embedding.py", "exec"), namespace)
    return namespace["PeriodicQMGeometry"], hashlib.sha256(source.encode()).hexdigest()


def water_box(numatoms, shape):
    import numpy as np

    nwaters = (numatoms - 1) // 3
    if numatoms != 1 + 3 * nwaters or nwaters < 1:
        raise ValueError("Each size must be one QM atom plus a positive integer number of three-site waters")
    edge = (nwaters / 0.0334) ** (1 / 3)
    box = np.eye(3) * edge
    if shape == "triclinic":
        box[1, 0], box[2, 0], box[2, 1] = edge * 0.18, edge * 0.09, edge * 0.13
    rng = np.random.default_rng(20260925 + numatoms)
    oxygens = rng.random((nwaters, 3)) @ box
    offsets = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.239987, 0.927297, 0]])
    waters = (oxygens[:, None, :] + offsets).reshape(-1, 3)
    waters -= np.floor(waters @ np.linalg.inv(box)) @ box
    coords = np.vstack((np.array([0.137, 0.213, 0.319]) @ box, waters))
    bonds = [(first, first + child) for first in range(1, numatoms, 3) for child in (1, 2)]
    return coords, bonds, box


def callback_setup(coords, bonds, box):
    import numpy as np
    import openmm
    from openmm import app

    from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
    from openmmqmmm.constants import ANG_TO_BOHR
    from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider

    class CoulombQM:
        numcores = 1
        theorytype = "QM"

        def run(self, *, current_coords, current_mm_coords, mm_charges, grad=False, **_kwargs):
            delta = (current_coords[:, None, :] - current_mm_coords[None, :, :]) * ANG_TO_BOHR
            radii = np.linalg.norm(delta, axis=-1)
            charges = np.asarray(mm_charges)
            energy = np.sum(charges / radii)
            if not grad:
                return energy
            pairs = -charges[None, :, None] * delta / radii[:, :, None] ** 3
            return energy, pairs.sum(axis=1), -pairs.sum(axis=0)

    elems = ["He", *(["O", "H", "H"] * ((len(coords) - 1) // 3))]
    fragment = Fragment(elems=elems, coords=coords, conncalc=False)
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HE", chain)
    topology.addAtom("He", app.element.helium, residue)
    for _ in range((len(coords) - 1) // 3):
        residue = topology.addResidue("HOH", chain)
        topology.addAtom("O", app.element.oxygen, residue)
        topology.addAtom("H1", app.element.hydrogen, residue)
        topology.addAtom("H2", app.element.hydrogen, residue)
    atoms = list(topology.atoms())
    for first, second in bonds:
        topology.addBond(atoms[first], atoms[second])
    helium = io.StringIO(
        '<ForceField><AtomTypes><Type name="He" class="He" element="He" mass="4.0026"/></AtomTypes>'
        '<Residues><Residue name="HE"><Atom name="He" type="He"/></Residue></Residues>'
        '<NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">'
        '<Atom type="He" charge="0" sigma="0.1" epsilon="0"/></NonbondedForce></ForceField>'
    )
    mm = OpenMMTheory(
        fragment=fragment,
        topology=topology,
        forcefield=app.ForceField("tip3p.xml", helium),
        topoforce=True,
        periodic=True,
        periodic_cell_vectors=box,
        periodic_nonbonded_cutoff=8,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=CoulombQM(),
        mm_theory=mm,
        qmatoms=[0],
        qm_charge=0,
        qm_mult=1,
        openmm_externalforce=True,
    )
    providers = {
        name: RPMDQMMMForceProvider(theory, elems, 0, 1, periodic=True, cache_size=1)
        for name in ("baseline", "optimized")
    }
    # The real callback consumes an OpenMM State. Its preparation and native MM
    # evaluation are not part of these Python callback timings.
    state_system = openmm.System()
    for _ in coords:
        state_system.addParticle(1)
    state_system.setDefaultPeriodicBoxVectors(*(box * 0.1))
    integrator = openmm.VerletIntegrator(0.001)
    context = openmm.Context(state_system, integrator, openmm.Platform.getPlatformByName("Reference"))
    return theory, providers, context, integrator


def timing_summary(samples):
    median = statistics.median(samples)
    return {
        "median_ms": median,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def benchmark_case(numatoms, shape, baseline_class, current_class, warmups, repeats, callback):
    import numpy as np

    coords, bonds, box = water_box(numatoms, shape)
    geometries = {"baseline": baseline_class(numatoms, bonds, [0]), "optimized": current_class(numatoms, bonds, [0])}
    samples = {"imaging": {name: [] for name in geometries}}
    differences = {"coordinates_angstrom": 0.0}
    if callback:
        theory, providers, context, integrator = callback_setup(coords, bonds, box)
        samples["callback"] = {name: [] for name in geometries}
        differences.update(energy_kj_mol=0.0, forces_kj_mol_nm=0.0)
    for iteration in range(warmups + repeats):
        # Both versions see the identical frame; successive frames are unique
        # so every callback is a cache miss, including the first measured frame.
        frame = coords.copy()
        frame[0, 0] += (iteration + 1) * 1e-5
        if callback:
            context.setPositions(frame * 0.1)
            state = context.getState(getPositions=True)
        order = ("baseline", "optimized") if iteration % 2 == 0 else ("optimized", "baseline")
        images, results = {}, {}
        for name in order:
            start = time.perf_counter_ns()
            images[name] = geometries[name].image(frame, box)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if iteration >= warmups:
                samples["imaging"][name].append(elapsed)
        if callback:
            for name in order:
                theory._periodic_geometry = geometries[name]
                start = time.perf_counter_ns()
                results[name] = providers[name](state)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if iteration >= warmups:
                    samples["callback"][name].append(elapsed)
        coord_error = float(np.max(np.abs(images["baseline"] - images["optimized"])))
        differences["coordinates_angstrom"] = max(differences["coordinates_angstrom"], coord_error)
        np.testing.assert_allclose(images["optimized"], images["baseline"], atol=1e-10, rtol=0)
        if callback:
            energy_error = abs(results["baseline"][0] - results["optimized"][0])
            force_error = float(np.max(np.abs(results["baseline"][1] - results["optimized"][1])))
            differences["energy_kj_mol"] = max(differences["energy_kj_mol"], energy_error)
            differences["forces_kj_mol_nm"] = max(differences["forces_kj_mol_nm"], force_error)
            np.testing.assert_allclose(results["optimized"][0], results["baseline"][0], atol=1e-8, rtol=0)
            np.testing.assert_allclose(results["optimized"][1], results["baseline"][1], atol=1e-7, rtol=0)
    result = {"atoms": numatoms, "cell": shape, "max_abs_difference": differences, "timings": {}}
    for scope, variants in samples.items():
        summary = {name: timing_summary(values) for name, values in variants.items()}
        summary["median_speedup"] = summary["baseline"]["median_ms"] / summary["optimized"]["median_ms"]
        result["timings"][scope] = summary
    if callback:
        result["callback_counts"] = {
            name: {"evaluations": provider.evaluation_count, "cache_hits": provider.cache_hits}
            for name, provider in providers.items()
        }
        if any(
            provider.cache_hits or provider.evaluation_count != warmups + repeats for provider in providers.values()
        ):
            raise AssertionError("Callback timing unexpectedly included a cache hit or missed an evaluation")
        del context, integrator
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION)
    parser.add_argument("--atoms", type=int, nargs="+", default=[3001, 30001, 99991])
    parser.add_argument("--cells", choices=["orthorhombic", "triclinic"], nargs="+", default=["orthorhombic"])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--skip-callback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 1:
        parser.error("--warmups and --repeats must both be positive")
    output = args.output.resolve() if args.output else None
    logging.disable(logging.CRITICAL)

    import ase
    import numpy as np
    import openmm

    from openmmqmmm.periodic_embedding import PeriodicQMGeometry

    baseline_class, baseline_hash = load_baseline(args.baseline_revision)
    cpu = (
        next(
            (
                line.partition(":")[2].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if "model name" in line
            ),
            platform.processor(),
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "cpu": cpu,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "numpy": np.__version__,
            "ase": ase.__version__,
            "openmm": openmm.__version__,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS")
            },
        },
        "baseline_revision": git_output("rev-parse", args.baseline_revision),
        "baseline_source_sha256": baseline_hash,
        "working_tree_head": git_output("rev-parse", "HEAD"),
        "optimized_source_sha256": hashlib.sha256(
            (REPO_ROOT / "openmmqmmm/periodic_embedding.py").read_bytes()
        ).hexdigest(),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "ordering": "Paired identical frames; baseline/optimized order alternates each iteration.",
        "callback_scope": (
            "Complete RPMDQMMMForceProvider.__call__ on precreated OpenMM States, fresh cache misses, "
            "real electrostatic QMMMTheory, analytic Coulomb QM backend, TIP3P charges. Excludes native "
            "OpenMM dispatcher, native MM force evaluation, integration, reporters, setup and State creation."
        ),
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="openmmqmmm-imaging-benchmark-") as scratch, contextlib.chdir(scratch):
        for shape in args.cells:
            for numatoms in args.atoms:
                case = benchmark_case(
                    numatoms,
                    shape,
                    baseline_class,
                    PeriodicQMGeometry,
                    args.warmups,
                    args.repeats,
                    not args.skip_callback,
                )
                report["cases"].append(case)
                print(json.dumps(case), flush=True)
    if output:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
