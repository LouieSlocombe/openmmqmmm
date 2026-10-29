"""Audit RPMD callback serialization and bead/contraction/reporting cost with a synthetic QM backend."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import runpy
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import openmm
from openmm import unit

from openmmqmmm import Fragment, MolecularDynamicsEngine, OpenMMTheory, QMMMTheory, constants
from openmmqmmm.openmm.rpmd_force import RPMDExternalQMForceProvider

_COMMON = runpy.run_path(str(Path(__file__).with_name("benchmark_qmmm.py")))


class HarmonicQM:
    """A deterministic harmonic backend with measured concurrent entry and optional latency."""

    numcores = 1
    theorytype = "QM"

    def __init__(self, delay_seconds: float = 0.002) -> None:
        self.delay_seconds = delay_seconds
        self.calls = 0
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()

    def run(self, *, current_coords: Any, grad: bool = False, **_: Any) -> Any:
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak_active = max(self.active, self.peak_active)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            coords = np.asarray(current_coords) * constants.ANG_TO_BOHR
            energy = 0.0005 * np.sum(coords**2)
            return (energy, 0.001 * coords) if grad else energy
        finally:
            with self.lock:
                self.active -= 1


def serialization_probe(workers: int = 4, delay_seconds: float = 0.01) -> dict[str, Any]:
    """Submit distinct states simultaneously and measure entry into one shared provider's backend."""
    _COMMON["_positive_integer"](workers, "workers")
    system = openmm.System()
    system.addParticle(1.0)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    states = []
    for index in range(workers):
        context.setPositions([[0.1 + 0.01 * index, 0, 0]])
        states.append(context.getState(getPositions=True))
    backend = HarmonicQM(delay_seconds)
    provider = RPMDExternalQMForceProvider(backend, ["H"], 0, 2)
    barrier = threading.Barrier(workers)

    def evaluate(state: openmm.State) -> Any:
        barrier.wait(timeout=30)
        return provider(state)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, states))
    elapsed = time.perf_counter() - started
    return {
        "submitted_workers": workers,
        "distinct_evaluations": provider.evaluation_count,
        "peak_simultaneous_backend_calls": backend.peak_active,
        "synthetic_delay_seconds": delay_seconds,
        "wall_seconds": elapsed,
        "energies_kj_mol": [result[0] for result in results],
    }


def _case(
    beads: int,
    qm_copies: int,
    energy_frequency: int,
    *,
    steps: int,
    warmup_steps: int,
    repeats: int,
    delay_seconds: float,
    platform: str,
    mm_threads: int,
) -> dict[str, Any]:
    coords = np.array([[-0.35, 0.02, 0], [0.35, 0, 0.03]])
    fragment = Fragment(elems=["H", "H"], coords=coords, charge=0, mult=1, conncalc=False)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform=platform,
        numcores=mm_threads,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    backend = HarmonicQM(delay_seconds)
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=backend,
        mm_theory=mm,
        qmatoms=[0, 1],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
    )
    row = {
        "beads": beads,
        "qm_copies": qm_copies,
        "energy_frequency": energy_frequency,
        "trajectory_frequency": 1,
        "steps_per_repeat": steps,
        "warmup_steps": warmup_steps,
        "platform": platform,
        "mm_threads": mm_threads,
        "samples": [],
    }
    with contextlib.closing(
        MolecularDynamicsEngine(
            fragment=fragment,
            theory=theory,
            integrator="RPMDIntegrator",
            rpmd_num_copies=beads,
            rpmd_qm_num_copies=qm_copies,
            rpmd_energy_frequency=energy_frequency,
            traj_frequency=1,
            trajectory_file_option="DCD",
            restartfile_frequency=0,
            timestep=0.0001,
            temperature=10,
        )
    ) as engine:

        def seed(current: MolecularDynamicsEngine) -> None:
            rng = np.random.default_rng(20260928)
            integrator = current.simulation.integrator
            integrator.setApplyThermostat(False)
            for bead in range(beads):
                integrator.setPositions(bead, (coords + rng.normal(0, 0.01, coords.shape)) * unit.angstrom)
                integrator.setVelocities(bead, rng.normal(0, 0.001, coords.shape) * unit.nanometer / unit.picosecond)

        engine.run(simulation_steps=warmup_steps, pre_dynamics_hook=seed)
        provider = engine.rpmd_force_provider
        row["warmup_evaluations"] = provider.evaluation_count
        for repeat in range(repeats):
            evaluations, hits = provider.evaluation_count, provider.cache_hits
            started = time.perf_counter()
            engine.run(simulation_steps=steps, restart=True)
            elapsed = time.perf_counter() - started
            row["samples"].append(
                {
                    "repeat": repeat,
                    "wall_seconds": elapsed,
                    "wall_seconds_per_step": elapsed / steps,
                    "fresh_qm_evaluations": provider.evaluation_count - evaluations,
                    "cache_hits": provider.cache_hits - hits,
                }
            )
        row["peak_simultaneous_backend_calls"] = backend.peak_active
        row["total_backend_calls"] = backend.calls
        row["wall_seconds_per_step"] = _COMMON["_distribution"](
            [sample["wall_seconds_per_step"] for sample in row["samples"]]
        )
    return row


def run_benchmark(
    *,
    beads: tuple[int, ...] = (1, 4, 8),
    energy_frequencies: tuple[int, ...] = (0, 1),
    steps: int = 4,
    warmup_steps: int = 1,
    repeats: int = 3,
    delay_seconds: float = 0.002,
    platform: str = "Reference",
    mm_threads: int = 1,
) -> dict[str, Any]:
    """Measure fresh evaluations and wall time for full-ring and centroid QM propagation."""
    for name, value in (
        ("steps", steps),
        ("warmup_steps", warmup_steps),
        ("repeats", repeats),
        ("mm_threads", mm_threads),
    ):
        _COMMON["_positive_integer"](value, name)
    for count in beads:
        _COMMON["_positive_integer"](count, "beads")
    if not beads or not energy_frequencies:
        raise ValueError("beads and energy_frequencies must not be empty")
    if not np.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("delay_seconds must be finite and nonnegative")
    for frequency in energy_frequencies:
        if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency < 0:
            raise ValueError("energy_frequencies must contain nonnegative integers")
    rows = []
    with tempfile.TemporaryDirectory(prefix="openmmqmmm-rpmd-benchmark-") as scratch:
        for count, frequency in itertools.product(beads, energy_frequencies):
            for copies in sorted({1, count}):
                directory = Path(scratch) / f"beads{count}-qm{copies}-energy{frequency}"
                directory.mkdir()
                with contextlib.chdir(directory):
                    rows.append(
                        _case(
                            count,
                            copies,
                            frequency,
                            steps=steps,
                            warmup_steps=warmup_steps,
                            repeats=repeats,
                            delay_seconds=delay_seconds,
                            platform=platform,
                            mm_threads=mm_threads,
                        )
                    )
    return {
        "schema_version": 1,
        "metadata": _COMMON["_metadata"](),
        "synthetic_delay_seconds": delay_seconds,
        "serialization_probe": serialization_probe(),
        "cases": rows,
    }


def main() -> None:
    """Write the reproducible synthetic scaling audit as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--energy-frequencies", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=0.002)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--mm-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("rpmd-benchmark.json"))
    args = vars(parser.parse_args())
    output = args.pop("output")
    report = run_benchmark(**args)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(report['cases'])} cases to {output}")


if __name__ == "__main__":
    main()
