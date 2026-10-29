"""Compare warm QM/MM accuracy and throughput; run with --help for JSON configuration."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import os
import platform
import runpy
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import openmm

import openmmqmmm
from openmmqmmm import Fragment, OpenMMTheory, ORCATheory, QMMMTheory

DEFAULT_CONFIG = {
    "warmup_cycles": 1,
    "repeats": 3,
    "reference_case": 0,
    "fixture": {},
    "matrix": {
        "orca_numcores": [1],
        "openmm_numcores": [1],
        "platform": ["Reference"],
        "scratch": [tempfile.gettempdir()],
        "numerical": [{"name": "hf-sto3g", "orcasimpleinput": "! HF STO-3G tightscf"}],
    },
}


def tiny_fixture(case: dict[str, Any], options: dict[str, Any]) -> tuple[QMMMTheory, np.ndarray]:
    """Return H2 in one external charge and three reproducible displaced geometries."""
    coords = np.array([[0.0, 0, 0], [0.0, 0, 0.74], [4.0, 0.5, 0.2]])
    fragment = Fragment(elems=["H", "H", "He"], coords=coords, conncalc=False)
    mm = OpenMMTheory(
        fragment=fragment,
        dummysystem=True,
        platform=case["platform"],
        numcores=case["openmm_numcores"],
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    qm = ORCATheory(
        orcadir=options.get("orcadir"),
        numcores=case["orca_numcores"],
        **{key: value for key, value in case["numerical"].items() if key != "name"},
    )
    theory = QMMMTheory(
        fragment=fragment,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0, 1],
        charges=[0.0, 0.0, 0.1],
        qm_charge=0,
        qm_mult=1,
    )
    rng = np.random.default_rng(20260924)
    snapshots = np.stack(
        [coords, coords + rng.normal(0, 0.015, coords.shape), coords + rng.normal(0, 0.015, coords.shape)]
    )
    return theory, snapshots


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _matrix(config: dict[str, Any], base_dir: Path) -> list[dict[str, Any]]:
    matrix = config.get("matrix", DEFAULT_CONFIG["matrix"])
    keys = list(DEFAULT_CONFIG["matrix"])
    if set(matrix) != set(keys) or any(not isinstance(matrix[key], list) or not matrix[key] for key in keys):
        raise ValueError(f"matrix must contain nonempty lists for {keys}")
    cases = [dict(zip(keys, values, strict=True)) for values in itertools.product(*(matrix[key] for key in keys))]
    for case in cases:
        for key in ("orca_numcores", "openmm_numcores"):
            _positive_integer(case[key], key)
        if not isinstance(case["numerical"], dict) or not case["numerical"].get("name"):
            raise ValueError("Each numerical setting requires a name and ORCA constructor options")
        case["scratch"] = str((base_dir / case["scratch"]).resolve())
    return cases


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": min(values),
        "max": max(values),
        "std": float(np.std(values)),
    }


def _metadata() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "openmm": openmm.__version__,
        "openmmqmmm": openmmqmmm.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "thread_environment": {
            key: os.getenv(key) for key in ("OMP_NUM_THREADS", "OPENMM_CPU_THREADS", "MKL_NUM_THREADS")
        },
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _evaluate(theory: Any, coords: np.ndarray) -> tuple[float, np.ndarray, float]:
    started = time.perf_counter()
    energy, gradient = theory.run(current_coords=coords.copy(), grad=True)
    elapsed = time.perf_counter() - started
    gradient = np.array(gradient, dtype=float, copy=True)
    if gradient.shape != coords.shape or not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
        raise ValueError("Benchmark requires finite energy and a full-system N x 3 gradient")
    return float(energy), -gradient, elapsed


def _case(
    case: dict[str, Any], config: dict[str, Any], factory: Callable, expected_snapshots: str | None
) -> tuple[dict[str, Any], str | None]:
    root = Path(case["scratch"])
    root.mkdir(parents=True, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="openmmqmmm-benchmark-", dir=root)
    row = {"settings": case, "scratch_directory": workdir, "status": "error", "samples": []}
    try:
        with contextlib.chdir(workdir):
            started = time.perf_counter()
            theory, snapshots = factory(case, config.get("fixture", {}).get("options", {}))
            snapshots = np.asarray(snapshots, dtype=float)
            if (
                snapshots.ndim != 3
                or snapshots.shape[-1] != 3
                or len(snapshots) < 2
                or not np.all(np.isfinite(snapshots))
            ):
                raise ValueError("Fixture must supply at least two finite (N, 3) snapshots")
            fingerprint = hashlib.sha256(snapshots.astype("<f8").tobytes()).hexdigest()
            if expected_snapshots is not None and fingerprint != expected_snapshots:
                raise ValueError("Every matrix case must use identical snapshots in identical atom order")
            expected_snapshots = fingerprint
            row["snapshot_sha256"] = fingerprint
            row["snapshots_angstrom"] = snapshots.tolist()
            row["setup_seconds"] = time.perf_counter() - started
            row["atoms"] = snapshots.shape[1]
            row["qm_atoms"] = len(theory.qmatoms) if hasattr(theory, "qmatoms") else None
            mm, qm = getattr(theory, "mm_theory", None), getattr(theory, "qm_theory", None)
            row["backend"] = {
                "qm_class": type(qm).__name__,
                "orca_directory": getattr(qm, "orcadir", None),
                "qm_numcores": getattr(qm, "numcores", None),
                "openmm_platform": getattr(mm, "platform_choice", None),
                "openmm_properties": getattr(mm, "properties", None),
            }
            row["elements"] = list(theory.elems) if hasattr(theory, "elems") else None
            row["qm_indices"] = list(theory.qmatoms) if hasattr(theory, "qmatoms") else None
            warmup_times = []
            for _ in range(config.get("warmup_cycles", 1)):
                warmup_times.extend(_evaluate(theory, coords)[2] for coords in snapshots)
            row["warmup_wall_seconds"] = warmup_times
            for repeat in range(config.get("repeats", 3)):
                for snapshot, coords in enumerate(snapshots):
                    energy, forces, elapsed = _evaluate(theory, coords)
                    row["samples"].append(
                        {
                            "repeat": repeat,
                            "snapshot": snapshot,
                            "energy_eh": energy,
                            "forces_eh_bohr": forces.tolist(),
                            "wall_seconds": elapsed,
                        }
                    )
            row["wall_seconds"] = _distribution([sample["wall_seconds"] for sample in row["samples"]])
            output = Path(f"{getattr(qm, 'filename', 'orca')}.out")
            row["backend"]["orca_version_line"] = None
            if output.is_file():
                with output.open(errors="replace") as stream:
                    row["backend"]["orca_version_line"] = next(
                        (line.strip() for line in stream if "Program Version" in line), None
                    )
            row["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - preserve diagnostics and continue the benchmark matrix
        row["error"] = f"{type(error).__name__}: {error}"
    finally:
        row["scratch_retained"] = bool(config.get("keep_scratch", False) or row["status"] != "ok")
        if not row["scratch_retained"]:
            shutil.rmtree(workdir)
    return row, expected_snapshots


def _compare(rows: list[dict[str, Any]], reference_index: int) -> None:
    reference = rows[reference_index]
    if reference["status"] != "ok":
        return
    reference_samples = {(sample["repeat"], sample["snapshot"]): sample for sample in reference["samples"]}
    for row in rows:
        if row["status"] != "ok":
            continue
        origins = {sample["repeat"]: sample["energy_eh"] for sample in row["samples"] if sample["snapshot"] == 0}
        for sample in row["samples"]:
            target = reference_samples[sample["repeat"], sample["snapshot"]]
            reference_origin = reference_samples[sample["repeat"], 0]["energy_eh"]
            delta = np.asarray(sample["forces_eh_bohr"]) - np.asarray(target["forces_eh_bohr"])
            norm = float(np.linalg.norm(target["forces_eh_bohr"]))
            sample["error_vs_reference"] = {
                "energy_eh": sample["energy_eh"] - target["energy_eh"],
                "relative_energy_eh": (sample["energy_eh"] - origins[sample["repeat"]])
                - (target["energy_eh"] - reference_origin),
                "force_max_component_eh_bohr": float(np.max(np.abs(delta))),
                "force_rms_component_eh_bohr": float(np.sqrt(np.mean(delta**2))),
                "force_relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
            }
        row["median_wall_ratio_vs_reference"] = row["wall_seconds"]["median"] / reference["wall_seconds"]["median"]


def run_benchmark(
    config: dict[str, Any], *, base_dir: Path | None = None, factory: Callable | None = None
) -> dict[str, Any]:
    """Run sequential warm cases and compare all force components against the reference case."""
    base_dir = Path.cwd() if base_dir is None else base_dir
    for key, default in (("warmup_cycles", 1), ("repeats", 3)):
        _positive_integer(config.get(key, default), key)
    cases = _matrix(config, base_dir)
    reference = config.get("reference_case", 0)
    if isinstance(reference, bool) or not isinstance(reference, int) or not 0 <= reference < len(cases):
        raise ValueError("reference_case must index the expanded matrix")
    if factory is None:
        specification = config.get("fixture", {}).get("factory")
        if specification:
            filename, name = specification.rsplit(":", 1)
            factory = runpy.run_path(str((base_dir / filename).resolve()))[name]
        else:
            factory = tiny_fixture
    rows, fingerprint = [], None
    metadata = _metadata()
    for case in cases:
        row, fingerprint = _case(case, config, factory, fingerprint)
        rows.append(row)
    _compare(rows, reference)
    return {"schema_version": 1, "metadata": metadata, "config": config, "reference_case": reference, "cases": rows}


def main() -> None:
    """Write a JSON report or print an editable default configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON matrix; local fixture factories are trusted Python code")
    parser.add_argument("--output", type=Path, default=Path("qmmm-benchmark.json"))
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.print_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2))
        return
    config = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG
    result = run_benchmark(config, base_dir=args.config.resolve().parent if args.config else Path.cwd())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['cases'])} cases to {args.output}")
    if any(case["status"] != "ok" for case in result["cases"]):
        parser.exit(1, "Some cases failed; their errors and retained scratch directories are recorded in the report.\n")


if __name__ == "__main__":
    main()
