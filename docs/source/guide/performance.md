# Measuring QM/MM performance

Run `examples/benchmark_qmmm.py` from an installed checkout to compare settings on the
same displaced geometries. The default is a small H2 QM region with one external charge;
it checks the workflow and is not a biomolecular performance benchmark.

```bash
python examples/benchmark_qmmm.py --print-config > benchmark.json
OPENMMQMMM_ORCADIR=/path/to/orca python examples/benchmark_qmmm.py \
    --config benchmark.json --output measurements.json
```

The JSON `matrix` expands in this order: `orca_numcores`, `openmm_numcores`, `platform`,
`scratch`, `numerical`. Each is a nonempty list. `reference_case` indexes the expanded
matrix, starting at zero. Numerical settings are dictionaries with a unique descriptive
`name` and `ORCATheory` options, for example:

```json
"numerical": [
  {"name": "pbe-nori", "orcasimpleinput": "! PBE def2-SVP NORI tightscf"},
  {"name": "pbe-ri", "orcasimpleinput": "! PBE def2-SVP def2/J RI tightscf"}
]
```

Use a scientifically appropriate reference for the method being compared. RI-J and
RIJCOSX may already be enabled by the input; a NORI comparison is useful only where
it tests the intended approximation. FMM and other options can be supplied through
`orcablocks`. Verify their meaning for your installed ORCA version before comparing.
The harness passes inputs through unchanged and does not claim that any setting is faster.

Each case constructs a fresh theory in a unique directory under its selected scratch
root. It warms every snapshot for `warmup_cycles` (default 1), then measures `repeats`
complete traversals (default 3), retaining the same theory and ORCA GBW between calls.
Setup and warmup times are separate from measured wall times. Cases run sequentially;
repeat the matrix in a different order to check drift and contention before interpreting
small differences. Resource counts describe one evaluation, not concurrent trajectories.

The report contains input settings, software and host metadata, thread environment,
snapshot coordinates and hashes, raw timings, and mean/median/p95/min/max/std wall time.
For every measured snapshot it reports total energy, all atomic forces, and errors against
the same repeat and snapshot of the reference: absolute energy, energy relative to snapshot
zero, maximum and RMS force-component error, and relative force L2 error. A zero reference
force norm yields `null` for the relative metric. Units are Hartree and Hartree/Bohr.
`median_wall_ratio_vs_reference` below one indicates lower measured time; assess accuracy
and measurement variability before treating it as a useful speedup.

Failed cases retain their scratch directory and error in the report, and the command exits
unsuccessfully. Successful cases remove scratch unless `keep_scratch` is true. A failed
reference disables all accuracy comparisons. Never mix different atom orderings or
geometries between cases.

## Production fixtures

Set `fixture.factory` to a local Python file and function, relative to the JSON file:

```json
"fixture": {
  "factory": "protein_fixture.py:build",
  "options": {"snapshot_file": "/absolute/path/snapshots.npy"}
}
```

The factory is trusted Python code. `build(case, options)` must return a fresh
`QMMMTheory` and an array of shape `(snapshots, atoms, 3)` in Å, containing at least two
geometries. Load saved production snapshots, including displaced QM and MM atoms, and use
absolute paths for topology and parameter files because construction happens inside scratch.
Apply `case["orca_numcores"]` to ORCA, `case["openmm_numcores"]` and `case["platform"]`
to OpenMM, and the selected `case["numerical"]` options to ORCA. Keep charge fields,
link-atom settings and force projection identical. The harness rejects differing snapshot
hashes and records the resulting backend configuration. Supply fixed box vectors appropriate
to the snapshots when testing a periodic fixture; compare variable-cell trajectories as
separate fixtures with their corresponding systems.

Use multiple displaced snapshots from the target workload, local and shared scratch where
relevant, several ORCA process counts, and representative OpenMM platforms and CPU thread
counts. Record CPU allocation, device selection and filesystem characteristics alongside the
JSON. This measures total evaluation throughput, including QM execution, point-charge
gradients, native MM and wrapper I/O; it does not establish long-trajectory accuracy or
production speedups from the small default fixture.

The timed operation is standalone `QMMMTheory.run`, including any MM Context construction
performed by that path. Dynamics reuses an OpenMM Context and invokes the QM provider from
inside the integrator, so these times are not persistent-context MD step throughput. Use
the MD profiling records to measure propagation, cache use and reporting in that workload.
The report records the Git revision and whether the checkout contained uncommitted changes.

## RPMD parallelism audit

The current PythonForce provider owns one QM/MM theory and holds a reentrant lock around
cache lookup, the complete QM evaluation, and insertion. Fresh bead evaluations therefore
enter that backend serially. The theory's coordinates, gradients, charge-site arrays and
ORCA scratch/GBW state are shared. `job_parallel` is not used by the RPMD callback path;
removing the provider lock would not create a safe bead-parallel implementation.

`ORCATheory(numcores=p)` instead requests `p` processes *within one electronic calculation*.
The ORCA driver launches its MPI modules; the package supplies `%pal nprocs` and waits for
that calculation to finish. See the [ORCA parallel execution manual](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/parallel.html).
OpenMM CPU `numcores` controls native MM work and does not allocate ORCA bead workers.

Ring-polymer contraction reduces the copies at which a force group is evaluated; it does
not make those evaluations concurrent. Full-ring energy includes every bead's potential,
kinetic energy and ring springs, so its cost can exceed contracted propagation. See
[OpenMM's RPMDIntegrator API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.RPMDIntegrator.html).
The exact-coordinate cache can reuse endpoint forces, so count fresh provider evaluations
and cache hits separately rather than assuming every force request starts an electronic job.

Run the scheduling and serialization probe without ORCA:

```bash
OPENMM_CPU_THREADS=1 OMP_NUM_THREADS=1 python examples/benchmark_rpmd.py \
    --beads 1 4 8 --energy-frequencies 0 1 --steps 4 --warmup-steps 1 \
    --repeats 3 --delay-seconds 0.002 --output rpmd-benchmark.json
```

The probe runs the real QM/MM PythonForce and RPMD engine with a two-atom harmonic backend,
zero native MM potential, distinct deterministic initial bead coordinates, and the thermostat
disabled. A synthetic 2 ms delay per fresh evaluation represents latency while releasing the
Python GIL. Each case measures three four-step blocks after a warmup step, retains the same
Context, and writes a positions-only DCD every step. `--energy-frequencies 0 1` compares
disabled diagnostics with full-ring energy every step. Wall time includes the public
`run(restart=True)` bookkeeping and trajectory I/O. JSON records each block's fresh calls,
cache hits, wall time, backend concurrency and software/host metadata. The timing is a
scheduling diagnostic, not an ORCA or biomolecular MM throughput estimate.

Measured on 29 October 2026 with Python 3.13.14, OpenMM 8.6.1, Reference platform, x86-64:

| Beads | QM copies | Energy every step | Fresh QM calls/step | Median ms/step |
| ---: | ---: | :---: | ---: | ---: |
| 1 | 1 | no | 1 | 2.19 |
| 4 | 4 | no | 4 | 8.51 |
| 4 | 1 | no | 1 | 2.23 |
| 4 | 1 | yes | 5 | 10.64 |
| 8 | 8 | no | 8 | 16.87 |
| 8 | 1 | no | 1 | 2.27 |
| 8 | 1 | yes | 9 | 19.18 |

All three blocks had the same call counts. These counts exclude startup and depend on cache
reuse; they are not universal integrator guarantees. A separate simultaneous four-thread
submission of distinct geometries to one provider measured peak backend concurrency **1**
and 41.7 ms total with a 10 ms delay per evaluation, confirming provider serialization.
Timing will vary by host; the regression test checks concurrency and relative call counts,
not elapsed-time thresholds. Contraction accuracy requires independent validation of the
physical system and is not established by this harmonic probe.

For a current production run, benchmark ORCA process count on the target QM region, allocate
OpenMM's CPU threads or GPU explicitly, and reserve memory for ORCA's per-process settings.
Record full-ring diagnostic cadence when comparing throughput. Additional cores do not
automatically add bead workers. Independent trajectories can use separate processes,
directories and theory objects under the scheduler's total CPU/GPU and memory allocation.

A future bead-worker design would need a batching interface beyond the present synchronous
single-state callback, independent mutable theory/MM state, unique scratch and output files,
controlled per-bead GBW reuse, ordered results and worker-failure cleanup. With `w` workers
and `p` ORCA processes each, budget at least `w*p` process slots plus native MM/runtime
resources, and validate energy/force equivalence and restart behavior before claiming speedup.
This audit establishes the existing serialization and contraction/reporting costs; it does
not introduce concurrent bead execution.
