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
