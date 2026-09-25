# Periodic imaging performance, 2026-09-25

The updated routine reduced isolated imaging time by 4.86–6.26× and complete fresh
Python QM/MM callback time by 4.01–4.94× in these synthetic water cases. These are
paired measurements of the baseline and updated implementation in the same
process. The audit's earlier 41/212/325 ms figures used a different benchmark and
are not the baseline for these speedups.

The baseline `PeriodicQMGeometry` is loaded from Git revision
`d599059eae5d4c7f56acc8758fbe4e1936838daf`. All other callback code comes from the
current checkout for both implementations. Source hashes, environment details,
all nine measured samples, minima and maxima are in
[periodic-imaging-benchmark.json](periodic-imaging-benchmark.json).

| Cell | Atoms | Baseline imaging (ms) | Updated imaging (ms) | Speedup | Baseline callback (ms) | Updated callback (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Orthorhombic | 3,001 | 5.948 | 1.143 | 5.21× | 6.275 | 1.521 | 4.12× |
| Orthorhombic | 30,001 | 59.059 | 10.875 | 5.43× | 61.631 | 14.111 | 4.37× |
| Orthorhombic | 99,991 | 193.229 | 32.944 | 5.87× | 202.246 | 43.893 | 4.61× |
| Triclinic | 3,001 | 6.038 | 1.243 | 4.86× | 6.395 | 1.594 | 4.01× |
| Triclinic | 30,001 | 56.419 | 9.017 | 6.26× | 59.524 | 12.061 | 4.94× |
| Triclinic | 99,991 | 191.484 | 33.495 | 5.72× | 201.197 | 44.319 | 4.54× |

Each entry is the median of nine measurements after three warmups. Baseline and
updated calls alternate order and receive the same coordinates in each pair.
Graph construction, force-field setup, OpenMM State creation and numerical
comparisons are excluded from timing. Every callback performs a fresh evaluation:
each input differs by a small displacement of the QM atom, and recorded counts
confirm 12 evaluations and zero cache hits for each implementation and case.

The systems contain one unbonded QM atom plus independently positioned three-site
waters at a nominal density of 0.0334 molecules/Å³. Water coordinates are wrapped
individually, so some molecules cross cell faces. The triclinic boxes have
off-diagonal components of 0.18, 0.09 and 0.13 times the box edge. These are
deterministic synthetic geometries, not equilibrated molecular dynamics frames.

The callback measurement calls `RPMDQMMMForceProvider.__call__` on real, precreated
OpenMM States using an actual electrostatic `QMMMTheory`. A cheap analytic Coulomb
backend computes nonzero energies and gradients with TIP3P point charges. Timing
includes cache-key construction, imaging, QM/MM preparation, the analytic backend,
gradient assembly, unit conversion and cache insertion. It excludes OpenMM's
native callback dispatch, native MM forces, integration, reporters and electronic
structure calculations. It therefore measures total Python callback work, and
does not establish an ORCA or end-to-end MD speedup.

All six cases had an observed maximum absolute difference of zero for cluster
coordinates, callback energy and every force component. The script checks absolute
tolerances of 1e-10 Å, 1e-8 kJ/mol and 1e-7 kJ/mol/nm respectively, with no relative
tolerance. These water timings do not exercise every topology: the separate
[regression tests](../../../tests/test_periodic_embedding.py) cover multiple QM
fragments, covalent boundaries, virtual sites, winding networks and image-switch
surfaces.

The implementation caches parent/child arrays, breadth-first graph layers,
component membership and reduction indices. Each graph layer unwraps all of its
atoms at once. For float64 coordinates, `bincount` computes component centers in
their original atom order, and indexed array operations translate whole
molecules. Custom site callbacks returning other dtypes retain their previous
mean accumulation precision through a fallback; native OpenMM sites use float64.

ASE `find_mic` and its original call grouping are retained. No orthorhombic fast
path was introduced: even in a 10 Å cubic cell, the floating-point value
`nextafter(-5.0, 0.0)` can select an opposite-signed image when evaluated alone
versus in a mixed batch that triggers ASE's general path. Preserving the
singleton QM-fragment calls and component summation order protects the existing
image convention at these boundaries.

The recorded final run used Python 3.13.15, NumPy 2.4.6, ASE 3.29.0 and OpenMM 8.6.1
on an Intel Core i7-14700KF running Linux. Thread settings were all one, and CPU
affinity was fixed to logical CPU 4. Reproduce from the repository root in the
development Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENMM_CPU_THREADS=1 \
  taskset -c 4 python docs/audits/embedding-repros/periodic_imaging_benchmark.py \
  --cells orthorhombic triclinic \
  --output docs/audits/embedding-repros/periodic-imaging-benchmark.json
```

The script requires Git history containing the baseline revision. Choose an
available CPU on other systems, or omit `taskset`; affinity and thread settings
are recorded. `--atoms`, `--warmups`, `--repeats` and `--skip-callback` permit
smaller diagnostic runs. Generated setup files are isolated in a temporary
directory and removed on exit.

Absolute times remain sensitive to machine load and scheduling. An earlier
un-pinned development run showed a sudden slowdown during its triclinic cases.
A subsequent run of the final code pinned to CPU 0 also showed large scheduling
variation, with 3,001-atom updated imaging samples ranging from 1.18 to 4.53 ms;
its full raw data is retained in
[periodic-imaging-core0-diagnostic.json](periodic-imaging-core0-diagnostic.json).
The final CPU 4 run above had a 1.11–1.22 ms range for that case. CPU 0's
30,001- and 99,991-atom orthorhombic callback speedups were 4.44× and 4.61×, close
to the final run's 4.37× and 4.61× despite approximately doubled absolute times.
The cause of the scheduling variation was not established; the reported gains
are specific to these workloads and environment.
