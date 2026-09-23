GitHub issue drafts from the QM/MM MD audit

Twelve independent drafts, ordered for triage. Each file contains a suggested title, labels, problem, proposed change and acceptance criteria. Priorities and labels are suggestions, not existing repository metadata. Source links are pinned to audited commit `b4495f805e29b210e26ef978b481f2883a07cbcc`.

| Draft | Suggested priority | Issue |
|---|---|---|
| 01 | High | [Fix OpenMMTheory.set_numcores() so it updates the CPU thread configuration](./01-fix-openmm-thread-setter.md) |
| 02 | High | [Make platform selection explicit when MD reuses an existing QM/MM theory](./02-make-md-platform-selection-explicit.md) |
| 03 | High | [Enforce absolute tolerances in real-ORCA QM/MM energy regression tests](./03-enforce-absolute-energy-tolerances.md) |
| 04 | High | [Expose structured QM/MM MD timing, SCF and cache statistics](./04-expose-qmmm-md-profiling.md) |
| 05 | High | [Batch classical QM/MM MD stepping and avoid unconditional state retrieval](./05-batch-classical-md-reporting.md) |
| 06 | Medium | [Reduce per-evaluation overhead in periodic molecule imaging](./06-vectorize-periodic-imaging.md) |
| 07 | Medium | [Consolidate ORCA output parsing and make output retention configurable](./07-reduce-orca-output-overhead.md) |
| 08 | Medium | [Add reproducible accuracy and throughput benchmarks for ORCA QM/MM settings](./08-benchmark-orca-accuracy-throughput.md) |
| 09 | Medium; larger project | [Prototype multiple-time-step QM/MM using a cheap reference and high-level correction](./09-prototype-delta-qm-mts.md) |
| 10 | Medium; RPMD workflows | [Separate RPMD full-ring energy diagnostics from trajectory reporting](./10-control-rpmd-energy-report-cost.md) |
| 11 | Medium; optimization and other supported callers | [Make cached truncated-PC energy and gradient corrections mutually consistent](./11-correct-cached-truncated-pc-derivatives.md) |
| 12 | Medium | [Add cell-size and image-switch validation for finite periodic QM embedding](./12-validate-periodic-embedding-convergence.md) |

Start with drafts 01–05, then use their profiling to prioritize implementation work. Draft 08 is a benchmark investigation; draft 09 is an experimental feature. Draft 10 is relevant to RPMD, and draft 11 addresses currently supported non-MD uses of truncation.

The audit's standalone OpenMM Context-reuse idea is deferred: MD already retains a Context and the exploratory standalone timing did not demonstrate a speedup. Exact-zero charge filtering and reference-based RPMD contraction can be separate follow-ups after profiling and the shared high/low potential interface exist.

Audit validation: 174 selected tests passed, including real ORCA integrations. These drafts are a rewrite of those findings; no new performance gains are claimed.

