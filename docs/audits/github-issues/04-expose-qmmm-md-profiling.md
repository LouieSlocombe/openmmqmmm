**Title:** Expose structured QM/MM MD timing, SCF and cache statistics

Suggested labels: `enhancement`, `performance`  
Suggested priority: High

The ORCA interface parses SCF iterations and component timings but primarily logs or discards them. This makes it difficult to determine whether a trajectory is limited by SCF, point-charge gradients, periodic imaging, native MM, I/O or reporting.

Add optional structured profiling records per force evaluation and per MD step. Distinguish fresh electronic calculations from cache hits and include QM atom count, charge-site count and execution settings. Use wall-clock instrumentation for wrapper phases and retain ORCA-reported component timings separately to avoid double counting.

Acceptance criteria:

- [ ] Retain SCF iteration count and available ORCA timing fields.
- [ ] Record callback wall time, imaging, input/output processing and reporting costs where measurable.
- [ ] Expose fresh QM evaluation counts and cache hits.
- [ ] Export machine-readable records with units and a documented schema.
- [ ] Profiling has low measured overhead and does not alter energies, forces or cache behavior.
- [ ] Examples profile warm displaced geometries, not only identical-geometry repeats.

Source: [ORCA run statistics](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/orca.py#L673), [timing parser](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/orca.py#L1065), [force cache](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/rpmd_force.py#L65).

