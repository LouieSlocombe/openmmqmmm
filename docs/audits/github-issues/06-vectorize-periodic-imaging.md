**Title:** Reduce per-evaluation overhead in periodic molecule imaging

Suggested labels: `enhancement`, `performance`  
Suggested priority: Medium

Every periodic QM/MM evaluation reconstructs a finite embedding cluster. The imaging routine loops over graph edges and molecules, computes a separate NumPy mean per component, and applies per-molecule translations.

The audit measured the existing routine at approximately 41 ms for 3,001 atoms, 212 ms for 30,001, and 325 ms for 99,991 in synthetic water boxes. These are baseline microbenchmarks, not demonstrated optimization gains.

Precompute reusable graph/index arrays and vectorize molecule centers and translations. Consider an orthorhombic fast path only if it preserves the current image-selection convention.

Acceptance criteria:

- [ ] Equivalent cluster coordinates and energy/force results within stated numerical tolerances.
- [ ] Preserve whole molecules, virtual sites, multiple QM fragments and covalent boundaries.
- [ ] Retain triclinic support and rejection of winding covalent networks.
- [ ] Cover image-switch boundaries explicitly.
- [ ] Report repeatable isolated timings and total callback impact.

Source: [PeriodicQMGeometry.image](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/periodic_embedding.py#L69), [per-evaluation call](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/qmmm.py#L824).

