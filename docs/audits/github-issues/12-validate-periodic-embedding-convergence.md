**Title:** Add cell-size and image-switch validation for finite periodic QM embedding

Suggested labels: `testing`, `scientific validation`  
Suggested priority: Medium

Periodic QM embedding constructs one molecule-imaged finite charge cluster; it does not include an Ewald/PME sum in the QM Hamiltonian. MM uses its separately configured periodic treatment. Changing a molecule's selected image can make the finite QM field discontinuous.

This limitation is already documented. Existing image-invariance and local derivative tests do not establish cell-size convergence or behavior across image-switching surfaces. Add targeted validation so future cutoff, multipole or timestep changes can be assessed against known baseline limitations.

Acceptance criteria:

- [ ] Add deterministic tests spanning an image-switching surface in a nonsymmetric QM environment.
- [ ] Provide a reproducible cell-size/embedding-extent convergence protocol.
- [ ] Measure relative energies, QM/MM forces and baseline NVE drift.
- [ ] Distinguish expected finite-cluster discontinuities from implementation regressions.
- [ ] Document applicability limits without presenting MM PME as periodic QM embedding.

Source: [periodic geometry model](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/periodic_embedding.py#L1), [documented limitation](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/docs/source/guide/qmmm.md#L77), [periodic tests](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/tests/test_qmmm_periodic_accuracy.py#L1).

