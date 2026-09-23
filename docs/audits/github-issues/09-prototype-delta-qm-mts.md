**Title:** Prototype multiple-time-step QM/MM using a cheap reference and high-level correction

Suggested labels: `enhancement`, `research`  
Suggested priority: Medium; larger project

Reducing expensive electronic evaluations could provide larger gains than wrapper tuning. Introduce an experimental force split:

`U = U_MM_modified + U_QM_low + (U_QM_high - U_QM_low)`

Evaluate MM and the cheap embedded QM reference every inner step; evaluate the high-minus-low correction less often. Both QM levels must use consistent charge fields, link atoms and force projection.

The current coupling removes internal QM bonded MM terms. Assigning the whole QM contribution to a slow force group would therefore also slow the stiff bond forces. The cheap reference must capture those motions and the relevant chemistry.

Acceptance criteria:

- [ ] Implement the split without double counting and recover the target potential when both levels are evaluated together.
- [ ] Validate QM, cap and MM charge-force contributions against finite differences.
- [ ] Converge inner/outer timestep choices with NVE drift and relevant NVT observables.
- [ ] Preserve full-potential energies for reporting and barostat trials.
- [ ] Benchmark total time including reference evaluations.
- [ ] Keep the feature opt-in and publish its limitations and reference-model requirements.

Source: [QM bonded-term removal](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/qmmm.py#L711). References: [OpenMM MTS](https://docs.openmm.org/latest/api-python/generated/openmm.mtsintegrator.MTSIntegrator.html), [ab initio MTS study](https://arxiv.org/abs/1312.1284). Dependencies: structured QM/MM profiling and the proposed ORCA accuracy/throughput benchmark suite.

