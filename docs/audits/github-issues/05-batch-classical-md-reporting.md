**Title:** Batch classical QM/MM MD stepping and avoid unconditional state retrieval

Suggested labels: `enhancement`, `performance`  
Suggested priority: High

The classical QM/MM MD loop calls `simulation.step(1)` and retrieves positions and energy every step, including steps with no scheduled output. This introduces Python overhead, host synchronization and native MM work.

Advance to the next trajectory, special-atom, energy or checkpoint event. Retrieve only the state fields needed for that event and share a State between consumers where possible.

The existing exact-coordinate cache usually lets the next integration step reuse the reported endpoint's QM result. This change should therefore be evaluated as an overhead reduction; it does not imply halving steady-state QM jobs.

Acceptance criteria:

- [ ] Preserve all reporter/checkpoint cadences, final state and restart behavior.
- [ ] Avoid positions/energy retrieval on steps without a consumer.
- [ ] Preserve force evaluation at integrator and barostat trial geometries.
- [ ] Validate deterministic propagation and outputs against the existing loop.
- [ ] Benchmark elapsed time, state retrievals and fresh QM evaluations separately.

Source: [classical MD loop](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/md.py#L1317), [state retrieval](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/md.py#L1184).

