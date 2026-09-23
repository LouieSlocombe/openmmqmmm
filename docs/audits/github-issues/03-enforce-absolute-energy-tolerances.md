**Title:** Enforce absolute tolerances in real-ORCA QM/MM energy regression tests

Suggested labels: `bug`, `testing`  
Suggested priority: High

The real-ORCA energy regressions use `np.isclose(..., atol=2e-6)` without setting `rtol`. Near -115.8 Eh, NumPy's default relative tolerance permits approximately 0.73 kcal/mol of error, despite the much tighter absolute tolerance written in the test.

Use `rtol=0` where the intended comparison is absolute. Review related energy and gradient assertions and state their units and intended tolerance. If a tighter assertion reveals a reference/version difference, investigate it instead of automatically widening the threshold.

Acceptance criteria:

- [ ] Absolute energy assertions explicitly set `rtol=0`.
- [ ] A perturbation larger than the intended absolute tolerance fails.
- [ ] Existing ORCA reference cases are checked against the installed supported version.
- [ ] Remaining relative tolerances are intentional and explained.

Source: [electrostatic energy assertion](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/tests/test_QM_MM_openmm.py#L52), [mechanical energy assertion](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/tests/test_QM_MM_openmm.py#L166).

