**Title:** Make cached truncated-PC energy and gradient corrections mutually consistent

Suggested labels: `bug`, `numerical accuracy`  
Suggested priority: Medium; optimization and other supported callers

Between full-field refreshes, `truncated_pc` returns a truncated energy plus a constant energy correction, while adding an independently cached gradient correction. The returned gradient is generally not the derivative of the reported energy.

Existing tests check full refreshes and repeated identical geometries. Using the existing synthetic boundary fixture with radius 0.1 Å and refresh interval 50 produced a maximum analytical-versus-finite-difference discrepancy of 1.676 Eh/bohr. This is a structural diagnostic, not a realistic molecular error estimate.

Add displaced-geometry coverage and define a conservative correction with consistent coordinate derivatives, or explicitly restrict unsupported derivative-based uses. A local correction must account for moving caps and virtual charge sites as well as real atoms; refresh transitions require separate treatment.

Acceptance criteria:

- [ ] Cover displaced cached evaluations between refreshes.
- [ ] Verify energy-gradient consistency while holding the correction state fixed.
- [ ] Define behavior at refreshes and when the selected charge set changes.
- [ ] Retain existing classical-MD and RPMD rejection until a history-independent, validated potential is available.
- [ ] Document remaining approximation and history dependence.

Source: [cached correction](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/qmmm.py#L1231), [existing boundary tests](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/tests/test_qmmm_boundary_accuracy.py#L127).

