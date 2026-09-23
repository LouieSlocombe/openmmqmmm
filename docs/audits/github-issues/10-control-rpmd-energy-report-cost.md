**Title:** Separate RPMD full-ring energy diagnostics from trajectory reporting

Suggested labels: `enhancement`, `performance`, `RPMD`  
Suggested priority: Medium; RPMD workflows

The RPMD state reporter always calls `getTotalEnergy()`. In contracted runs this evaluates the potential on the full ring, potentially launching many more QM calculations than propagation needs. An eight-bead centroid-contracted probe required eight additional full-bead QM evaluations for energy reporting.

Expose a separate cadence or opt-out for full-ring energy diagnostics and share computed results between reporters. Avoid full-potential evaluation for positions-only output.

Acceptance criteria:

- [ ] Independently control trajectory output and full-ring energy diagnostics.
- [ ] Avoid duplicate total-energy requests at the same reporting event.
- [ ] Preserve propagation and accurately label any reported energy definition.
- [ ] Test fresh QM evaluation counts for contracted and uncontracted runs.
- [ ] Update cost documentation to distinguish callback invocations from fresh jobs: a four-bead probe used 8, 4, 4 fresh evaluations over three steps because endpoints were cached.

Source: [RPMD state reporter](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/md.py#L122), [QM contraction configuration](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/md.py#L867).

