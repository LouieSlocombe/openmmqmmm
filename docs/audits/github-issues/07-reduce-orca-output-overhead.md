**Title:** Consolidate ORCA output parsing and make output retention configurable

Suggested labels: `enhancement`, `performance`  
Suggested priority: Medium

Each ORCA evaluation scans output separately for failures, warnings, termination, energy, optional properties and timings. The interface also hardcodes `keep_last_output=True` and copies the complete output after each evaluation.

Consolidate common parsing where practical, skip irrelevant optional-property extraction, and expose a documented output-retention policy. Keep useful failure diagnostics and preserve GBW reuse across MD steps.

Acceptance criteria:

- [ ] Preserve error detection, termination checks and scientific result parsing.
- [ ] Preserve requested optional properties.
- [ ] Make output-copy behavior configurable with a backward-compatible default.
- [ ] Preserve restart/orbital files independently of output-retention policy.
- [ ] Verify parsing with existing captured ORCA outputs.
- [ ] Measure overhead on representative outputs before claiming a total MD speedup.

Source: [output copy](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/orca.py#L432), [optional-property parsing](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/orca.py#L483), [run result processing](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/orca.py#L668).

