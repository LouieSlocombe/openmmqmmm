**Title:** Make platform selection explicit when MD reuses an existing QM/MM theory

Suggested labels: `API`, `documentation`  
Suggested priority: High

`openmm_md(platform="CUDA", theory=qmmm)` reuses `qmmm.mm_theory` without applying the supplied platform. A caller can request CUDA while continuing to run the MM component on CPU.

Clarify which object owns platform configuration and prevent explicit conflicting requests from being silently ignored. Preserve the existing theory's platform when no override is supplied; distinguish that case from an explicit request. Applying overrides safely before Context creation or rejecting them with an actionable message are both reasonable designs.

Acceptance criteria:

- [ ] Omitted platform arguments inherit the existing MM theory's platform.
- [ ] Explicit conflicting arguments are honored safely or produce a clear diagnostic.
- [ ] Documentation demonstrates configuring platform, precision and CPU threads on `OpenMMTheory`.
- [ ] Tests cover inherited and explicitly conflicting platform settings without requiring a GPU.
- [ ] Documentation explains that changing the OpenMM platform affects MM execution, not the separately launched ORCA calculation.

Source: [theory selection](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/md.py#L768).

