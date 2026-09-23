# QM/MM embedding audit — 23 September 2026

Audited checkout: `b4495f8`. This audit found **nine reproducible defects**, including
two cases where supported options silently return incorrect physical gradients.
No implementation files were changed. Reproduction programs are in
[embedding-repros](embedding-repros/).

This document records the original audit. The subsequent corrections, supported
workflow restrictions, and validation are recorded in
[the remediation report](2026-09-23-qmmm-embedding-fixes.md).

The ordinary, fixed-charge paths have useful coverage: mechanical and electrostatic
embedding, simple/ratio caps, shift/RCD boundary charges, and periodic molecule
imaging pass the existing derivative and invariance tests. The defects below concern
native OpenMM virtual sites, updating QM charges, unusual boundary arrangements,
and incomplete MM force removal. Passing these tests does not establish chemical
accuracy for a production QM region or convergence of a periodic embedding field.

## Confirmed findings

### 1. [P1] Standalone embedding omits native virtual-site placement and force projection

[qmmm.py:495](../../openmmqmmm/qmmm.py#L495) assigns every native MM point-charge
gradient to its corresponding particle row. The subsequent mappings cover only
the extra shift/RCD sites created by QMMMTheory. A native OpenMM virtual site is
therefore treated as an independent atom in standalone energies and gradients.
Its position is also read directly from the supplied coordinate row rather than
recomputed from its hosts before constructing the QM field.

For a `TwoParticleAverageSite` carrying charge −1e, central finite differences
along the host/site geometry give a host derivative of **0.003888636743 Eh/bohr**;
the returned host derivative is **−0.000000648261 Eh/bohr**, containing only the MM
contribution. Moving a host by 1 Å while leaving the dependent row unchanged gives
QM energy −0.088196201757 Eh instead of −0.081411878545 Eh.

This affects standalone single points, optimization, and numerical derivatives
for systems containing native virtual charge sites. In the tested native
`PythonForce` path, OpenMM redistributes the QM force correctly: the host derivative
0.003889285002 Eh/bohr agrees with finite differences to 7e−13 Eh/bohr. Preserve
that path and avoid projecting the same force twice when fixing standalone calls.
Recompute native sites and apply their placement Jacobians consistently, or reject
unsupported standalone use. These sites depend on host coordinates by definition
([OpenMM virtual-site specification](https://docs.openmm.org/latest/userguide/theory/05_other_features.html#virtual-sites)).

Reproduction: `virtual_sites.py`.

### 2. [P1] Updating QM charges omits their coordinate derivatives

[qmmm.py:1074](../../openmmqmmm/qmmm.py#L1074) adds QM and fixed-charge MM gradients
after the charges have been recomputed from the current QM geometry. For
`update_qm_region_charges=True`, the reported energy contains `E_MM(R, q(R))`, but
the gradient omits `Σ_i (∂E_MM/∂q_i)(∂q_i/∂R)`. Population charges are not generally
variational parameters of this combined energy.

A deterministic QM backend with charge-conserving populations
`q0 = 0.1*r01` and `q1 = -q0` gives **0.001312633689 Eh/bohr** for one returned
gradient component versus **0.003412847592 Eh/bohr** from finite differences.
Thus optimization and frequency calculations can use forces inconsistent with
their objective. Classical MD and RPMD already reject this option. Until a
consistent energy/response model exists, restrict gradient-dependent jobs too,
or support charges fixed across the entire job explicitly.

Reproduction: `mm.py`, `charge_response` output. This probe uses a custom backend;
the separate ORCA compatibility defect below prevents that backend reaching this
stage without an adapter.

### 3. [P2] QM–QM nonbonded parameter offsets survive exclusion

[theory.py:1199](../../openmmqmmm/openmm/theory.py#L1199) replaces base exception
parameters with zeros but does not clear `ExceptionParameterOffset` entries
attached to the same pair. On a nonperiodic two-particle system entirely assigned
to QM, an active offset leaves **51.515588 kJ/mol** of mechanical MM energy.
Electrostatic setup clears the charge offset later, but leaves the LJ offset and
**−0.585209 kJ/mol** of MM energy. Both cases also retain nonzero MM forces.
The expected MM contribution for these fully excluded nonperiodic particles is
zero. Clear all offsets for excluded QM–QM exceptions while preserving legitimate
QM–MM LJ terms.

Reproduction: `mm.py`, `offsets` outputs.

### 4. [P2] Nonperiodic covalent boundaries ignore available OpenMM topology

[qmmm.py:678](../../openmmqmmm/qmmm.py#L678) uses topology only when the system is
periodic. Otherwise it infers covalent bonds from coordinate distances; MM charge
recipients are inferred the same way at
[qmmm.py:319](../../openmmqmmm/qmmm.py#L319).

A nonperiodic carbon chain with topology `0–1–2`, positions `0, 2.0, 3.4 Å`, and
`qmatoms=[0]` gets no link atom despite its explicit crossing bond. Conversely,
positions `0, 1.4, 2.8 Å` with only topology bond `1–2` create a spurious QM0–MM1
cap. Stretched bonds and close nonbonded contacts can therefore alter the chosen
QM Hamiltonian. Use available OpenMM topology for both boundary and recipient
selection independently of periodicity, with distance inference as a fallback
when topology is unavailable.

Reproduction: `boundary.py`, `missed_topology_bond` and `invented_topology_bond`.

### 5. [P2] Charge redistribution can recharge capped MM1 atoms

[qmmm.py:325](../../openmmqmmm/qmmm.py#L325) excludes QM atoms from the recipient
list but permits another boundary MM1 atom. In a carbon chain `0–1–2–3`, with
`qmatoms=[0,3]` and initial charges `[0,0.3,0.4,0]`, the recipient mapping is
`{1:[2], 2:[1]}`. Shift puts `[0.4,0.3]` directly back on the two capped hosts;
RCD puts `[-0.4,-0.3]` there. Default shift dipole correction also creates nearby
charge pairs with magnitudes 2.5e and 3.333e.

Total charge and dipole still agree, so those invariants do not detect the
failure to remove the proximal MM1 charges. Exclude the complete MM1 set from
recipients and reject boundaries without valid recipients, or define a separate
treatment for overlapping boundary regions. Moving the boundary charge away from
MM1 is the purpose of these schemes
([ORCA boundary charge schemes](https://orca-manual.mpi-muelheim.mpg.de/contents/multiscalesimulations/qmmm-general.html)).

Reproduction: `boundary.py`, `adjacent_mm1_shift` and `adjacent_mm1_rcd`.

### 6. [P2] Updating particle charges leaves Coulomb exception products stale

[theory.py:1868](../../openmmqmmm/openmm/theory.py#L1868) updates particle charges
but not the independently stored exception charge products. Updating a QM charge
from 0.2e to 0.8e, with an MM charge of −0.5e and half-scaled cross exception,
leaves the product at **−0.05e² instead of −0.2e²**. At 4 Å the MM energy is
**−17.366932 instead of −69.467729 kJ/mol**.

This affects mechanical QM–MM 1–4 coupling when charges are updated, as well as
explicit replacement charges where existing nonzero exceptions should retain
their force-field scaling. Preserve the exception scaling when recomputing the
products, with a defined policy for initially zero charges and custom exceptions.

Reproduction: `mm.py`, `charge_update`.

### 7. [P2] The QM charge-update interface is incompatible with ORCATheory

[qmmm.py:1020](../../openmmqmmm/qmmm.py#L1020) requires `qm_theory.charges`, but
[orca.py:444](../../openmmqmmm/orca.py#L444) stores parsed populations in
`properties["Mulliken_charges"]`. An actual HF/STO-3G calculation on H2 with
`print_population_analysis=True` converges at −1.116759307204 Eh and records
Mulliken populations `[0,0]`, then QMMMTheory raises `InputError` because
`.charges` does not exist.

Define a shared population-charge accessor with an explicit charge model, and
validate backend support before launching an expensive QM calculation. This API
fix alone does not resolve the response-gradient and exception defects above.

Reproduction: `orca_charge_updates.py`.

### 8. [P2] Custom torsion removal assumes exactly two parameters

[theory.py:1988](../../openmmqmmm/openmm/theory.py#L1988) writes `(0.0, 0.0)` into
every selected `CustomTorsionForce`. The package's own
`add_custom_torsion_force()` uses zero per-torsion parameters, so removing a
torsion with three or four QM atoms produces
`OpenMMException: CustomTorsionForce: Wrong number of parameters for torsion 0`
when the Context is created. Other parameter layouts or expressions need not
vanish when two values are zeroed either. Rebuild supported forces without the
selected terms, or recognize their expressions and reject unsupported forms
before mutation.

Reproduction: `mm.py`, `custom_torsion`.

### 9. [P2] Multiple boundary cuts bypass the unusual-boundary guard

[coords.py:2188](../../openmmqmmm/coords.py#L2188) accepts the multi-neighbor branch
before the non-C–C validation in the single-neighbor branch. With default
`unusualboundary=False`, one O–C cut raises `InputError`, while a QM oxygen with
two carbon neighbors is accepted as `{0:[1,2]}`. Apply the element/cut validation
to every accepted pair, including multiple cuts. Otherwise the documented
requirement for explicit acceptance of unusual boundaries is not enforced.

Reproduction: `boundary.py`, `nonpolar_guard_1` and `nonpolar_guard_2`.

## Existing approximations and unresolved model policy

- **Cached truncated point charges are not a conservative approximation between
  refreshes.** Separate constant energy and gradient corrections generally do
  not differentiate to one another. Reusing the existing analytic cap fixture
  with radius 0.1 Å and refresh interval 50 reproduced a maximum mismatch of
  **1.675966 Eh/bohr**. This magnitude is fixture-specific; the inconsistency is
  structural. It is already documented in the earlier MD audit and is not counted
  as a new defect here. Keep the existing MD/RPMD rejection and validate uses in
  optimization or numerical frequencies.
- **Periodic electrostatic embedding is a finite molecule-imaged field.** It has
  no Ewald sum in the QM Hamiltonian, and image switches can make that field
  discontinuous. Existing image-invariance tests pass; they do not establish
  cell-size convergence or long-time energy conservation. This is already
  documented behavior, not a newly discovered implementation defect.
- **Native periodic MM image/tail contributions need an explicit policy.**
  QM–QM exceptions remove direct interactions but do not make the all-QM periodic
  MM contribution zero. For two QM LJ sites in a 30 Å cell, dispersion correction
  leaves −3.445283e−7 Eh; LJPME leaves −1.514391e−6 Eh plus nonzero forces.
  Mechanical embedding with QM charges `+1e,−1e` and PME leaves −0.001067828 Eh
  (about −0.670 kcal/mol). These contributions change with the cell. Classical
  interactions between periodic QM images may be a deliberate model choice, so
  these measurements alone do not establish a bug. Document and test that choice
  explicitly, or remove the residuals if “remove QM–QM MM interactions” is intended
  to include periodic images and tails. Reproduction: `periodic_residuals.py`.

## Validation and limits

Runtime: Python 3.13.15, OpenMM 8.6.1, local ORCA 6.1.1, Reference platform for
the synthetic OpenMM probes. The focused suite passed **140 tests**:

```text
python -m pytest -q tests/test_qmmm_core.py tests/test_qmmm_boundary_accuracy.py \
  tests/test_qmmm_mm_accuracy.py tests/test_qmmm_periodic_accuracy.py \
  tests/test_qmmm_md_accuracy.py tests/test_orca_input.py tests/test_orca_parsers.py
```

The real ORCA/OpenMM test
`tests/test_QM_MM_openmm.py::test_qm_mm_orca_openmm_meoh_h2o` also passed. These
are **141 existing tests** in total; overlapping agent test runs are not added
again. All nine findings were demonstrated separately outside the passing suite.

Additional actual ORCA probes verified an empty embedding field and selected
QM/point-charge derivatives for H2 in a two-charge field. Point-charge derivative
errors were below 2e−10 Eh/bohr; the selected QM derivative differed by 9.3e−7
Eh/bohr at the chosen 0.001 Å finite-difference step. The summed gradient was
zero to roundoff. These validate the small-system backend contract, not general
electronic-structure accuracy. See `orca_derivatives.py`.

Run the saved probes with the project's dependency environment; ORCA probes
also need an installed ORCA binary. They use synthetic systems and print the
observations described above. Prior files under `docs/audits/` were preserved.
