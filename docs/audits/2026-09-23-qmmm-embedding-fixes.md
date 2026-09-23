# QM/MM embedding audit remediation

All nine defects in the [embedding audit](2026-09-23-qmmm-embedding.md) have been
addressed through corrections or explicit rejection of unsupported calculations.
The implementation also closes the cached-truncation gradient gap and defines the
existing periodic MM image/tail policy. The original audit and numerical outputs
remain historical records of checkout `b4495f8`.

| Audit finding | Implemented behavior | Regression coverage |
| --- | --- | --- |
| 1. Native virtual-site coordinates and gradients | Recompute sites before evaluating the QM field; use OpenMM's native Jacobians to redistribute standalone QM forces. Standalone MM forces already redistributed by OpenMM have zero dependent rows. Native callbacks retain raw site forces so OpenMM redistributes exactly once. All four native site types and nested dependencies are supported. | `test_embedding_virtual_sites.py`, `test_embedding_virtual_site_callback.py` |
| 2. Missing population-charge response | Reject gradients with `update_qm_region_charges=True` before QM execution. Energy-only updates remain available; fixed charges are required for optimization, frequencies, and dynamics. | `test_embedding_charge_regressions.py` |
| 3. Active QM–QM exception offsets | Clear all parameter offsets associated with excluded QM–QM exceptions. Preserve valid QM–MM LJ coupling. | `test_embedding_mm_regressions.py` |
| 4. Nonperiodic topology ignored | Use OpenMM covalent topology for boundary and charge-recipient selection in both periodic and nonperiodic systems. Fall back to distance inference only without topology. | `test_embedding_boundary_regressions.py` |
| 5. Adjacent boundary hosts recharged | Exclude the complete MM1 set from redistribution recipients; reject electrostatic cuts without eligible MM neighbours. Mechanical cuts remain permitted. | `test_embedding_boundary_regressions.py` |
| 6. Stale exception charge products | Preserve inferred Coulomb scaling when assigning fixed charges, including subsequent zero crossings. Validate ambiguous custom exceptions before mutation. Retain LJ parameters. | `test_embedding_mm_regressions.py` |
| 7. ORCA population interface mismatch | Add `ORCATheory.get_atomic_charges()` for fresh Mulliken populations, independent of logging. Validate charge count, finiteness, order contract, and total charge. Support custom accessors and initialized legacy `.charges`. | `test_embedding_charge_regressions.py`, including a real ORCA calculation |
| 8. Custom torsion parameter corruption | Rebuild arbitrary custom torsion forces without removed terms, preserving expressions, parameters, derivatives, names, groups, and PBC settings. Refresh cached force references. | `test_embedding_mm_regressions.py` |
| 9. Multiple-cut validation bypass | Validate every inferred non-C–C cut against `unusualboundary`, including multiple neighbours. | `test_embedding_boundary_regressions.py` |

The virtual-site helper serializes only its force-free site metadata and the
auxiliary linear force definition; native contexts are recreated after copying
or unpickling. Tests cover standalone/callback energy and force agreement, finite
differences with nonzero native MM forces, wrapped host coordinates, and rejected
partitions leaving the supplied MM System and charges intact.

Native sites and their hosts must remain entirely in MM. A QM selection containing
a site or any of its host dependencies is rejected before MM mutation, because
retaining part of the same molecule's virtual charge in the embedding field would
be inconsistent. Likewise, population updates for capped regions are rejected
until a charge-conserving mapping of cap populations is defined. These restrictions
are documented in the [QM/MM guide](../source/guide/qmmm.md).

Cached truncated-PC calculations now support energy-only evaluations. Gradients,
optimization, and numerical frequencies require `truncated_pc=False` or a refresh
interval of 1, which restores the full field every call. Numerical frequencies
reject incompatible settings before creating displacement workspaces. MD/RPMD
retain their existing blanket truncation rejection. Radius and refresh interval
validation also prevents invalid control values. These cases are covered in
`test_embedding_truncation_regressions.py`; no unvalidated far-field potential or
population-response approximation was introduced.

Periodic electrostatic embedding remains a finite molecule-imaged field. Native
MM exclusions remove direct primary-cell pairs while retaining configured image
interactions and dispersion-tail corrections. Tests establish the retained Ewald,
LJPME, and analytical-tail behavior, including its cell scaling. The guide now
states that even an all-QM periodic system can retain MM energy, forces, and
pressure contributions. Full periodic QM electrostatics and production cell-size
convergence are separate scientific capabilities; this change does not claim to
provide either.

## Validation

- **103 new regression cases** across six embedding test files.
- **1,070 passed and 3 skipped** across the full suite run in batches. The skips
  require `openmmnqe`, forcefill/AmberTools, and OpenMM-PLUMED/OPES.
- The broad non-OpenMM batch initially found one real-ORCA link-atom fixture that
  omitted all methanol topology bonds. It now supplies its known covalent graph
  explicitly; both affected ORCA integration tests pass. The fixture continues
  using its original nonbonded-only MM potential.
- Full repository Ruff lint and formatting checks pass; `git diff --check` passes.
- Sphinx builds with warnings treated as errors when run offline with external
  inventories disabled. The normal build reached only five external-inventory DNS
  failures in the restricted environment; no source-document warnings occurred.

Tests used Python 3.13.15, OpenMM 8.6.1, and local ORCA 6.1.1. Numerical derivative
probes use the Reference platform. Existing geomeTRIC dummy-system warnings and
multiprocessing fork warnings remain unrelated to these changes.
