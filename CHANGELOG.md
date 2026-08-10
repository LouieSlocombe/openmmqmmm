# Changelog

## 1.0.1 (2026-08-10)

Bug-fix release from a full audit of the 1.0.0 tree. The 1.0.0 rename pass left several call
sites pointing at the old names; the parallel-execution path did not run at all.

### Fixed
- `job_parallel` raised `TypeError: worker_par() got an unexpected keyword argument 'Grad'` on
  every worker — the kwarg was renamed on the callee but not at the four `apply_async` call
  sites. This made `job_parallel`, `numerical_frequencies(runmode="parallel")` and
  `NumGrad(runmode="parallel")` completely unusable.
- RCD charge shifting (`QMMMTheory(chargeboundary_method="rcd")`) kept only the last RCD site
  per step instead of all of them, so the point-charge field had more charges than coordinates;
  an empty MM boundary raised `UnboundLocalError`. `QMMMTheory` now also verifies that charges
  and coordinates are the same length before handing the field to the QM code.
- `UnboundLocalError` in the parallel worker when `mofilesdir` was combined with a string label;
  unsupported label/theory combinations now raise `InputError`.
- Parallel workers restore their working directory when a job fails, so a failure no longer
  corrupts subsequent jobs assigned to the same pool worker.
- `job_parallel` validated `theories`/`numcores` after already indexing them, raising `TypeError`
  instead of `InputError`.
- `single_point_theories` crashed with `TypeError` while printing its summary table when charge
  and multiplicity were passed as arguments rather than set on the fragment.
- `check_gradient_for_bad_atoms` printed the z coordinate twice and never the y coordinate.
- `NumGrad.run` ignored `grad=` and always returned an (energy, gradient) tuple.
- Electrostatic embedding with a QM-free theory built flat `(3,)` zero-gradients instead of
  per-atom `(natoms, 3)` arrays, breaking the QM/MM gradient assembly.

### Changed
- Results files are now named `results_singlepoint.json`, `results_singlepoint_theories.json`,
  `results_singlepoint_fragments.json`, `results_singlepoint_fragments_theories.json`,
  `results_singlepoint_reaction.json`, `results_anfreq.json`, `results_numfreq.json` and
  `results_optimizer.json`. 1.0.0 documented JSON results but every call site still wrote the
  old `ASH_*.result` names.

### Added
- `openmmqmmm.__version__`.
- Google-style docstrings on the remaining undocumented public coordinate helpers.
- Regression tests for `job_parallel` (the gap that let the kwarg bug ship) and for the
  `Results` write/read round-trip.

## 1.0.0 (2026-08-09)

Full modernization of the codebase. **Breaking release**: the ASH-compatible API is gone —
see the migration table in README.md.

### Changed
- Public API renamed to PEP8 style: `Singlepoint` → `single_point`, `geomeTRICOptimizer` →
  `optimize_geometry`, `NumFreq` → `numerical_frequencies`, `OpenMM_MD` → `openmm_md`,
  `ASH_Results` → `Results`, and ~90 further symbol renames; keyword arguments snake_cased
  (`Grad=` → `grad=`, `ActiveRegion=` → `active_region=`, ...). The `Optimizer`/`Opt`/
  `MolecularDynamics`/`MetaDynamics` aliases are gone.
- Flat module layout: `openmmqmmm.orca`, `openmmqmmm.openmm`, `openmmqmmm.coords`, ... (the
  `modules/`, `functions/`, `interfaces/` subpackages and their name prefixes are gone).
- All output goes through the `logging` module (per-module loggers under `openmmqmmm.*`).
  Importing the package is silent; `configure_logging()` restores the classic console output.
  The `printlevel` verbosity system was removed in favor of logging levels.
- Errors raise typed exceptions (`OpenMMQMMMError` base with `InputError`,
  `MissingDependencyError`, `ExternalProgramError`, `FileFormatError`, `InternalError`)
  instead of printing and exiting the interpreter.
- ORCA discovery: `orcadir` argument → `OPENMMQMMM_ORCADIR` environment variable → validated
  `orca` in `PATH`. The `~/ash_user_settings.ini` machinery was removed. Discovery rejects
  unrelated `orca` binaries (e.g. the GNOME screen reader) and cannot hang on them.
- Fragment files use the `.frag` extension (was `.ygg`); results are written to
  `results.json` (was `ASH.result`); openmm is a declared dependency.

### Fixed
- Crash in `print_dummy_orca_file` for 2-column Hessian remainders (malformed format string)
  and a wrong branch for 4-column remainders.
- `optimize_geometry(num_grad=True)` numerical-gradient wrapping (parameter shadowed the class).
- Silently dropped second label column in `print_coords_all`/`write_coords_all`.
- `TypeError` for scalar labels with `mofilesdir` in parallel runs.
- `Element` objects discarding their `name` attribute.
- `solvate_small_molecule` now fails clearly for unsupported forcefield XMLs instead of
  pointing at a directory that does not exist (the same applies to the removed
  `Fragment(databasefile=...)` route).
- `QMMMTheory.get_dipole_moment`/`get_polarizability_tensor` no longer hit
  `UnboundLocalError` when the QM theory lacks the property.

### Removed
- Dead code throughout: superseded `ShiftMMCharges` variants, unused `Fragment` methods, the
  unused `Theory`/`QMTheory` base classes, never-assigned `Results` fields, the `mm_elems`
  parameter chain, ~600 lines of commented-out code, and the import-time banner/atexit/cwd-scan
  machinery.

### Added
- Ruff lint + format enforcement (config in pyproject.toml) and GitHub Actions CI
  (lint + tests on Python 3.10/3.13 with the conda-forge OpenMM stack).
- Google-style docstrings on the public API, typed `Results` dataclass, `py.typed` marker.
- `find_orca()` helper and validated-ORCA test skip logic; silent-import and discovery tests.
