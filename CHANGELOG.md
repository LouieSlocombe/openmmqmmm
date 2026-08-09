# Changelog

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
