# Changelog

## Unreleased

A second audit pass, this time driven by writing tests for the untested modules. Statement
coverage went from 28% to 43% (36% without ORCA installed) across 289 tests, and the new
tests found eight defects — most of them silent wrong answers rather than crashes.

### Fixed
- **ORCA input files were malformed whenever `extraline` or `orcablocks` was set on a
  gas-phase calculation.** `create_orca_input_plain` wrote `extraline` with no trailing
  newline and `orcablocks` with no leading one, so directives ran together on a single
  line: `! OPT %scf maxiter 200 end`, or `! TightSCF! Engrad` for a gradient run. ORCA
  exits with an error. The point-charge variant used by QM/MM had the separators right, so
  only gas-phase runs were affected. The two near-duplicate writers are now one
  implementation, differing only by the `%pointcharges` line.
- **`ORCATheory.opt()` could not be called twice** and leaked into later calculations: it
  appended `! OPT` to `self.extraline`, so a second call emitted `! OPT ! OPT` and any
  subsequent `run()` single point silently inherited the optimization directive. It also
  now returns the optimized energy instead of None.
- **The thermal vibrational energy used the wrong Bose-Einstein factor** — `1/exp(x - 1)`
  instead of `1/(exp(x) - 1)` — making that term 2.7x too large for water and, worse,
  sending it to zero rather than the classical RT limit for the low-frequency modes that
  dominate floppy biomolecular systems. The corrected value reproduces ORCA's to the
  printed precision. Zero-frequency modes are now excluded rather than dividing by zero.
- **`enforce_periodic_box` was passed to OpenMM and mdtraj**, whose APIs spell it
  `enforcePeriodicBox`. The 1.0.0 snake_case rename pass had renamed keyword arguments of
  the external libraries, not just the package's own. `openmm_minimize` raised TypeError
  after doing all its work but before updating the fragment; the MD state retrieval,
  trajectory reporters, metadynamics and the geomeTRIC QM/MM path were also affected.
- **`elemlisttoformula` returned a different string in every process**, because it iterated
  a set. The formula is embedded in the calculation labels `single_point_fragments` builds,
  so those were not reproducible between runs. Formulas are now in Hill notation.
- **`write_orca_hessfile` produced files `grab_hessian` could not read**: nothing marked
  the end of the `$hessian` block, so the reader ran on into `$atoms` and raised IndexError.
- **`grab_orca_timings` matched one of its nine labels** against ORCA 6.x output, including
  the point-charge gradient timing that the QM/MM path reports. It now tolerates the
  varying column widths.
- **`NumGrad(npoint=...)` silently returned a zero gradient** for any value other than 1 or
  2 — which an optimizer reads as an already-converged structure. Unsupported stencils now
  raise `InputError` at construction.
- `single_point_theories`, `single_point_fragments` and `single_point_reaction` raised
  `AttributeError` for any theory without a `cleanup()` method, `ZeroTheory` included.
- `OpenMMTheory.write_pdbfile` ignored an explicit `positions=` argument whenever the
  object had its own, silently writing the original coordinates.
- `Reaction` now rejects a stoichiometry that does not match the fragment count at
  construction, rather than after every fragment has been through the QM program.
- `ORCATheory.cleanup()` never deleted the ORCA `*tmp` scratch files: the glob pattern was
  the literal string `"self.filename*tmp"` rather than an f-string.

### Changed
- **Severity now lives in the log level, not the message text.** 50 warnings and errors
  were emitted through `logger.info` with a `WARNING:`/`Error:` prefix, so filtering by
  level did not surface them; section banners and routine progress lines were conversely
  emitted at WARNING. Leftover debug output (`logger.info("here")`, internal attribute
  dumps) is gone or demoted to DEBUG.
- **`openmm.py` (6169 lines) is now a package**: `openmm/theory.py`, `openmm/systemsetup.py`,
  `openmm/md.py` and `openmm/metadynamics.py`. Every name is re-exported, so imports are
  unchanged.
- `inertia` uses `np.array` rather than the pending-deprecation `np.matrix`.

### Added
- Google-style docstrings on all 128 public methods of the exported classes — `Fragment`,
  `OpenMMTheory`, `QMMMTheory`, `ORCATheory` and the rest had none.
- Return annotations on every exported function, so the shipped `py.typed` marker is no
  longer a promise the package does not keep.
- Test suites for the ORCA output parsers (against committed ORCA 6.1.1 reference output),
  the frequency/thermochemistry engine, `NumGrad`, the coordinate helpers, the periodic-cell
  helpers, `OpenMMTheory` and the single-point job functions.
- Tests that enforce the conventions above: no warning text at INFO, no undocumented public
  method, no exported function without a return annotation.
- Coverage measurement (`pytest --cov`, `pytest-cov` in the `test` extra), a coverage floor
  in CI, a wheel-build-and-import CI job, `CLAUDE.md`, and runnable `examples/`.

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
  old `*.result` names.
- Removed the last legacy naming from the code: the `write_pdbfile` default output name, the
  geomeTRIC engine local variable, and the remaining legacy references in log and error text.
  The upstream migration table was dropped from README.md; the rename summary below covers it.

### Added
- `openmmqmmm.__version__`.
- Google-style docstrings on the remaining undocumented public coordinate helpers.
- Regression tests for `job_parallel` (the gap that let the kwarg bug ship) and for the
  `Results` write/read round-trip.

## 1.0.0 (2026-08-09)

Full modernization of the codebase. **Breaking release**: the legacy 0.x API is gone.

### Changed
- Public API renamed to PEP8 style: `Singlepoint` → `single_point`, `geomeTRICOptimizer` →
  `optimize_geometry`, `NumFreq` / `AnFreq` → `numerical_frequencies` /
  `analytic_frequencies`, `OpenMM_MD` → `openmm_md`, `OpenMM_Opt` → `openmm_minimize`,
  `OpenMM_Modeller` → `openmm_modeller`, `OpenMM_metadynamics` → `openmm_metadynamics`,
  `Job_parallel` / `Simple_parallel` → `job_parallel` / `simple_parallel`, `ReactionEnergy` →
  `reaction_energy`, `actregiondefine` → `define_active_region`, `ORCA_External_Optimizer` →
  `orca_external_optimizer`, and ~90 further symbol renames. The `Results` dataclass and the
  `Plot` class took over from their prefixed predecessors. Keyword arguments were snake_cased
  (`Grad=` → `grad=`, `ActiveRegion=` → `active_region=`, `TruncatedPC=` → `truncated_pc=`,
  ...). The `Optimizer`/`Opt`/`MolecularDynamics`/`MetaDynamics` aliases are gone. Class names
  that were already CapWords (`ORCATheory`, `OpenMMTheory`, `QMMMTheory`, `Fragment`,
  `Reaction`, `ZeroTheory`) are unchanged.
- Flat module layout: `openmmqmmm.orca`, `openmmqmmm.openmm`, `openmmqmmm.coords`, ... (the
  `modules/`, `functions/`, `interfaces/` subpackages and their name prefixes are gone).
- All output goes through the `logging` module (per-module loggers under `openmmqmmm.*`).
  Importing the package is silent; `configure_logging()` restores the classic console output.
  The `printlevel` verbosity system was removed in favor of logging levels.
- Errors raise typed exceptions (`OpenMMQMMMError` base with `InputError`,
  `MissingDependencyError`, `ExternalProgramError`, `FileFormatError`, `InternalError`)
  instead of printing and exiting the interpreter.
- ORCA discovery: `orcadir` argument → `OPENMMQMMM_ORCADIR` environment variable → validated
  `orca` in `PATH`. The `~/*_user_settings.ini` machinery was removed. Discovery rejects
  unrelated `orca` binaries (e.g. the GNOME screen reader) and cannot hang on them.
- Fragment files use the `.frag` extension (was `.ygg`); results are written as JSON (was a
  `.result` file); openmm is a declared dependency.

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
