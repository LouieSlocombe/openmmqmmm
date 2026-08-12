# openmmqmmm — working notes

ORCA + OpenMM QM/MM package, derived from ASH and reduced to the ORCA + OpenMM stack.
See [README.md](README.md) for the user-facing documentation.

## Environment

The package is installed editable into the conda env `openmmqmmm`:

```bash
conda env create -f environment.yml && conda activate openmmqmmm && pip install -e .
```

ORCA is not part of the environment. It is found via the `orcadir` argument, then the
`OPENMMQMMM_ORCADIR` environment variable, then a validated `orca` in `PATH`. Discovery
validates candidates (the directory must hold `orca` plus its `orca_*` helpers) because
`/usr/bin/orca` is the GNOME screen reader on many Linux systems.

## Checks to run before calling a change done

```bash
ruff check . && ruff format --check . && pytest
```

Set `OPENMMQMMM_ORCADIR` to also run the two end-to-end QM/MM tests; without it they skip
and coverage drops from ~43% to ~36%. `pytest --cov` reports coverage (config in
pyproject.toml). The full suite takes about 4 minutes — the OpenMM tests dominate, since
each builds a solvated-protein system.

## Conventions

These were settled in the 1.0.0 modernization and are load-bearing — changing them breaks
user scripts:

- Job functions are snake_case (`single_point`, `optimize_geometry`,
  `numerical_frequencies`, `openmm_md`); classes are CapWords (`ORCATheory`,
  `OpenMMTheory`, `QMMMTheory`, `Fragment`, `Results`); keyword arguments are snake_case
  (`grad=`, `active_region=`, `num_grad=`).
- **Only the package's own keyword arguments are snake_case.** OpenMM's and mdtraj's APIs
  are camelCase (`enforcePeriodicBox`, `getPositions`, `reportInterval`); renaming those
  at a call site is a silent `TypeError` waiting to happen, and has happened.
- Importing the package is silent and side-effect free. Output goes through `logging`
  under per-module `openmmqmmm.*` loggers; `configure_logging()` sets up console output.
- **Severity lives in the log level, not the message text.** A message about a problem
  uses `logger.warning`/`logger.error` — not `logger.info("WARNING: ...")`. Enforced by
  `openmmqmmm/tests/test_logging_levels.py`.
- Errors raise typed exceptions from `openmmqmmm.exceptions`, never `sys.exit`.
- Every public method of an exported class needs a Google-style docstring, and every
  exported function needs a return annotation (the package ships `py.typed`). Both are
  enforced by `openmmqmmm/tests/test_api_docs.py`.
- Fragment files use `.frag`; each job function writes its `Results` to `results_*.json`.

## Testing notes

- `conftest.py` runs every test in its own temporary directory — the ORCA/OpenMM runs
  scatter output files into the working directory.
- The `fake_orca_dir` / `make_fake_orca_install` fixtures let ORCA-dependent code paths be
  tested without a real installation, which is how the input writers and output parsers
  stay covered in CI.
- `openmmqmmm/tests/orca_outputs/` holds real ORCA 6.1.1 reference output. The output
  parsers are the layer that breaks silently when ORCA changes its formatting, so
  regressions there show up as parse mismatches rather than errors.
- Test data (~2.5 MB) stays in the repository and is excluded from wheels.
