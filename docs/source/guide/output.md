# Output, logging and errors

## Turning on output

Importing the package is silent and free of side effects, which is what a library should do:
it installs a `NullHandler` and nothing else. One call gives you the classic calculation
output:

```python
import openmmqmmm

openmmqmmm.configure_logging()  # INFO to the console
openmmqmmm.configure_logging(level="DEBUG", file="calc.log")  # and to a file
```

{func}`~openmmqmmm.configure_logging` configures both the `openmmqmmm` logger and geomeTRIC's,
so an optimization's output appears in the same stream as everything else. It replaces only
the handlers it installed itself, so a logging setup your application already made survives.

The `OPENMMQMMM_LOGLEVEL` environment variable overrides the `level` argument, which is how
you raise the verbosity of a script you would rather not edit:

```bash
OPENMMQMMM_LOGLEVEL=DEBUG python run.py
```

## Levels

`INFO`
: The calculation output — headers, energies, convergence tables, what each step did.

`DEBUG`
: Adds internal detail, and step timings on the `openmmqmmm.timings` logger. To keep the
  timings without the rest:

  ```python
  import logging

  openmmqmmm.configure_logging(level="DEBUG")
  logging.getLogger("openmmqmmm").setLevel(logging.INFO)
  logging.getLogger("geometric").setLevel(logging.INFO)
  logging.getLogger("openmmqmmm.timings").setLevel(logging.DEBUG)
  ```

`WARNING` and above
: Severity is carried by the level, never by a `"WARNING: "` prefix in the message, so
  filtering by level is reliable.

## Using the package's loggers directly

Every module logs to `logging.getLogger(__name__)`, so the standard library's own tools work:
attach your own handler to the `openmmqmmm` logger, filter by module
(`openmmqmmm.orca`, `openmmqmmm.openmm.md`), or route the output into an application's
logging configuration and skip `configure_logging` entirely.

## ORCA output files

ORCA always writes `<filename>.out`, which the next evaluation overwrites. The constructor
options control additional copies:

| Option | Default | Additional output |
|---|---|---|
| `keep_last_output` | `True` | `<filename>_last.out`, replaced after each completed run |
| `keep_each_run_output` | `False` | `<filename>_run<N>.out`, one file per evaluation |
| `save_output_with_label` | `False` | `<filename>_<label>_<charge>_<mult>.out` |

For trajectories where the current output is enough, use `ORCATheory(..., keep_last_output=False)`.
This skips the full-file copy; it keeps the current output and GBW orbital reuse. Disabling
copies does not delete files left by earlier runs. A failed subprocess or abnormal termination
leaves its diagnostics in `<filename>.out` and preserves previous successful copies. Explicitly
ignored errors follow the usual copy policy.

The common run summary reads termination, SCF iterations, energy, warnings, errors and timings
in one pass. Optional property tables are read only when requested or detected in the output.

## ORCA evaluation statistics

`ORCATheory(..., profile=True)` sets `theory.last_evaluation_stats` after each `run()`.
The record contains the current QM atom and point-charge counts, execution settings, SCF
iterations and status. It is JSON serializable and replaced on every call, including failures;
profiling is disabled by default and the attribute is then `None`.

`orca_timings_s` contains ORCA's reported component times. `wall_times_s` measures input
preparation, subprocess execution, common parsing, remaining wrapper work, output copies and
postprocessing with a monotonic clock. Its `total` includes these phases: do not add it to
them, or add ORCA component timings to the subprocess wall time. A failed call records only
the phases it reached. Explicitly ignored ORCA failures have status `ignored_error`.

## QM/MM dynamics profiling

Pass `profile_file="profile.jsonl"` to `MolecularDynamicsEngine` or `openmm_md` to append
structured records for every force callback and each MD step. `profile_interval=10` groups
step records into ten-step intervals; the final partial interval is also recorded. Profiling
is off by default. Enabling it adds clock, file-I/O and scheduling overhead, so compare
throughput separately with profiling disabled.

`force_evaluation` records identify cache hits versus fresh calculations, atom/QM/cap/charge-site
counts, the platform and its properties, timestep, bead/contraction settings and callback wall
time. QM/MM wrapper phases measure periodic imaging, embedding preparation, charge-site
construction, the QM backend and gradient projection. Fresh ORCA evaluations include the
backend record described above; cache hits have no backend timings. The backend's original
`profile` setting is restored after each callback. Failed callbacks produce an error record.

`md_step` records contain the ending step, number of steps, elapsed wall time, fresh-evaluation
and cache-hit counts, callback wall time and `other_wall_s`. The latter is the remainder,
including native MM integration, reporting and I/O; it is not an isolated MM kernel timer.
Step timing starts after initialization and restarts exclude time spent idle between runs.
Neither record type requests additional energies or forces. Full-ring diagnostic work, when
enabled, appears as ordinary force evaluations.

Force records carry `simulation_time_ps` and `context_step`. OpenMM's RPMD integrator does not
advance the Context step counter within a batch, so `context_step` can lag the actual progress;
use the time and `md_step` records when examining RPMD intervals. Files append across restarts;
serialized Systems do not retain profiling destinations or timer state. Wrapper/backend totals
are inclusive: do not add them to their nested phases or ORCA component timings.

## Errors

Every error the package raises derives from {exc}`~openmmqmmm.OpenMMQMMMError`, and each
subclass also inherits the closest builtin, so existing `except ValueError` handlers keep
working:

| Exception | Also inherits | Raised when |
|---|---|---|
| {exc}`~openmmqmmm.InputError` | `ValueError` | An argument combination cannot work |
| {exc}`~openmmqmmm.MissingDependencyError` | `ImportError` | An optional dependency is not installed |
| {exc}`~openmmqmmm.ExternalProgramError` | `RuntimeError` | ORCA, PLUMED or another external program failed |
| {exc}`~openmmqmmm.FileFormatError` | `ValueError` | A file could not be parsed as its format |
| {exc}`~openmmqmmm.InternalError` | `RuntimeError` | An invariant inside the package was violated |

```python
from openmmqmmm import ExternalProgramError, OpenMMQMMMError

try:
    single_point(theory=orca, fragment=fragment)
except ExternalProgramError:
    ...  # ORCA itself failed; its output is on the logger
except OpenMMQMMMError:
    ...  # anything else this package raises
```

Nothing calls `sys.exit`: a failure is always an exception you can catch, which is what makes
the package usable inside a larger script.

{func}`~openmmqmmm.require` is the helper behind the optional-dependency case: it imports a
module and returns it, or raises {exc}`~openmmqmmm.MissingDependencyError` with a message
naming the feature that needed it and the command that installs it.
