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

  openmmqmmm.configure_logging()
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
