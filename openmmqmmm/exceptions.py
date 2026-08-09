"""Exception hierarchy for openmmqmmm.

Every error raised by this package derives from OpenMMQMMMError, so callers can
catch that one type to handle any package failure. The subclasses also inherit
from the closest builtin exception (ValueError, ImportError, RuntimeError) so
generic handlers keep working.
"""


class OpenMMQMMMError(Exception):
    """Base class for all openmmqmmm errors."""


class InputError(OpenMMQMMMError, ValueError):
    """Bad or missing user input: arguments, keywords, invalid combinations."""


class MissingDependencyError(OpenMMQMMMError, ImportError):
    """An optional dependency is required for the requested feature.

    Raised via require(), which attaches an installation hint.
    """


class ExternalProgramError(OpenMMQMMMError, RuntimeError):
    """An external program (ORCA, OpenMPI) is missing, broken, or failed."""


class FileFormatError(OpenMMQMMMError, ValueError):
    """A file (xyz/pdb/psf/Hessian/ORCA output...) could not be parsed."""


class InternalError(OpenMMQMMMError, RuntimeError):
    """An internal consistency check failed ("should never happen")."""


def require(module_name, hint=None, feature=None):
    """Import and return an optional dependency, or raise MissingDependencyError.

    Args:
        module_name: importable module name, e.g. "matplotlib" or "openff.toolkit".
        hint: install command suggestion, e.g. "conda install -c conda-forge matplotlib".
        feature: short description of what needs the dependency, for the error message.

    Returns:
        The imported module object.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as err:
        parts = [f"The '{module_name}' library is required"]
        if feature:
            parts.append(f"for {feature}")
        message = " ".join(parts) + "."
        if hint:
            message += f" Install it with: {hint}"
        raise MissingDependencyError(message) from err
