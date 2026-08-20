from types import ModuleType


class OpenMMQMMMError(Exception):
    """Base class for all openmmqmmm errors."""


class InputError(OpenMMQMMMError, ValueError):
    """Bad or missing user input: arguments, keywords, invalid combinations."""


class MissingDependencyError(OpenMMQMMMError, ImportError):
    """An optional dependency is required for the requested feature."""


class ExternalProgramError(OpenMMQMMMError, RuntimeError):
    """An external program (ORCA, OpenMPI) is missing, broken, or failed."""


class FileFormatError(OpenMMQMMMError, ValueError):
    """A file (xyz/pdb/psf/Hessian/ORCA output...) could not be parsed."""


class InternalError(OpenMMQMMMError, RuntimeError):
    """An internal consistency check failed ("should never happen")."""


def require(module_name: str, hint: str | None = None, feature: str | None = None) -> ModuleType:
    """Import and return an optional dependency, or raise MissingDependencyError."""
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
