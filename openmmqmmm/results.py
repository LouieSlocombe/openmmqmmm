from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from openmmqmmm.coords import Fragment
from openmmqmmm.exceptions import FileFormatError

logger = logging.getLogger(__name__)


class _NonFiniteValueError(ValueError):
    """Internal signal used to omit a field that JSON cannot represent safely."""


class _UnsupportedValueError(TypeError):
    """Internal signal used to omit a field containing a known unsupported value."""


_OMIT = object()


def _json_compatible(value: Any) -> Any:
    """Recursively convert common scientific Python values to strict JSON values."""
    if isinstance(value, Fragment):
        raise _UnsupportedValueError("Fragment objects are not part of the on-disk Results schema")
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise _NonFiniteValueError
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            converted = float(value)
            if not np.isfinite(converted):
                raise _NonFiniteValueError
            return converted
        if isinstance(value, np.complexfloating):
            raise _UnsupportedValueError("complex NumPy values have no JSON number representation")
        converted = value.item()
        if isinstance(converted, np.generic):
            raise _UnsupportedValueError(f"NumPy scalar type {value.dtype} has no lossless JSON representation")
        return _json_compatible(converted)
    if isinstance(value, float) and not np.isfinite(value):
        raise _NonFiniteValueError
    if isinstance(value, complex):
        raise _UnsupportedValueError("complex values have no JSON number representation")
    if isinstance(value, Mapping):
        converted_mapping = {}
        original_keys = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in converted_mapping:
                raise _UnsupportedValueError(
                    f"mapping keys {original_keys[normalized_key]!r} and {key!r} both normalize to {normalized_key!r}"
                )
            converted_mapping[normalized_key] = _json_compatible(item)
            original_keys[normalized_key] = key
        return converted_mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, PathLike):
        return os.fspath(value)
    return value


def _serialize_field(name: str, value: Any) -> Any:
    """Convert one Results field, returning a sentinel when policy omits it."""
    try:
        return _json_compatible(value)
    except _NonFiniteValueError:
        logger.warning("Non-finite value found in %s; omitting that field from the results file", name)
        return _OMIT
    except _UnsupportedValueError as error:
        logger.warning("Cannot serialize %s; omitting that field from the results file: %s", name, error)
        return _OMIT


@dataclass
class Results:
    """Container for job results (energies, gradients, frequencies, thermochemistry)."""

    label: str | None = None
    energy: float | None = None
    qm_energy: float | None = None
    mm_energy: float | None = None
    qmmm_energy: float | None = None
    gradient: np.ndarray | None = None
    reaction_energy: float | None = None

    energies: list[Any] | None = None
    reaction_energies: list[float] | None = None
    relative_energies: list[float] | None = None
    labels: list[str | None] | None = None
    gradients: list[np.ndarray] | None = None
    energies_dict: dict[Any, float] | None = None
    gradients_dict: dict[Any, np.ndarray] | None = None
    # Name of worker directories that could be accessed later
    worker_dirnames: dict[Any, str] | None = None
    charge: int | None = None
    mult: int | None = None
    properties: dict[Any, Any] | None = None
    hessian: np.ndarray | None = None
    frequencies: list[float] | None = None
    freq_masses: list[float] | None = None
    freq_elems: list[str] | None = None
    freq_coords: np.ndarray | None = None
    freq_atoms: list[int] | None = None
    freq_tr_modenum: int | None = None
    freq_projection: bool | None = None
    freq_scaling_factor: float | None = None
    freq_dipole_derivs: np.ndarray | None = None
    freq_polarizability_derivs: np.ndarray | None = None
    freq_raman: bool | None = None
    normal_modes: np.ndarray | None = None
    raman_activities: np.ndarray | None = None
    ir_intensities: np.ndarray | None = None
    depolarization_ratios: np.ndarray | None = None
    vib_eigenvectors: np.ndarray | None = None
    thermochemistry: dict[str, Any] | None = None
    displacement_dipole_dictionary: dict[Any, Any] | None = None
    displacement_polarizability_dictionary: dict[Any, Any] | None = None

    def write_to_disk(self, filename: str | PathLike[str] = "results.json") -> None:
        """Write defined attributes atomically as strict JSON.

        NumPy values are converted recursively, including arrays nested inside
        dictionaries. Fields containing NaN or infinity are omitted because JSON has
        no portable representation for them. A serialization failure leaves any
        existing results file untouched.
        """
        import json

        logger.info("Writing defined Results attributes to %s", filename)

        serialized_fields: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            serialized_value = _serialize_field(name, value)
            if serialized_value is not _OMIT:
                serialized_fields[name] = serialized_value

        # Serialize before opening a destination so unsupported user-defined values
        # cannot truncate a valid result from an earlier calculation.
        try:
            document = json.dumps(serialized_fields, allow_nan=False, indent=2) + "\n"
        except (TypeError, ValueError) as error:
            logger.error("Failed to serialize Results; leaving %s untouched: %s", filename, error)
            return

        destination = Path(filename)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
            ) as temporary:
                temporary.write(document)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)


# Taken from the annotations, so an ndarray field added to Results is restored without a
# second list to keep in step. The dict-valued ndarray fields are not distinguishable from
# the plain dict fields by annotation, so those stay explicit.
_ARRAY_FIELDS = frozenset(f.name for f in fields(Results) if f.type == "np.ndarray | None")
_ARRAY_MAPPING_FIELDS = frozenset(
    {"displacement_dipole_dictionary", "displacement_polarizability_dictionary", "gradients_dict"}
)


def _restore_array_fields(data: dict[str, Any]) -> None:
    """Restore the ndarray fields promised by the Results annotations."""
    for name in _ARRAY_FIELDS.intersection(data):
        if data[name] is not None:
            data[name] = np.asarray(data[name])
    if data.get("gradients") is not None:
        data["gradients"] = [np.asarray(gradient) for gradient in data["gradients"]]
    for name in _ARRAY_MAPPING_FIELDS.intersection(data):
        if data[name] is not None:
            data[name] = {key: None if value is None else np.asarray(value) for key, value in data[name].items()}


def read_results_from_file(filename: str | PathLike[str] = "results.json") -> Results:
    """Read a Results object from a JSON file written by Results.write_to_disk."""
    import json

    logger.info("Reading Results data from file:")
    with open(filename) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise FileFormatError(f"Results file must contain a JSON object, not {type(data).__name__}")
    logger.debug("Results fields read: %s", sorted(data))

    # Ignore keys from files written by older versions with more fields
    known_fields = {f.name for f in fields(Results)}
    known_data = {k: v for k, v in data.items() if k in known_fields}
    _restore_array_fields(known_data)
    return Results(**known_data)
