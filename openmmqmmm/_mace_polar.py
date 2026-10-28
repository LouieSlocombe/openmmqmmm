from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import openmm

from openmmqmmm.exceptions import InputError, MissingDependencyError, require

_MODELS = {
    "mace-polar-1-small": "polar-1-s",
    "mace-polar-1-medium": "polar-1-m",
    "mace-polar-1-large": "polar-1-l",
}
_SYSTEM_OPTIONS = frozenset(
    {"charge", "multiplicity", "precision", "device", "returnEnergyType", "mlLongRange", "forceGroup"}
)
_INSTALL_HINT = "conda env update -n openmmqmmm -f build_tools/environment_mace_polar.yml"


def _polar_name(potential: Any) -> str | None:
    # OpenMM-ML 1.8 stores the model identity on its implementation object.
    name = potential if isinstance(potential, str) else getattr(getattr(potential, "_impl", None), "name", None)
    if name == "mace-polar":
        return "mace-polar-1-medium"
    if isinstance(name, str) and name.startswith("mace-polar"):
        if name not in _MODELS:
            raise InputError(f"Unknown MACE-POLAR model {name!r}; choose mace-polar or one of {sorted(_MODELS)}")
        return name
    return None


def _polar_options(
    name: str,
    potential_options: Mapping[str, Any],
    system_options: Mapping[str, Any],
    system: openmm.System,
    periodic: bool,
) -> tuple[str, dict[str, Any]]:
    unknown = set(potential_options) - {"modelPath"}
    if unknown:
        raise InputError(
            f"Unsupported MACE-POLAR potential_kwargs: {sorted(unknown)}; use system_kwargs for model settings"
        )
    unknown = set(system_options) - _SYSTEM_OPTIONS
    if unknown:
        raise InputError(f"Unsupported MACE-POLAR system_kwargs: {sorted(unknown)}")
    options = dict(system_options)
    if options.get("returnEnergyType", "energy") != "energy":
        raise InputError("MACE-POLAR requires returnEnergyType='energy' for energies consistent with its forces")
    if options.get("mlLongRange", True) is not True:
        raise InputError("MACE-POLAR requires mlLongRange=True")
    options["precision"] = options.get("precision", "double")
    if options["precision"] not in ("single", "double"):
        raise InputError("MACE-POLAR precision must be 'single' or 'double'")
    for key, default in (("charge", 0), ("multiplicity", 1)):
        value = options.get(key, default)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise InputError(f"MACE-POLAR {key} must be an integer")
        if key == "multiplicity" and value < 1:
            raise InputError("MACE-POLAR multiplicity must be a positive integer")
        options[key] = int(value)

    if periodic:
        nonbonded = [force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)]
        if len(nonbonded) != 1 or nonbonded[0].getNonbondedMethod() not in (
            openmm.NonbondedForce.PME,
            openmm.NonbondedForce.Ewald,
        ):
            raise InputError(
                "Periodic MACE-POLAR ML/MM requires one NonbondedForce using PME or Ewald; "
                "cutoff-only electrostatics and LJPME are unsupported"
            )

    model = _MODELS[name]
    if potential_options.get("modelPath") is not None:
        model_path = potential_options["modelPath"]
        if not isinstance(model_path, (str, os.PathLike)) or not Path(model_path).is_file():
            raise InputError("MACE-POLAR modelPath must identify an existing local checkpoint file")
        model = str(Path(model_path).resolve())
    return model, options


def _create_polar_potential(openmmml: Any, model: str, options: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    require("graph_longrange", feature="MACE-POLAR", hint=_INSTALL_HINT)
    calculators = require("mace.calculators", feature="MACE-POLAR", hint=_INSTALL_HINT)
    if not callable(getattr(calculators, "mace_polar", None)):
        raise MissingDependencyError(
            f"MACE-POLAR requires a MACE release with mace_polar. Install it with: {_INSTALL_HINT}"
        )
    try:
        calculator = calculators.mace_polar(
            model=model,
            device=options.get("device", ""),
            default_dtype="float64" if options["precision"] == "double" else "float32",
        )
    except ImportError as exc:
        raise MissingDependencyError(
            f"MACE-POLAR backend dependencies are unavailable. Install with: {_INSTALL_HINT}"
        ) from exc

    # The public ASE calculator supplies full PolarMACE energies and batched cell
    # tensors. OpenMM-ML 1.8's raw MACE callback uses a scalar cell volume and
    # defaults to interaction_energy, which omits PolarMACE electrostatics.
    kwargs = {
        "calculator": calculator,
        "info": {"charge": options["charge"], "spin": options["multiplicity"]},
        "mlLongRange": True,
    }
    if "forceGroup" in options:
        kwargs["forceGroup"] = options["forceGroup"]
    return openmmml.MLPotential("ase"), kwargs
