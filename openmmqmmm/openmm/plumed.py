from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy.typing as npt

from openmmqmmm.coords import Fragment
from openmmqmmm.exceptions import (
    InputError,
    MissingDependencyError,
)
from openmmqmmm.openmm.md import MolecularDynamicsEngine, engine_kwargs_from
from openmmqmmm.utils import (
    main_header,
    write_string_to_file,
)

logger = logging.getLogger(__name__)


def openmm_md_plumed(
    *,
    fragment: Fragment | None = None,
    theory: Any = None,
    timestep: float = 0.001,
    simulation_steps: int | None = None,
    simulation_time: float | None = None,
    traj_frequency: int = 1000,
    temperature: float = 300,
    integrator: str = "LangevinMiddleIntegrator",
    rpmd_num_copies: int | None = None,
    rpmd_qm_num_copies: int | None = None,
    specialatoms: Sequence[int] | None = None,
    specialtraj_frequency: int = 1000,
    barostat: str | None = None,
    pressure: float = 1,
    trajectory_file_option: str = "DCD",
    trajfilename: str = "trajectory",
    coupling_frequency: float = 1,
    charge: int | None = None,
    mult: int | None = None,
    platform: str = "CPU",
    hydrogenmass: float | None = 1.5,
    constraints: Sequence[Sequence[float | int]] | None = None,
    anderson_thermostat: bool = False,
    restraints: Sequence[Sequence[float | int]] | None = None,
    enforce_periodic_box: bool = True,
    special_wrapping: bool = False,
    special_wrapping_updatepos: bool = False,
    wrapping_atoms: Sequence[int] | None = None,
    dummyatomrestraint: bool = False,
    center_on_atoms: Sequence[int] | None = None,
    solute_indices: Sequence[int] | None = None,
    datafilename: str | None = None,
    dummy_mm: bool = False,
    add_centerforce: bool = False,
    centerforce_atoms: Sequence[int] | None = None,
    centerforce_distance: float = 10.0,
    centerforce_constant: float = 1.0,
    centerforce_center: npt.ArrayLike | None = None,
    barostat_frequency: int = 25,
    chkfile: str | None = None,
    statefile: str | None = None,
    plumed_input_string: str | None = None,
) -> None:
    """Run MD with a PLUMED bias (requires the openmm-plumed plugin)."""
    # Captured before any local is bound; the PLUMED-specific parameters are filtered out.
    engine_kwargs = engine_kwargs_from(locals())

    logger.info(main_header("OpenMM MD using the OpenMM-Plumed interface"))

    try:
        # Imported for the side effect: this registers the PLUMED plugin with OpenMM, so
        # find_spec would report availability without actually making it available.
        import openmmplumed  # noqa: F401
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "openmmplumed module plugin not found. The current conda-forge build requires OpenMM <8.5 and is "
            "incompatible with this project's OpenMM 8.5.2 requirement. Install from scratch with "
            "`bash build_tools/conda_install.sh`, or build it into the active environment with the "
            "build_plumed function in build_tools/build_plumed.sh; see build_tools/README.md and "
            "https://github.com/openmm/openmm-plumed"
        ) from None

    md = MolecularDynamicsEngine(**engine_kwargs)

    logger.debug("Setting up Plumed")
    # The PLUMED input is the whole bias specification; there is nothing to fall back on.
    if plumed_input_string is None:
        raise InputError("plumed_input_string is required: it defines the PLUMED bias to apply.")
    logger.info(
        "plumed_input_string provided. Will read all options from this string (make sure to provide atom indices "
        "in 1-based indexing)"
    )
    write_string_to_file(plumed_input_string, "plumedinput.in")

    logger.debug("Now starting PLUMED-biased simulation")
    md.run(
        simulation_steps=simulation_steps,
        simulation_time=simulation_time,
        restraints=restraints,
        plumedinput=plumed_input_string,
    )
    logger.info("PLUMED-biased simulation done")

    md.finalize_simulation()

    logger.info(
        "You can now analyze/plot the data with plumed's own tools (requires presence of HILLS and COLVAR "
        "files in directory)"
    )
    logger.info("\n")
