from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from openmmqmmm.exceptions import (
    InputError,
    require,
)
from openmmqmmm.openmm.md import MolecularDynamicsEngine, engine_kwargs_checked
from openmmqmmm.utils import (
    main_header,
    write_string_to_file,
)

logger = logging.getLogger(__name__)


def openmm_md_plumed(
    *,
    plumed_input_string: str | None = None,
    simulation_steps: int | None = None,
    simulation_time: float | None = None,
    restraints: Sequence[Sequence[float | int]] | None = None,
    **md_options: Any,
) -> None:
    """Run MD with a PLUMED bias (requires the openmm-plumed plugin).

    Every keyword of :class:`~openmmqmmm.MolecularDynamicsEngine` is accepted through
    ``md_options`` and forwarded unchanged; an unrecognised one raises ``InputError``.
    """
    engine_kwargs = engine_kwargs_checked(md_options, restraints=restraints)

    logger.info(main_header("OpenMM MD using the OpenMM-Plumed interface"))

    # Imported for the side effect: this registers the PLUMED plugin with OpenMM, so
    # find_spec would report availability without actually making it available.
    require(
        "openmmplumed",
        hint=(
            "`bash build_tools/conda_install.sh` from scratch, or the build_plumed function in "
            "build_tools/build_plumed.sh for the active environment; see build_tools/README.md and "
            "https://github.com/openmm/openmm-plumed"
        ),
        feature=(
            "PLUMED-biased dynamics. The current conda-forge build requires OpenMM <8.5 and is "
            "incompatible with this project's OpenMM 8.6 requirement"
        ),
    )

    # The PLUMED input is the whole bias specification; there is nothing to fall back on.
    if plumed_input_string is None:
        raise InputError("plumed_input_string is required: it defines the PLUMED bias to apply.")
    md = MolecularDynamicsEngine(**engine_kwargs)
    try:
        logger.debug("Setting up Plumed")
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
    finally:
        md.close()

    logger.info(
        "You can now analyze/plot the data with plumed's own tools (requires presence of HILLS and COLVAR "
        "files in directory)"
    )
