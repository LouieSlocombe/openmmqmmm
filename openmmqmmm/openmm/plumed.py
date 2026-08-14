import logging

from openmmqmmm.exceptions import (
    InputError,
    MissingDependencyError,
)
from openmmqmmm.openmm.md import MolecularDynamicsEngine, engine_kwargs_from
from openmmqmmm.utils import (
    main_header,
    writestringtofile,
)

logger = logging.getLogger(__name__)


def openmm_md_plumed(
    *,
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
    rpmd_num_copies=None,
    specialatoms=None,
    specialtraj_frequency=1000,
    barostat=None,
    pressure=1,
    trajectory_file_option="DCD",
    trajfilename="trajectory",
    coupling_frequency=1,
    charge=None,
    mult=None,
    platform="CPU",
    hydrogenmass=1.5,
    constraints=None,
    anderson_thermostat=False,
    restraints=None,
    enforce_periodic_box=True,
    special_wrapping=False,
    special_wrapping_updatepos=False,
    wrapping_atoms=None,
    dummyatomrestraint=False,
    center_on_atoms=None,
    solute_indices=None,
    datafilename=None,
    dummy_mm=False,
    add_centerforce=False,
    centerforce_atoms=None,
    centerforce_distance=10.0,
    centerforce_constant=1.0,
    centerforce_center=None,
    barostat_frequency=25,
    chkfile=None,
    statefile=None,
    plumed_input_string=None,
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
            "openmmplumed module plugin not found. See https://github.com/openmm/openmm-plumed \nYou can install via "
            "conda: \nconda install -c conda-forge openmm-plumed"
        ) from None

    md = MolecularDynamicsEngine(**engine_kwargs)

    logger.info("Setting up Plumed")
    # The PLUMED input is the whole bias specification; there is nothing to fall back on.
    if plumed_input_string is None:
        raise InputError("plumed_input_string is required: it defines the PLUMED bias to apply.")
    logger.info(
        "plumed_input_string provided. Will read all options from this string (make sure to provide atom indices "
        "in 1-based indexing)"
    )
    writestringtofile(plumed_input_string, "plumedinput.in")

    logger.info("Now starting PLUMED-biased simulation")
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
