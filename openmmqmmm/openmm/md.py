from __future__ import annotations

import contextlib
import functools
import inspect
import io
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Integral
from sys import stdout
from typing import Any, TextIO

import numpy as np
import numpy.typing as npt
import openmm
import openmm.app
import openmm.unit

import openmmqmmm
import openmmqmmm.constants
from openmmqmmm.coords import (
    Fragment,
    check_charge_mult,
    check_gradient_for_bad_atoms,
    get_centroid,
    write_xyzfile,
)
from openmmqmmm.exceptions import (
    InputError,
    MissingDependencyError,
)
from openmmqmmm.mdtraj import mdtraj_image_trajectory, mdtraj_load, mdtraj_rmsf
from openmmqmmm.openmm.nqe_export import attach_qmmm_rpmd_force
from openmmqmmm.openmm.rpmd_force import (
    RPMDExternalQMForceProvider,
    add_rpmd_python_force,
)
from openmmqmmm.openmm.systemsetup import openmm_minimize
from openmmqmmm.openmm.theory import NUCLEAR_QUANTUM_INTEGRATORS, ForceReporter, OpenMMTheory
from openmmqmmm.singlepoint import single_point
from openmmqmmm.utils import (
    log_time_since,
    main_header,
    small_header,
)

logger = logging.getLogger(__name__)

RPMD_RESTART_FILENAME = "OpenMM_MD_rpmd_restart.npz"
RPMD_FINAL_RESTART_FILENAME = "OpenMM_MD_final_rpmd_restart.npz"
RPMD_RESTART_FORMAT_VERSION = 1


class _LoggerWriter(io.TextIOBase):
    """Present a line-buffered file interface that emits records to a logger."""

    def __init__(self, target: logging.Logger, level: int = logging.INFO) -> None:
        self._logger = target
        self._level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if line:
                self._logger.log(self._level, "%s", line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._logger.log(self._level, "%s", self._buffer)
            self._buffer = ""

    def writable(self) -> bool:
        return True


class _RPMDStateDataReporter:
    """Write state data for one RPMD copy plus the full ring-polymer energy."""

    def __init__(
        self,
        output: TextIO,
        copy_index: int,
        degrees_of_freedom: int | None,
        separator: str = ",",
        append: bool = False,
    ) -> None:
        self._output = output
        self._copy_index = copy_index
        self._degrees_of_freedom = degrees_of_freedom
        self._separator = separator
        self._header_written = append

    def report(self, simulation: openmm.app.Simulation, state: openmm.State) -> None:
        if not self._header_written:
            headers = (
                "Step",
                "Time (ps)",
                "RPMD Copy",
                "Copy Potential Energy (kJ/mole)",
                "Copy Kinetic Energy (kJ/mole)",
                "Copy Temperature (K)",
                "Ring Polymer Total Energy (kJ/mole)",
            )
            header = ('"' + self._separator + '"').join(headers)
            print(f'#"{header}"', file=self._output)
            self._header_written = True

        potential_energy = state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        kinetic_energy = state.getKineticEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        gas_constant = openmm.unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            openmm.unit.kilojoule_per_mole / openmm.unit.kelvin
        )
        if self._degrees_of_freedom and self._degrees_of_freedom > 0:
            temperature = 2 * kinetic_energy / (self._degrees_of_freedom * gas_constant)
        else:
            temperature = float("nan")
        total_energy = simulation.integrator.getTotalEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        simulation_time = state.getTime().value_in_unit(openmm.unit.picosecond)
        values = (
            simulation.currentStep,
            simulation_time,
            self._copy_index,
            potential_energy,
            kinetic_energy,
            temperature,
            total_energy,
        )
        print(self._separator.join(str(value) for value in values), file=self._output)
        self._output.flush()


def engine_kwargs_from(caller_locals: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    engine_parameters = set(inspect.signature(MolecularDynamicsEngine.__init__).parameters) - {"self"}
    return {name: value for name, value in caller_locals.items() if name in engine_parameters} | overrides


def _close_on_error(method: Callable[..., Any]) -> Callable[..., Any]:
    """Close an MD engine's owned output resources if an operation fails."""

    @functools.wraps(method)
    def guarded(engine: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(engine, *args, **kwargs)
        except BaseException:
            engine.close()
            raise

    return guarded


def read_npt_statefile(npt_output: str | os.PathLike[str]) -> dict[str, npt.NDArray[np.generic]]:
    import csv
    from collections import defaultdict

    columns = defaultdict(list)

    with open(npt_output) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                columns[k].append(v)
    steps = np.array(columns['#"Step"'])
    volume = np.array(columns["Box Volume (nm^3)"]).astype(float)
    density = np.array(columns["Density (g/mL)"]).astype(float)

    return {"steps": steps, "volume": volume, "density": density}


def openmm_md(
    *,
    fragment: Fragment | None = None,
    theory: Any = None,
    timestep: float = 0.001,
    simulation_steps: int | None = None,
    simulation_time: float | None = None,
    traj_frequency: int = 1000,
    restartfile_frequency: int = 1000,
    temperature: float = 300,
    integrator: str = "LangevinMiddleIntegrator",
    rpmd_num_copies: int | None = None,
    rpmd_qm_num_copies: int | None = None,
    barostat: str | None = None,
    pressure: float = 1,
    trajectory_file_option: str = "DCD",
    trajfilename: str = "trajectory",
    specialtraj_frequency: int = 1000,
    specialatoms: Sequence[int] | None = None,
    energy_file_option: str | os.PathLike[str] | None = None,
    force_file_option: str | os.PathLike[str] | None = None,
    atomic_units_force_reporter: bool = False,
    coupling_frequency: float = 1,
    charge: int | None = None,
    mult: int | None = None,
    hydrogenmass: float | None = 1.5,
    force_periodic: bool | None = None,
    periodic_cell_dimensions: npt.ArrayLike | None = None,
    anderson_thermostat: bool = False,
    platform: str = "CPU",
    constraints: Sequence[Sequence[float | int]] | None = None,
    restraints: Sequence[Sequence[float | int]] | None = None,
    enforce_periodic_box: bool = True,
    special_wrapping: bool = False,
    special_wrapping_updatepos: bool = False,
    wrapping_atoms: Sequence[int] | None = None,
    dummyatomrestraint: bool = False,
    center_on_atoms: Sequence[int] | None = None,
    solute_indices: Sequence[int] | None = None,
    datafilename: str | os.PathLike[str] | None = None,
    dummy_mm: bool = False,
    add_centerforce: bool = False,
    centerforce_atoms: Sequence[int] | None = None,
    centerforce_constant: float = 1.0,
    centerforce_distance: float = 10.0,
    centerforce_center: npt.ArrayLike | None = None,
    barostat_frequency: int = 25,
    chkfile: str | os.PathLike[str] | None = None,
    statefile: str | os.PathLike[str] | None = None,
) -> None:
    """Run molecular dynamics of a fragment with OpenMM (also drives QM/MM MD)."""
    engine_kwargs = engine_kwargs_from(locals())

    logger.info(main_header("OpenMM MD wrapper function"))
    md = MolecularDynamicsEngine(**engine_kwargs)
    try:
        if simulation_steps is not None:
            md.run(simulation_steps=simulation_steps)
        elif simulation_time is not None:
            md.run(simulation_time=simulation_time)
        else:
            raise InputError("Either simulation_steps or simulation_time need to be defined (not both).")

        md.finalize_simulation()
    finally:
        md.close()


class MolecularDynamicsEngine:
    """Driver for OpenMM molecular-dynamics simulations (also used for QM/MM MD)."""

    def __init__(
        self,
        *,
        fragment: Fragment | None = None,
        theory: Any = None,
        charge: int | None = None,
        mult: int | None = None,
        timestep: float = 0.001,
        traj_frequency: int = 1000,
        restartfile_frequency: int = 1000,
        temperature: float = 300,
        integrator: str = "LangevinMiddleIntegrator",
        rpmd_num_copies: int | None = None,
        rpmd_qm_num_copies: int | None = None,
        barostat: str | None = None,
        pressure: float = 1,
        trajectory_file_option: str = "DCD",
        trajfilename: str = "trajectory",
        specialtraj_frequency: int = 1000,
        specialatoms: Sequence[int] | None = None,
        energy_file_option: str | os.PathLike[str] | None = None,
        force_file_option: str | os.PathLike[str] | None = None,
        atomic_units_force_reporter: bool = False,
        coupling_frequency: float = 1,
        platform: str = "CPU",
        anderson_thermostat: bool = False,
        hydrogenmass: float | None = 1.5,
        constraints: Sequence[Sequence[float | int]] | None = None,
        restraints: Sequence[Sequence[float | int]] | None = None,
        force_periodic: bool | None = False,
        periodic_cell_dimensions: npt.ArrayLike | None = None,
        enforce_periodic_box: bool = True,
        special_wrapping: bool = False,
        special_wrapping_updatepos: bool = False,
        wrapping_atoms: Sequence[int] | None = None,
        dummyatomrestraint: bool = False,
        center_on_atoms: Sequence[int] | None = None,
        solute_indices: Sequence[int] | None = None,
        datafilename: str | os.PathLike[str] | None = None,
        dummy_mm: bool = False,
        add_centerforce: bool = False,
        centerforce_atoms: Sequence[int] | None = None,
        centerforce_constant: float = 1.0,
        centerforce_distance: float = 10.0,
        centerforce_center: npt.ArrayLike | None = None,
        barostat_frequency: int = 25,
        chkfile: str | os.PathLike[str] | None = None,
        statefile: str | os.PathLike[str] | None = None,
    ) -> None:
        module_init_time = time.time()

        logger.info(main_header("OpenMM Molecular Dynamics Initialization"))

        if fragment is None:
            raise InputError("No fragment object. Exiting.")
        self.fragment = fragment

        is_rpmd = integrator == "RPMDIntegrator"
        if is_rpmd and barostat is not None:
            raise InputError(
                "RPMDIntegrator cannot be used with a barostat. Remove the barostat or explicitly select a "
                "classical integrator."
            )

        self.charge, self.mult = check_charge_mult(
            charge, mult, theory.theorytype, fragment, "OpenMM_MD", theory=theory
        )

        # Trajectory filename. Used for trajs in DCD, PDB etc. format, also single PDB snapshots
        self.trajfilename = trajfilename

        self.specialatoms = specialatoms
        self.specialtraj_frequency = specialtraj_frequency

        if os.path.exists("wrapped_special_traj.xyz"):
            os.remove("wrapped_special_traj.xyz")
        if os.path.exists("OpenMMMD_traj_wrapped.xyz"):
            os.remove("OpenMMMD_traj_wrapped.xyz")

        self.dummy_mm = dummy_mm

        self.theory_runtype = None

        self._resolve_theory(
            theory,
            platform=platform,
            hydrogenmass=hydrogenmass,
            constraints=constraints,
            force_periodic=force_periodic,
            periodic_cell_dimensions=periodic_cell_dimensions,
        )

        if is_rpmd and self.theory_runtype in {"QMMM", "QM"}:
            if special_wrapping or special_wrapping_updatepos:
                raise InputError(
                    "RPMD does not support special_wrapping. Use OpenMM periodic wrapping through the "
                    "PythonForce callback instead."
                )
            if dummyatomrestraint:
                raise InputError("RPMD does not support dummyatomrestraint.")

        if integrator in NUCLEAR_QUANTUM_INTEGRATORS:
            self.openmmobject._disable_hydrogen_mass_repartitioning()

        self._configure_rpmd_copies(
            is_rpmd=is_rpmd, rpmd_num_copies=rpmd_num_copies, rpmd_qm_num_copies=rpmd_qm_num_copies
        )

        self._attach_qm_force(is_rpmd)
        self._add_restraints(restraints)

        self.chkfile = chkfile
        self.statefile = statefile

        self.temperature = temperature
        self.pressure = pressure
        self.integrator = integrator
        self.rpmd_report_copy = 0
        self._rpmd_reporters = []
        self._rpmd_reporter_owner: openmm.app.Simulation | None = None
        self._simulation_reporters: list[Any] = []
        self._simulation_reporter_owner: openmm.app.Simulation | None = None
        self.coupling_frequency = coupling_frequency
        self.timestep = timestep
        self.traj_frequency = int(traj_frequency)
        self.restartfile_frequency = restartfile_frequency
        self.barostat_frequency = barostat_frequency
        self.trajectory_file_option = trajectory_file_option
        self.force_file_option = force_file_option  # Gradients/forces as a file
        self.energy_file_option = energy_file_option  # Energies as a file
        self.atomic_units_force_reporter = atomic_units_force_reporter  # Forces in atomic units
        if self.openmmobject.periodic is True:
            # Generally we want True except sometimes we do our own wrapping
            self.enforce_periodic_box = enforce_periodic_box
        else:
            logger.info("System is non-periodic. Setting enforcePeriodicBox to False")
            # Non-periodic. Setting enforcePeriodicBox to False (otherwise nonsense)
            self.enforce_periodic_box = False

        self.special_wrapping = special_wrapping
        self.special_wrapping_updatepos = (
            special_wrapping_updatepos  # Testing: update positions in simulation object after wrapping
        )
        self.wrapping_atoms = wrapping_atoms

        self._log_system_parameters(
            anderson_thermostat=anderson_thermostat,
            barostat=barostat,
            special_wrapping_updatepos=special_wrapping_updatepos,
        )

        if self.openmmobject.autoconstraints is None:
            logger.warning(
                "Autoconstraints have not been set in OpenMMTheory. By default no bonds are constrained in the MD "
                "simulation, which usually requires a timestep around 0.5 fs. autoconstraints='HBonds' is recommended "
                "for 2 fs timesteps with LangevinIntegrator and 4 fs with LangevinMiddleIntegrator; 'AllBonds' or "
                "'HAngles' permits larger timesteps. See OpenMM issues 2754 and 2520 for guidance."
            )
            logger.debug("Will continue...")
        if (self.openmmobject.rigidwater is True and len(self.openmmobject.user_frozen_atoms) != 0) or (
            self.openmmobject.autoconstraints is not None and len(self.openmmobject.user_frozen_atoms) != 0
        ):
            logger.warning(
                "Frozen_atoms options selected but there are general constraints defined in "
                "the OpenMM object (either rigidwater=True or autoconstraints is not None)"
                "\nOpenMM will crash if constraints and frozen atoms involve the same atoms"
            )

        self._set_initial_positions(dummyatomrestraint=dummyatomrestraint, solute_indices=solute_indices)

        # https://github.com/openmm/openmm/issues/1854 -- the translation was computed and then
        # discarded, so the option has never done anything. Fail rather than silently ignore it.
        if center_on_atoms is not None:
            raise InputError("center_on_atoms is accepted but not implemented. Remove it from the call.")

        self._configure_integrator_and_barostat(barostat=barostat, anderson_thermostat=anderson_thermostat)
        if add_centerforce is True:
            self._add_centerforce(
                centerforce_atoms=centerforce_atoms,
                centerforce_center=centerforce_center,
                centerforce_constant=centerforce_constant,
                centerforce_distance=centerforce_distance,
            )
        self._open_data_output(datafilename)

        logger.info("enforcePeriodicBox: %s", self.enforce_periodic_box)
        logger.info("OpenMM Forces defined: %s", self.openmmobject.system.getForces())

        log_time_since(module_init_time, "OpenMM_MD setup")

    @staticmethod
    def _is_rpmd_simulation(simulation: openmm.app.Simulation) -> bool:
        return isinstance(simulation.integrator, openmm.RPMDIntegrator)

    def _get_simulation_state(self, **kwargs: Any) -> openmm.State:
        if self._is_rpmd_simulation(self.simulation):
            return self.simulation.integrator.getState(self.rpmd_report_copy, **kwargs)
        return self.simulation.context.getState(**kwargs)

    def _set_rpmd_reporters(self, simulation: openmm.app.Simulation, restart: bool = False) -> None:
        old_reporters = self._rpmd_reporters
        self._rpmd_reporters = []
        self._rpmd_reporter_owner = simulation
        old_reporters.clear()

        logger.debug("Creating RPMD-aware state and trajectory reporters for copy %s", self.rpmd_report_copy)
        self._rpmd_reporters.append(
            _RPMDStateDataReporter(_LoggerWriter(logger), self.rpmd_report_copy, self.openmmobject.dof, separator=",")
        )
        if self.dataoutputoption != stdout:
            self._rpmd_reporters.append(
                _RPMDStateDataReporter(
                    self.dataoutputoption,
                    self.rpmd_report_copy,
                    self.openmmobject.dof,
                    separator=",",
                    append=restart,
                )
            )

        if self.trajectory_file_option == "PDB":
            self._rpmd_reporters.append(
                openmm.app.PDBReporter(
                    self.trajfilename + ".pdb", self.traj_frequency, enforcePeriodicBox=self.enforce_periodic_box
                )
            )
        elif self.trajectory_file_option == "DCD":
            if restart and not os.path.isfile(f"{self.trajfilename}.dcd"):
                logger.warning("Restart requested without an existing DCD trajectory; creating a new file")
                restart = False
            self._rpmd_reporters.append(
                openmm.app.DCDReporter(
                    self.trajfilename + ".dcd",
                    self.traj_frequency,
                    append=restart,
                    enforcePeriodicBox=self.enforce_periodic_box,
                )
            )
        elif self.trajectory_file_option == "NetCDFReporter":
            mdtraj = mdtraj_load()
            self._rpmd_reporters.append(mdtraj.reporters.NetCDFReporter(self.trajfilename + ".nc", self.traj_frequency))
        elif self.trajectory_file_option == "HDF5Reporter":
            mdtraj = mdtraj_load()
            self._rpmd_reporters.append(
                mdtraj.reporters.HDF5Reporter(
                    self.trajfilename + ".lh5", self.traj_frequency, enforcePeriodicBox=self.enforce_periodic_box
                )
            )

        if self.force_file_option is not None:
            self._rpmd_reporters.append(
                ForceReporter(
                    self.trajfilename + "_force.txt",
                    self.traj_frequency,
                    atomic_units=self.atomic_units_force_reporter,
                )
            )
        logger.info("RPMD restart data will be written to %s", RPMD_RESTART_FILENAME)

    def _report_rpmd_state(self) -> None:
        state = self.simulation.integrator.getState(
            self.rpmd_report_copy,
            getPositions=True,
            getVelocities=True,
            getForces=self.force_file_option is not None,
            getEnergy=True,
            enforcePeriodicBox=self.enforce_periodic_box,
        )
        for reporter in self._rpmd_reporters:
            reporter.report(self.simulation, state)

    def _run_rpmd_mm(self, simulation_steps: int) -> None:
        target_step = self.simulation.currentStep + simulation_steps
        while self.simulation.currentStep < target_step:
            current_step = self.simulation.currentStep
            steps_to_event = [target_step - current_step]
            if self.traj_frequency > 0:
                steps_to_event.append(self.traj_frequency - current_step % self.traj_frequency)
            if self.restartfile_frequency > 0:
                steps_to_event.append(self.restartfile_frequency - current_step % self.restartfile_frequency)
            steps = min(steps_to_event)
            self.simulation.integrator.step(steps)
            # Calling an RPMDIntegrator directly advances time but, unlike Simulation.step(),
            # does not update Context's step counter.  Keep Simulation.currentStep in sync so
            # this event loop terminates and reporters/restarts receive the correct step.
            self.simulation.currentStep = current_step + steps

            current_step = self.simulation.currentStep
            if self.traj_frequency > 0 and current_step % self.traj_frequency == 0:
                self._report_rpmd_state()
            if self.restartfile_frequency > 0 and current_step % self.restartfile_frequency == 0:
                self.write_state_and_chk_files(current_step)

    def _save_rpmd_restart(self, filename: str | os.PathLike[str]) -> None:
        integrator = self.simulation.integrator
        num_copies = integrator.getNumCopies()
        positions = []
        velocities = []
        for copy_index in range(num_copies):
            state = integrator.getState(copy_index, getPositions=True, getVelocities=True)
            positions.append(state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer))
            velocities.append(
                state.getVelocities(asNumpy=True).value_in_unit(openmm.unit.nanometer / openmm.unit.picosecond)
            )

        report_state = integrator.getState(self.rpmd_report_copy)
        box_vectors = report_state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(openmm.unit.nanometer)
        simulation_time = report_state.getTime().value_in_unit(openmm.unit.picosecond)
        with open(filename, "wb") as restart_file:
            np.savez(
                restart_file,
                format_version=RPMD_RESTART_FORMAT_VERSION,
                num_copies=num_copies,
                positions_nm=np.asarray(positions),
                velocities_nm_per_ps=np.asarray(velocities),
                box_vectors_nm=np.asarray(box_vectors),
                current_step=self.simulation.currentStep,
                time_ps=simulation_time,
            )
        logger.info("Saved all %s RPMD copies to %s", num_copies, filename)

    def _load_rpmd_restart(self, filename: str | os.PathLike[str]) -> None:
        try:
            with np.load(filename, allow_pickle=False) as restart:
                format_version = int(restart["format_version"])
                num_copies = int(restart["num_copies"])
                positions = np.array(restart["positions_nm"])
                velocities = np.array(restart["velocities_nm_per_ps"])
                box_vectors = np.array(restart["box_vectors_nm"])
                current_step = int(restart["current_step"])
                simulation_time = float(restart["time_ps"])
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise InputError(f"Invalid RPMD restart file '{filename}': {error}") from error

        integrator = self.simulation.integrator
        expected_copies = integrator.getNumCopies()
        expected_particles = self.openmmobject.system.getNumParticles()
        if format_version != RPMD_RESTART_FORMAT_VERSION:
            raise InputError(
                f"Unsupported RPMD restart format version {format_version}; expected {RPMD_RESTART_FORMAT_VERSION}."
            )
        if num_copies != expected_copies:
            raise InputError(f"RPMD restart contains {num_copies} copies, but this simulation uses {expected_copies}.")
        if positions.shape != (expected_copies, expected_particles, 3) or velocities.shape != positions.shape:
            raise InputError("RPMD restart positions or velocities have incompatible dimensions.")

        for copy_index in range(expected_copies):
            copy_positions = [openmm.Vec3(*xyz) for xyz in positions[copy_index]] * openmm.unit.nanometer
            copy_velocities = [openmm.Vec3(*xyz) for xyz in velocities[copy_index]] * (
                openmm.unit.nanometer / openmm.unit.picosecond
            )
            integrator.setPositions(copy_index, copy_positions)
            integrator.setVelocities(copy_index, copy_velocities)
        if box_vectors.shape == (3, 3):
            vectors = [openmm.Vec3(*vector) for vector in box_vectors] * openmm.unit.nanometer
            self.simulation.context.setPeriodicBoxVectors(*vectors)
        self.simulation.context.setStepCount(current_step)
        self.simulation.context.setTime(simulation_time * openmm.unit.picosecond)
        if getattr(self, "rpmd_force_provider", None) is not None:
            self.rpmd_force_provider.clear_cache()
        logger.info("Restored all %s RPMD copies from %s", expected_copies, filename)

    # Set sim reporters. Needs to be done after simulation is created and not modified anymore
    def _remove_simulation_reporters(self) -> None:
        """Detach reporters owned by this engine while preserving caller reporters."""
        owned_reporters = self._simulation_reporters
        self._simulation_reporters = []
        simulation = self._simulation_reporter_owner
        self._simulation_reporter_owner = None
        if simulation is not None:
            for reporter in owned_reporters:
                with contextlib.suppress(ValueError):
                    simulation.reporters.remove(reporter)
        # Dropping the last references closes OpenMM trajectory reporters through
        # their destructors. StateDataReporter does not close caller-owned streams.
        owned_reporters.clear()

    def _add_simulation_reporter(self, simulation: openmm.app.Simulation, reporter: Any) -> None:
        """Attach and track one reporter owned by this engine."""
        simulation.reporters.append(reporter)
        self._simulation_reporters.append(reporter)
        self._simulation_reporter_owner = simulation

    def set_sim_reporters(self, simulation: openmm.app.Simulation, restart: bool = False) -> None:
        """Configure trajectory, state-data, and restart reporting for the simulation."""
        if self._is_rpmd_simulation(simulation):
            self._set_rpmd_reporters(simulation, restart=restart)
            return

        self._remove_simulation_reporters()
        logger.debug("Creating CheckpointReporter that will write a restartable checkpointfile every X steps")
        checkpointfilename = "OpenMM_MD.chk"
        self._add_simulation_reporter(
            simulation, openmm.app.CheckpointReporter(checkpointfilename, self.traj_frequency * 1)
        )
        logger.debug("Creating StateDataReporter that will write through the package logger")
        statedatareporter_log = openmm.app.StateDataReporter(
            _LoggerWriter(logger),
            self.traj_frequency,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            volume=self.volume,
            density=self.density,
            temperature=True,
            separator=",",
        )
        self._add_simulation_reporter(simulation, statedatareporter_log)
        if self.dataoutputoption != stdout:
            logger.info("Creating StateDataReporter that will write to file: %s", self.datafilename)
            logger.info("restart: %s", restart)
            statedatareporter_file = openmm.app.StateDataReporter(
                self.dataoutputoption,
                self.traj_frequency,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                volume=self.volume,
                density=self.density,
                temperature=True,
                separator=",",
                append=restart,
            )
            self._add_simulation_reporter(simulation, statedatareporter_file)

        if self.trajectory_file_option == "PDB":
            self._add_simulation_reporter(
                simulation,
                openmm.app.PDBReporter(
                    self.trajfilename + ".pdb", self.traj_frequency, enforcePeriodicBox=self.enforce_periodic_box
                ),
            )
        elif self.trajectory_file_option == "DCD":
            # Note: using append keyword here if restarting
            # Check first if file exists for restart (OpenMM errors otherwise)
            if restart is True and os.path.isfile(f"{self.trajfilename}.dcd") is False:
                logger.warning("Restart option was active but trajectory file not existing. Will create new file")
                restart = False

            self._add_simulation_reporter(
                simulation,
                openmm.app.DCDReporter(
                    self.trajfilename + ".dcd",
                    self.traj_frequency,
                    append=restart,
                    enforcePeriodicBox=self.enforce_periodic_box,
                ),
            )
            logger.info("DCDReporter added")
        elif self.trajectory_file_option == "NetCDFReporter":
            logger.info("NetCDFReporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = mdtraj_load()
            self._add_simulation_reporter(
                simulation, mdtraj.reporters.NetCDFReporter(self.trajfilename + ".nc", self.traj_frequency)
            )
        elif self.trajectory_file_option == "HDF5Reporter":
            logger.info("HDF5Reporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = mdtraj_load()
            self._add_simulation_reporter(
                simulation,
                mdtraj.reporters.HDF5Reporter(
                    self.trajfilename + ".lh5", self.traj_frequency, enforcePeriodicBox=self.enforce_periodic_box
                ),
            )
        elif self.trajectory_file_option == "XYZ":
            logger.info("XYZ trajectory format selected (not available for classical MD). Warning: not very fast")
            logger.info("Deleting possible old trajectory-file (OpenMMMD_traj.xyz)")
            with contextlib.suppress(OSError):
                os.remove("OpenMMMD_traj.xyz")
            # Done manually by write_xyzfile

        if self.force_file_option is not None:
            logger.info("ForceReporter traj format selected.")
            self._add_simulation_reporter(
                simulation,
                ForceReporter(
                    self.trajfilename + "_force.txt", self.traj_frequency, atomic_units=self.atomic_units_force_reporter
                ),
            )
        if self.energy_file_option is not None:
            logger.info("Energyfile  selected.")
            with contextlib.suppress(OSError):
                os.remove(self.energy_file_option)
        logger.debug("Simulation reporters: %s", simulation.reporters)

    def write_state_and_chk_files(self, step: int) -> None:
        """Write restart data for the current integrator."""
        if self._is_rpmd_simulation(self.simulation):
            logger.info("Step %s. Saving all RPMD copy positions and velocities", step)
            self._save_rpmd_restart(RPMD_RESTART_FILENAME)
            return

        logger.info(
            f"Step {step}. Saving a statefile and checkpointfile : OpenMM_MD_state.xml and OpenMM_MD_checkpoint.chk"
        )
        logger.info(
            "Can be used to restart a simulation (statefile and chkfile keywords) using the same coordinates and "
            "velocities."
        )
        self.simulation.saveState("OpenMM_MD_state.xml")
        self.simulation.saveCheckpoint("OpenMM_MD_checkpoint.chk")

    def _resolve_theory(
        self,
        theory: Any,
        *,
        platform: str,
        hydrogenmass: float | None,
        constraints: Sequence[Sequence[float | int]] | None,
        force_periodic: bool | None,
        periodic_cell_dimensions: npt.ArrayLike | None,
    ) -> None:
        """Classify the theory object and bind the OpenMM system this engine will drive."""
        self.openmmobject = None
        self.QM_MM_object = None
        self.rpmd_force_provider = None
        self.rpmd_python_force = None
        self.rpmd_external_force_group = None

        logger.debug("Analyzing theory input to OpenMM_MDclass")
        if isinstance(theory, OpenMMTheory):
            logger.debug("This is an OpenMMTheory object")
            self.openmmobject = theory
            self.theory_runtype = "dummy_MM" if self.dummy_mm is True else "MM"
            return

        if isinstance(theory, openmmqmmm.QMMMTheory):
            logger.debug("This is an QMMMTheory object")
            self.QM_MM_object = theory
            self.openmmobject = theory.mm_theory
            self.theory_runtype = "QMMM"
            # Making sure QM/MM object will exit before calculating MM part
            self.QM_MM_object.exit_after_customexternalforce_update = True
            logger.debug("Turning on externalforce option.")
            self.QM_MM_object.openmm_externalforce = True
            return

        logger.info(
            "Unrecognized theory. Will assume to be QM theory and will continue.\n"
            "QM-program forces will be added as a custom external force to OpenMM.\n"
            "Now creating OpenMMTheory object on platform: %s",
            platform,
        )
        # Creating dummy OpenMMTheory (basic topology, particle masses, no forces except CMMRemoval)
        self.openmmobject = OpenMMTheory(
            fragment=self.fragment,
            dummysystem=True,
            platform=platform,
            hydrogenmass=hydrogenmass,
            constraints=constraints,
            periodic=force_periodic,
            periodic_cell_dimensions=periodic_cell_dimensions,
        )
        self.qmtheory = theory
        self.theory_runtype = "QM"

    def _configure_rpmd_copies(
        self, *, is_rpmd: bool, rpmd_num_copies: int | None, rpmd_qm_num_copies: int | None
    ) -> None:
        """Resolve the bead count and how many beads the QM force is evaluated on."""
        if rpmd_num_copies is not None:
            self.openmmobject.set_rpmd_num_copies(rpmd_num_copies)
        self.rpmd_num_copies = self.openmmobject.rpmd_num_copies

        if rpmd_qm_num_copies is not None and (not is_rpmd or self.theory_runtype not in {"QMMM", "QM"}):
            raise InputError("rpmd_qm_num_copies is only valid for QM/MM or external-QM RPMD dynamics.")
        if rpmd_qm_num_copies is None:
            self.rpmd_qm_num_copies = self.rpmd_num_copies
            return
        if (
            isinstance(rpmd_qm_num_copies, bool)
            or not isinstance(rpmd_qm_num_copies, Integral)
            or not 1 <= rpmd_qm_num_copies <= self.rpmd_num_copies
        ):
            raise InputError(
                f"rpmd_qm_num_copies must be a positive integer no larger than rpmd_num_copies "
                f"({self.rpmd_num_copies})."
            )
        self.rpmd_qm_num_copies = int(rpmd_qm_num_copies)

    def _attach_qm_force(self, is_rpmd: bool) -> None:
        """Give OpenMM access to the QM gradient, bead-resolved for RPMD and frozen otherwise."""
        if self.theory_runtype not in {"QMMM", "QM"}:
            return
        if not is_rpmd:
            logger.info("Creating the classical-MD CustomExternalForce for external QM gradients")
            self.openmm_externalforceobject = self.openmmobject.add_custom_external_force()
            return

        if self.theory_runtype == "QMMM":
            (
                self.rpmd_force_provider,
                self.rpmd_python_force,
                self.rpmd_external_force_group,
            ) = attach_qmmm_rpmd_force(
                theory=self.QM_MM_object,
                elems=self.fragment.elems,
                charge=self.charge,
                mult=self.mult,
                num_beads=self.rpmd_num_copies,
                periodic=self.openmmobject.periodic,
            )
        else:
            self.rpmd_force_provider = RPMDExternalQMForceProvider(
                self.qmtheory,
                self.fragment.elems,
                self.charge,
                self.mult,
                periodic=self.openmmobject.periodic,
                cache_size=2 * self.rpmd_num_copies + 4,
            )
            self.rpmd_python_force, self.rpmd_external_force_group = add_rpmd_python_force(
                self.openmmobject.system,
                self.rpmd_force_provider,
                periodic=self.openmmobject.periodic,
            )

        if self.rpmd_qm_num_copies < self.rpmd_num_copies:
            contractions = dict(getattr(self.openmmobject, "rpmd_contractions", {}))
            contractions[self.rpmd_external_force_group] = self.rpmd_qm_num_copies
            self.openmmobject.set_rpmd_contractions(contractions)
            logger.info("QM force contracted from %s to %s RPMD copies", self.rpmd_num_copies, self.rpmd_qm_num_copies)

    def _add_restraints(self, restraints: Sequence[Sequence[float | int]] | None) -> None:
        """Add the user's bond, angle and torsion restraints to the OpenMM system."""
        if restraints is None:
            return
        logger.info("Restraints defined. Will add to OpenMMTheory object")
        adders = {
            4: (self.openmmobject.add_custom_bond_force, "Bond", "kcal/mol/Angstrom^2"),
            5: (self.openmmobject.add_custom_angle_force, "Angle", "kcal/mol/radian^2"),
            6: (self.openmmobject.add_custom_torsion_force, "Torsion", "kcal/mol/radian^2"),
        }
        for restraint in restraints:
            entry = adders.get(len(restraint))
            if entry is None:
                raise InputError(
                    f"Restraint {restraint} has {len(restraint)} entries; expected 4 (bond), 5 (angle) or 6 (torsion)."
                )
            add_force, kind, unit = entry
            logger.info("%s restraint assumed. Atoms/value/force-constant: %s (%s)", kind, restraint, unit)
            add_force(*restraint)

    def _log_system_parameters(
        self, *, anderson_thermostat: bool, barostat: str | None, special_wrapping_updatepos: bool
    ) -> None:
        """Echo the resolved MD system parameters so the log is a record of what actually ran."""
        logger.info(small_header("MD system parameters"))
        logger.info(
            f"Temperature: {self.temperature} K\nTimestep: {self.timestep} ps\n"
            f"Integrator: {self.integrator}\nBarostat: {barostat}\n"
            f"Anderson thermostat: {anderson_thermostat}\n"
            f"coupling_frequency: {self.coupling_frequency} ps^-1 (for Nose-Hoover and Langevin integrators)"
        )
        if self.integrator == "RPMDIntegrator":
            logger.info("RPMD number of copies (beads): %s", self.rpmd_num_copies)
            if self.theory_runtype in {"QMMM", "QM"}:
                logger.info("RPMD copies used for the QM force: %s", self.rpmd_qm_num_copies)
        logger.info(
            f"OpenMM autoconstraints: {self.openmmobject.autoconstraints}\n"
            f"OpenMM hydrogenmass: {self.openmmobject.hydrogenmass}\n"
            f"OpenMM rigidwater constraints: {self.openmmobject.rigidwater}\n"
            f"User Constraints: {self.openmmobject.user_constraints}\n"
            f"User Restraints: {self.openmmobject.user_restraints}\n"
            f"Number of atoms: {self.fragment.numatoms}\n"
            f"Number of frozen atoms: {len(self.openmmobject.user_frozen_atoms)}"
        )
        if len(self.openmmobject.user_frozen_atoms) < 50:
            logger.info("Frozen atoms %s", self.openmmobject.user_frozen_atoms)
        logger.info(
            f"\nWill write trajectory in format: {self.trajectory_file_option}\n"
            f"Trajectory write frequency: {self.traj_frequency}\n"
            f"enforcePeriodicBox: {self.enforce_periodic_box}\n"
            f"special_wrapping: {self.special_wrapping}\n"
            f"special_wrapping_updatepos: {special_wrapping_updatepos}\n"
            f"wrapping_atoms: {self.wrapping_atoms}\n"
        )

    def _set_initial_positions(self, *, dummyatomrestraint: bool, solute_indices: Sequence[int] | None) -> None:
        """Take the starting positions from the fragment, adding the restraint dummy atom if asked."""
        logger.debug("Defining atom positions from fragment")
        # self.positions rather than the fragment's, because a dummy atom may be appended below
        self.positions = self.fragment.coords
        self.dummyatomrestraint = dummyatomrestraint
        if dummyatomrestraint is not True:
            return

        if solute_indices is None:
            raise InputError("Dummyatomrestraint requires solute_indices to be set")
        logger.warning(
            "Using dummyatomrestraints: a dummy atom is added to the topology and to the OpenMM coordinates "
            "but not to the fragment, so visualizing the trajectory needs the PDB-file containing the dummy "
            "atom that is written at the end."
        )
        dummypos = get_centroid(np.take(self.fragment.coords, solute_indices, axis=0))
        logger.info("Dummy atom will be added to position: %s", dummypos)
        self.positions = np.append(self.positions, [dummypos], axis=0)
        self.openmmobject.add_dummy_atom_to_restrain_solute(atomindices=solute_indices)

    def _configure_integrator_and_barostat(self, *, barostat: str | None, anderson_thermostat: bool) -> None:
        """Add or drop the barostat/thermostat, then hand the integrator settings to OpenMM."""
        forceclassnames = [force.__class__.__name__ for force in self.openmmobject.system.getForces()]
        if barostat is not None:
            if "MonteCarloBarostat" in forceclassnames:
                logger.info("Barostat already present. Skipping.")
            else:
                montecarlobarostat = openmm.MonteCarloBarostat(
                    self.pressure * openmm.unit.bar, self.temperature * openmm.unit.kelvin
                )
                montecarlobarostat.setFrequency(self.barostat_frequency)
                self.openmmobject.system.addForce(montecarlobarostat)
                logger.info("Barostat added")
            self.integrator = "LangevinMiddleIntegrator"
            logger.info("Barostat requires using integrator: %s", self.integrator)
        elif anderson_thermostat is True:
            logger.info("Anderson thermostat is on.")
            if "AndersenThermostat" not in forceclassnames:
                self.openmmobject.system.addForce(
                    openmm.AndersenThermostat(self.temperature * openmm.unit.kelvin, 1 / openmm.unit.picosecond)
                )
            self.integrator = "VerletIntegrator"
            logger.debug("Now using integrator: %s", self.integrator)
        else:
            # Highest index first: removeForce renumbers every force above the one it removes
            for index, forcename in reversed(list(enumerate(forceclassnames))):
                if forcename in {"MonteCarloBarostat", "AndersenThermostat"}:
                    logger.debug("Removing old force: %s", forcename)
                    self.openmmobject.system.removeForce(index)

        # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
        # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
        self.openmmobject.set_simulation_parameters(
            timestep=self.timestep,
            temperature=self.temperature,
            integrator=self.integrator,
            coupling_frequency=self.coupling_frequency,
        )
        self.volume = self.density = barostat is not None

    def _open_data_output(self, datafilename: str | os.PathLike[str] | None) -> None:
        """Select package logging, or a freshly truncated append-mode data file."""
        self.datafilename = datafilename
        if datafilename is None:
            self.dataoutputoption = stdout
            return
        # The reporter writes its header on every open, so a stale file would accumulate several
        with contextlib.suppress(FileNotFoundError):
            os.remove(datafilename)
        # An open file object, not a name: a name does not survive stepping the simulation one step at a time
        self.dataoutputoption = open(datafilename, "a", encoding="utf-8")  # noqa: SIM115 - owned until close()
        logger.info("Will write data to file: %s", datafilename)

    def _add_centerforce(
        self,
        *,
        centerforce_atoms: Sequence[int] | None,
        centerforce_center: npt.ArrayLike | None,
        centerforce_constant: float,
        centerforce_distance: float,
    ) -> None:
        """Add the flat-bottom force that keeps the solute near a chosen centre."""
        logger.info("Centerforce option active")
        if centerforce_atoms is None:
            centerforce_atoms = self.QM_MM_object.qmatoms
            logger.info("centerforce_atoms unset. Using QM atoms: %s", centerforce_atoms)
        if centerforce_center is None:
            centerforce_center = self.fragment.get_coordinate_center()
            logger.debug("No center coordinates set. Using geometric center of fragment: %s", centerforce_center)
        self.openmmobject.add_centerforce(
            center_coords=centerforce_center,
            atomindices=centerforce_atoms,
            forceconstant=centerforce_constant,
            distance=centerforce_distance,
        )

    def _log_run_parameters(self, simulation_time: float, simulation_steps: int) -> None:
        """Log the run-level parameters and the forces the System carries."""
        logger.info(small_header("MD run parameters"))
        logger.info(
            f"Simulation time: {simulation_time} ps\nSimulation steps: {simulation_steps}\n"
            f"Timestep: {self.timestep} ps\nSet temperature: {self.temperature} K"
        )
        logger.info("OpenMM integrator: %s", self.openmmobject.integrator_name)
        forceclassnames = [force.__class__.__name__ for force in self.openmmobject.system.getForces()]
        logger.info("OpenMM System forces present before run: %s", forceclassnames)

    def _attach_reporters(self, continuing: bool, extra_reporters: Iterable[Any] | None) -> None:
        """Attach the state and trajectory reporters, plus any caller-supplied extras."""
        is_rpmd = self._is_rpmd_simulation(self.simulation)
        if is_rpmd:
            reusing_reporters = self._rpmd_reporter_owner is self.simulation and bool(self._rpmd_reporters)
        else:
            reusing_reporters = self._simulation_reporter_owner is self.simulation and bool(self._simulation_reporters)

        if continuing and reusing_reporters:
            logger.debug("Continuing the existing simulation with its current reporters")
        elif continuing:
            logger.debug("Continuing from restored state. Rebuilding engine-owned reporters in append mode")
            if self.datafilename is not None and self.dataoutputoption.closed:
                self.dataoutputoption = open(  # noqa: SIM115 - owned until close()
                    self.datafilename, "a", encoding="utf-8"
                )
            self.set_sim_reporters(self.simulation, restart=True)
        else:
            logger.info("New run. Creating simulation reporters")
            if self.datafilename is not None:
                logger.info("Deleting old datafile: %s", self.datafilename)
                if self.dataoutputoption != stdout:
                    self.dataoutputoption.close()
                with contextlib.suppress(OSError):
                    os.remove(self.datafilename)
                self.dataoutputoption = open(  # noqa: SIM115 - owned until close()
                    self.datafilename, "a", encoding="utf-8"
                )
            self.set_sim_reporters(self.simulation)
            self.openmmobject.set_positions(self.positions, self.simulation)

        if extra_reporters is None:
            return
        extra_reporters = list(extra_reporters)
        if is_rpmd:
            # The RPMD event loop drives these at traj_frequency alongside the
            # engine's own reporters; describeNextReport is not consulted.
            self._rpmd_reporters.extend(extra_reporters)
        else:
            self.simulation.reporters.extend(extra_reporters)
        logger.info("Attached %s extra reporter(s)", len(extra_reporters))

    def _restart_from_file(
        self,
        restart_file: str | os.PathLike[str],
        description: str,
        load: Callable[[openmm.app.Simulation, str | os.PathLike[str]], None],
    ) -> None:
        """Create the Simulation and restore positions and velocities from a restart file."""
        self.simulation = self.openmmobject.create_simulation()
        if self._is_rpmd_simulation(self.simulation):
            logger.info("RPMD restart file provided via %s", description)
            self._load_rpmd_restart(restart_file)
            return
        logger.info("%s provided. Restarting simulation using position and velocity data in file", description)
        logger.info(
            "Simulation velocities before: %s",
            self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True),
        )
        load(self.simulation, restart_file)
        logger.info(
            "Simulation velocities after loading %s: %s",
            description,
            self.simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True),
        )

    def _prepare_wrapping(
        self,
    ) -> tuple[openmm.unit.Quantity | None, Any | None, Sequence[int] | None]:
        """Return the (box vectors, mdtraj topology, wrapping atoms) that per-step wrapping needs."""
        if self.openmmobject.periodic is not True:
            logger.info("System is not periodic")
            return None, None, None

        logger.info("Periodic Boundary Conditions used.")
        if self.enforce_periodic_box is True:
            logger.info("EnforcePeriodic Box is True. Wrapping enforced by OpenMM.")
            logger.warning("In case of problematic wrapping for e.g. QM/MM, try enabling special_wrapping=True")
        if self.special_wrapping is not True:
            return None, None, None

        logger.info("special_wrapping is True. Wrapping will be handled in each step by mdtraj library")
        try:
            import mdtraj
        except ImportError:
            raise MissingDependencyError(
                "Error: mdtraj not found, needs to be installed (pip install mdtraj)"
            ) from None
        boxvectors = self._get_simulation_state().getPeriodicBoxVectors(asNumpy=True)
        mdtrajtopology = mdtraj.Topology.from_openmm(self.openmmobject.topology)

        if self.wrapping_atoms is not None:
            logger.debug("Will use atoms %s for wrapping", self.wrapping_atoms)
            return boxvectors, mdtrajtopology, self.wrapping_atoms

        logger.debug("No wrapping_atoms keyword has been set to center on.")
        if self.theory_runtype == "QMMM":
            logger.info("Theory-runtype is QMMM. Using QMatoms as wrapping_atoms")
            wrapping_atoms = self.QM_MM_object.qmatoms
        elif self.theory_runtype == "MM":
            logger.info("Theory_runtype is MM. No anchor atoms needed")
            wrapping_atoms = None
        else:
            raise InputError(f"Theory_runtype is {self.theory_runtype} but no wrapping_atoms have been set.\nExiting")
        logger.info("wrapping_atoms have been set to: %s", wrapping_atoms)
        return boxvectors, mdtrajtopology, wrapping_atoms

    def _write_first_frame(self) -> None:
        """Write the initial frame next to the trajectory as both PDB and PDBx/mmCIF."""
        state = self._get_simulation_state(
            getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforce_periodic_box
        )
        positions = state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        topology = self.openmmobject.topology

        pdb_filename = self.trajfilename + "_firstframe.pdb"
        logger.info("Writing initial frame to disk as PDB-file: %s", pdb_filename)
        with open(pdb_filename, "w") as f:
            openmm.app.pdbfile.PDBFile.writeHeader(topology, f)
            openmm.app.pdbfile.PDBFile.writeModel(topology, positions, f)
            openmm.app.pdbfile.PDBFile.writeFooter(topology, f)

        pdbx_filename = self.trajfilename + "_firstframe.cif"
        logger.info("Writing initial frame to disk as PDBx/mmCIF-file: %s", pdbx_filename)
        with open(pdbx_filename, "w") as f:
            openmm.app.pdbxfile.PDBxFile.writeHeader(topology, f)
            openmm.app.pdbxfile.PDBxFile.writeModel(topology, positions, f)

    def _finish_rpmd_run(self, simulation_steps: int, description: str, module_init_time: float) -> None:
        """Run the bead-resolved PythonForce path and log its evaluation statistics."""
        logger.debug("Running bead-resolved %s through OpenMM PythonForce", description)
        self._run_rpmd_mm(simulation_steps)
        logger.info(
            "%s RPMD external evaluations: %s (%s cache hits)",
            description,
            self.rpmd_force_provider.evaluation_count,
            self.rpmd_force_provider.cache_hits,
        )
        logger.info(small_header("OpenMM MD simulation finished!"))
        log_time_since(module_init_time, "OpenMM_MD run")

    def _current_step_coords(
        self,
        checkpoint: float,
        boxvectors: openmm.unit.Quantity | None,
        mdtrajtopology: Any | None,
        wrapping_atoms: Sequence[int] | None,
    ) -> tuple[openmm.State, npt.NDArray[np.float64]]:
        """Return this step's OpenMM state and its coordinates in Angstrom, wrapped if requested."""
        current_state = self.simulation.context.getState(
            getPositions=True, enforcePeriodicBox=self.enforce_periodic_box, getEnergy=True
        )
        log_time_since(checkpoint, "get OpenMM state")
        checkpoint = time.time()
        current_coords = np.array(current_state.getPositions(asNumpy=True)) * 10
        log_time_since(checkpoint, "get current_coords")

        if self.openmmobject.periodic is True and self.special_wrapping is True:
            logger.info("special_wrapping is True. Wrapping handled by mdtraj")
            checkpoint = time.time()
            current_coords = diff_wrap_box_coords(current_coords / 10.0, boxvectors, mdtrajtopology, wrapping_atoms)
            log_time_since(checkpoint, "wrapping via diff_wrap_box_coords")
            if self.special_wrapping_updatepos is True:
                logger.info("special_wrapping_update is True. Updating positions")
                checkpoint = time.time()
                self.openmmobject.set_positions(current_coords, self.simulation)
                log_time_since(checkpoint, "set positions update")
        return current_state, current_coords

    def _write_special_atoms_frame(self, step: int, current_coords: npt.ArrayLike) -> None:
        """Append this step's special-atom subset to its own XYZ trajectory."""
        if self.specialatoms is None or step % self.specialtraj_frequency != 0:
            return
        specialelems = [self.fragment.elems[i] for i in self.specialatoms]
        special_coords = np.take(current_coords, self.specialatoms, axis=0)
        logger.info("Writing wrapped coords to trajfile: only for special atoms")
        write_xyzfile(specialelems, special_coords, "wrapped_special_traj", writemode="a")

    @_close_on_error
    def run(
        self,
        simulation_steps: int | None = None,
        simulation_time: float | None = None,
        plumedinput: str | None = None,
        restraints: Sequence[Sequence[float | int]] | None = None,
        restart: bool = False,
        chkfile: str | os.PathLike[str] | None = None,
        statefile: str | os.PathLike[str] | None = None,
        *,
        extra_reporters: Iterable[Any] | None = None,
        pre_dynamics_hook: Callable[[MolecularDynamicsEngine], None] | None = None,
    ) -> None:
        """Run the molecular dynamics simulation.

        extra_reporters is an iterable of OpenMM-protocol reporters attached for this
        run only: in RPMD runs the engine calls them directly every traj_frequency
        steps (describeNextReport scheduling is not consulted), classically they join
        simulation.reporters with native scheduling. pre_dynamics_hook is called once
        with the engine after the Simulation exists, positions are set, and any
        restart data is loaded, but before dynamics -- the place to seed RPMD bead
        distributions from an external tool. The hook also runs on restarts, so skip
        bead seeding when continuing from chkfile/statefile.
        """
        module_init_time = time.time()
        logger.info(main_header("OpenMM Molecular Dynamics Run"))

        if simulation_steps is None and simulation_time is None:
            raise InputError("Either simulation_steps or simulation_time needs to be set.")
        if simulation_time is not None:
            simulation_steps = int(simulation_time / self.timestep)
        if simulation_steps is not None:
            simulation_time = simulation_steps * self.timestep

        if chkfile is None and self.chkfile is not None:
            logger.info("chkfile provided to init. Will use this for restart.")
            chkfile = self.chkfile
        if statefile is None and self.statefile is not None:
            logger.info("statefile provided to init. Will use this for restart.")
            statefile = self.statefile

        if plumedinput is not None:
            import openmmplumed

            logger.info("Plumed active. Adding Plumedforce to system")
            logger.info("plumedinput: %s", plumedinput)
            plumed_force = openmmplumed.PlumedForce(plumedinput)
            # The plugin defaults to -1 K, which it reports to PLUMED as "undefined". Every
            # quantity PLUMED derives from kT then breaks: OPES_METAD's default
            # BIASFACTOR=BARRIER/kT becomes infinite and aborts, and reweighting goes wrong
            # silently.
            plumed_force.setTemperature(self.temperature)
            logger.info("Plumed temperature: %s K", self.temperature)
            self.openmmobject.system.addForce(plumed_force)

        if restraints is not None:
            logger.debug("Adding restraints")
            self.openmmobject.add_bondrestraints(restraints=restraints)

        new_simulation = False
        if chkfile is not None:
            self._restart_from_file(chkfile, "Checkpoint file", openmm.app.Simulation.loadCheckpoint)
            new_simulation = True
        elif statefile is not None:
            self._restart_from_file(statefile, "State file", openmm.app.Simulation.loadState)
            new_simulation = True
        elif restart is True:
            logger.info("Restart true. Reusing already-defined simulation object")
        else:
            logger.info("Restart false and no chkfile/statefile set. This is a new simulation")
            self.simulation = self.openmmobject.create_simulation()
            new_simulation = True
            logger.info("Simulation created.")
        self._log_run_parameters(simulation_time, simulation_steps)

        if self.openmmobject.periodic is True:
            logger.debug("Checking Initial PBC vectors.")
            self.state = self._get_simulation_state()
            a, b, c = self.state.getPeriodicBoxVectors()
            logger.info("A:  %s\nB:  %s\nC:  %s", a, b, c)
            logger.info(f"Boxlength: {a[0].value_in_unit(openmm.unit.angstrom)} Angstrom")

        self._attach_reporters(
            continuing=restart is True or statefile is not None or chkfile is not None,
            extra_reporters=extra_reporters,
        )

        if pre_dynamics_hook is not None:
            logger.debug("Calling pre_dynamics_hook before dynamics")
            pre_dynamics_hook(self)

        boxvectors, mdtrajtopology, wrapping_atoms = self._prepare_wrapping()
        if new_simulation:
            self._write_first_frame()

        if self.theory_runtype == "QMMM":
            logger.info("QM/MM MD run beginning")
            if self._is_rpmd_simulation(self.simulation):
                self._finish_rpmd_run(simulation_steps, "QM/MM", module_init_time)
                return

            # Classical QM/MM uses a frozen-gradient CustomExternalForce updated before every step.
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                logger.debug("Step: %s", step)
                if step % self.traj_frequency == 0:
                    logger.debug("Step: %s", step)

                _, current_coords = self._current_step_coords(checkpoint, boxvectors, mdtrajtopology, wrapping_atoms)

                checkpoint = time.time()
                self.QM_MM_object.run(
                    current_coords=current_coords,
                    elems=self.fragment.elems,
                    grad=True,
                    exit_after_customexternalforce_update=True,
                    charge=self.charge,
                    mult=self.mult,
                )
                log_time_since(checkpoint, "QM/MM run")

                if step % self.restartfile_frequency == 0:
                    self.write_state_and_chk_files(step)

                # NOTE: Manual per-step info is not possible here because the MM-energy has not been
                # calculated yet when using the customexternalforceupdate option
                if step % self.traj_frequency == 0:
                    logger.info("Writing wrapped coords to trajfile: OpenMMMD_traj_wrapped.xyz (for debugging)")
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj_wrapped", writemode="a")
                self._write_special_atoms_frame(step, current_coords)

                # The QM_PC gradient (link-atom projected, from QM_MM object) is provided to OpenMM external force
                checkpoint = time.time()
                self.openmmobject.update_custom_external_force(
                    self.openmm_externalforceobject, self.QM_MM_object.QM_PC_gradient, self.simulation
                )
                log_time_since(checkpoint, "update custom external force")

                checkpoint = time.time()
                self.simulation.step(1)
                log_time_since(checkpoint, "openmmobject sim step")
                log_time_since(checkpoint_begin_step, "Total sim step")
        elif self.theory_runtype == "QM":
            logger.info("External QM with OpenMM option")
            if self._is_rpmd_simulation(self.simulation):
                self._finish_rpmd_run(simulation_steps, "External-QM", module_init_time)
                return

            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                logger.debug("Step: %s", step)
                if step % self.traj_frequency == 0:
                    logger.debug("Step: %s", step)

                current_state, current_coords = self._current_step_coords(
                    checkpoint, boxvectors, mdtrajtopology, wrapping_atoms
                )

                checkpoint = time.time()
                energy, gradient = self.qmtheory.run(
                    current_coords=current_coords,
                    elems=self.fragment.elems,
                    grad=True,
                    charge=self.charge,
                    mult=self.mult,
                )
                logger.info("Energy: %s", energy)
                log_time_since(checkpoint, "QM run")
                self.openmmobject.update_custom_external_force(
                    self.openmm_externalforceobject, gradient, self.simulation
                )

                # Calculate energy associated with external force so that we can subtract it later
                extforce_energy = 3 * np.mean(sum(gradient * current_coords * openmmqmmm.constants.ANG_TO_BOHR))
                logger.info("extforce_energy: %s", extforce_energy)

                if step % self.traj_frequency == 0:
                    print_current_step_info(step, current_state, self.openmmobject, qm_energy=energy)

                    if self.energy_file_option is not None:
                        with open(self.energy_file_option, "a") as f:
                            f.write(f"{energy}\n")

                    if self.trajectory_file_option == "XYZ":
                        write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", writemode="a")
                self._write_special_atoms_frame(step, current_coords)

                if step % self.restartfile_frequency == 0:
                    self.write_state_and_chk_files(step)

                checkpoint = time.time()
                self.simulation.step(1)
                log_time_since(checkpoint, "OpenMM sim step")
                log_time_since(checkpoint_begin_step, "Total sim step")
        elif self.theory_runtype == "MM":
            logger.info("OpenMM MM dynamics option chosen.")
            if self._is_rpmd_simulation(self.simulation):
                self._run_rpmd_mm(simulation_steps)
            else:
                self.simulation.step(simulation_steps)
        else:
            raise InputError(
                f"Error: Unrecognized Theory runtype ({self.theory_runtype}) for MD. This might mean that this theory "
                f"object is not yet supported for running MD. Exiting."
            )

        logger.info(small_header("OpenMM MD simulation finished!"))
        log_time_since(module_init_time, "OpenMM_MD run")

    def close(self) -> None:
        """Close engine-owned output streams and detach engine-owned reporters."""
        data_output = getattr(self, "dataoutputoption", stdout)
        if data_output != stdout and not data_output.closed:
            data_output.close()

        if hasattr(self, "_simulation_reporters"):
            self._remove_simulation_reporters()

        rpmd_reporters = getattr(self, "_rpmd_reporters", [])
        self._rpmd_reporters = []
        self._rpmd_reporter_owner = None
        rpmd_reporters.clear()

    def finalize_simulation(self) -> None:
        """Write the final structure and trajectory files and log the timing summary."""
        try:
            self._finalize_simulation()
        finally:
            self.close()

    def _finalize_simulation(self) -> None:
        """Implement finalization while the public wrapper guarantees resource cleanup."""
        logger.info("Finalizing simulation data")

        if self.datafilename is not None:
            self.dataoutputoption.close()

        self.state = self._get_simulation_state(
            getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforce_periodic_box
        )

        if self.openmmobject.periodic is True:
            logger.debug("Checking PBC vectors:")
            a, b, c = self.state.getPeriodicBoxVectors()
            logger.info("A:  %s", a)
            logger.info("B:  %s", b)
            logger.info("C:  %s", c)
            logger.info("a 0 %s", a[0])
            logger.debug("Updating PBC vectors in simulation.context, OpenMM system and OpenMM topology")
            self.simulation.context.setPeriodicBoxVectors(a, b, c)
            # System. Necessary
            self.openmmobject.system.setDefaultPeriodicBoxVectors(a, b, c)
            # Topology (for header in PDB-files). Necessary
            self.openmmobject.topology.setPeriodicBoxVectors(self.state.getPeriodicBoxVectors())

        pdb_filename = self.trajfilename + "_lastframe.pdb"
        logger.info("Writing final frame to disk as PDB-file: %s", pdb_filename)
        with open(pdb_filename, "w") as f:
            openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbfile.PDBFile.writeModel(
                self.openmmobject.topology, self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
            openmm.app.pdbfile.PDBFile.writeFooter(self.openmmobject.topology, f)
        logger.info(f"Trajectory : {self.trajfilename}.{self.trajectory_file_option}")
        pdbx_filename = self.trajfilename + "_lastframe.cif"
        logger.info("Writing final frame to disk as PDBx/mmCIF-file: %s", pdbx_filename)
        with open(pdbx_filename, "w") as f:
            openmm.app.pdbxfile.PDBxFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbxfile.PDBxFile.writeModel(
                self.openmmobject.topology, self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
        logger.info(f"Trajectory : {self.trajfilename}.{self.trajectory_file_option}")

        if self._is_rpmd_simulation(self.simulation):
            logger.info("Saving all RPMD copies to %s", RPMD_FINAL_RESTART_FILENAME)
            self._save_rpmd_restart(RPMD_FINAL_RESTART_FILENAME)
        else:
            # Can be used to restart using statefile option
            logger.info(
                "Saving a statefile and checkpointfile of the final frame of the simulation: "
                "OpenMM_MD_final_state.xml and OpenMM_MD_final_checkpoint.chk"
            )
            logger.info(
                "These files can be used to restart a simulation (statefile and chkfile keywords) using the same "
                "coordinates and velocities."
            )
            self.simulation.saveState("OpenMM_MD_final_state.xml")
            self.simulation.saveCheckpoint("OpenMM_MD_final_checkpoint.chk")

        newcoords = self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        logger.debug("Updating coordinates in fragment.")
        self.fragment.coords = newcoords
        # Updating positions array also in case we call run again
        self.positions = newcoords


def openmm_box_equilibration(
    *,
    fragment: Fragment | None = None,
    theory: OpenMMTheory | None = None,
    datafilename: str | os.PathLike[str] = "nptsim.csv",
    numsteps_per_npt: int = 10000,
    max_npt_cycles: int = 10,
    pressure: float = 1,
    volume_threshold: float = 1.3,
    density_threshold: float = 0.005,
    temperature: float = 300,
    timestep: float = 0.001,
    traj_frequency: int = 100,
    trajfilename: str = "equilibration_NPT",
    trajectory_file_option: str = "DCD",
    coupling_frequency: float = 1,
    enforce_periodic_box: bool = True,
    use_mdtraj: bool = True,
    dummyatomrestraint: bool = False,
    solute_indices: Sequence[int] | None = None,
    barostat_frequency: int = 25,
) -> tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3]:
    """Run NPT simulations in cycles until box volume and density stop changing."""
    # Captured before any local is bound, so this is the caller's arguments and nothing
    # else. The integrator and barostat are fixed by what this function does: NPT cycles.
    engine_kwargs = engine_kwargs_from(locals(), integrator="LangevinMiddleIntegrator", barostat="MonteCarloBarostat")

    logger.info(main_header("Periodic Box Size Equilibration"))
    module_init_time = time.time()

    if fragment is None or theory is None:
        raise InputError("Fragment and theory required.")

    if numsteps_per_npt < traj_frequency:
        raise InputError(
            "Parameter 'numpsteps_per_NPT' must be greater than 'traj_frequency', otherwise no data will be written "
            "during the equilibration!"
        )

    numpoints_for_convergence_check = numsteps_per_npt // traj_frequency

    logger.info(small_header("Equilibration Parameters"))
    logger.info("Steps per NPT cycle: %s", numsteps_per_npt)
    logger.info("Max NPT cycles: %s", max_npt_cycles)
    logger.info(f"Timestep: {timestep * 1000} fs")
    logger.info("Density threshold: %s", density_threshold)
    logger.info("Volume threshold: %s", volume_threshold)
    logger.info("Intermediate MD data file: %s", datafilename)
    logger.info("Number of datapoints used for convergence check in each cycle: %s", numpoints_for_convergence_check)

    if len(theory.user_frozen_atoms) > 0:
        logger.info("Frozen_atoms: %s", theory.user_frozen_atoms)
        logger.warning(
            "OpenMM object has frozen atoms defined. This is known to cause strange issues for NPT simulations."
        )
        logger.warning("Check the results carefully!")

    steps = 0
    volume_std = 10
    density_std = 1

    md = MolecularDynamicsEngine(**engine_kwargs)
    try:
        restart = False
        for i in range(max_npt_cycles):
            logger.info("%s", "-" * 100)
            logger.debug("Starting NPT cycle %s with %s MD steps", i, numsteps_per_npt)
            logger.info(
                "Simulation data (timestep, energy, temperature, volume,density etc.) is also written to %s",
                datafilename,
            )
            md.run(numsteps_per_npt, restart=restart)
            restart = True

            steps += numsteps_per_npt

            NPTresults = read_npt_statefile(datafilename)
            volume = NPTresults["volume"][-numpoints_for_convergence_check:]
            density = NPTresults["density"][-numpoints_for_convergence_check:]
            logger.info("Total number of volume datapoints available: %s", len(NPTresults["volume"]))
            logger.info("Total number of density datapoints available: %s", len(NPTresults["density"]))
            logger.info(
                "Number of datapoints (last) used for convergence check in each cycle: %s",
                numpoints_for_convergence_check,
            )
            volume_std = np.std(volume)
            density_std = np.std(density)

            logger.info(small_header("Equilibration Status"))
            logger.info("Total steps taken: %s", steps)
            logger.info(f"Total simulation time: {timestep * steps} ps")
            logger.info("Current Volume: %s", volume[-1])
            logger.info(f"Current Density: {density[-1]}")
            logger.info(f"\nCurrent Volume SD: {volume_std}   (threshold: {volume_threshold})")
            logger.info(f"Current Density SD: {density_std} (threshold: {density_threshold})")

            if volume_std < volume_threshold and density_std < density_threshold:
                logger.info(f"Equilibration of periodic box finished after {steps} and {timestep * steps} ps !\n")
                break

            if i == max_npt_cycles - 1:
                logger.warning(
                    f"Max NPT cycles reached ({max_npt_cycles}). Total steps taken: {steps} and "
                    f"{timestep * steps} ps !\n"
                )
                logger.warning("The NPT simulation may not be properly converged")
                break

        md.finalize_simulation()

        logger.info(f"Final PDB file: {trajfilename}.pdb")
        logger.info(f"NPT trajectory: {trajfilename}.{trajectory_file_option.lower()}")

        if use_mdtraj is True:
            logger.debug("Trying to load mdtraj for reimaging trajectory")
            try:
                logger.info("Imaging trajectory")
                mdtraj_image_trajectory(f"{trajfilename}.dcd", f"{trajfilename}_lastframe.pdb")
            except ImportError:
                logger.warning("MDTraj could not be imported; skipping trajectory reimaging")
            except ValueError as e:
                logger.warning("MDTraj reimaging failed; skipping it: %s", e)

        log_time_since(module_init_time, "OpenMM_box_equilibration")
        return md.state.getPeriodicBoxVectors()
    finally:
        md.close()


def print_current_step_info(
    step: int, state: openmm.State, openmmobject: OpenMMTheory, qm_energy: float | None = None
) -> None:
    kinetic_energy = state.getKineticEnergy()
    kinetic_energy_eh = (
        kinetic_energy.value_in_unit(openmm.unit.kilojoules_per_mole) / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
    )

    # Potential energy from the theory level instead
    if qm_energy is not None:
        dummy_warning = "(correct)"
        pot_energy = qm_energy
    else:
        dummy_warning = "(dummy)"
        pot_energy = (
            state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)
            / openmmqmmm.constants.HARTREE_TO_KJ_PER_MOL
        )

    temp = (2 * kinetic_energy / (openmmobject.dof * openmm.unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(
        openmm.unit.kelvin
    )

    logger.info("%s", "=" * 50)
    logger.info(f"SIMULATION STATUS (STEP {step})")
    logger.info("%s", "_" * 50)
    logger.info(f"Time: {state.getTime()}")
    logger.info(f"Potential energy {dummy_warning}: {pot_energy} Eh")
    logger.info(f"Kinetic energy: {kinetic_energy_eh} Eh")
    logger.info(f"Temperature: {temp}")
    logger.info("%s", "=" * 50)


def gentle_warmup_md(
    *,
    theory: OpenMMTheory | None = None,
    fragment: Fragment | None = None,
    time_steps: Sequence[float] | None = None,
    steps: Sequence[int] | None = None,
    temperatures: Sequence[float] | None = None,
    check_gradient_first: bool = True,
    gradient_threshold: float = 100,
    use_mdtraj: bool = True,
    trajfilename: str = "warmup_MD",
    initial_opt: bool = True,
    traj_frequencies: Sequence[int] | None = None,
    maxoptsteps: int = 10,
    coupling_frequency: float = 1,
) -> None:
    """Gradually warm up an MD system in stages (short timesteps first, then longer)."""
    if traj_frequencies is None:
        traj_frequencies = [1, 1, 100]
    if temperatures is None:
        temperatures = [1, 10, 300]
    if steps is None:
        steps = [10, 50, 10000]
    if time_steps is None:
        time_steps = [0.0005, 0.001, 0.004]
    logger.info(main_header("Gentle_warm_up_MD"))
    module_init_time = time.time()
    logger.info("Trajectory filename: %s", trajfilename)
    if theory is None or fragment is None:
        raise InputError("Gentle_warm_up_MD requires theory (OpenMM object) and fragment")

    if len(time_steps) != len(steps) or len(time_steps) != len(temperatures):
        raise InputError("Error: Lists time_steps, steps and temperatures all need to be the same length. Exiting")

    if check_gradient_first is True:
        logger.info("check_gradient_first is True")
        logger.info("Will run singlepoint gradient calculation to check for large forces")
        theory.force_run = True
        SP_result = single_point(theory=theory, fragment=fragment, grad=True)
        badindices = check_gradient_for_bad_atoms(
            fragment=fragment, gradient=SP_result.gradient, threshold=gradient_threshold
        )
        if len(badindices) > 0:
            logger.info(f"\nNumber of atoms with large forces: {len(badindices)}")
            logger.info("Suggests a bad system geometry or that atoms need constraints (might be present already)")
            logger.info("Gentle_warm_up_MD will go on")

    if initial_opt is True:
        logger.info(f"\ninitial_opt is True (default). Will attempt initial {maxoptsteps}-step minimization first")
        logger.info("If this step runs forever something is wrong. Select initial_opt=False to avoid in this case")
        try:
            openmm_minimize(fragment=fragment, theory=theory, maxiter=maxoptsteps, tolerance=1)
            logger.info("Minimization successful")
        except Exception as e:  # noqa: BLE001 - MD warm-up continues even if pre-minimization fails
            logger.info("Problem minimizing system")
            logger.error("message: %s", e)
            logger.debug("Will go on to do MD")

    logger.info(f"\n{len(steps)} MD-runs have been defined")
    for num, (ts, step, temp) in enumerate(zip(time_steps, steps, temperatures, strict=False)):
        logger.info(f"MD-step {num} Number of simulation steps: {step} with timestep: {ts} and temperature: {temp} K")

    for num, (ts, step, temp, traj_frequency) in enumerate(
        zip(time_steps, steps, temperatures, traj_frequencies, strict=False)
    ):
        # Name of PDB and DCD filename: i.e. warmup_MD_cycle1.pdb and warmup_MD_cycle1.dcd
        MDcyclename = trajfilename + f"_cycle{num}"
        logger.info(
            f"\n\nNow running MD-run {num}. Number of steps: {step} with timestep:{ts} and temperature: {temp} K"
        )
        logger.info(f"Will write trajectory to file: {MDcyclename}.dcd")
        openmm_md(
            fragment=fragment,
            theory=theory,
            timestep=ts,
            simulation_steps=step,
            traj_frequency=traj_frequency,
            temperature=temp,
            integrator="LangevinMiddleIntegrator",
            coupling_frequency=coupling_frequency,
            trajfilename=MDcyclename,
            trajectory_file_option="DCD",
        )

        if use_mdtraj is True:
            logger.debug("Trying to load mdtraj for basic analysis of trajectory")
            try:
                logger.info("Imaging trajectory")
                mdtraj_image_trajectory(f"{MDcyclename}.dcd", f"{MDcyclename}_lastframe.pdb")
                logger.debug("\nRunning RMS Fluctuation analysis on trajectory")
                mdtraj_rmsf(
                    f"{MDcyclename}.dcd",
                    f"{MDcyclename}_lastframe.pdb",
                    print_largest_values=True,
                    threshold=0.005,
                    largest_values=10,
                )
            except ImportError:
                logger.warning("MDTraj could not be imported; skipping trajectory analysis")
            except ValueError as e:
                logger.warning("MDTraj trajectory analysis failed; skipping it: %s", e)

    logger.info("Gentle_warm_up_MD finished successfully!")
    log_time_since(module_init_time, "Gentle_warm_up_MD")


def diff_wrap_box_coords(
    coords_nm: npt.ArrayLike,
    boxvectors: npt.ArrayLike,
    mdtrajtopology: Any,
    anchoratoms: Sequence[int],
) -> npt.NDArray[np.float64]:
    import mdtraj

    traj = mdtraj.Trajectory(coords_nm, mdtrajtopology)
    traj.unitcell_vectors = np.array(boxvectors).reshape(1, 3, 3)
    # Anchoratoms (usually QM-region or similar)
    anchors = [{traj.topology.atom(i) for i in anchoratoms}]
    imaged = traj.image_molecules(anchor_molecules=anchors)
    return imaged._xyz[0] * 10.0
