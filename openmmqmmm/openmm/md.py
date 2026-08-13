"""Molecular dynamics: the simulation engine, MD drivers and box equilibration."""

import contextlib
import inspect
import logging
import os
import time
from sys import stdout

import numpy as np
import openmm
import openmm.app
import openmm.unit

import openmmqmmm
from openmmqmmm.coords import (
    change_origin_to_centroid,
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
from openmmqmmm.openmm.systemsetup import openmm_minimize
from openmmqmmm.openmm.theory import ForceReporter, OpenMMTheory
from openmmqmmm.singlepoint import single_point
from openmmqmmm.utils import (
    create_conn_dict,
    log_time_since,
    main_header,
    small_header,
)

logger = logging.getLogger(__name__)


def engine_kwargs_from(caller_locals, **overrides):
    """Pick the MolecularDynamicsEngine arguments out of a wrapper's own arguments.

    The MD entry points restate the engine's ~45 parameters in their own signatures and
    used to restate them a second time in the constructor call. Keeping two hand-written
    lists in step failed exactly as expected: for a release every one of them passed
    ``enforcePeriodicBox`` to a class whose parameter is ``enforce_periodic_box``, so
    every call raised TypeError.

    Call this as the first statement of a wrapper, with ``locals()``, so that what it sees
    is the bound arguments and nothing else. Parameters that belong to the wrapper rather
    than the engine are left behind; ``overrides`` sets values the wrapper fixes itself.
    """
    engine_parameters = set(inspect.signature(MolecularDynamicsEngine.__init__).parameters) - {"self"}
    return {name: value for name, value in caller_locals.items() if name in engine_parameters} | overrides


def read_npt_statefile(npt_output):
    import csv
    from collections import defaultdict

    # Read in CSV file of last NPT simulation and store in lists
    columns = defaultdict(list)

    with open(npt_output) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                columns[k].append(v)
    # Extract step number, volume and density and cast as floats
    steps = np.array(columns['#"Step"'])
    volume = np.array(columns["Box Volume (nm^3)"]).astype(float)
    density = np.array(columns["Density (g/mL)"]).astype(float)

    return {"steps": steps, "volume": volume, "density": density}


def openmm_md(
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    restartfile_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
    barostat=None,
    pressure=1,
    trajectory_file_option="DCD",
    trajfilename="trajectory",
    specialtraj_frequency=1000,
    specialatoms=None,
    energy_file_option=None,
    force_file_option=None,
    atomic_units_force_reporter=False,
    coupling_frequency=1,
    charge=None,
    mult=None,
    hydrogenmass=1.5,
    force_periodic=None,
    periodic_cell_dimensions=None,
    anderson_thermostat=False,
    platform="CPU",
    constraints=None,
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
    centerforce_constant=1.0,
    centerforce_distance=10.0,
    centerforce_center=None,
    barostat_frequency=25,
    chkfile=None,
    statefile=None,
) -> None:
    """Run molecular dynamics of a fragment with OpenMM (also drives QM/MM MD).

    Simulation length is set via simulation_steps or simulation_time (ps);
    thermostat/barostat, trajectory format and restraints are configurable.
    """
    engine_kwargs = engine_kwargs_from(locals())

    logger.info(main_header("OpenMM MD wrapper function"))
    md = MolecularDynamicsEngine(**engine_kwargs)
    if simulation_steps is not None:
        md.run(simulation_steps=simulation_steps)
    elif simulation_time is not None:
        md.run(simulation_time=simulation_time)
    else:
        raise InputError("Either simulation_steps or simulation_time need to be defined (not both).")

    # Now calling finalize_simulation: writing final files etc.
    md.finalize_simulation()

    # TODO: Return a Results object here?


class MolecularDynamicsEngine:
    """Driver for OpenMM molecular-dynamics simulations (also used for QM/MM MD).

    Usually created via the openmm_md / openmm_md_plumed functions.
    """

    def __init__(
        self,
        fragment=None,
        theory=None,
        charge=None,
        mult=None,
        timestep=0.001,
        traj_frequency=1000,
        restartfile_frequency=1000,
        temperature=300,
        integrator="LangevinMiddleIntegrator",
        barostat=None,
        pressure=1,
        trajectory_file_option="DCD",
        trajfilename="trajectory",
        specialtraj_frequency=1000,
        specialatoms=None,
        energy_file_option=None,
        force_file_option=None,
        atomic_units_force_reporter=False,
        coupling_frequency=1,
        platform="CPU",
        anderson_thermostat=False,
        hydrogenmass=1.5,
        constraints=None,
        restraints=None,
        force_periodic=False,
        periodic_cell_dimensions=None,
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
        centerforce_constant=1.0,
        centerforce_distance=10.0,
        centerforce_center=None,
        barostat_frequency=25,
        chkfile=None,
        statefile=None,
    ):
        module_init_time = time.time()

        logger.info(main_header("OpenMM Molecular Dynamics Initialization"))

        if fragment is None:
            raise InputError("No fragment object. Exiting.")
        self.fragment = fragment

        # Check charge/mult
        self.charge, self.mult = check_charge_mult(
            charge, mult, theory.theorytype, fragment, "OpenMM_MD", theory=theory
        )

        # External QM option off by default
        self.externalqm = False

        # Trajectory filename. Used for trajs in DCD, PDB etc. format, also single PDB snapshots
        self.trajfilename = trajfilename

        # Specialatoms and specialtraj_frequency for special printing
        self.specialatoms = specialatoms
        self.specialtraj_frequency = specialtraj_frequency

        # Delete previous special and wrapping trajectory file
        if os.path.exists("wrapped_special_traj.xyz"):
            os.remove("wrapped_special_traj.xyz")
        if os.path.exists("OpenMMMD_traj_wrapped.xyz"):
            os.remove("OpenMMMD_traj_wrapped.xyz")

        # Distinguish between OpenMM theory QM/MM theory or QM theory
        self.dummy_mm = dummy_mm

        # Printlevel

        # Determine centroid of original fragment coordinates
        self.centroid_system = get_centroid(fragment.coords)

        # Theory_runtype
        self.theory_runtype = None

        self.openmmobject = None
        self.QM_MM_object = None
        logger.info("Analyzing theory input to OpenMM_MDclass")
        if isinstance(theory, OpenMMTheory):
            logger.info("This is an OpenMMTheory object")
            self.openmmobject = theory
            self.QM_MM_object = None
            if self.dummy_mm is True:
                self.theory_runtype = "dummy_MM"
            else:
                self.theory_runtype = "MM"
        # Case: QM/MM theory with OpenMM mm_theory
        elif isinstance(theory, openmmqmmm.QMMMTheory):
            logger.info("This is an QMMMTheory object")
            self.QM_MM_object = theory
            self.openmmobject = theory.mm_theory
            self.theory_runtype = "QMMM"

            # Making sure QM/MM object will exit before calculating MM part
            self.QM_MM_object.exit_after_customexternalforce_update = True
            logger.info("Turning on externalforce option.")
            self.QM_MM_object.openmm_externalforce = True
            # NOTE: Now creating externalforceobject as part of this MD object instead (previously QM/MM object)
            self.openmm_externalforceobject = self.openmmobject.add_custom_external_force()
            # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
            # (it would calculate OpenMM energy twice) so turning on in case forgotten
        # Case: OpenMM with external QM
        else:
            # NOTE: Recognize QM theories here ??
            logger.info("Unrecognized theory.")
            logger.info("Will assume to be QM theory and will continue")
            logger.info("QM-program forces will be added as a custom external force to OpenMM")
            self.externalqm = True
            logger.info("Now creating OpenMMTheory object")
            logger.info("OpenMM platform: %s", platform)
            # Creating dummy OpenMMTheory (basic topology, particle masses, no forces except CMMRemoval)
            self.openmmobject = OpenMMTheory(
                fragment=fragment,
                dummysystem=True,
                platform=platform,
                hydrogenmass=hydrogenmass,
                constraints=constraints,
                periodic=force_periodic,
                periodic_cell_dimensions=periodic_cell_dimensions,
            )  # NOTE: might add more options here
            logger.info("Creating new OpenMM custom external force for external QM theory.")
            self.openmm_externalforceobject = self.openmmobject.add_custom_external_force()
            self.QM_MM_object = None
            self.qmtheory = theory
            self.theory_runtype = "QM"

        # Basic restraints (bond,angle,torsion)
        if restraints is not None:
            logger.info("Restraints defined. Will add to OpenMMTheory object")
            logger.info("All restraints: %s", restraints)
            for restraint in restraints:
                logger.info("Restraint: %s", restraint)
                if len(restraint) == 4:
                    logger.info("Bond restraint assumed")
                    logger.info(
                        f"Atoms: {restraint[0]} {restraint[1]} Value: {restraint[2]} Force-constant: {restraint[3]} "
                        f"kcal/mol/Angstrom^2"
                    )
                    self.openmmobject.add_custom_bond_force(restraint[0], restraint[1], restraint[2], restraint[3])
                elif len(restraint) == 5:
                    logger.info("Angle restraint assumed")
                    logger.info(
                        f"Atoms: {restraint[0]} {restraint[1]} {restraint[2]} Value: {restraint[3]} Force-constant: "
                        f"{restraint[4]} kcal/mol/radian^2"
                    )
                    self.openmmobject.add_custom_angle_force(
                        restraint[0], restraint[1], restraint[2], restraint[3], restraint[4]
                    )
                elif len(restraint) == 6:
                    logger.info("Torsion restraint assumed")
                    logger.info(
                        f"Atoms: {restraint[0]} {restraint[1]} {restraint[2]} {restraint[3]} Value: {restraint[4]} "
                        f"Force-constant: {restraint[5]} kcal/mol/radian^2"
                    )
                    self.openmmobject.add_custom_torsion_force(
                        restraint[0], restraint[1], restraint[2], restraint[3], restraint[4], restraint[5]
                    )

        # RESTART options
        self.chkfile = chkfile
        self.statefile = statefile

        # Assigning some basic variables
        self.temperature = temperature
        self.pressure = pressure
        self.integrator = integrator
        self.coupling_frequency = coupling_frequency
        self.timestep = timestep
        self.traj_frequency = int(traj_frequency)
        self.restartfile_frequency = restartfile_frequency
        self.barostat_frequency = barostat_frequency
        self.trajectory_file_option = trajectory_file_option
        self.force_file_option = force_file_option  # Gradients/forces as a file
        self.energy_file_option = energy_file_option  # Energies as a file
        self.atomic_units_force_reporter = atomic_units_force_reporter  # Forces in atomic units
        # PERIODIC or not
        if self.openmmobject.periodic is True:
            # Generally we want True except sometimes we do our own wrapping
            self.enforce_periodic_box = enforce_periodic_box
        else:
            logger.info("System is non-periodic. Setting enforcePeriodicBox to False")
            # Non-periodic. Setting enforcePeriodicBox to False (otherwise nonsense)
            self.enforce_periodic_box = False

        # Optional wrapping_atoms (anchoratoms)
        self.special_wrapping = special_wrapping
        self.special_wrapping_updatepos = (
            special_wrapping_updatepos  # Testing: update positions in simulation object after wrapping
        )
        self.wrapping_atoms = wrapping_atoms

        logger.info(small_header("MD system parameters"))
        logger.info(f"Temperature: {self.temperature} K")
        logger.info("OpenMM autoconstraints: %s", self.openmmobject.autoconstraints)
        logger.info("OpenMM hydrogenmass: %s", self.openmmobject.hydrogenmass)
        logger.info("OpenMM rigidwater constraints: %s", self.openmmobject.rigidwater)
        logger.info("User Constraints: %s", self.openmmobject.user_constraints)
        logger.info("User Restraints: %s", self.openmmobject.user_restraints)
        logger.info("Number of atoms: %s", self.fragment.numatoms)
        logger.info("Number of frozen atoms: %s", len(self.openmmobject.user_frozen_atoms))
        if len(self.openmmobject.user_frozen_atoms) < 50:
            logger.info("Frozen atoms %s", self.openmmobject.user_frozen_atoms)
        logger.info("Integrator: %s", self.integrator)
        logger.info(f"Timestep: {self.timestep} ps")
        logger.info("Anderon Thermostat: %s", anderson_thermostat)
        logger.info(f"coupling_frequency: {self.coupling_frequency} ps^-1 (for Nose-Hoover and Langevin integrators)")
        logger.info("Barostat: %s", barostat)

        logger.info("")
        logger.info("Will write trajectory in format: %s", self.trajectory_file_option)
        logger.info("Trajectory write frequency: %s", self.traj_frequency)
        logger.info("enforcePeriodicBox: %s", self.enforce_periodic_box)
        logger.info("special_wrapping: %s", self.special_wrapping)
        logger.info("special_wrapping_updatepos: %s", special_wrapping_updatepos)
        logger.info("wrapping_atoms: %s", self.wrapping_atoms)
        logger.info("")

        if self.openmmobject.autoconstraints is None:
            logger.warning("""
                WARNING: Autoconstraints have not been set in OpenMMTheory object definition. This means that by
                         default no bonds are constrained in the MD simulation. This usually requires a small
                         timestep: 0.5 fs or so.
                         autoconstraints='HBonds' is recommended for 2 fs timesteps with
                         LangevinIntegrator and 4fs with LangevinMiddleIntegrator).
                         autoconstraints='AllBonds' or autoconstraints='HAngles' allows even
                         larger timesteps to be used.
                         See : https://github.com/openmm/openmm/pull/2754 and https://github.com/openmm/openmm/issues/2520
                         for recommended simulation settings in OpenMM.
                         """)
            logger.info("Will continue...")
        if (self.openmmobject.rigidwater is True and len(self.openmmobject.user_frozen_atoms) != 0) or (
            self.openmmobject.autoconstraints is not None and len(self.openmmobject.user_frozen_atoms) != 0
        ):
            logger.warning(
                "WARNING: Frozen_atoms options selected but there are general constraints defined in "
                "the OpenMM object (either rigidwater=True or autoconstraints is not None)"
                "\nOpenMM will crash if constraints and frozen atoms involve the same atoms"
            )
        logger.info("")

        logger.info("Defining atom positions from fragment")
        # Note: using self.positions as we may add dummy atoms (e.g. dummyatomrestraint below)
        self.positions = self.fragment.coords

        # Dummy-atom restraint to deal with NPT simulations that contain constraints/restraints/frozen_atoms
        self.dummyatomrestraint = dummyatomrestraint
        if self.dummyatomrestraint is True:
            if solute_indices is None:
                raise InputError("Dummyatomrestraint requires solute_indices to be set")
            logger.warning(
                "Warning: Using dummyatomrestraints. This means that we will add a dummy atom to topology and OpenMM "
                "coordinates"
            )
            logger.info("We do not add the dummy atom to the fragment")
            logger.info(
                "Affects visualization of trajectory (make sure to use PDB-file that contains the dummy-atom, printed "
                "in the end)"
            )
            # Should be centroid of solute or something rather
            solute_coords = np.take(self.fragment.coords, solute_indices, axis=0)
            dummypos = get_centroid(solute_coords)
            logger.info("Dummy atom will be added to position: %s", dummypos)
            # Adding dummy-atom coordinates to self.positions
            self.positions = np.append(self.positions, [dummypos], axis=0)
            logger.info("len self.pos %s", len(self.positions))
            logger.info("len self.fragment.coords %s", len(self.fragment.coords))

            # Restraining solute atoms to dummy-atom
            self.openmmobject.add_dummy_atom_to_restrain_solute(atomindices=solute_indices)

        # TRANSLATE solute: #https://github.com/openmm/openmm/issues/1854
        # Translate solute to geometric center on origin
        if center_on_atoms is not None:
            solute_coords = np.take(self.fragment.coords, solute_indices, axis=0)
            changed_origin_coords = change_origin_to_centroid(self.fragment.coords, subsetcoords=solute_coords)
            logger.info("changed_origin_coords %s", changed_origin_coords)

        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        # Set up system with chosen barostat, thermostat, integrator
        if barostat is not None:
            logger.info("Checking for barostat")
            if "MonteCarloBarostat" not in forceclassnames:
                montecarlobarostat = openmm.MonteCarloBarostat(
                    self.pressure * openmm.unit.bar, self.temperature * openmm.unit.kelvin
                )
                # Setting barostat frequency to chosen value or default (25)
                montecarlobarostat.setFrequency(self.barostat_frequency)
                self.openmmobject.system.addForce(montecarlobarostat)
                logger.info("Barostat added")
            else:
                logger.info("Barostat already present. Skipping.")

            self.integrator = "LangevinMiddleIntegrator"
            logger.info("Barostat requires using integrator: %s", self.integrator)
            self.openmmobject.set_simulation_parameters(
                timestep=self.timestep,
                temperature=self.temperature,
                integrator=self.integrator,
                coupling_frequency=self.coupling_frequency,
            )
        elif anderson_thermostat is True:
            logger.info("Anderson thermostat is on.")
            if "AndersenThermostat" not in forceclassnames:
                self.openmmobject.system.addForce(
                    openmm.AndersenThermostat(self.temperature * openmm.unit.kelvin, 1 / openmm.unit.picosecond)
                )
            self.integrator = "VerletIntegrator"
            logger.info("Now using integrator: %s", integrator)
            self.openmmobject.set_simulation_parameters(
                timestep=self.timestep,
                temperature=self.temperature,
                integrator=self.integrator,
                coupling_frequency=self.coupling_frequency,
            )
        else:
            # Deleting barostat and Andersen thermostat if present from previous sims
            for i, forcename in enumerate(forceclassnames):
                if forcename in {"MonteCarloBarostat", "AndersenThermostat"}:
                    logger.info("Removing old force: %s", forcename)
                    self.openmmobject.system.removeForce(i)

            # Regular thermostat or integrator without barostat
            # Integrators: LangevinIntegrator, LangevinMiddleIntegrator, NoseHooverIntegrator, VerletIntegrator,
            # BrownianIntegrator, VariableLangevinIntegrator, VariableVerletIntegrator
            self.openmmobject.set_simulation_parameters(
                timestep=self.timestep,
                temperature=self.temperature,
                integrator=self.integrator,
                coupling_frequency=self.coupling_frequency,
            )

        if barostat is not None:
            self.volume = self.density = True
        else:
            self.volume = self.density = False

        # If statedatareporter filename set:
        self.datafilename = datafilename
        if self.datafilename is not None:
            # Remove old file
            # Added because of problems (19 May 2023 by CVS) in read NPT data file (OpenMM box equilibration) as header
            # is printed each time
            # Now removing file before starting. Possibly better to put this elsewhere as we may sometimes
            # want to keep running simulation while appending to datafile
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.datafilename)

            # Now doing open file object in append mode instead of just filename.
            # Just filename does not play nice when running simulation step by step
            # Future OpenMM update may do this automatically?
            self.dataoutputoption = open(self.datafilename, "a")  # noqa: SIM115 - handed to OpenMM reporter
            logger.info("Will write data to file: %s", self.datafilename)
        # otherwise stdout:
        else:
            self.dataoutputoption = stdout

        # QM/MM MD
        # if self.QM_MM_object is not None:
        #    #True sometimes means we end up with solute in corner of box (wrong for nonPBC QM code)
        #
        #    #Making sure QM/MM object will exit before calculating MM part
        #
        #    # OpenMM_MD with QM/MM object does not make sense without openmm_externalforce
        #    # (it would calculate OpenMM energy twice) so turning on in case forgotten
        #    if self.QM_MM_object.openmm_externalforce is False:
        #        #NOTE: Now creating externalforceobject as part of this MD object instead (previously QM/MM object)
        # CENTER COORDINATES HERE on SOLUTE HERE ??
        # NOTE: Deprecated most likely
        # if centercoordinates is True:
        #    # Solute atoms assumed to be QM-region

        # Adding (flat-bottom) center force acting on solute
        if add_centerforce is True:
            logger.info("Centerforce option active")
            if centerforce_atoms is None:
                logger.info("centerforce_atoms unset. Trying to use QM atoms: %s", self.QM_MM_object.qmatoms)
                centerforce_atoms = self.QM_MM_object.qmatoms
            if centerforce_center is None:
                logger.info("No center coordinates set. Using geometric center of whole fragment.")
                # Get geometric center of system (Angstrom)
                centerforce_center = self.fragment.get_coordinate_center()
                logger.info("centerforce_center: %s", centerforce_center)
            # An alternative is add_flatbottom_centerforce(mol_a_indices=centerforce_atoms,
            # mol_b_indices=rest_system, ...), but it runs into PBC wrapping issues.
            self.openmmobject.add_centerforce(
                center_coords=centerforce_center,
                atomindices=centerforce_atoms,
                forceconstant=centerforce_constant,
                distance=centerforce_distance,
            )

        # After adding possible QM/MM force, possible Plumed force, possible center force
        # Let's list all OpenMM object system forces for sanity
        logger.info("enforcePeriodicBox: %s", self.enforce_periodic_box)
        logger.info("OpenMM Forces defined: %s", self.openmmobject.system.getForces())

        log_time_since(module_init_time, "OpenMM_MD setup")

    # Set sim reporters. Needs to be done after simulation is created and not modified anymore
    def set_sim_reporters(self, simulation, restart=False):
        # CheckpointReporter
        """Attach the trajectory, state-data and checkpoint reporters to the simulation."""
        logger.info("Creating CheckpointReporter that will write a restartable checkpointfile every X steps")
        checkpointfilename = "OpenMM_MD.chk"
        simulation.reporters.append(openmm.app.CheckpointReporter(checkpointfilename, self.traj_frequency * 1))
        # StateDataReporter
        logger.info("Creating StateDataReporter that will write to stdout")
        statedatareporter_stdout = openmm.app.StateDataReporter(
            stdout,
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
        simulation.reporters.append(statedatareporter_stdout)
        # Another reporter for writing to file
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
            simulation.reporters.append(statedatareporter_file)
            self.dataoutputoption = open(self.datafilename, "a")  # noqa: SIM115 - handed to OpenMM reporter

        # TODO: See if this can be made to work for simulations with step-by-step
        if self.trajectory_file_option == "PDB":
            simulation.reporters.append(
                openmm.app.PDBReporter(
                    self.trajfilename + ".pdb", self.traj_frequency, enforcePeriodicBox=self.enforce_periodic_box
                )
            )
        elif self.trajectory_file_option == "DCD":
            # Note: using append keyword here if restarting
            # Check first if file exists for restart (OpenMM errors otherwise)
            if restart is True and os.path.isfile(f"{self.trajfilename}.dcd") is False:
                logger.warning("Restart option was active but trajectory file not existing. Will create new file")
                restart = False

            simulation.reporters.append(
                openmm.app.DCDReporter(
                    self.trajfilename + ".dcd",
                    self.traj_frequency,
                    append=restart,
                    enforcePeriodicBox=self.enforce_periodic_box,
                )
            )
            logger.info("DCDReporter added")
        elif self.trajectory_file_option == "NetCDFReporter":
            logger.info("NetCDFReporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = mdtraj_load()
            simulation.reporters.append(mdtraj.reporters.NetCDFReporter(self.trajfilename + ".nc", self.traj_frequency))
        elif self.trajectory_file_option == "HDF5Reporter":
            logger.info("HDF5Reporter traj format selected. This requires mdtraj. Importing.")
            mdtraj = mdtraj_load()
            simulation.reporters.append(
                mdtraj.reporters.HDF5Reporter(
                    self.trajfilename + ".lh5", self.traj_frequency, enforcePeriodicBox=self.enforce_periodic_box
                )
            )
        elif self.trajectory_file_option == "XYZ":
            logger.info("XYZ trajectory format selected (not available for classical MD). Warning: not very fast")
            logger.info("Deleting possible old trajectory-file (OpenMMMD_traj.xyz)")
            with contextlib.suppress(OSError):
                os.remove("OpenMMMD_traj.xyz")
            # Done manually by write_xyzfile

        if self.force_file_option is not None:
            logger.info("ForceReporter traj format selected.")
            simulation.reporters.append(
                ForceReporter(
                    self.trajfilename + "_force.txt", self.traj_frequency, atomic_units=self.atomic_units_force_reporter
                )
            )
        if self.energy_file_option is not None:
            logger.info("Energyfile  selected.")
            with contextlib.suppress(OSError):
                os.remove(self.energy_file_option)
        logger.debug("Simulation reporters: %s", simulation.reporters)

    def write_state_and_chk_files(self, step):
        # Saving state and chkfile to disk
        """Write the OpenMM state (XML) and checkpoint files so a run can be restarted."""
        logger.info(
            f"Step {step}. Saving a statefile and checkpointfile : OpenMM_MD_state.xml and OpenMM_MD_checkpoint.chk"
        )
        logger.info(
            "Can be used to restart a simulation (statefile and chkfile keywords) using the same coordinates and "
            "velocities."
        )
        self.simulation.saveState("OpenMM_MD_state.xml")
        self.simulation.saveCheckpoint("OpenMM_MD_checkpoint.chk")

    # Simulation loop.
    def run(
        self,
        simulation_steps=None,
        simulation_time=None,
        plumedinput=None,
        restraints=None,
        restart=False,
        chkfile=None,
        statefile=None,
    ):
        """Run the molecular dynamics simulation.

        Args:
            simulation_steps: number of steps to run; overrides simulation_time.
            simulation_time: simulation length in ps, converted using the timestep.
            plumedinput: Plumed input as a string, defining the bias to apply.
            restraints: bond restraints to add to the system before running.
            restart: reuse the already-defined simulation object and append to the
                existing reporter files instead of starting fresh.
            chkfile: OpenMM checkpoint file to restore positions and velocities from.
            statefile: OpenMM state XML file to restore positions and velocities from.
                Used when chkfile is not given.

        Returns:
            The final Results object for the trajectory.
        """
        module_init_time = time.time()
        logger.info(main_header("OpenMM Molecular Dynamics Run"))

        if simulation_steps is None and simulation_time is None:
            raise InputError("Either simulation_steps or simulation_time needs to be set.")
        if simulation_time is not None:
            simulation_steps = int(simulation_time / self.timestep)
        if simulation_steps is not None:
            simulation_time = simulation_steps * self.timestep

        # Checking whether chkfile has been provided to run method or init
        if chkfile is None and self.chkfile is not None:
            logger.info("chkfile provided to init. Will use this for restart.")
            chkfile = self.chkfile
        if statefile is None and self.statefile is not None:
            logger.info("statefile provided to init. Will use this for restart.")
            statefile = self.statefile

        ##################################
        # CREATE SIMULATION OBJECT
        ##################################

        # If using Plumed then now we add Plumed-force to system from plumedinput string
        if plumedinput is not None:
            import openmmplumed

            logger.info("Plumed active. Adding Plumedforce to system")
            logger.info("plumedinput: %s", plumedinput)
            self.openmmobject.system.addForce(openmmplumed.PlumedForce(plumedinput))

        # Possible restraints added
        if restraints is not None:
            logger.info("Adding restraints")
            self.openmmobject.add_bondrestraints(restraints=restraints)

        # Creating simulation object and
        if chkfile is not None:
            self.simulation = self.openmmobject.create_simulation()
            logger.info("Checkpoint file provided. Restarting simulation using position and velocity data in file")
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities before: %s", state.getVelocities(asNumpy=True))
            self.simulation.loadCheckpoint(chkfile)
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities after loading checkpoint file: %s", state.getVelocities(asNumpy=True))
        elif statefile is not None:
            self.simulation = self.openmmobject.create_simulation()
            logger.info("State file provided. Restarting simulation using position and velocity data in file")
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities before: %s", state.getVelocities(asNumpy=True))
            self.simulation.loadState(statefile)
            state = self.simulation.context.getState(getVelocities=True)
            logger.info("Simulation velocities after loading statefile: %s", state.getVelocities(asNumpy=True))
        elif restart is True:
            logger.info("Restart true. Reusing already-defined simulation object")
        else:
            logger.info("Restart false and no chkfile/statefile set. This is a new simulation")
            self.simulation = self.openmmobject.create_simulation()
            logger.info("Simulation created.")
        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        ##################################
        # PRINT BASICS
        ##################################
        logger.info(small_header("MD run parameters"))
        logger.info(f"Simulation time: {simulation_time} ps")
        logger.info(f"Simulation steps: {simulation_steps}")
        logger.info(f"Timestep: {self.timestep} ps")
        logger.info(f"Set temperature: {self.temperature} K")
        logger.info("OpenMM integrator: %s", self.openmmobject.integrator_name)
        logger.info("")
        forceclassnames = [i.__class__.__name__ for i in self.openmmobject.system.getForces()]
        logger.info("OpenMM System forces present before run: %s", forceclassnames)

        # Printing PBCs
        if self.openmmobject.periodic is True:
            logger.info("Checking Initial PBC vectors.")
            self.state = self.simulation.context.getState()
            a, b, c = self.state.getPeriodicBoxVectors()
            logger.info("A:  %s", a)
            logger.info("B:  %s", b)
            logger.info("C:  %s", c)
            boxlength = a[0].value_in_unit(openmm.unit.angstrom)  # Box length in Angstrom
            logger.info(f"Boxlength: {boxlength} Angstrom")
        else:
            logger.info("System is not periodic")
        # Delete old traj
        ## Crashes when permissions not present or file is folder. Should never occur.
        #    pass

        # Make sure file associated with StateDataReporter is open
        if restart is True:
            logger.info("Restart true. Reusing simulation reporters")
            # if self.datafilename is not None:
            # Setting simulation reporters
            # Seems to be necessary to do this again after restart
            # restart option means that StateDatareport and DCDReporter will append to files
            self.set_sim_reporters(self.simulation, restart=True)
        elif statefile is not None:
            logger.info("statefile is used")
            # if self.datafilename is not None:
            # Setting simulation reporters
            # Seems to be necessary to do this again after restart
            # restart option means that StateDatareport and DCDReporter will append to files
            self.set_sim_reporters(self.simulation, restart=True)
        elif chkfile is not None:
            logger.info("chkfile is used")
            # if self.datafilename is not None:
            # Setting simulation reporters
            # Seems to be necessary to do this again after restart
            # restart option means that StateDatareport and DCDReporter will append to files
            self.set_sim_reporters(self.simulation, restart=True)
        else:
            logger.info("Restart false")
            if self.datafilename is not None:
                # RB addition: Delete file after each run
                logger.info("Deleting old datafile: %s", self.datafilename)
                with contextlib.suppress(OSError):
                    os.remove(self.datafilename)
                self.dataoutputoption = open(self.datafilename, "a")  # noqa: SIM115 - handed to OpenMM reporter
            # Setup data and simulation reporters for simulation object
            self.set_sim_reporters(self.simulation)

            # Setting coordinates of OpenMM object from current fragment.coords
            self.openmmobject.set_positions(self.positions, self.simulation)
        logger.info("")

        ###########################################
        # PBC and Wrapping
        ###########################################
        # Defining boxvectors in case we need
        if self.openmmobject.periodic is True:
            logger.info("Periodic Boundary Conditions used.")

            if self.enforce_periodic_box is True:
                logger.info("EnforcePeriodic Box is True. Wrapping enforced by OpenMM.")
                logger.warning(
                    "Warning: in case of problematic wrapping for e.g. QM/MM, try enabling special_wrapping=True"
                )
            # Wrapping handled by mdtraj
            if self.special_wrapping is True:
                logger.info("special_wrapping is True. Wrapping will be handled in each step by mdtraj library")
                logger.info("Importing mdtraj")
                try:
                    import mdtraj
                except ImportError:
                    raise MissingDependencyError(
                        "Error: mdtraj not found, needs to be installed (pip install mdtraj)"
                    ) from None
                # Defining boxvectors for wrapping
                boxvectors = self.simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
                # Convert topology from openmm format to mdtraj format
                mdtrajtopology = mdtraj.Topology.from_openmm(self.openmmobject.topology)
                # Choosing wrapping_atoms depending on theory-type
                if self.wrapping_atoms is None:
                    logger.info("No wrapping_atoms keyword has been set to center on.")
                    if self.theory_runtype == "QMMM":
                        logger.info("Theory-runtype is QMMM. Using QMatoms as wrapping_atoms")
                        wrapping_atoms = self.QM_MM_object.qmatoms
                    elif self.theory_runtype == "QM":
                        raise InputError("Theory_runtype is QM but no wrapping_atoms have been set.\nExiting")
                    elif self.theory_runtype == "dummy_MM":
                        raise InputError("Theory_runtype is dummy_MM but no wrapping_atoms have been set.\nExiting")
                    elif self.theory_runtype == "MM":
                        logger.info("Theory_runtype is MM. No achoratoms needed")
                        wrapping_atoms = None
                    logger.info("wrapping_atoms have been set to: %s", wrapping_atoms)
                else:
                    wrapping_atoms = self.wrapping_atoms
                    logger.info(f"Will use atoms {wrapping_atoms} for wrapping")

        ########################################
        # Writing intial frame to disk as PDB.
        ########################################
        pdb_filename = self.trajfilename + "_firstframe.pdb"
        logger.info("Writing intial frame to disk as PDB-file: %s", pdb_filename)
        blastate = self.simulation.context.getState(
            getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforce_periodic_box
        )
        with open(pdb_filename, "w") as f:
            openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbfile.PDBFile.writeModel(
                self.openmmobject.topology, blastate.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
            openmm.app.pdbfile.PDBFile.writeFooter(self.openmmobject.topology, f)
        # PDBx/mmCIF
        pdbx_filename = self.trajfilename + "_firstframe.cif"
        logger.info("Writing intial frame to disk as PDBx/mmCIF-file: %s", pdbx_filename)
        with open(pdbx_filename, "w") as f:
            openmm.app.pdbxfile.PDBxFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbxfile.PDBxFile.writeModel(
                self.openmmobject.topology, blastate.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )

        ###############################################################################
        # MD LOOP for each Theory-Runtype: QMMM, QM, dummy_MM, MM
        ###############################################################################
        if self.theory_runtype == "QMMM":
            # if self.QM_MM_object is not None:
            logger.info("QM/MM MD run beginning")
            # CASE: QM/MM. Custom external force needs to have been created in OpenMMTheory (should be handled by init)

            # Get connectivity from OpenMM topology
            connectivity = []
            for resi in self.openmmobject.topology.residues():
                resatoms = [i.index for i in list(resi.atoms())]
                connectivity.append(resatoms)
            # Convert to dict
            create_conn_dict(connectivity)

            # MD LOOP
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                logger.debug("Step: %s", step)
                if step % self.traj_frequency == 0:
                    logger.info("Step: %s", step)

                # Get state of simulation. Gives access to coords, velocities, forces, energy etc.
                current_state = self.simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=self.enforce_periodic_box, getEnergy=True
                )
                log_time_since(checkpoint, "get OpenMM state")
                checkpoint = time.time()
                # Get current coordinates from state to use for QM/MM step
                current_coords = np.array(current_state.getPositions(asNumpy=True)) * 10
                checkpoint = time.time()
                log_time_since(checkpoint, "get current_coords")

                # Periodic wrapping handling
                if self.openmmobject.periodic is True and self.special_wrapping is True:
                    logger.info("special_wrapping is True. Wrapping handled by mdtraj")
                    checkpoint = time.time()
                    # Wrapping
                    current_coords = diff_wrap_box_coords(
                        current_coords / 10.0, boxvectors, mdtrajtopology, wrapping_atoms
                    )
                    log_time_since(checkpoint, "wrapping via diff_wrap_box_coords")
                    checkpoint = time.time()
                    # Optional position update
                    if self.special_wrapping_updatepos is True:
                        logger.info("special_wrapping_update is True. Updating positions")
                        self.openmmobject.set_positions(current_coords, self.simulation)
                        log_time_since(checkpoint, "set positions update")
                        checkpoint = time.time()

                # Run QM/MM step to get full system QM+PC gradient.
                self.QM_MM_object.run(
                    current_coords=current_coords,
                    elems=self.fragment.elems,
                    grad=True,
                    exit_after_customexternalforce_update=True,
                    charge=self.charge,
                    mult=self.mult,
                )
                log_time_since(checkpoint, "QM/MM run")
                checkpoint = time.time()

                if step % self.restartfile_frequency == 0:
                    # Writing state and chk files
                    self.write_state_and_chk_files(step)

                # Printing step-info or write-trajectory at regular intervals
                # NOTE: Manual per-step info is not possible here because the MM-energy has not been
                # calculated yet when using the customexternalforceupdate option
                if step % self.traj_frequency == 0:
                    logger.info("Writing wrapped coords to trajfile: OpenMMMD_traj_wrapped.xyz (for debugging)")
                    write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj_wrapped", writemode="a")

                if self.specialatoms is not None and step % self.specialtraj_frequency == 0:
                    specialelems = [self.fragment.elems[i] for i in self.specialatoms]
                    special_coords = np.take(current_coords, self.specialatoms, axis=0)
                    logger.info("Writing wrapped coords to trajfile: only for special atoms")
                    write_xyzfile(specialelems, special_coords, "wrapped_special_traj", writemode="a")

                # Now need to update OpenMM external force with new QM-PC force
                # The QM_PC gradient (link-atom projected, from QM_MM object) is provided to OpenMM external force
                CheckpointTime = time.time()
                self.openmmobject.update_custom_external_force(
                    self.openmm_externalforceobject, self.QM_MM_object.QM_PC_gradient, self.simulation
                )
                log_time_since(CheckpointTime, "update custom external force")

                # NOTE: Think about energy correction (currently skipped above)
                # Now take OpenMM step (E+G + displacement etc.)
                checkpoint = time.time()

                self.simulation.step(1)
                log_time_since(checkpoint, "openmmobject sim step")
                log_time_since(checkpoint_begin_step, "Total sim step")

        # External QM for OpenMMtheory
        # TODO: Think about possible wrapping
        elif self.theory_runtype == "QM":
            logger.info("External QM with OpenMM option")
            for step in range(simulation_steps):
                checkpoint_begin_step = time.time()
                checkpoint = time.time()
                logger.info("Step: %s", step)
                # Get state of simulation. Gives access to coords, velocities, forces, energy etc.
                current_state = self.simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=self.enforce_periodic_box, getEnergy=True
                )
                log_time_since(checkpoint, "get OpenMM state")
                checkpoint = time.time()
                # Get current coordinates from state to use for QM/MM step
                current_coords = np.array(current_state.getPositions(asNumpy=True)) * 10
                log_time_since(checkpoint, "get current coords")
                checkpoint = time.time()

                # Periodic wrapping handling
                if self.openmmobject.periodic is True and self.special_wrapping is True:
                    logger.info("special_wrapping is True. Wrapping handled by mdtraj")
                    checkpoint = time.time()
                    # Wrapping
                    current_coords = diff_wrap_box_coords(
                        current_coords / 10.0, boxvectors, mdtrajtopology, wrapping_atoms
                    )
                    log_time_since(checkpoint, "wrapping via diff_wrap_box_coords")
                    checkpoint = time.time()
                    # Optional position update
                    if self.special_wrapping_updatepos is True:
                        logger.info("special_wrapping_update is True. Updating positions")
                        self.openmmobject.set_positions(current_coords, self.simulation)
                        log_time_since(checkpoint, "set positions update")
                        checkpoint = time.time()

                # Run QM step to get full system QM gradient.
                # Updates OpenMM object with QM forces
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
                # TODO: take this and QM energy and add to print_current_step_info
                extforce_energy = 3 * np.mean(sum(gradient * current_coords * 1.88972612546))
                logger.info("extforce_energy: %s", extforce_energy)

                # Printing step-info or write-trajectory at regular intervals
                if step % self.traj_frequency == 0:
                    # Manual step info option
                    print_current_step_info(step, current_state, self.openmmobject, qm_energy=energy)

                    if self.energy_file_option is not None:
                        with open(self.energy_file_option, "a") as f:
                            f.write(f"{energy}\n")

                    # Manual trajectory option
                    if self.trajectory_file_option == "XYZ":
                        write_xyzfile(self.fragment.elems, current_coords, "OpenMMMD_traj", writemode="a")

                if self.specialatoms is not None and step % self.specialtraj_frequency == 0:
                    specialelems = [self.fragment.elems[i] for i in self.specialatoms]
                    special_coords = np.take(current_coords, self.specialatoms, axis=0)
                    logger.info("Writing wrapped coords to trajfile: only for special atoms")
                    write_xyzfile(specialelems, special_coords, "wrapped_special_traj", writemode="a")

                if step % self.restartfile_frequency == 0:
                    # Writing state and chk files
                    self.write_state_and_chk_files(step)

                self.simulation.step(1)
                log_time_since(checkpoint, "OpenMM sim step")
                log_time_since(checkpoint_begin_step, "Total sim step")

        elif self.theory_runtype == "MM":
            logger.info("Regular classical OpenMM MD option chosen.")
            # Running all steps in one go
            self.simulation.step(simulation_steps)
        else:
            raise InputError(
                f"Error: Unrecognized Theory runtype ({self.theory_runtype}) for MD. This might mean that this theory "
                f"object is not yet supported for running MD. Exiting."
            )

        logger.info(small_header("OpenMM MD simulation finished!"))
        log_time_since(module_init_time, "OpenMM_MD run")

    def finalize_simulation(self):
        """Write the final structure and trajectory files and log the timing summary.

        Called once the requested number of steps has been taken.
        """
        logger.info("Finalizing simulation data")

        #######################
        # CLOSING OPEN FILES
        #######################
        # Close Statadatareporter file if open
        if self.datafilename is not None:
            self.dataoutputoption.close()

        # GETTING positions, forces and energy of final frame
        self.state = self.simulation.context.getState(
            getEnergy=True, getPositions=True, getForces=True, enforcePeriodicBox=self.enforce_periodic_box
        )

        ##########################
        # PERIODIC BOX VECTORS
        ##########################
        if self.openmmobject.periodic is True:
            logger.info("Checking PBC vectors:")
            a, b, c = self.state.getPeriodicBoxVectors()
            logger.info("A:  %s", a)
            logger.info("B:  %s", b)
            logger.info("C:  %s", c)
            logger.info("a 0 %s", a[0])
            # Set new PBC vectors since they may have changed
            logger.info("Updating PBC vectors in simulation.context, OpenMM system and OpenMM topology")
            # Context. Used?
            self.simulation.context.setPeriodicBoxVectors(a, b, c)
            # System. Necessary
            self.openmmobject.system.setDefaultPeriodicBoxVectors(a, b, c)
            # Topology (for header in PDB-files). Necessary
            self.openmmobject.topology.setPeriodicBoxVectors(self.state.getPeriodicBoxVectors())

        ################################################
        # Writing final frame to disk as PDB and PDBx
        ################################################
        pdb_filename = self.trajfilename + "_lastframe.pdb"
        logger.info("Writing final frame to disk as PDB-file: %s", pdb_filename)
        with open(pdb_filename, "w") as f:
            openmm.app.pdbfile.PDBFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbfile.PDBFile.writeModel(
                self.openmmobject.topology, self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
            openmm.app.pdbfile.PDBFile.writeFooter(self.openmmobject.topology, f)
        logger.info(f"Trajectory : {self.trajfilename}.{self.trajectory_file_option}")
        # PDBx/mmCIF
        pdbx_filename = self.trajfilename + "_lastframe.cif"
        logger.info("Writing final frame to disk as PDBx/mmCIF-file: %s", pdbx_filename)
        with open(pdbx_filename, "w") as f:
            openmm.app.pdbxfile.PDBxFile.writeHeader(self.openmmobject.topology, f)
            openmm.app.pdbxfile.PDBxFile.writeModel(
                self.openmmobject.topology, self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom), f
            )
        logger.info(f"Trajectory : {self.trajfilename}.{self.trajectory_file_option}")

        # Saving state to disk
        # Can be used to restart using statefile option
        logger.info(
            "Saving a statefile and checkpointfile of the final frame of the simulation: OpenMM_MD_final_state.xml and "
            "OpenMM_MD_final_checkpoint.chk"
        )
        logger.info(
            "These file can be used to restart a simulation (statefile and chkfile keywords) using the same "
            "coordinates and velocities."
        )
        self.simulation.saveState("OpenMM_MD_final_state.xml")
        self.simulation.saveCheckpoint("OpenMM_MD_final_checkpoint.chk")

        ########################
        # Updating fragment
        ########################
        newcoords = self.state.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        logger.info("Updating coordinates in fragment.")
        self.fragment.coords = newcoords
        # Updating positions array also in case we call run again
        self.positions = newcoords


def openmm_box_equilibration(
    fragment=None,
    theory=None,
    datafilename="nptsim.csv",
    numsteps_per_npt=10000,
    max_npt_cycles=10,
    pressure=1,
    volume_threshold=1.3,
    density_threshold=0.005,
    temperature=300,
    timestep=0.001,
    traj_frequency=100,
    trajfilename="equilibration_NPT",
    trajectory_file_option="DCD",
    coupling_frequency=1,
    enforce_periodic_box=True,
    use_mdtraj=True,
    dummyatomrestraint=False,
    solute_indices=None,
    barostat_frequency=25,
) -> list:
    """Run NPT simulations in cycles until box volume and density stop changing.

    Args:
        fragment: Fragment with the periodic system.
        theory: OpenMMTheory object (periodic).
        datafilename: CSV file for per-cycle state data.
        numsteps_per_npt: MD steps per NPT cycle.
        max_npt_cycles: maximum number of cycles.
        pressure: barostat pressure in bar.
        volume_threshold: convergence threshold for the box-volume change.
        density_threshold: convergence threshold for the density change.
        temperature: thermostat temperature in K.
        timestep: MD timestep in ps.
        traj_frequency: trajectory write interval in steps.
        trajfilename: base name of the trajectory file; the extension comes from
            trajectory_file_option.
        trajectory_file_option: trajectory format ("DCD", ...).
        coupling_frequency: thermostat coupling frequency in ps^-1.
        enforce_periodic_box: wrap coordinates into the primary box.
        use_mdtraj: after the run, reimage the trajectory with mdtraj and write
            <trajfilename>_lastframe.pdb. Skipped silently if mdtraj is unavailable.
        dummyatomrestraint: add a dummy atom to the topology and restrain the solute to
            it, keeping the solute centred as the box changes size. Requires
            solute_indices.
        solute_indices: atom indices of the solute; required when dummyatomrestraint
            is True.
        barostat_frequency: barostat attempt interval in timesteps.

    Returns:
        Fragment updated with the equilibrated coordinates and box vectors.
    """
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

    # Number of points used in each cycle to calculate stdev

    if len(theory.user_frozen_atoms) > 0:
        logger.info("Frozen_atoms: %s", theory.user_frozen_atoms)
        logger.warning(
            "OpenMM object has frozen atoms defined. This is known to cause strange issues for NPT simulations."
        )
        logger.warning("Check the results carefully!")

    # Starting parameters
    steps = 0
    volume_std = 10
    density_std = 1

    md = MolecularDynamicsEngine(**engine_kwargs)
    restart = False
    # while volume_std >= volume_threshold and density_std >= density_threshold:
    for i in range(max_npt_cycles):
        logger.info("")
        logger.info("%s", "-" * 100)
        logger.info(f"Now starting  NPT cycle {i} with {numsteps_per_npt} MD steps")
        logger.info(
            f"Simulation data (timestep, energy, temperature, volume,density etc.) is also written to {datafilename}"
        )
        if restart is False:
            # Call MD object run method for the first
            md.run(numsteps_per_npt, restart=restart)
            # Setting restart to True for next iteration
            restart = True
        else:
            # Easier and safer to continue by call simulation step directly instead of md.run
            md.simulation.step(numsteps_per_npt)

        steps += numsteps_per_npt

        # Read reporter file and calculate stdev

        NPTresults = read_npt_statefile(datafilename)
        volume = NPTresults["volume"][-numpoints_for_convergence_check:]
        density = NPTresults["density"][-numpoints_for_convergence_check:]
        logger.info("Total number of volume datapoints available: %s", len(NPTresults["volume"]))
        logger.info("Total number of density datapoints available: %s", len(NPTresults["density"]))
        logger.info(
            "Number of datapoints (last) used for convergence check in each cycle: %s", numpoints_for_convergence_check
        )
        volume_std = np.std(volume)
        density_std = np.std(density)

        logger.info(small_header("Equilibration Status"))
        logger.info("Total steps taken: %s", steps)
        logger.info(f"Total simulation time: {timestep * steps} ps")
        logger.info("Current Volume: %s", volume[-1])
        logger.info(f"Current Density: {density[-1]}")
        logger.info("")
        logger.info(f"Current Volume SD: {volume_std}   (threshold: {volume_threshold})")
        logger.info(f"Current Density SD: {density_std} (threshold: {density_threshold})")

        if volume_std < volume_threshold and density_std < density_threshold:
            logger.info(f"Equilibration of periodic box finished after {steps} and {timestep * steps} ps !\n")
            break

        if i == max_npt_cycles - 1:
            logger.warning(
                f"Warning: Max NPT cycles reached ({max_npt_cycles}). Total steps taken: {steps} and "
                f"{timestep * steps} ps !\n"
            )
            logger.warning("The NPT simulation may not be properly converged")
            break

    # Finalizing simulation (writes and updates files)
    md.finalize_simulation()

    logger.info(f"Final PDB file: {trajfilename}.pdb")
    logger.info(f"NPT trajectory: {trajfilename}.{trajectory_file_option.lower()}")

    # Running mdtraj
    if use_mdtraj is True:
        logger.info("Trying to load mdtraj for reimaging trajectory")
        try:
            logger.info("Imaging trajectory")
            mdtraj_image_trajectory(f"{trajfilename}.dcd", f"{trajfilename}_lastframe.pdb")
        except ImportError:
            logger.info("mdtraj library could not be imported. Skipping")
        except ValueError as e:
            logger.info(f"mdtraj reimaging failed. Skipping. Error: {e}")

    log_time_since(module_init_time, "OpenMM_box_equilibration")
    return md.state.getPeriodicBoxVectors()


def print_current_step_info(step, state, openmmobject, qm_energy=None):

    # Kinetic energy directly from MD-state
    kinetic_energy = state.getKineticEnergy()
    kinetic_energy_eh = kinetic_energy.value_in_unit(openmm.unit.kilojoules_per_mole) / 2625.5002

    # Potential energy from the theory level instead
    if qm_energy is not None:
        dummy_warning = "(correct)"
        pot_energy = qm_energy
    else:
        dummy_warning = "(dummy)"
        pot_energy = state.getPotentialEnergy()

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
    theory=None,
    fragment=None,
    time_steps=None,
    steps=None,
    temperatures=None,
    check_gradient_first=True,
    gradient_threshold=100,
    use_mdtraj=True,
    trajfilename="warmup_MD",
    initial_opt=True,
    traj_frequencies=None,
    maxoptsteps=10,
    coupling_frequency=1,
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

    # Gradient check before we proceed
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

    # Try a simple minimization first or simple MD

    if initial_opt is True:
        logger.info(f"\ninitial_opt is True (default). Will attempt initial {maxoptsteps}-step minimization first")
        logger.info("If this step runs forever something is wrong. Select initial_opt=False to avoid in this case")
        try:
            openmm_minimize(fragment=fragment, theory=theory, maxiter=maxoptsteps, tolerance=1)
            logger.info("Minimization successful")
        except Exception as e:  # noqa: BLE001 - MD warm-up continues even if pre-minimization fails
            logger.info("Problem minimizing system")
            logger.error("message: %s", e)
            logger.info("Will go on to do MD")

    logger.info(f"\n{len(steps)} MD-runs have been defined")
    for num, (ts, step, temp) in enumerate(zip(time_steps, steps, temperatures, strict=False)):
        logger.info(f"MD-step {num} Number of simulation steps: {step} with timestep: {ts} and temperature: {temp} K")

    logger.info("")
    logger.info("")
    # Gentle heating up protocol
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

        # Running mdtraj after each sim
        if use_mdtraj is True:
            logger.info("Trying to load mdtraj for basic analysis of trajectory")
            try:
                logger.info("Imaging trajectory")
                mdtraj_image_trajectory(f"{MDcyclename}.dcd", f"{MDcyclename}_lastframe.pdb")
                logger.info("\nRunning RMS Fluctuation analysis on trajectory")
                mdtraj_rmsf(
                    f"{MDcyclename}.dcd",
                    f"{MDcyclename}_lastframe.pdb",
                    print_largest_values=True,
                    threshold=0.005,
                    largest_values=10,
                )
            except ImportError:
                logger.info("mdtraj library could not be imported. Skipping")
            except ValueError as e:
                logger.info(f"mdtraj reimaging failed. Skipping. Error: {e}")

    logger.info("Gentle_warm_up_MD finished successfully!")
    log_time_since(module_init_time, "Gentle_warm_up_MD")


def diff_wrap_box_coords(coords_nm, boxvectors, mdtrajtopology, anchoratoms):
    # Import mdtraj library
    import mdtraj

    # Creating Trajectory object for geometry
    traj = mdtraj.Trajectory(coords_nm, mdtrajtopology)
    # Setting PBC vectors
    traj.unitcell_vectors = np.array(boxvectors).reshape(1, 3, 3)
    # Anchoratoms (usually QM-region or similar)
    anchors = [{traj.topology.atom(i) for i in anchoratoms}]
    # Re-imaging trajectory
    imaged = traj.image_molecules(anchor_molecules=anchors)
    return imaged._xyz[0] * 10.0
