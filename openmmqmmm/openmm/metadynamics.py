"""Metadynamics, PLUMED-biased MD and the free-energy surface helpers."""

import logging
import os

import numpy as np

import openmmqmmm.constants
import openmmqmmm.parallel
import openmmqmmm.plotting
from openmmqmmm.exceptions import (
    FileFormatError,
    InputError,
    MissingDependencyError,
)
from openmmqmmm.openmm.md import MolecularDynamicsEngine
from openmmqmmm.utils import (
    main_header,
    writestringtofile,
)

logger = logging.getLogger(__name__)


def openmm_metadynamics(
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
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
    flatbottom_restraint_cv1=None,
    flatbottom_restraint_cv2=None,
    funnel_restraint=None,
    funnel_parameters=None,
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
    cv1_atoms=None,
    cv2_atoms=None,
    cv1_type=None,
    cv2_type=None,
    biasfactor=6,
    height=1,
    reference_xyzfile=None,
    cv1_biaswidth=0.5,
    cv2_biaswidth=0.5,
    cv1_range=None,
    cv2_range=None,
    cv1_parameters=None,
    cv2_parameters=None,
    user_cvforce1=None,
    user_biasvar1=None,
    user_cvforce2=None,
    user_biasvar2=None,
    frequency=1,
    savefrequency=10,
    chkfile=None,
    statefile=None,
    biasdir=".",
    multiplewalkers=False,
    numcores=1,
    walkerid=None,
) -> None:
    """Run metadynamics MD using OpenMM's native metadynamics implementation.

    Collective variables are defined via cv1_atoms/cv1_type (and optionally CV2).
    """
    logger.info(main_header("OpenMM metadynamics"))

    # Biasdirectory
    logger.info("biasdirectory chosen to be: %s", biasdir)
    biasdir_full_path = os.path.abspath(biasdir)
    logger.info("Full path to biasdirectory is: %s", biasdir_full_path)
    if not os.path.isdir(biasdir_full_path):
        raise FileFormatError(f"Error: Biasdirectory: {biasdir_full_path} does not exist")

    if cv2_type is None:
        logger.info("CV2 not specified. Assuming only 1 CV in simulation.")
        numCVs = 1
        if user_cvforce1 is None and (cv1_atoms is None or cv1_type is None):
            raise InputError("Error: You must specify both CV1_atoms and CV1_type keywords")
    else:
        numCVs = 2
        if user_cvforce1 is None and (cv1_atoms is None or cv1_type is None):
            raise InputError("Error: You must specify both CV1_atoms and CV1_type keywords")
        if user_cvforce2 is None and (cv2_atoms is None or cv2_type is None):
            raise InputError("Error: You must specify both CV2_atoms and CV2_type keywords")

    # Parallelization
    if multiplewalkers is True and numcores == 1:
        raise InputError("Error: For multiplewalkers=True  you must set numcores to the number of walkers")

    # Creating MDclass
    md = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        charge=charge,
        mult=mult,
        timestep=timestep,
        traj_frequency=traj_frequency,
        temperature=temperature,
        integrator=integrator,
        constraints=constraints,
        specialatoms=specialatoms,
        specialtraj_frequency=specialtraj_frequency,
        barostat=barostat,
        pressure=pressure,
        trajectory_file_option=trajectory_file_option,
        coupling_frequency=coupling_frequency,
        anderson_thermostat=anderson_thermostat,
        enforcePeriodicBox=enforce_periodic_box,
        special_wrapping=special_wrapping,
        special_wrapping_updatepos=special_wrapping_updatepos,
        wrapping_atoms=wrapping_atoms,
        dummyatomrestraint=dummyatomrestraint,
        center_on_atoms=center_on_atoms,
        solute_indices=solute_indices,
        datafilename=datafilename,
        dummy_mm=dummy_mm,
        platform=platform,
        hydrogenmass=hydrogenmass,
        add_centerforce=add_centerforce,
        trajfilename=trajfilename,
        chkfile=chkfile,
        statefile=statefile,
        centerforce_atoms=centerforce_atoms,
        centerforce_constant=centerforce_constant,
        centerforce_distance=centerforce_distance,
        centerforce_center=centerforce_center,
        barostat_frequency=barostat_frequency,
    )

    if user_cvforce1 is not None:
        logger.info("User CV-force 1 was given: %s", user_cvforce1)
        md.user_cvforce1 = user_cvforce1
    if user_biasvar1 is not None:
        logger.info("User Biasvar CV1 was given: %s", user_biasvar1)
        md.user_biasvar1 = user_biasvar1
    if user_cvforce2 is not None:
        logger.info("User CV-force 2 was given: %s", user_cvforce2)
        md.user_cvforce2 = user_cvforce2
    if user_biasvar2 is not None:
        logger.info("User Biasvar CV2 was given: %s", user_biasvar2)
        md.user_biasvar2 = user_biasvar2

    # Load OpenMM.app

    # If RMSD CV
    if cv1_type == "rmsd" or cv2_type == "rmsd":
        # Reference position. For now just use initial cooordinates as reference positions
        # reference_pos = [openmm.Vec3(coords_nm[i, 0], coords_nm[i, 1], coords_nm[i, 2]) for i in
        #       range(len(coords_nm))] * openmm.unit.nanometer
        logger.info("rmsd_CV1_reference_indices: %s", cv1_atoms)
        logger.info("rmsd_CV2_reference_indices: %s", cv2_atoms)
    else:
        pass
    # Setting up collective variables for native case
    native_MTD = True
    # Creating dictionary with MTD parameters that will be passed to MD function
    if numCVs == 1:
        # Create metadynamics dict for 1 CV
        metadyn_settings = {
            "numCVs": numCVs,
            "temperature": temperature,
            "biasfactor": biasfactor,
            "height": height,
            "frequency": frequency,
            "saveFrequency": savefrequency,
            "biasdir": biasdir_full_path,
            "CV1_type": cv1_type,
            "CV2_type": None,
            "reference_xyzfile": reference_xyzfile,
            "CV1_atoms": cv1_atoms,
            "CV2_atoms": cv2_atoms,
            "CV1_range": cv1_range,
            "CV2_range": cv2_range,
            "CV1_biaswidth": cv1_biaswidth,
            "CV2_biaswidth": cv2_biaswidth,
            "CV2_minvalue": None,
            "CV2_maxvalue": None,
            "CV1_parameters": cv1_parameters,
            "flatbottom_restraint_CV1": flatbottom_restraint_cv1,
            "flatbottom_restraint_CV2": flatbottom_restraint_cv2,
        }
    elif numCVs == 2:
        # Create metadynamics object for 2 CVs
        metadyn_settings = {
            "numCVs": numCVs,
            "temperature": temperature,
            "biasfactor": biasfactor,
            "height": height,
            "frequency": frequency,
            "saveFrequency": savefrequency,
            "biasdir": biasdir_full_path,
            "CV1_type": cv1_type,
            "CV2_type": cv2_type,
            "reference_xyzfile": reference_xyzfile,
            "CV1_range": cv1_range,
            "CV2_range": cv2_range,
            "CV1_parameters": cv1_parameters,
            "CV2_parameters": cv2_parameters,
            "CV1_atoms": cv1_atoms,
            "CV2_atoms": cv2_atoms,
            "CV1_biaswidth": cv1_biaswidth,
            "CV2_biaswidth": cv2_biaswidth,
            "flatbottom_restraint_CV1": flatbottom_restraint_cv1,
            "flatbottom_restraint_CV2": flatbottom_restraint_cv2,
        }

    # Add restraining funnel for funnel metadynamics
    if funnel_restraint is not None:
        if funnel_parameters is None:
            raise InputError(
                "Error: funnel_restraint requires passing a dictionary with funnel definition parameters.\nExample: "
                "funnel_parameters = {'ligand_indices':[0,1,2], 'k_xyz':10.0, 'z_cc':11.0, 'alpha':35.0, "
                "'R_cylinder':1.0, 'force_group':10}"
            )

        # Getting atom indices for host and guess
        guest_indices = funnel_parameters["ligand_indices"]
        logger.info("guest_indices: %s", guest_indices)
        if "host_indices" in funnel_parameters:
            logger.info("Found host indices in funnel_parameters")
            host_indices = funnel_parameters["host_indices"]
            logger.info("host_indices: %s", host_indices)
        else:
            raise InputError("No host_indices found in funnel_parameters")

        md.openmmobject.add_funnel_restraint(
            host_indices,
            guest_indices,
            k_xy=funnel_parameters["k_xy"],
            z_cc=funnel_parameters["z_cc"],
            alpha=funnel_parameters["alpha"],
            r_cylinder=funnel_parameters["R_cylinder"],
            force_group=funnel_parameters["force_group"],
        )

    # Calling md.run with either native option active or false
    logger.info("Now starting metadynamics simulation")

    if multiplewalkers is True:
        raise InputError("{}\nError: Disabled".format(f"Now launching Metadynamics job with {numcores} walkers"))
        # Input parameters passed as dictionary to Simple_parallel
        # NOTE: multiprocess library (instead of multiprocessing) is necessary.
        # Otherwise pickling problem involving _io.TextIOWrapper
        openmmqmmm.parallel.simple_parallel(
            jobfunction=md.run,
            parameter_dict={
                "simulation_steps": simulation_steps,
                "simulation_time": simulation_time,
                "metadynamics": native_MTD,
                "metadyn_settings": metadyn_settings,
            },
            numcores=numcores,
            version="multiprocess",
            separate_dirs=True,
            restraints=restraints,
        )
    else:
        md.run(
            simulation_steps=simulation_steps,
            simulation_time=simulation_time,
            metadynamics=native_MTD,
            metadyn_settings=metadyn_settings,
            restraints=restraints,
        )
    logger.info("Metadynamics simulation done")

    # Finalizing simulation (writes and updates files)
    md.finalize_simulation()

    # Data plotting
    logger.info("\nAll bias-files have been written to biasdirectory: %s", biasdir_full_path)
    logger.info("Dir also contains: MTD_parameters.txt")
    logger.info("Use function  get_free_energy_from_biasfiles  to create free-energy surface")
    logger.info("and function metadynamics_plot_data to plot the data")
    logger.info("")
    return


def openmm_md_plumed(
    fragment=None,
    theory=None,
    timestep=0.001,
    simulation_steps=None,
    simulation_time=None,
    traj_frequency=1000,
    temperature=300,
    integrator="LangevinMiddleIntegrator",
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
    numcores=1,
) -> None:
    """Run MD with a PLUMED bias (requires the openmm-plumed plugin)."""
    logger.info(main_header("OpenMM metadynamics using OpenMM-Plumed interface"))

    logger.info("Using metadynamics via OpenMM Plumed plugin")
    try:
        import openmmplumed
    except ModuleNotFoundError:
        raise MissingDependencyError(
            "openmmplumed module plugin not found. See https://github.com/openmm/openmm-plumed \nYou can install via "
            "conda: \nconda install -c conda-forge openmm-plumed"
        ) from None

    # Creating MDclass
    md = MolecularDynamicsEngine(
        fragment=fragment,
        theory=theory,
        charge=charge,
        mult=mult,
        timestep=timestep,
        traj_frequency=traj_frequency,
        temperature=temperature,
        integrator=integrator,
        constraints=constraints,
        specialatoms=specialatoms,
        specialtraj_frequency=specialtraj_frequency,
        barostat=barostat,
        pressure=pressure,
        trajectory_file_option=trajectory_file_option,
        coupling_frequency=coupling_frequency,
        anderson_thermostat=anderson_thermostat,
        enforcePeriodicBox=enforce_periodic_box,
        special_wrapping=special_wrapping,
        special_wrapping_updatepos=special_wrapping_updatepos,
        wrapping_atoms=wrapping_atoms,
        dummyatomrestraint=dummyatomrestraint,
        center_on_atoms=center_on_atoms,
        solute_indices=solute_indices,
        datafilename=datafilename,
        dummy_mm=dummy_mm,
        platform=platform,
        hydrogenmass=hydrogenmass,
        add_centerforce=add_centerforce,
        trajfilename=trajfilename,
        centerforce_atoms=centerforce_atoms,
        centerforce_constant=centerforce_constant,
        chkfile=chkfile,
        statefile=statefile,
        centerforce_distance=centerforce_distance,
        centerforce_center=centerforce_center,
        barostat_frequency=barostat_frequency,
    )

    # Load OpenMM.app

    logger.info("Setting up Plumed")
    # OPTION to provide the full Plumed input as string instead
    if plumed_input_string is not None:
        logger.info(
            "plumed_input_string provided. Will read all options from this string (make sure to provide atom indices "
            "in 1-based indexing)"
        )
        writestringtofile(plumed_input_string, "plumedinput.in")
        plumedinput = plumed_input_string

    logger.info("Now starting metadynamics simulation")
    md.run(
        simulation_steps=simulation_steps,
        simulation_time=simulation_time,
        restraints=restraints,
        plumedinput=plumedinput,
    )
    logger.info("Metadynamics simulation done")

    # Finalizing simulation (writes and updates files)
    md.finalize_simulation()

    os.path.dirname(os.path.dirname(os.path.dirname(openmmplumed.mm.pluginLoadedLibNames[0])))
    logger.info(
        "You can now analyze/plot the metadynamics data with plumed's own tools (requires presence of HILLS and COLVAR "
        "files in directory)"
    )
    logger.info("\n")

    return


def free_energy_from_bias_array(temperature, bias_factor, total_bias) -> np.ndarray:
    """Convert a metadynamics bias array to a free-energy surface."""
    deltaT = temperature * (bias_factor - 1)
    kjpermoleconversion = 1
    free_energy = -((temperature + deltaT) / deltaT) * total_bias * kjpermoleconversion
    return free_energy


def get_free_energy_from_biasfiles(temperature, biasfactor, cv1_gridwidth, cv2_gridwidth, directory=".") -> tuple:
    """Reconstruct a free-energy surface from metadynamics bias files on disk."""
    import glob

    # Checking gridwiths
    full_bias = np.zeros(cv1_gridwidth) if cv2_gridwidth is None else np.zeros((cv2_gridwidth, cv1_gridwidth))

    # Looping over bias-files
    logger.info("full_bias shape: %s", full_bias.shape)
    list_of_biases = []
    for biasfile in glob.glob(f"{directory}/*.npy"):
        logger.info("Loading biasfile: %s", biasfile)
        try:
            data = np.load(biasfile)
            logger.info("data shape: %s", data.shape)
            full_bias += data
            list_of_biases.append(data)
        except FileNotFoundError:
            logger.info("File not found error: Simulation probably still running. skipping file")

    # Get final free energy (sum of all)
    free_energy = free_energy_from_bias_array(temperature, biasfactor, full_bias)

    # Get free-energy per biasfile
    list_of_free_energies = []
    for bias_array in list_of_biases:
        fe = free_energy_from_bias_array(temperature, biasfactor, bias_array)
        list_of_free_energies.append(fe)

    # Return final free_energy array and also list of free-energy-arrays for each biasfile
    return free_energy, list_of_free_energies


def metadynamics_plot_data(biasdir=None, dpi=200, imageformat="png", plot_xlim=None, plot_ylim=None) -> None:
    """Plot metadynamics results (free-energy surface and CV trajectories) from a bias directory."""
    import json

    # Read mtd settings dict from file
    with open(f"{biasdir}/MTD_parameters.txt") as mtdfh:
        metadyn_settings = json.load(mtdfh)

    cv1_type = metadyn_settings["CV1_type"]
    cv2_type = metadyn_settings["CV2_type"]
    temperature = metadyn_settings["temperature"]
    biasfactor = metadyn_settings["biasfactor"]
    CV1_gridwidth = metadyn_settings["CV1_gridwidth"]
    logger.info("metadyn_settings: %s", metadyn_settings)
    CV2_gridwidth = metadyn_settings["CV2_gridwidth"]

    CV1_minvalue = metadyn_settings["CV1_minvalue"]
    CV1_maxvalue = metadyn_settings["CV1_maxvalue"]
    CV2_minvalue = metadyn_settings["CV2_minvalue"]
    CV2_maxvalue = metadyn_settings["CV2_maxvalue"]
    logger.info(f"Using CV1_minvalue:{CV1_minvalue} CV1_maxvalue:{CV1_maxvalue}")
    logger.info(f"Using CV2_minvalue:{CV2_minvalue} CV2_maxvalue:{CV2_maxvalue}")

    e_conversionfactor = 4.184  # kJ/mol to kcal/mol
    numCVs = 2 if cv2_type is not None else 1
    if numCVs == 2:
        cv1_conversionfactor = 1.0
        cv2_conversionfactor = 1.0
        if cv1_type == "dihedral" or cv1_type == "torsion" or cv1_type == "angle":
            cv1_conversionfactor = 180 / np.pi
            CV1_unit_label = "°"
        elif cv1_type == "bond" or cv1_type == "distance" or cv1_type == "rmsd":
            cv1_conversionfactor = 10.0
            CV1_unit_label = "Å"
        if cv2_type == "dihedral" or cv2_type == "angle" or cv1_type == "torsion":
            cv2_conversionfactor = 180 / np.pi
            CV2_unit_label = "°"
        elif cv2_type == "bond" or cv2_type == "distance" or cv2_type == "rmsd":
            cv2_conversionfactor = 10.0
            CV2_unit_label = "Å"
        else:
            CV1_unit_label = ""
            CV2_unit_label = ""

        # Get free energy surface from biasfiles
        free_energy, _list_of_fes_from_biasfiles = get_free_energy_from_biasfiles(
            temperature, biasfactor, CV1_gridwidth, CV2_gridwidth, directory=biasdir
        )
        # Relative free energy in kcal/mol
        rel_free_energy = (free_energy - np.min(free_energy)) / e_conversionfactor
        # Coordinates in correct unit
        xvalues = [
            cv1_conversionfactor * (CV1_minvalue + ((CV1_maxvalue - CV1_minvalue) / (CV1_gridwidth - 1)) * i)
            for i in range(CV1_gridwidth)
        ]
        yvalues = [
            cv2_conversionfactor * (CV2_minvalue + ((CV2_maxvalue - CV2_minvalue) / (CV2_gridwidth - 1)) * i)
            for i in range(CV2_gridwidth)
        ]
        np.savetxt("MTD_free_energy.txt", free_energy)
        np.savetxt("MTD_free_energy_rel.txt", rel_free_energy)
        np.savetxt("CV1_coord_values.txt", xvalues)
        np.savetxt("CV2_coord_values.txt", yvalues)

        # Plot
        logger.info("Now plotting:")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.info("Problem importing matplotlib")
            return
        # 2D CV plotting uisng scatter with colormap
        # Colormap to use in 2CV plots.
        # Perceptually uniform sequential: viridis, plasma, inferno, magma, cividis
        # Others: # RdYlBu_r
        # See https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
        colormap_option3 = "RdYlBu_r"
        X2, Y2 = np.meshgrid(xvalues, yvalues)
        option3fig, option3ax = plt.subplots()
        cm = plt.cm.get_cmap(colormap_option3)
        colorscatter = option3ax.scatter(X2, Y2, c=rel_free_energy, marker="o", linestyle="-", linewidth=1, cmap=cm)
        # Colorbar
        cbar = plt.colorbar(colorscatter)
        cbar.set_label("ΔG (kcal/mol)", fontweight="bold", fontsize="xx-small")
        # Limits
        if plot_xlim is not None:
            option3ax.set_xlim(plot_xlim[0], plot_xlim[1])
        if plot_ylim is not None:
            option3ax.set_ylim(plot_ylim[0], plot_ylim[1])
        option3ax.set_xlabel(f"CV1:{cv1_type}  ({CV1_unit_label})")
        option3ax.set_ylabel(f"CV2:{cv2_type}  ({CV2_unit_label})")
        option3fig.savefig("MTD_CV1_CV2_.png", format=imageformat, dpi=dpi)
        logger.info("Created file: MTD_CV1_CV2_.png")
        return

    elif numCVs == 1:
        cv1_conversionfactor = 1.0
        if cv1_type == "dihedral" or cv1_type == "torsion" or cv1_type == "angle":
            cv1_conversionfactor = 180 / np.pi
            CV1_unit_label = "°"
        elif cv1_type == "bond" or cv1_type == "distance" or cv1_type == "rmsd":
            cv1_conversionfactor = 10.0
            CV1_unit_label = "Ang"
        else:
            CV1_unit_label = ""
        free_energy, _bla = get_free_energy_from_biasfiles(
            temperature, biasfactor, CV1_gridwidth, None, directory=biasdir
        )

        # X-values
        full_range = CV1_maxvalue - CV1_minvalue
        increment = full_range / (CV1_gridwidth - 1)
        xvalues = [cv1_conversionfactor * (CV1_minvalue + increment * i) for i in range(CV1_gridwidth)]
        np.savetxt("CV1_coord_values.txt", xvalues)
        # Relative energy in kcal/mol
        rel_free_energy = (free_energy - min(free_energy)) / e_conversionfactor
        logger.info("rel_free_energy: %s", rel_free_energy)
        # Save stuff
        np.savetxt("MTD_free_energy.txt", free_energy)
        np.savetxt("MTD_free_energy_rel.txt", rel_free_energy)

        # Plot object
        logger.info("Now plotting:")
        CVlabel = f"{cv1_type} ({CV1_unit_label})"
        y_axislabel = "Energy (kcal(/mol))"
        eplot = openmmqmmm.plotting.Plot("Metadynamics", num_subplots=1, x_axislabel=CVlabel, y_axislabel=y_axislabel)
        eplot.addseries(0, x_list=xvalues, y_list=rel_free_energy, legend=None, color="blue", line=True, scatter=False)
        eplot.savefig("MTD_CV1", imageformat=imageformat, dpi=dpi)
        return
