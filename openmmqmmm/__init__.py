"""
ASH - A MULTISCALE MODELLING PROGRAM
R. Bjornsson

Trimmed distribution: ORCA + OpenMM QM/MM functionality for biomolecular calculations.
"""
import atexit
import glob
# Python libraries
import numpy as np
import os
import pathlib
import sys

# Getting ASH-path
ashpath = str(pathlib.Path(__file__).parent.resolve())
print("ashpath:", ashpath)
###############
# ASH modules
###############
# Adding modules,interfaces directories to sys.path
sys.path.insert(0, ashpath)
print("Sys path:", sys.path)

from .functions.functions_general import create_ash_env_file, blankline, BC, listdiff, print_time_rel, \
    print_time_rel_and_tot, pygrep, \
    printdebug, read_intlist_from_file, frange, writelisttofile, load_julia_interface, read_datafile, write_datafile, \
    ashexit, natural_sort, numlines_in_file

# Test if inputfile has a bad name
inputfile_base = os.path.splitext(sys.argv[0])[0]
pyfiles_in_dir = glob.glob('*.py')
forbidden_inputfilenames = ['openmmqmmm', 'openmm', 'geometric', 'mdtraj', 'openbabel']
for pyfile in pyfiles_in_dir:
    if os.path.splitext(pyfile)[0] in forbidden_inputfilenames:
        print(f"Error: Current directory contains file : {inputfile_base}.py with a forbidden name. Please rename it")
        print("Forbidden names:", forbidden_inputfilenames)
        ashexit()

# Results dataclass
from .modules.module_results import ASH_Results, read_results_from_file

# Fragment class and coordinate functions
import openmmqmmm.modules.module_coords
from .modules.module_coords import get_molecules_from_trajectory, eldict_covrad, write_pdbfile, Fragment, read_xyzfile, \
    write_xyzfile, make_cluster_from_box, read_ambercoordinates, read_gromacsfile, split_multimolxyzfile, \
    distance_between_atoms, \
    angle_between_atoms, dihedral_between_atoms
from .modules.module_coords import remove_atoms_from_system_CHARMM, add_atoms_to_system_CHARMM, getwaterconstraintslist, \
    QMregionfragexpand, cut_sphere, cut_cubic_box, QMPC_fragexpand, read_xyzfiles, Reaction, define_XH_constraints, \
    simple_get_water_constraints, print_internal_coordinate_table, \
    flexible_align_pdb, flexible_align_xyz, flexible_align, insert_solute_into_solvent, nuc_nuc_repulsion, \
    calculate_RMSD

# Singlepoint
import openmmqmmm.modules.module_singlepoint
from .modules.module_singlepoint import Singlepoint, newSinglepoint, ZeroTheory, ScriptTheory, Singlepoint_fragments, \
    Singlepoint_theories, Singlepoint_fragments_and_theories, Singlepoint_reaction, ReactionEnergy

# Parallel
import openmmqmmm.functions.functions_parallel
from .functions.functions_parallel import Job_parallel, Simple_parallel

Singlepoint_parallel = Job_parallel

# Freq
from .modules.module_freq import AnFreq, NumFreq, approximate_full_Hessian_from_smaller, calc_rotational_constants, \
    get_dominant_atoms_in_mode, write_normalmode, wigner_distribution, write_hessian, read_hessian

# Constants
import openmmqmmm.constants

# functions related to electronic structure
import openmmqmmm.functions.functions_elstructure
from .functions.functions_elstructure import read_cube, write_cube, write_cube_diff, diffdens_tool, \
    create_cubefile_from_orbfile, diffdens_of_cubefiles, \
    NOCV_density_ORCA, difference_density_ORCA, NOCV_Multiwfn, write_cube_sum, write_cube_product, \
    create_density_from_orb, make_molden_file, \
    diagonalize_DM_AO, diagonalize_DM, DM_AO_to_MO, DM_AO_to_MO, DM_MO_to_AO, select_space_from_occupations, \
    select_indices_from_occupations, ASH_write_integralfile, \
    density_sensitivity_metric

# multiwfn interface
import openmmqmmm.interfaces.interface_multiwfn
from .interfaces.interface_multiwfn import multiwfn_run

# # QMcode interfaces
from .interfaces.interface_ORCA import ORCATheory, counterpoise_calculation_ORCA, ORCA_External_Optimizer, \
    run_orca_plot, MolecularOrbitalGrab, \
    run_orca_mapspc, make_molden_file_ORCA, grab_coordinates_from_ORCA_output, ICE_WF_CFG_CI_size, orca_frag_guess, \
    orblocfind, ORCAfinalenergygrab, \
    read_ORCA_json_file, write_ORCA_json_file, create_GBW_from_json_file, create_ORCA_json_file, \
    get_densities_from_ORCA_json, grab_ORCA_wfn, \
    new_ORCA_natorbsfile_from_density, ORCA_orbital_setup, create_ORCA_FCIDUMP, print_gradient_in_ORCAformat
import openmmqmmm.interfaces.interface_ORCA

from .interfaces.interface_openbabel import OpenBabelTheory, pdb_to_smiles, mol_to_pdb, sdf_to_pdb, \
    writepdb_with_connectivity, \
    xyz_to_pdb_with_connectivity

# MM: external and internal
from .interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD, OpenMM_MDclass, OpenMM_Opt, OpenMM_Modeller, \
    OpenMM_box_equilibration, write_nonbonded_FF_for_ligand, solvate_small_molecule, small_molecule_parameterizer, \
    OpenMM_metadynamics, OpenMM_MD_plumed, Gentle_warm_up_MD, check_gradient_for_bad_atoms, \
    get_free_energy_from_biasfiles, \
    free_energy_from_bias_array, metadynamics_plot_data, merge_pdb_files

# General aliases
MolecularDynamics = OpenMM_MD
MetaDynamics = OpenMM_metadynamics

# TODO: Temporary aliases, to be deleted
OpenMM_box_relaxation = OpenMM_box_equilibration
small_molecule_parameterizor = small_molecule_parameterizer

from .modules.module_MM import NonBondedTheory, UFFdict, UFF_modH_dict, LJCoulpy, coulombcharge, LennardJones, \
    LJCoulombv2, LJCoulomb, MMforcefield_read
# MDtraj
from .interfaces.interface_mdtraj import MDtraj_imagetraj, MDtraj_slice, MDtraj_RMSF, MDtraj_RMSD, MDtraj_coord_analyze

# Theory, Numgrad
from .modules.module_theory import Theory, QMTheory, NumGradclass, MECPGradclass

# QM/MM
from .modules.module_QMMM import QMMMTheory, actregiondefine, read_charges_from_psf, compute_decomposed_QM_MM_energy

# geomeTRIC interface
from .interfaces.interface_geometric_new import geomeTRICOptimizer, GeomeTRICOptimizerClass

Optimizer = geomeTRICOptimizer
Opt = geomeTRICOptimizer

# Plotting
import openmmqmmm.modules.module_plotting
from .modules.module_plotting import reactionprofile_plot, contourplot, volumeplot, plot_Spectrum, MOplot_vertical, \
    ASH_plot

# Initialize settings
import openmmqmmm.settings_ash

# Print header
import openmmqmmm.ash_header

openmmqmmm.ash_header.print_header()

# Exit command (footer)
if openmmqmmm.settings_ash.settings_dict["print_exit_footer"] is True:
    atexit.register(openmmqmmm.ash_header.print_footer)
    if openmmqmmm.settings_ash.settings_dict["print_full_timings"] is True:
        atexit.register(openmmqmmm.ash_header.print_timings)
