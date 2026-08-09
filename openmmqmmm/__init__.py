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

from .functions.functions_general import blankline, BC, listdiff, print_time_rel, pygrep, \
    printdebug, read_intlist_from_file, writelisttofile, ashexit, natural_sort

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
    write_xyzfile, read_ambercoordinates, read_gromacsfile, split_multimolxyzfile, \
    distance_between_atoms, \
    angle_between_atoms, dihedral_between_atoms
from .modules.module_coords import getwaterconstraintslist, \
    QMregionfragexpand, QMPC_fragexpand, read_xyzfiles, Reaction, define_XH_constraints, \
    simple_get_water_constraints, print_internal_coordinate_table, \
    flexible_align_pdb, flexible_align_xyz, flexible_align, insert_solute_into_solvent, nuc_nuc_repulsion, \
    calculate_RMSD

# Singlepoint
import openmmqmmm.modules.module_singlepoint
from .modules.module_singlepoint import Singlepoint, ZeroTheory, Singlepoint_fragments, \
    Singlepoint_theories, Singlepoint_fragments_and_theories, Singlepoint_reaction, ReactionEnergy

# Parallel
import openmmqmmm.functions.functions_parallel
from .functions.functions_parallel import Job_parallel, Simple_parallel

# Freq
from .modules.module_freq import AnFreq, NumFreq, approximate_full_Hessian_from_smaller, calc_rotational_constants, \
    write_hessian, read_hessian

# Constants
import openmmqmmm.constants

# # QMcode interfaces
from .interfaces.interface_ORCA import ORCATheory, ORCA_External_Optimizer
import openmmqmmm.interfaces.interface_ORCA

from .interfaces.interface_openbabel import pdb_to_smiles, mol_to_pdb, sdf_to_pdb, \
    writepdb_with_connectivity, \
    xyz_to_pdb_with_connectivity

# MM: external and internal
from .interfaces.interface_OpenMM import OpenMMTheory, OpenMM_MD, OpenMM_MDclass, OpenMM_Opt, OpenMM_Modeller, \
    OpenMM_box_equilibration, solvate_small_molecule, small_molecule_parameterizer, \
    OpenMM_metadynamics, OpenMM_MD_plumed, Gentle_warm_up_MD, check_gradient_for_bad_atoms, \
    get_free_energy_from_biasfiles, \
    free_energy_from_bias_array, metadynamics_plot_data, merge_pdb_files

# General aliases
MolecularDynamics = OpenMM_MD
MetaDynamics = OpenMM_metadynamics

# MDtraj
from .interfaces.interface_mdtraj import MDtraj_imagetraj, MDtraj_slice, MDtraj_RMSF, MDtraj_RMSD, MDtraj_coord_analyze

# Theory, Numgrad
from .modules.module_theory import Theory, QMTheory, NumGradclass

# QM/MM
from .modules.module_QMMM import QMMMTheory, actregiondefine, read_charges_from_psf, compute_decomposed_QM_MM_energy

# geomeTRIC interface
from .interfaces.interface_geometric_new import geomeTRICOptimizer, GeomeTRICOptimizerClass

Optimizer = geomeTRICOptimizer
Opt = geomeTRICOptimizer

# Plotting
from .modules.module_plotting import ASH_plot

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
