"""
ASH - A MULTISCALE MODELLING PROGRAM
R. Bjornsson

Trimmed distribution: ORCA + OpenMM QM/MM functionality for biomolecular calculations.
"""

import logging as _logging

# Exceptions
from .exceptions import (
    ExternalProgramError,
    FileFormatError,
    InputError,
    InternalError,
    MissingDependencyError,
    OpenMMQMMMError,
    require,
)

# Logging setup helper
from .utils import configure_logging

# Results dataclass
# Parallel
from .parallel import Job_parallel, Simple_parallel

# MDtraj
# geomeTRIC interface
from .geometric import GeomeTRICOptimizerClass, geomeTRICOptimizer
from .mdtraj import MDtraj_coord_analyze, MDtraj_imagetraj, MDtraj_RMSD, MDtraj_RMSF, MDtraj_slice
from .openbabel import (
    mol_to_pdb,
    pdb_to_smiles,
    sdf_to_pdb,
    writepdb_with_connectivity,
    xyz_to_pdb_with_connectivity,
)
from .openmm import (
    Gentle_warm_up_MD,
    OpenMM_box_equilibration,
    OpenMM_MD,
    OpenMM_MD_plumed,
    OpenMM_MDclass,
    OpenMM_metadynamics,
    OpenMM_Modeller,
    OpenMM_Opt,
    OpenMMTheory,
    check_gradient_for_bad_atoms,
    free_energy_from_bias_array,
    get_free_energy_from_biasfiles,
    merge_pdb_files,
    metadynamics_plot_data,
    small_molecule_parameterizer,
    solvate_small_molecule,
)

# Constants
# # QMcode interfaces
from .orca import ORCA_External_Optimizer, ORCATheory

# Fragment class and coordinate functions
from .coords import (
    Fragment,
    QMPC_fragexpand,
    QMregionfragexpand,
    Reaction,
    angle_between_atoms,
    calculate_RMSD,
    define_XH_constraints,
    dihedral_between_atoms,
    distance_between_atoms,
    flexible_align,
    flexible_align_pdb,
    flexible_align_xyz,
    get_molecules_from_trajectory,
    getwaterconstraintslist,
    insert_solute_into_solvent,
    nuc_nuc_repulsion,
    print_internal_coordinate_table,
    read_ambercoordinates,
    read_gromacsfile,
    read_xyzfile,
    read_xyzfiles,
    simple_get_water_constraints,
    split_multimolxyzfile,
    write_pdbfile,
    write_xyzfile,
)

# Freq
from .freq import (
    AnFreq,
    NumFreq,
    approximate_full_Hessian_from_smaller,
    calc_rotational_constants,
    read_hessian,
    write_hessian,
)

# Plotting
from .plotting import ASH_plot

# QM/MM
from .qmmm import QMMMTheory, actregiondefine, compute_decomposed_QM_MM_energy, read_charges_from_psf
from .results import ASH_Results, read_results_from_file

# Singlepoint
from .singlepoint import (
    ReactionEnergy,
    Singlepoint,
    Singlepoint_fragments,
    Singlepoint_fragments_and_theories,
    Singlepoint_reaction,
    Singlepoint_theories,
    ZeroTheory,
)

# Numerical gradient
from .numgrad import NumGradclass

# Library convention: silent unless the application configures logging.
# configure_logging() sets up ASH-style console output in one call.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# General aliases
MolecularDynamics = OpenMM_MD
MetaDynamics = OpenMM_metadynamics
Optimizer = geomeTRICOptimizer
Opt = geomeTRICOptimizer

# Public API — the only names star-imports provide
__all__ = [
    # Results
    "ASH_Results",
    # Plotting
    "ASH_plot",
    # Frequencies
    "AnFreq",
    "ExternalProgramError",
    "FileFormatError",
    # Fragment and coordinate functions
    "Fragment",
    "Gentle_warm_up_MD",
    "GeomeTRICOptimizerClass",
    "InputError",
    "InternalError",
    # Parallel
    "Job_parallel",
    "MDtraj_RMSD",
    "MDtraj_RMSF",
    "MDtraj_coord_analyze",
    # MDtraj
    "MDtraj_imagetraj",
    "MDtraj_slice",
    "MetaDynamics",
    "MissingDependencyError",
    "MolecularDynamics",
    "NumFreq",
    # Numerical gradient
    "NumGradclass",
    # ORCA
    "ORCATheory",
    "ORCA_External_Optimizer",
    # Exceptions
    "OpenMMQMMMError",
    # OpenMM
    "OpenMMTheory",
    "OpenMM_MD",
    "OpenMM_MD_plumed",
    "OpenMM_MDclass",
    "OpenMM_Modeller",
    "OpenMM_Opt",
    "OpenMM_box_equilibration",
    "OpenMM_metadynamics",
    "Opt",
    "Optimizer",
    # QM/MM
    "QMMMTheory",
    "QMPC_fragexpand",
    "QMregionfragexpand",
    "Reaction",
    "ReactionEnergy",
    "Simple_parallel",
    # Single-point
    "Singlepoint",
    "Singlepoint_fragments",
    "Singlepoint_fragments_and_theories",
    "Singlepoint_reaction",
    "Singlepoint_theories",
    "ZeroTheory",
    "actregiondefine",
    "angle_between_atoms",
    "approximate_full_Hessian_from_smaller",
    "calc_rotational_constants",
    "calculate_RMSD",
    "check_gradient_for_bad_atoms",
    "compute_decomposed_QM_MM_energy",
    "configure_logging",
    "define_XH_constraints",
    "dihedral_between_atoms",
    "distance_between_atoms",
    "flexible_align",
    "flexible_align_pdb",
    "flexible_align_xyz",
    "free_energy_from_bias_array",
    # geomeTRIC optimizer
    "geomeTRICOptimizer",
    "get_free_energy_from_biasfiles",
    "get_molecules_from_trajectory",
    "getwaterconstraintslist",
    "insert_solute_into_solvent",
    "merge_pdb_files",
    "metadynamics_plot_data",
    "mol_to_pdb",
    "nuc_nuc_repulsion",
    # openbabel helpers
    "pdb_to_smiles",
    "print_internal_coordinate_table",
    "read_ambercoordinates",
    "read_charges_from_psf",
    "read_gromacsfile",
    "read_hessian",
    "read_results_from_file",
    "read_xyzfile",
    "read_xyzfiles",
    "require",
    "sdf_to_pdb",
    "simple_get_water_constraints",
    "small_molecule_parameterizer",
    "solvate_small_molecule",
    "split_multimolxyzfile",
    "write_hessian",
    "write_pdbfile",
    "write_xyzfile",
    "writepdb_with_connectivity",
    "xyz_to_pdb_with_connectivity",
]
