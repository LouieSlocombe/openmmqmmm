"""
ASH - A MULTISCALE MODELLING PROGRAM
R. Bjornsson

Trimmed distribution: ORCA + OpenMM QM/MM functionality for biomolecular calculations.
"""

# Results dataclass
from .modules.module_results import ASH_Results, read_results_from_file

# Fragment class and coordinate functions
from .modules.module_coords import (
    get_molecules_from_trajectory,
    write_pdbfile,
    Fragment,
    read_xyzfile,
    write_xyzfile,
    read_ambercoordinates,
    read_gromacsfile,
    split_multimolxyzfile,
    distance_between_atoms,
    angle_between_atoms,
    dihedral_between_atoms,
)
from .modules.module_coords import (
    getwaterconstraintslist,
    QMregionfragexpand,
    QMPC_fragexpand,
    read_xyzfiles,
    Reaction,
    define_XH_constraints,
    simple_get_water_constraints,
    print_internal_coordinate_table,
    flexible_align_pdb,
    flexible_align_xyz,
    flexible_align,
    insert_solute_into_solvent,
    nuc_nuc_repulsion,
    calculate_RMSD,
)

# Singlepoint
from .modules.module_singlepoint import (
    Singlepoint,
    ZeroTheory,
    Singlepoint_fragments,
    Singlepoint_theories,
    Singlepoint_fragments_and_theories,
    Singlepoint_reaction,
    ReactionEnergy,
)

# Parallel
from .functions.functions_parallel import Job_parallel, Simple_parallel

# Freq
from .modules.module_freq import (
    AnFreq,
    NumFreq,
    approximate_full_Hessian_from_smaller,
    calc_rotational_constants,
    write_hessian,
    read_hessian,
)

# Constants

# # QMcode interfaces
from .interfaces.interface_ORCA import ORCATheory, ORCA_External_Optimizer

from .interfaces.interface_openbabel import (
    pdb_to_smiles,
    mol_to_pdb,
    sdf_to_pdb,
    writepdb_with_connectivity,
    xyz_to_pdb_with_connectivity,
)

from .interfaces.interface_OpenMM import (
    OpenMMTheory,
    OpenMM_MD,
    OpenMM_MDclass,
    OpenMM_Opt,
    OpenMM_Modeller,
    OpenMM_box_equilibration,
    solvate_small_molecule,
    small_molecule_parameterizer,
    OpenMM_metadynamics,
    OpenMM_MD_plumed,
    Gentle_warm_up_MD,
    check_gradient_for_bad_atoms,
    get_free_energy_from_biasfiles,
    free_energy_from_bias_array,
    metadynamics_plot_data,
    merge_pdb_files,
)

# General aliases
MolecularDynamics = OpenMM_MD
MetaDynamics = OpenMM_metadynamics

# MDtraj
from .interfaces.interface_mdtraj import MDtraj_imagetraj, MDtraj_slice, MDtraj_RMSF, MDtraj_RMSD, MDtraj_coord_analyze

# Numerical gradient
from .modules.module_theory import NumGradclass

# QM/MM
from .modules.module_QMMM import QMMMTheory, actregiondefine, read_charges_from_psf, compute_decomposed_QM_MM_energy

# geomeTRIC interface
from .interfaces.interface_geometric_new import geomeTRICOptimizer, GeomeTRICOptimizerClass

Optimizer = geomeTRICOptimizer
Opt = geomeTRICOptimizer

# Plotting
from .modules.module_plotting import ASH_plot

# Public API — the only names star-imports provide
__all__ = [
    # Results
    "ASH_Results",
    "read_results_from_file",
    # Fragment and coordinate functions
    "Fragment",
    "Reaction",
    "read_xyzfile",
    "read_xyzfiles",
    "write_xyzfile",
    "write_pdbfile",
    "read_ambercoordinates",
    "read_gromacsfile",
    "split_multimolxyzfile",
    "get_molecules_from_trajectory",
    "distance_between_atoms",
    "angle_between_atoms",
    "dihedral_between_atoms",
    "getwaterconstraintslist",
    "QMregionfragexpand",
    "QMPC_fragexpand",
    "define_XH_constraints",
    "simple_get_water_constraints",
    "print_internal_coordinate_table",
    "flexible_align",
    "flexible_align_pdb",
    "flexible_align_xyz",
    "insert_solute_into_solvent",
    "nuc_nuc_repulsion",
    "calculate_RMSD",
    # Single-point
    "Singlepoint",
    "ZeroTheory",
    "Singlepoint_fragments",
    "Singlepoint_theories",
    "Singlepoint_fragments_and_theories",
    "Singlepoint_reaction",
    "ReactionEnergy",
    # Parallel
    "Job_parallel",
    "Simple_parallel",
    # Frequencies
    "AnFreq",
    "NumFreq",
    "approximate_full_Hessian_from_smaller",
    "calc_rotational_constants",
    "write_hessian",
    "read_hessian",
    # ORCA
    "ORCATheory",
    "ORCA_External_Optimizer",
    # openbabel helpers
    "pdb_to_smiles",
    "mol_to_pdb",
    "sdf_to_pdb",
    "writepdb_with_connectivity",
    "xyz_to_pdb_with_connectivity",
    # OpenMM
    "OpenMMTheory",
    "OpenMM_MD",
    "OpenMM_MDclass",
    "OpenMM_Opt",
    "OpenMM_Modeller",
    "OpenMM_box_equilibration",
    "OpenMM_metadynamics",
    "OpenMM_MD_plumed",
    "Gentle_warm_up_MD",
    "solvate_small_molecule",
    "small_molecule_parameterizer",
    "check_gradient_for_bad_atoms",
    "get_free_energy_from_biasfiles",
    "free_energy_from_bias_array",
    "metadynamics_plot_data",
    "merge_pdb_files",
    "MolecularDynamics",
    "MetaDynamics",
    # MDtraj
    "MDtraj_imagetraj",
    "MDtraj_slice",
    "MDtraj_RMSF",
    "MDtraj_RMSD",
    "MDtraj_coord_analyze",
    # Numerical gradient
    "NumGradclass",
    # QM/MM
    "QMMMTheory",
    "actregiondefine",
    "read_charges_from_psf",
    "compute_decomposed_QM_MM_energy",
    # geomeTRIC optimizer
    "geomeTRICOptimizer",
    "GeomeTRICOptimizerClass",
    "Optimizer",
    "Opt",
    # Plotting
    "ASH_plot",
]
