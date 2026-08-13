"""openmmqmmm: ORCA + OpenMM QM/MM for biomolecular calculations.

Derived from the ASH multiscale modelling program by R. Bjornsson.
"""

import logging as _logging
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

# Fragment class and coordinate functions
from .coords import (
    Fragment,
    Reaction,
    angle_between_atoms,
    calculate_rmsd,
    define_xh_constraints,
    dihedral_between_atoms,
    distance_between_atoms,
    expand_qm_pc_region,
    expand_qm_region,
    flexible_align,
    flexible_align_pdb,
    flexible_align_xyz,
    get_molecules_from_trajectory,
    get_water_constraints,
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

# Freq
from .freq import (
    analytic_frequencies,
    approximate_full_hessian_from_smaller,
    calc_rotational_constants,
    numerical_frequencies,
    read_hessian,
    write_hessian,
)

# MDtraj
# geomeTRIC interface
from .geometric import GeometricOptimizer, optimize_geometry
from .mdtraj import mdtraj_coord_analyze, mdtraj_image_trajectory, mdtraj_rmsd, mdtraj_rmsf, mdtraj_slice

# Numerical gradient
from .numgrad import NumGrad
from .openbabel import (
    mol_to_pdb,
    pdb_to_smiles,
    sdf_to_pdb,
    write_pdb_with_connectivity,
    xyz_to_pdb_with_connectivity,
)
from .openmm import (
    MolecularDynamicsEngine,
    OpenMMTheory,
    check_gradient_for_bad_atoms,
    gentle_warmup_md,
    merge_pdb_files,
    openmm_box_equilibration,
    openmm_md,
    openmm_md_plumed,
    openmm_minimize,
    openmm_modeller,
    small_molecule_parameterizer,
    solvate_small_molecule,
)

# Constants
# # QMcode interfaces
from .orca import ORCATheory, orca_external_optimizer

# Results dataclass
# Parallel
from .parallel import job_parallel

# Plotting
from .plotting import Plot

# QM/MM
from .qmmm import QMMMTheory, compute_decomposed_qm_mm_energy, define_active_region, read_charges_from_psf
from .results import Results, read_results_from_file

# Singlepoint
from .singlepoint import (
    ZeroTheory,
    reaction_energy,
    single_point,
    single_point_fragments,
    single_point_fragments_and_theories,
    single_point_reaction,
    single_point_theories,
)

# Logging setup helper
from .utils import configure_logging

# Library convention: silent unless the application configures logging.
# configure_logging() sets up the classic console output in one call.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# Version comes from the package metadata; pyproject.toml is the single source of truth.
try:
    __version__ = _get_version("openmmqmmm")
except _PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

# Public API — the only names star-imports provide
__all__ = [
    "__version__",
    # Core classes
    "Fragment",
    "GeometricOptimizer",
    "MolecularDynamicsEngine",
    "NumGrad",
    "ORCATheory",
    "OpenMMTheory",
    "Plot",
    "QMMMTheory",
    "Reaction",
    "Results",
    "ZeroTheory",
    # Exceptions and setup
    "ExternalProgramError",
    "FileFormatError",
    "InputError",
    "InternalError",
    "MissingDependencyError",
    "OpenMMQMMMError",
    "configure_logging",
    "require",
    # Job functions
    "analytic_frequencies",
    "numerical_frequencies",
    "optimize_geometry",
    "orca_external_optimizer",
    "reaction_energy",
    "single_point",
    "single_point_fragments",
    "single_point_fragments_and_theories",
    "single_point_reaction",
    "single_point_theories",
    # Parallel
    "job_parallel",
    # OpenMM workflows
    "gentle_warmup_md",
    "openmm_box_equilibration",
    "openmm_md",
    "openmm_md_plumed",
    "openmm_minimize",
    "openmm_modeller",
    "check_gradient_for_bad_atoms",
    "merge_pdb_files",
    "small_molecule_parameterizer",
    "solvate_small_molecule",
    # QM/MM helpers
    "compute_decomposed_qm_mm_energy",
    "define_active_region",
    "expand_qm_pc_region",
    "expand_qm_region",
    "read_charges_from_psf",
    # Coordinates and fragments
    "angle_between_atoms",
    "calculate_rmsd",
    "define_xh_constraints",
    "dihedral_between_atoms",
    "distance_between_atoms",
    "flexible_align",
    "flexible_align_pdb",
    "flexible_align_xyz",
    "get_molecules_from_trajectory",
    "get_water_constraints",
    "insert_solute_into_solvent",
    "nuc_nuc_repulsion",
    "print_internal_coordinate_table",
    "read_ambercoordinates",
    "read_gromacsfile",
    "read_xyzfile",
    "read_xyzfiles",
    "simple_get_water_constraints",
    "split_multimolxyzfile",
    "write_pdbfile",
    "write_xyzfile",
    # Frequencies
    "approximate_full_hessian_from_smaller",
    "calc_rotational_constants",
    "read_hessian",
    "write_hessian",
    # Results I/O
    "read_results_from_file",
    # Trajectory analysis (mdtraj)
    "mdtraj_coord_analyze",
    "mdtraj_image_trajectory",
    "mdtraj_rmsd",
    "mdtraj_rmsf",
    "mdtraj_slice",
    # Ligand conversion (openbabel)
    "mol_to_pdb",
    "pdb_to_smiles",
    "sdf_to_pdb",
    "write_pdb_with_connectivity",
    "xyz_to_pdb_with_connectivity",
]
