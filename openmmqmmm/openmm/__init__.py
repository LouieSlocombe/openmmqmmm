"""OpenMM interface: OpenMMTheory, MD drivers, system preparation (modeller/solvation),
metadynamics and free-energy helpers.

Split across submodules for navigability; every name is re-exported here, so
`from openmmqmmm.openmm import X` keeps working exactly as before:

- `theory` — OpenMMTheory and the forces it manages
- `systemsetup` — modeller, solvation, minimization, ligand parameterization
- `md` — the MD engine and the molecular-dynamics drivers
- `metadynamics` — metadynamics, PLUMED and free-energy surfaces
"""

# Defined in coords but part of the OpenMM-facing API since 1.0: re-exported here so
# `from openmmqmmm.openmm import check_gradient_for_bad_atoms` keeps working.
from openmmqmmm.coords import check_gradient_for_bad_atoms
from openmmqmmm.openmm.md import (
    MolecularDynamicsEngine,
    create_cv_bias,
    diff_wrap_box_coords,
    gentle_warmup_md,
    openmm_box_equilibration,
    openmm_md,
    print_current_step_info,
    read_npt_statefile,
)
from openmmqmmm.openmm.metadynamics import (
    free_energy_from_bias_array,
    get_free_energy_from_biasfiles,
    metadynamics_plot_data,
    openmm_md_plumed,
    openmm_metadynamics,
)
from openmmqmmm.openmm.systemsetup import (
    calc_nonbonding_energy_exceptions,
    create_sys_and_check_14_scaling_nonbonding,
    find_alternate_locations_residues,
    merge_pdb_files,
    openmm_add_bonds_to_topology,
    openmm_minimize,
    openmm_modeller,
    print_systemsize,
    small_molecule_parameterizer,
    solvate_small_molecule,
    write_pdbfile_openmm_topology,
    write_pdbxfile_openmm_topology,
    write_xmlfile_parmed,
)
from openmmqmmm.openmm.theory import (
    ForceReporter,
    OpenMMTheory,
    clean_up_constraints_list,
    create_cnb,
    write_xmlfile_nonbonded,
)

__all__ = [
    "ForceReporter",
    "MolecularDynamicsEngine",
    "OpenMMTheory",
    "calc_nonbonding_energy_exceptions",
    "check_gradient_for_bad_atoms",
    "clean_up_constraints_list",
    "create_cnb",
    "create_cv_bias",
    "create_sys_and_check_14_scaling_nonbonding",
    "diff_wrap_box_coords",
    "find_alternate_locations_residues",
    "free_energy_from_bias_array",
    "gentle_warmup_md",
    "get_free_energy_from_biasfiles",
    "merge_pdb_files",
    "metadynamics_plot_data",
    "openmm_add_bonds_to_topology",
    "openmm_box_equilibration",
    "openmm_md",
    "openmm_md_plumed",
    "openmm_metadynamics",
    "openmm_minimize",
    "openmm_modeller",
    "print_current_step_info",
    "print_systemsize",
    "read_npt_statefile",
    "small_molecule_parameterizer",
    "solvate_small_molecule",
    "write_pdbfile_openmm_topology",
    "write_pdbxfile_openmm_topology",
    "write_xmlfile_nonbonded",
    "write_xmlfile_parmed",
]
