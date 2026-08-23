import inspect

import pytest

import openmmqmmm
import openmmqmmm.openmm

PUBLIC_NAMESPACES = (openmmqmmm, openmmqmmm.openmm)
EXPECTED_PUBLIC_EXPORTS = {
    "openmmqmmm": frozenset(
        {
            "__version__",
            "ExternalProgramError",
            "FileFormatError",
            "Fragment",
            "GeometricOptimizer",
            "InputError",
            "InternalError",
            "MissingDependencyError",
            "MolecularDynamicsEngine",
            "NumGrad",
            "ORCATheory",
            "OpenMMQMMMCalculator",
            "OpenMMQMMMError",
            "OpenMMTheory",
            "QMMMTheory",
            "RPMDPotentialExport",
            "Reaction",
            "Results",
            "ZeroTheory",
            "analytic_frequencies",
            "angle_between_atoms",
            "approximate_full_hessian_from_smaller",
            "calc_rotational_constants",
            "calculate_rmsd",
            "check_gradient_for_bad_atoms",
            "compute_decomposed_qm_mm_energy",
            "configure_logging",
            "define_active_region",
            "define_xh_constraints",
            "dihedral_between_atoms",
            "distance_between_atoms",
            "expand_qm_pc_region",
            "expand_qm_region",
            "export_rpmd_potential",
            "flexible_align",
            "flexible_align_pdb",
            "flexible_align_xyz",
            "gentle_warmup_md",
            "get_molecules_from_trajectory",
            "get_water_constraints",
            "insert_solute_into_solvent",
            "job_parallel",
            "mdtraj_image_trajectory",
            "mdtraj_rmsf",
            "merge_pdb_files",
            "modeller_from_topology",
            "nuc_nuc_repulsion",
            "numerical_frequencies",
            "openmm_box_equilibration",
            "openmm_md",
            "openmm_md_plumed",
            "openmm_minimize",
            "openmm_modeller",
            "optimize_geometry",
            "orca_external_optimizer",
            "print_internal_coordinate_table",
            "reaction_energy",
            "read_ambercoordinates",
            "read_charges_from_psf",
            "read_gromacsfile",
            "read_hessian",
            "read_results_from_file",
            "read_xyzfile",
            "read_xyzfiles",
            "require",
            "simple_get_water_constraints",
            "single_point",
            "single_point_fragments",
            "single_point_fragments_and_theories",
            "single_point_reaction",
            "single_point_theories",
            "solvate_small_molecule",
            "split_multimolxyzfile",
            "write_hessian",
            "write_pdbfile",
            "write_xyzfile",
        }
    ),
    "openmmqmmm.openmm": frozenset(
        {
            "ForceReporter",
            "MolecularDynamicsEngine",
            "OpenMMTheory",
            "RPMDPotentialExport",
            "check_gradient_for_bad_atoms",
            "clean_up_constraints_list",
            "diff_wrap_box_coords",
            "export_rpmd_potential",
            "find_alternate_locations_residues",
            "gentle_warmup_md",
            "merge_pdb_files",
            "modeller_from_topology",
            "openmm_add_bonds_to_topology",
            "openmm_box_equilibration",
            "openmm_md",
            "openmm_md_plumed",
            "openmm_minimize",
            "openmm_modeller",
            "print_current_step_info",
            "print_systemsize",
            "read_npt_statefile",
            "solvate_small_molecule",
            "write_pdbfile_openmm_topology",
            "write_pdbxfile_openmm_topology",
            "write_xmlfile_nonbonded",
        }
    ),
}


def _exported_objects(predicate):
    return [
        pytest.param(namespace, name, id=f"{namespace.__name__}.{name}")
        for namespace in PUBLIC_NAMESPACES
        for name in namespace.__all__
        if predicate(getattr(namespace, name, None))
        and getattr(getattr(namespace, name), "__module__", "").startswith("openmmqmmm")
    ]


EXPORTED_CLASSES = _exported_objects(inspect.isclass)
EXPORTED_FUNCTIONS = _exported_objects(inspect.isfunction)


def _own_public_methods(cls):
    return [
        (name, func)
        for name, func in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_") and func.__module__.startswith("openmmqmmm")
    ]


@pytest.mark.parametrize(("namespace", "class_name"), EXPORTED_CLASSES)
def test_exported_classes_are_documented(namespace, class_name):
    cls = getattr(namespace, class_name)
    assert inspect.getdoc(cls), f"{class_name} has no class docstring"


@pytest.mark.parametrize(("namespace", "class_name"), EXPORTED_CLASSES)
def test_public_methods_are_documented(namespace, class_name):
    cls = getattr(namespace, class_name)
    undocumented = [name for name, func in _own_public_methods(cls) if not inspect.getdoc(func)]
    assert not undocumented, f"{class_name} has undocumented public methods: {undocumented}"


@pytest.mark.parametrize(("namespace", "name"), EXPORTED_FUNCTIONS)
def test_exported_functions_are_documented(namespace, name):
    assert inspect.getdoc(getattr(namespace, name)), f"{name} has no docstring"


@pytest.mark.parametrize("namespace", PUBLIC_NAMESPACES, ids=lambda namespace: namespace.__name__)
def test_explicit_exports_are_unique_and_defined(namespace):
    assert len(namespace.__all__) == len(set(namespace.__all__))
    missing = [name for name in namespace.__all__ if not hasattr(namespace, name)]
    assert not missing, f"{namespace.__name__} exports undefined names: {missing}"

    actual = set(namespace.__all__)
    expected = EXPECTED_PUBLIC_EXPORTS[namespace.__name__]
    assert actual == expected, (
        f"{namespace.__name__} public API changed; missing={sorted(expected - actual)}, "
        f"unexpected={sorted(actual - expected)}"
    )
