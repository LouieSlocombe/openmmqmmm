"""Run mechanically embedded ML/MM dynamics for complete molecules in a PDB system."""

import argparse

from openmmqmmm import Fragment, MLMMTheory, OpenMMTheory, configure_logging, openmm_md, single_point


def main() -> None:
    """Build a mixed potential and run a short trajectory with user-supplied parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdbfile", help="Prepared PDB with the same atom order as the force-field topology")
    parser.add_argument("--forcefield", nargs="+", required=True, help="OpenMM XML files for the complete system")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--ml-atoms", nargs="+", type=int, help="Zero-based topology atom indices")
    selection.add_argument(
        "--ml-residues", nargs="+", type=int, help="Zero-based topology residue indices (not PDB residue IDs)"
    )
    parser.add_argument("--potential", default="ani2x", help="Installed OpenMM-ML potential name (default: ani2x)")
    parser.add_argument("--platform", default="CPU", help="OpenMM platform (default: CPU)")
    parser.add_argument("--device", help="Optional ML backend device, for example cpu or cuda")
    parser.add_argument("--model-path", help="Local checkpoint (use --potential mace-polar for a PolarMACE model)")
    parser.add_argument("--precision", choices=["single", "double"], help="ML backend numerical precision")
    parser.add_argument("--ml-charge", type=int, help="Total ML-region charge for charge-aware models")
    parser.add_argument("--ml-multiplicity", type=int, help="ML-region spin multiplicity for spin-aware models")
    parser.add_argument("--periodic", action="store_true", help="Use the PDB's periodic box and PME")
    parser.add_argument("--steps", type=int, default=1000, help="Number of MD steps (default: 1000)")
    parser.add_argument("--timestep", type=float, default=0.0005, help="Timestep in ps (default: 0.0005)")
    parser.add_argument("--temperature", type=float, default=300, help="Temperature in K (default: 300)")
    args = parser.parse_args()
    if args.steps <= 0 or args.timestep <= 0 or args.temperature <= 0:
        parser.error("steps, timestep and temperature must be positive")

    configure_logging()
    fragment = Fragment(pdbfile=args.pdbfile)
    mm = OpenMMTheory(
        xmlfiles=args.forcefield,
        pdbfile=args.pdbfile,
        periodic=args.periodic,
        platform=args.platform,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    mlatoms = args.ml_atoms
    if args.ml_residues is not None:
        residues = list(mm.topology.residues())
        if any(index < 0 or index >= len(residues) for index in args.ml_residues):
            parser.error(f"ML residue indices must be between 0 and {len(residues) - 1}")
        if len(set(args.ml_residues)) != len(args.ml_residues):
            parser.error("ML residue indices must be unique")
        mlatoms = [atom.index for index in args.ml_residues for atom in residues[index].atoms()]

    system_kwargs = {
        key: value
        for key, value in (
            ("device", args.device),
            ("precision", args.precision),
            ("charge", args.ml_charge),
            ("multiplicity", args.ml_multiplicity),
        )
        if value is not None
    }
    mlmm = MLMMTheory(
        mm_theory=mm,
        mlatoms=mlatoms,
        potential=args.potential,
        potential_kwargs={"modelPath": args.model_path} if args.model_path is not None else None,
        system_kwargs=system_kwargs,
    )
    result = single_point(theory=mlmm, fragment=fragment)
    print(f"Initial ML/MM energy: {result.energy:.8f} Eh")
    openmm_md(
        fragment=fragment,
        theory=mlmm,
        timestep=args.timestep,
        simulation_steps=args.steps,
        temperature=args.temperature,
        traj_frequency=min(100, args.steps),
        trajfilename="mlmm_trajectory",
    )


if __name__ == "__main__":
    main()
