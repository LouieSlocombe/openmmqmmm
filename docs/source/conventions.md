# Conventions

## Naming

Job functions are snake_case ({func}`~openmmqmmm.single_point`,
{func}`~openmmqmmm.optimize_geometry`, {func}`~openmmqmmm.numerical_frequencies`,
{func}`~openmmqmmm.openmm_md`). Classes are CapWords
({class}`~openmmqmmm.ORCATheory`, {class}`~openmmqmmm.OpenMMTheory`,
{class}`~openmmqmmm.QMMMTheory`, {class}`~openmmqmmm.Fragment`,
{class}`~openmmqmmm.Results`). Keyword arguments are snake_case (`grad=`, `active_region=`,
`num_grad=`).

Where a name comes from an external protocol it keeps that protocol's spelling instead —
OpenMM's `enforcePeriodicBox` and `biasFactor`, geomeTRIC's `logIni`. Those are call-site
compatibility, not style.

## Files a run leaves behind

`.frag`
: A fragment written by `Fragment.print_system`, and read back through `Fragment(fragfile=)`.
  It round-trips coordinates, charge, multiplicity and connectivity, which no standard
  coordinate format does.

`results_*.json`
: One per job function — `results_singlepoint.json`, `results_optimizer.json`,
  `results_numfreq.json` and so on. See {doc}`guide/jobs` for the full list, and
  {func}`~openmmqmmm.read_results_from_file` to read one back.

## Units

Hartree for energies, Hartree/Bohr for gradients, Angstrom for coordinates, cm⁻¹ for
frequencies, kelvin for temperature, bar for pressure, picoseconds for MD time. The two
exceptions are both deliberate: {class}`~openmmqmmm.OpenMMQMMMCalculator` speaks eV and eV/Å
because ASE does, and PLUMED input is in PLUMED's own units.

## Importing

Importing the package is silent and side-effect free: no logging configuration, no files
read, no environment inspected. Call {func}`~openmmqmmm.configure_logging` to get output —
see {doc}`guide/output`.

Errors raise exceptions and never exit the interpreter, so the package composes inside a
larger script.

## Type annotations

The package ships inline type annotations and a `py.typed` marker, so a type checker follows
its signatures into your code. The docstrings are one-line summaries by design; the
annotation is where a parameter's type is stated, which is why the API reference keeps
annotations in the signature.

## Changes from ASH

Version 1.0 is the first release, and it is not a drop-in replacement for
[ASH](https://github.com/RagnarB83/ash).

The API was renamed: snake_case job functions, no import-time side effects, `logging`
instead of `print`, exceptions instead of process exit, and no `~/ash_user_settings.ini`. The
naming described above dates from here. ASH drop-in compatibility was abandoned deliberately.

The in-house ligand parameterization was also removed —
`small_molecule_parameterizer`, `write_xmlfile_parmed`,
`create_sys_and_check_14_scaling_nonbonding` and `calc_nonbonding_energy_exceptions` — in
favour of [forcefill](https://github.com/LouieSlocombe/forcefill). See
{doc}`guide/system_setup` for what replaced them.

The native OpenMM metadynamics driver and the multiwalker implementation were also removed;
{func}`~openmmqmmm.openmm_md_plumed` is the only biased-dynamics route
({doc}`guide/dynamics`).

## Upstream

This package is derived from [ASH](https://github.com/RagnarB83/ash) and reduced to the
ORCA + OpenMM QM/MM stack. If you are coming from ASH, expect the same physics under
different names. See {doc}`citing`.
