# Jobs and results

Every job function takes `theory=` and `fragment=`, returns a {class}`~openmmqmmm.Results`
object, and writes that object to JSON next to the script. Energies are in Hartree, gradients
in Hartree/Bohr, coordinates in Angstrom, frequencies in cm⁻¹.

## Single points

```python
from openmmqmmm import single_point

result = single_point(theory=theory, fragment=fragment)
result = single_point(theory=theory, fragment=fragment, grad=True)  # result.gradient too
```

Four variants cover the usual sweeps, each returning one `Results` holding all of them:

| Function | Sweeps over |
|---|---|
| {func}`~openmmqmmm.single_point_theories` | one fragment, several theories |
| {func}`~openmmqmmm.single_point_fragments` | one theory, several fragments |
| {func}`~openmmqmmm.single_point_fragments_and_theories` | every combination of both |
| {func}`~openmmqmmm.single_point_reaction` | the species of a {class}`~openmmqmmm.Reaction` |

A `Reaction` is fragments plus stoichiometry, so a reaction energy falls out of the sweep:

```python
from openmmqmmm import Reaction, single_point_reaction

reaction = Reaction(fragments=[reactant, product], stoichiometry=[-1, 1], unit="kcal/mol")
result = single_point_reaction(theory=theory, reaction=reaction)
print(result.reaction_energy)
```

{func}`~openmmqmmm.reaction_energy` does the same arithmetic on energies you already have.

## Geometry optimization

{func}`~openmmqmmm.optimize_geometry` drives [geomeTRIC](https://github.com/leeping/geomeTRIC)
and updates the fragment in place:

```python
from openmmqmmm import optimize_geometry

result = optimize_geometry(theory=theory, fragment=fragment, maxiter=250)
```

`coordsystem=`
: `"tric"` by default; geomeTRIC's other internal-coordinate systems are available, and
  `force_coordsystem=True` stops the automatic fallback to Cartesians for a system it thinks
  needs them.

`actatoms=` / `frozenatoms=`
: The active region, or its complement. For anything protein-sized this is what makes the
  optimization finish — see {doc}`qmmm`.

`constraints=` / `constraintsinputfile=`
: Bonds, angles, dihedrals and Cartesian freezes. `constrainvalue=True` holds them at a value
  you give rather than at their current one.

`ts_opt=True`, `hessian=`, `partial_hessian_atoms=`, `modelhessian=`
: Transition-state search, and where its Hessian comes from.

`irc=True`
: Follow the intrinsic reaction coordinate from a transition state.

`convergence_setting=` / `conv_criteria=`
: A named geomeTRIC convergence set, or explicit thresholds.

{class}`~openmmqmmm.GeometricOptimizer` is the same machinery as an object, for driving an
optimization step by step. {func}`~openmmqmmm.orca_external_optimizer` hands the optimization
to ORCA instead, with openmmqmmm supplying energies and gradients.

## Frequencies and thermochemistry

```python
from openmmqmmm import numerical_frequencies

result = numerical_frequencies(theory=theory, fragment=fragment, hessatoms=qmatoms, numcores=8)
print(result.frequencies)
print(result.thermochemistry)
```

`hessatoms=` restricts the Hessian to a subset of atoms — mandatory for QM/MM, where a full
Hessian is out of reach. `npoint=2` is a central difference (`1` is forward),
`displacement=0.005` is the step in Angstrom, and `runmode="parallel"` with `numcores` spreads
the displacements over workers.

Thermochemistry comes back in the same `Results`: ZPE, enthalpy, entropy and Gibbs energy at
`temp` and `pressure`, with Grimme's quasi-RRHO treatment on by default (`qrrho=False` turns
it off). `IR=True` gives intensities, `Raman=True` activities, and `scaling_factor` applies a
uniform frequency scaling.

{func}`~openmmqmmm.analytic_frequencies` uses a theory's analytic Hessian where it has one.
{func}`~openmmqmmm.read_hessian` and {func}`~openmmqmmm.write_hessian` move a Hessian between
runs, and {func}`~openmmqmmm.approximate_full_hessian_from_smaller` extends a partial Hessian
to the full system.

## Running many jobs at once

{func}`~openmmqmmm.job_parallel` runs whole jobs in parallel, each in its own directory:

```python
from openmmqmmm import job_parallel

results = job_parallel(fragments=fragments, theories=[theory], numcores=8)
```

It parallelizes over fragments and theories, which is the efficient layout when each job is
small. `allow_theory_parallelization=True` additionally lets each job use the theory's own
parallelism — only worth it when the outer loop is short. `opt=True` runs optimizations
rather than single points.

Use a separate theory instance and directory for anything you run concurrently by hand: a
theory object owns its scratch directory.

## Results

{class}`~openmmqmmm.Results` is a plain container. Which attributes are populated depends on
the job:

```python
result.energy  # Hartree
result.gradient  # (natoms, 3) array, Hartree/Bohr
result.energies  # a sweep's energies, in the order the fragments were given
result.reaction_energy
result.frequencies  # cm^-1
result.hessian
result.normal_modes
result.thermochemistry
result.properties  # whatever else the theory reported
```

QM/MM adds `qm_energy`, `mm_energy` and `qmmm_energy`.

Each job function writes its results to a file named for the job:

| File | Written by |
|---|---|
| `results_singlepoint.json` | {func}`~openmmqmmm.single_point` |
| `results_singlepoint_theories.json` | {func}`~openmmqmmm.single_point_theories` |
| `results_singlepoint_fragments.json` | {func}`~openmmqmmm.single_point_fragments` |
| `results_singlepoint_fragments_theories.json` | {func}`~openmmqmmm.single_point_fragments_and_theories` |
| `results_singlepoint_reaction.json` | {func}`~openmmqmmm.single_point_reaction` |
| `results_optimizer.json` | {func}`~openmmqmmm.optimize_geometry` |
| `results_numfreq.json` | {func}`~openmmqmmm.numerical_frequencies` |
| `results_anfreq.json` | {func}`~openmmqmmm.analytic_frequencies` |

Read one back with {func}`~openmmqmmm.read_results_from_file`, or write one yourself with
`Results.write_to_disk`. `result_write_to_disk=False` turns the writing off where a job runs
in a loop.
