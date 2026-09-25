# Theory objects

A theory turns coordinates into an energy and, on request, a gradient. Every job function in
{doc}`jobs` takes one, and {doc}`qmmm` combines two of them.

## ORCATheory

{class}`~openmmqmmm.ORCATheory` writes ORCA input, runs the binary and parses the output.
The two arguments that matter most are passed through verbatim, so anything ORCA accepts
works:

```python
from openmmqmmm import ORCATheory

orca = ORCATheory(
    orcasimpleinput="! r2SCAN-3c tightscf",  # the ! line
    orcablocks="%scf maxiter 200 end",  # the % blocks
    numcores=8,
)
```

Useful beyond those:

`orcadir=`
: Where ORCA lives, if neither `OPENMMQMMM_ORCADIR` nor `PATH` should decide — see
  {doc}`../install`.

`numcores=`
: Parallel ORCA. Needs a working OpenMPI matched to the ORCA version.

`moreadfile=` / `autostart=`
: Start from existing orbitals. `autostart` reuses the previous step's GBW file within a run,
  which is what makes an optimization or a numerical Hessian affordable.

`brokensym=`, `hs_mult=`, `atomstoflip=`
: Broken-symmetry starting guesses for open-shell metal centres.

`extrabasis=` / `extrabasisatoms=` / `basis_per_element=` / `ecp_dict=`
: Per-element and per-atom basis sets and ECPs.

`label=`, `filename=`, `save_output_with_label=`, `keep_each_run_output=`
: Naming and retention of the ORCA input and output files. By default each run overwrites the
  last; keep them when you need to inspect what ORCA actually did.

The theory raises {exc}`~openmmqmmm.ExternalProgramError` when ORCA fails, having scanned the
output for the error and warning patterns ORCA prints. `ignore_orca_error=True` turns that
off, for the rare case where a non-zero exit is expected.

## OpenMMTheory

{class}`~openmmqmmm.OpenMMTheory` wraps an OpenMM `System`. It accepts every topology format
OpenMM does — pick exactly one:

```python
from openmmqmmm import OpenMMTheory

# OpenMM XML force fields, with a PDB file for the topology
OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile="system.pdb", periodic=True)

# CHARMM
OpenMMTheory(charmm_files=True, psffile="system.psf", charmmtopfile="top.rtf", charmmprmfile="par.prm")

# Amber
OpenMMTheory(amber_files=True, amberprmtopfile="system.prmtop")

# GROMACS
OpenMMTheory(gromacs_files=True, gromacstopfile="topol.top", grofile="conf.gro")

# A serialized OpenMM system
OpenMMTheory(xmlsystemfile="system.xml", pdbfile="system.pdb")
```

### Periodicity

`periodic=True` turns on periodic boundary conditions and, with it,
`nonbonded_method_pbc` (`"PME"` by default), `periodic_nonbonded_cutoff` (in Angstrom),
`switching_function_distance`, `dispersion_correction` and `ewalderrortolerance`. The box
comes from the input files where they carry one; `periodic_cell_dimensions` or
`periodic_cell_vectors` override that.

For a non-periodic system the corresponding knobs are `nonbonded_method_no_pbc` and
`nonbonded_cutoff_no_pbc`.

### Constraints and masses

`autoconstraints` is `"HBonds"` (the default), `"AllBonds"`, `"HAngles"` or `None`.
Together with the default `hydrogenmass=1.5` (hydrogen-mass repartitioning) and
`rigidwater=True`, `"HBonds"` supports 2 fs timesteps with `LangevinIntegrator` and 4 fs with
`LangevinMiddleIntegrator`; `"AllBonds"` and `"HAngles"` permit larger ones. With
`autoconstraints=None` nothing is constrained and the timestep has to come down to around
0.5 fs — the MD engine warns when it sees that combination.

`constraints=`, `bondconstraints=`, `restraints=` and `frozen_atoms=` add your own.
{func}`~openmmqmmm.define_xh_constraints` and {func}`~openmmqmmm.get_water_constraints`
build the common lists.

:::{warning}
Ring-polymer and adQTB dynamics need physical masses and no constraints. Build the theory
with `autoconstraints=None`, `rigidwater=False` and `hydrogenmass=None` for those — see
{doc}`rpmd`.
:::

### Platform

`platform=` selects the OpenMM platform (`"CPU"`, `"CUDA"`, `"OpenCL"`, `"Reference"`), and
`numcores=` the thread count for `"CPU"`. In QM/MM the MM side is rarely the bottleneck, so
the CPU platform is usually the right choice.

`mm.set_numcores(n)` updates both `mm.numcores` and the CPU `Threads` property used
by new Contexts. It raises `InputError` if a Context created by `mm.create_simulation()`
(including one owned by an MD engine) is still alive with a different thread count.
Release all such Simulations and any retained Context references before changing the
count, then create a new Simulation. Existing Contexts are never reconfigured;
setting the same count again is allowed. Contexts created directly through OpenMM
are managed by the caller and are outside this check.

`QMMMTheory.set_numcores(n)` applies the same restriction to its MM theory before
updating the QM or wrapper core count. On non-CPU platforms, the setter updates
`numcores` without adding the CPU-only `Threads` property.

### Inspecting the system

`do_energy_decomposition=True` logs the energy of every force group, which is the fastest way
to find out why a system has the energy it has. {func}`~openmmqmmm.openmm.print_systemsize`
and {func}`~openmmqmmm.openmm.write_xmlfile_nonbonded` cover the other two common questions:
how big is it, and what nonbonded parameters did the force field actually assign.

## NumGrad

{class}`~openmmqmmm.NumGrad` wraps any theory and produces a gradient by finite differences,
for a method that has no analytic one:

```python
from openmmqmmm import NumGrad, optimize_geometry

optimize_geometry(theory=NumGrad(theory=orca), fragment=fragment)
```

`npoint=1` is a forward difference, `npoint=2` (the default) central. With
`runmode="parallel"` the displacements are spread over `numcores` workers.
`optimize_geometry(num_grad=True)` and {func}`~openmmqmmm.numerical_frequencies` do the same
thing internally, so reach for `NumGrad` when you need the gradient somewhere else.

## ZeroTheory

{class}`~openmmqmmm.ZeroTheory` returns zero energy and a zero gradient. It exists to
exercise a workflow — a reporter, a restart, a parallel job layout — without paying for a
calculation.
