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

`label=`, `filename=`, `save_output_with_label=`, `keep_each_run_output=`, `keep_last_output=`
: Naming and retention of ORCA files. Each evaluation overwrites `<filename>.out` and by
  default copies it to `<filename>_last.out`. Set `keep_last_output=False` to avoid that
  copy in long trajectories. Labelled and numbered copies remain independently configurable;
  see {doc}`output`. Output retention does not disable GBW orbital reuse.

`profile=True`
: Retain JSON-serializable timing, SCF and execution metadata for the latest evaluation in
  `last_evaluation_stats`; see {doc}`output` for the separate wall and ORCA timing fields.

The theory raises {exc}`~openmmqmmm.ExternalProgramError` when ORCA fails, having scanned the
output for the error and warning patterns ORCA prints. `ignore_orca_error=True` turns that
off, for the rare case where a non-zero exit is expected.

## MACESCFTheory

{class}`~openmmqmmm.MACESCFTheory` adapts a MACE-SCF ASE calculator for standalone jobs
or the QM side of {class}`~openmmqmmm.QMMMTheory`. Unlike the mechanically embedded
OpenMM-ML route in {doc}`mlmm`, MM point charges enter the model's electrostatic
calculation and receive the corresponding forces. Install the optional backend as
described in {ref}`install-mace-scf`.

```python
from openmmqmmm import MACESCFTheory, QMMMTheory

backend = MACESCFTheory(
    model_path="/path/to/fixedpoint.model",
    model_type="fixedpoint",  # use "qeq" for a QEq checkpoint
    device="cpu",  # or "cuda" with a compatible PyTorch installation
    default_dtype="float64",
)
qm_mm = QMMMTheory(
    qm_theory=backend,
    mm_theory=mm,
    fragment=fragment,
    qmatoms=qmatoms,
    qm_charge=0,
    qm_mult=1,
    embedding="elstat",
)
```

Supply either `model_path` or a preconfigured ASE calculator via
`MACESCFTheory(calculator=calculator)`. FixedPoint and QEq are the supported model
families. A supplied calculator must implement energy, forces and the MACE-SCF
`embed(charges, positions=...)` point-charge protocol, including forces on the external
charges and a constrained `total_charge` input. Local-source models are not supported
by this adapter because their charge conventions differ.

`calculator_kwargs` passes additional constructor settings to the checkpoint-based
calculator. The adapter defaults to real-space electrostatics (`pbc_handling="realspace"`),
eager evaluation, and fixed total charge. Keep these settings for point-charge embedding;
compiled evaluation and a variable-charge reservoir are unsupported. For a supplied
calculator, configure the same finite-environment behavior before wrapping it. OpenMM's
`platform` and the MACE-SCF `device` are separate settings. `numcores` records the theory's
core-count metadata; configure PyTorch's thread count separately. Do not share an injected
calculator between concurrent calculations, because the adapter owns its embedding state.

`qm_charge` becomes the ASE `total_charge` information field (or the calculator's
configured total-charge key). MACE-SCF has no multiplicity input in this interface;
`qm_mult=1` is required, but it does not impose a spin state on the model. Use a checkpoint
whose elements, charge states and training domain cover the intended system, including
any link atoms.

The adapter uses the model's **total energy**, which already contains QM–MM electrostatic
coupling, and converts eV to Hartree. It converts both atomic and point-charge forces
from eV/Å to gradients in Hartree/Bohr, reversing their sign. Do not add
`point_charge_energy` a second time. OpenMM retains MM interactions and the configured
QM–MM Lennard-Jones terms through the normal QM/MM partition.

The model always sees nonperiodic QM atoms and a finite MM environment, even when the
OpenMM system is periodic. The molecule imaging, link-atom projection and MM virtual-site
force redistribution described in {doc}`qmmm` still apply, subject to model suitability;
this is not periodic MACE-SCF/Ewald embedding. The
{doc}`../examples` water-dimer script runs an energy and gradient with a local checkpoint.

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
`switching_function_distance` (applied to CHARMM files only), `dispersion_correction` and
`ewalderrortolerance`. The box
comes from the input files where they carry one; `periodic_cell_dimensions` or
`periodic_cell_vectors` override that.

For a non-periodic system the corresponding knobs are `nonbonded_method_no_pbc` and
`nonbonded_cutoff_no_pbc`.

### Constraints and masses

`autoconstraints` is `"HBonds"` (the default), `"AllBonds"`, `"HAngles"` or `None`.
Together with the default `hydrogenmass=1.5` (hydrogen-mass repartitioning) and
`rigidwater=True`, `"HBonds"` supports 2 fs timesteps with `LangevinIntegrator` and 4 fs with
`LangevinMiddleIntegrator`; `"AllBonds"` and `"HAngles"` permit larger ones. With
`autoconstraints=None` no bonds are constrained (water stays rigid unless `rigidwater=False`)
and the timestep has to come down to around 0.5 fs — the MD engine warns whenever
`autoconstraints` is `None`.

Single points, optimizations and frequencies refuse automatic constraints: build the MM
theory for those, including one inside a `QMMMTheory`, with `autoconstraints=None` and
`rigidwater=False`, or `OpenMMTheory.run` raises {exc}`~openmmqmmm.InputError`.

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
`numcores=` the thread count for `"CPU"`:

```python
mm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml"],
    pdbfile="system.pdb",
    periodic=True,
    platform="CPU",
    numcores=4,
)
```

Configure platform properties on this object too, for example
`platform="CUDA", properties={"Precision": "mixed"}`. MD reuses the MM theory's platform
and properties when its own `platform` argument is omitted or `None`; an explicit
conflicting platform raises `InputError`. This includes the MM theory inside a
`QMMMTheory`. See {doc}`dynamics` for the full selection rules and a CUDA example.
The OpenMM platform controls MM and integration; ORCA runs separately with its own core
count configured on `ORCATheory`.

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
logs how many atoms a `Modeller` holds, and {func}`~openmmqmmm.openmm.write_xmlfile_nonbonded`
writes per-residue nonbonded parameters out as an OpenMM force-field XML.

## MLMMTheory

{class}`~openmmqmmm.MLMMTheory` combines an OpenMM-ML potential with an existing MM
system. Select complete molecules using zero-based `mlatoms` indices; the ML model replaces
their internal force-field interactions while classical interactions couple them to the
environment. The source `OpenMMTheory` is copied and remains available independently.

Use it with single points, geometry optimization, numerical frequencies and OpenMM dynamics.
OpenMM-ML and the model backend are optional dependencies; ORCA is not needed. See
{doc}`mlmm` for setup, model limitations and constraints.

## NumGrad

{class}`~openmmqmmm.NumGrad` wraps any theory and produces a gradient by finite differences,
for a method that has no analytic one:

```python
from openmmqmmm import NumGrad, optimize_geometry

optimize_geometry(theory=NumGrad(theory=orca), fragment=fragment)
```

`npoint=1` is a forward difference, `npoint=2` (the default) central. With
`runmode="parallel"` the displacements are spread over `numcores` workers.
`optimize_geometry(num_grad=True)` does this wrapping for you, so reach for `NumGrad`
directly when you need the gradient somewhere else. {func}`~openmmqmmm.numerical_frequencies`
is different: it finite-differences the theory's gradients to build a Hessian.
`NumGrad.displacement` is supplied in Å and defaults to 0.005 Bohr converted to Å
(about 0.002646 Å). `numerical_frequencies` retains its separate 0.005 Å default.

## ZeroTheory

{class}`~openmmqmmm.ZeroTheory` returns zero energy and a zero gradient. It exists to
exercise a workflow — a reporter, a restart, a parallel job layout — without paying for a
calculation.
