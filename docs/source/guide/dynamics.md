# Molecular dynamics

{func}`~openmmqmmm.openmm_md` runs dynamics with OpenMM. It takes any theory, so the same
call runs classical MM dynamics or QM/MM dynamics depending on what you hand it:

```python
from openmmqmmm import openmm_md

# Classical MM
openmm_md(fragment=fragment, theory=mm, timestep=0.002, simulation_time=100)

# QM/MM, with the QM force evaluated every step
openmm_md(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=2)
```

`timestep` and `simulation_time` are in picoseconds. Give `simulation_steps` instead when a
step count is the natural unit.

## Integrators and temperature

`integrator=` selects one of `"LangevinMiddleIntegrator"` (the default),
`"LangevinIntegrator"`, `"VerletIntegrator"`, `"VariableVerletIntegrator"`,
`"VariableLangevinIntegrator"`, `"NoseHooverIntegrator"`, `"DrudeLangevinIntegrator"`,
`"RPMDIntegrator"` or `"QTBIntegrator"`. The last two are for nuclear quantum effects —
see {doc}`rpmd`.

`temperature=` is in kelvin and `coupling_frequency=` is the thermostat coupling in ps⁻¹.
`anderson_thermostat=True` uses an Andersen thermostat with a Verlet integrator instead of a
Langevin one.

The timestep has to match the constraints on the MM theory. With the default
`autoconstraints="HBonds"` and `hydrogenmass=1.5`, 4 fs works with the middle integrator;
with `autoconstraints=None`, expect to drop to around 0.5 fs. The engine warns when it sees
an unconstrained system.

## Pressure

`barostat="MonteCarloBarostat"` runs NPT at `pressure` (bar), applied every
`barostat_frequency` steps. Selecting a barostat forces `LangevinMiddleIntegrator`, and it
cannot be combined with `RPMDIntegrator`.

## Output

`trajectory_file_option=`
: `"DCD"` (the default), `"PDB"`, `"XYZ"`, `"NetCDFReporter"` or `"HDF5Reporter"`. The file is
  named by `trajfilename`, written every `traj_frequency` steps.

`specialatoms=` with `specialtraj_frequency=`
: A second, more frequent trajectory of a subset of atoms — the QM region, usually — without
  paying to write the solvent that often.

`energy_file_option=` / `force_file_option=`
: Energy and force files alongside the trajectory. `atomic_units_force_reporter=True` writes
  forces in Hartree/Bohr rather than OpenMM's units.

`datafilename=`
: The step-by-step state report. Left unset, that report goes to the package logger; set it
  and it goes to a file instead.

`restartfile_frequency=` with `chkfile=` / `statefile=`
: Checkpoint and state files for restarting. A checkpoint is binary and tied to the exact
  System; a state file is portable XML.

## Keeping the solute in the box

Periodic wrapping can leave a solute straddling a box edge:

`enforce_periodic_box=`
: OpenMM's own wrapping when writing coordinates.

`special_wrapping=` with `wrapping_atoms=`
: Wrapping driven by a chosen set of atoms rather than by whole molecules.

`center_on_atoms=` / `solute_indices=`
: Re-centre the box on a chosen group when writing frames.

`add_centerforce=True`
: A flat-bottom restraint (`centerforce_atoms`, `centerforce_center`, `centerforce_distance`,
  `centerforce_constant`) that keeps a solute near the middle of the box. Useful for a small
  solute that would otherwise diffuse into the periodic boundary.

`dummyatomrestraint=True`
: Restrain to a dummy atom instead.

After the run, {func}`~openmmqmmm.mdtraj_image_trajectory` re-images a finished trajectory and
{func}`~openmmqmmm.mdtraj_rmsf` computes per-atom fluctuations.

## Driving the engine directly

{func}`~openmmqmmm.openmm_md` is a thin wrapper around
{class}`~openmmqmmm.MolecularDynamicsEngine`. Use the engine when you need to reach into the
run:

```python
from openmmqmmm import MolecularDynamicsEngine

engine = MolecularDynamicsEngine(fragment=fragment, theory=qm_mm, timestep=0.001, temperature=300)
engine.run(simulation_steps=1000)
```

`run` takes two hooks that `openmm_md` does not expose:

`extra_reporters=`
: OpenMM reporters attached alongside the engine's own. In an RPMD run they are driven every
  `traj_frequency` steps.

`pre_dynamics_hook=`
: Called once with the engine after the `Simulation` exists but before dynamics start — the
  place to seed bead positions or touch the context.

Both exist so another package can drive an openmmqmmm run without either importing the other;
see {doc}`nqe_interop`.

`run` also takes `restart=True` with `chkfile=` or `statefile=` to continue a previous run.

## Metadynamics and other biases

Biased dynamics goes through PLUMED, and only through PLUMED:
{func}`~openmmqmmm.openmm_md_plumed` takes the same arguments as `openmm_md` plus a
`plumed_input_string`:

```python
from openmmqmmm import openmm_md_plumed

plumed_input = """
d1: DISTANCE ATOMS=93,134
METAD ARG=d1 SIGMA=0.05 HEIGHT=1.2 PACE=500 BIASFACTOR=10 FILE=HILLS
PRINT ARG=d1 FILE=COLVAR STRIDE=100
"""

openmm_md_plumed(
    fragment=fragment,
    theory=qm_mm,
    timestep=0.001,
    simulation_time=100,
    plumed_input_string=plumed_input,
)
```

Anything PLUMED can express works, including OPES, which is why the installer builds PLUMED
from source rather than taking the conda-forge package — that build omits the `opes` module.
Single-walker only: this package has no multiwalker driver, and the native OpenMM
metadynamics implementation it used to carry was removed.

This function needs PLUMED and openmm-plumed, and raises
{exc}`~openmmqmmm.MissingDependencyError` without them. See {doc}`../install`.
