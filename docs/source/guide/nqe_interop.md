# Working with openmmnqe

[openmmnqe](https://github.com/LouieSlocombe/openmmnqe) provides staged nuclear-quantum-effect
workflows — RPMD equilibration and production, adQTB, ring-polymer reporters, deuteration.
Neither package imports the other: they meet at plain OpenMM objects, so either one can drive
and the other supply.

## openmmnqe driving an openmmqmmm potential

{func}`~openmmqmmm.export_rpmd_potential` hands over the MM `System` with the bead-specific
`PythonForce` already attached, plus a matching `Modeller`. openmmnqe's `PreparedSystem`
routes that System through its stages unchanged:

```python
export = export_rpmd_potential(theory=qm_mm, num_beads=32)
prepared = openmmnqe.PreparedSystem(export.system)

openmmnqe.run_openmm_rpmd_equilibration(export.modeller, prepared, n_beads=32)
openmmnqe.run_openmm_rpmd_prod(
    export.modeller, prepared, checkpoint_file="rpmd_ready.chk", n_beads=32, barostat_freq=None
)
```

Four things to get right:

Always pass `barostat_freq=None`
: openmmnqe's RPMD and adQTB production stages add a barostat by default, and it refuses one
  on a System carrying a `PythonForce`.

One export per System-mutating stage
: Build a fresh export for each stage that changes the System — barostat, PLUMED bias,
  deuteration. An export is a snapshot, not a live handle.

Bridge through the stage-final PDB
: Start openmmnqe's classical preparation stages from a plain force field, then move into the
  QM/MM RPMD stages through the PDB each stage writes. A binary checkpoint does not survive
  the System change; the bead archive does.

Do not use `run_openmm_rpmd_contracted` for QM-force contraction
: It leaves the QM `PythonForce` in its own group, evaluated on every bead. Use openmmqmmm's
  own `rpmd_qm_num_copies` instead — see {doc}`rpmd`.

The export restores physical hydrogen masses and refuses a constrained System for
`num_beads > 1`, so build the underlying {class}`~openmmqmmm.OpenMMTheory` with
`autoconstraints=None` and `rigidwater=False`.

## openmmqmmm driving, with openmmnqe utilities

The reverse direction goes through
{class}`~openmmqmmm.MolecularDynamicsEngine`. `run` takes `extra_reporters` (attached
alongside the engine's own, and in RPMD runs driven every `traj_frequency` steps) and
`pre_dynamics_hook` (called once with the engine after the `Simulation` exists, before
dynamics start):

```python
modeller = modeller_from_topology(topology=mm.topology, coords_angstrom=fragment.coords)
openmmnqe.deuterate_system(modeller, mm.system, option="water")

engine = MolecularDynamicsEngine(fragment=fragment, theory=qm_mm, integrator="RPMDIntegrator", rpmd_num_copies=32)
engine.run(
    simulation_steps=1000,
    extra_reporters=[
        openmmnqe.RPMDCentroidReporter(
            topology=modeller.topology, file_name="centroid.pdb", reportInterval=100, num_beads=32
        )
    ],
    pre_dynamics_hook=lambda md: openmmnqe.init_beads(modeller, md.simulation, 32),
)
```

:::{warning}
Do not seed beads from the hook when restarting from a checkpoint. It would overwrite the
bead state the checkpoint just loaded.
:::

{func}`~openmmqmmm.modeller_from_topology` is the small adapter that turns an
`OpenMMTheory`'s topology plus a fragment's coordinates into the `Modeller` openmmnqe's
utilities expect.

## The reference implementation

[tests/test_nqe_interop.py](https://github.com/LouieSlocombe/openmmqmmm/blob/main/tests/test_nqe_interop.py)
exercises all of this with an analytic QM stand-in instead of ORCA, and runs whenever both
packages share an environment. `build_tools/README.md` has the recipe for building that
environment, and
[examples/qmmm_rpmd_nqe_stages.py](https://github.com/LouieSlocombe/openmmqmmm/blob/main/examples/qmmm_rpmd_nqe_stages.py)
is the same pattern as a runnable script.
