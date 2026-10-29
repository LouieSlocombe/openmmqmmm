# Ring-polymer and quantum-thermal-bath dynamics

Nuclear quantum effects — zero-point energy and tunnelling — matter for hydrogen transfer,
and classical MD misses both. Two integrators cover them, and both work with a QM/MM theory:
`RPMDIntegrator` for ring-polymer molecular dynamics, and `QTBIntegrator` for the adaptive
quantum thermal bath.

OpenMM's `PythonForce` is what makes QM/MM RPMD possible: the RPMD integrator asks for
the QM/MM energy and gradient of the bead it is currently propagating, and gets them.

## A QM/MM RPMD run

The MM theory must be built without constraints — OpenMM's `RPMDIntegrator` does not support
them — and with physical masses:

```python
from openmmqmmm import OpenMMTheory, QMMMTheory, openmm_md

mm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"],
    pdbfile="system.pdb",
    periodic=True,
    autoconstraints=None,
    rigidwater=False,
    hydrogenmass=None,
)
qm_mm = QMMMTheory(qm_theory=qm, mm_theory=mm, fragment=fragment, qmatoms=qmatoms, qm_charge=-1, qm_mult=6)

openmm_md(
    fragment=fragment,
    theory=qm_mm,
    integrator="RPMDIntegrator",
    rpmd_num_copies=32,
    timestep=0.0005,
    simulation_steps=100,
)
```

A constrained System with more than one bead is refused outright, naming the number of
constraints it found. Selecting either nuclear-quantum integrator also disables OpenMM's
automatic hydrogen-mass repartitioning and restores the mass each bonded heavy atom gave
away, because both methods need physical nuclear masses.

## Cost, and contracting the QM force

By default the QM force is evaluated independently on every bead. `RPMDIntegrator` evaluates
the potential twice per step, so the example above performs 64 QM/MM evaluations per MD step.

`rpmd_qm_num_copies=` applies OpenMM's ring-polymer contraction to the QM force alone:
`rpmd_qm_num_copies=1` evaluates it at the centroid, and any value up to `rpmd_num_copies`
is allowed. The MM forces stay on the full ring.

Final bead evaluations are cached, so the state and force reporting an exact-RPMD run does
between steps does not relaunch identical QM jobs.

## Energy diagnostics and trajectory output

`rpmd_energy_frequency=` controls the state report, including the full-ring total energy,
independently of `traj_frequency`. Its default, `None`, uses `traj_frequency` for compatibility.
Set a positive integer to sample energies less often, or `0` to disable these diagnostics.
For example, `traj_frequency=10, rpmd_energy_frequency=1000` writes coordinates every ten
steps and energies every thousand steps. The log and optional `datafilename` share one
full-ring evaluation at each diagnostic event.

Full-ring energy evaluates the potential on every bead even when propagation contracts the
QM force. An eight-bead centroid-contracted system can therefore need eight additional QM
calculations per diagnostic event. With `rpmd_energy_frequency=0`, DCD/PDB trajectory output
and initial/final structures request positions only. Explicit force output and trajectory
formats or extra reporters that request energies still evaluate the reported bead's potential.
Restart archives continue to store every bead's positions and velocities.

## What RPMD does not support

These are rejected rather than silently approximated, because each keeps state that is shared
across beads and cannot represent all of them at once:

- `truncated_pc` — its point-charge correction history is one shared set.
- `update_qm_region_charges` — one shared MM charge set and missing population-charge
  response derivatives.
- `special_wrapping` — use OpenMM's own periodic wrapping through the `PythonForce` state.
- `dummyatomrestraint`.
- A barostat. `RPMDIntegrator` and `MonteCarloBarostat` cannot be combined.

RPMD restart files carry positions and velocities for every bead.

## adQTB

```python
openmm_md(
    fragment=fragment,
    theory=qm_mm,
    integrator="QTBIntegrator",
    timestep=0.0005,
    simulation_time=10,
)
```

The adaptive quantum thermal bath reaches similar physics through a coloured-noise thermostat
on a single copy of the system, so it costs one QM/MM evaluation per step rather than 2N. It
takes the same `temperature`, `coupling_frequency` and `timestep` options as the Langevin
integrators, and the same physical-mass requirement applies.

## Staged workflows

For equilibration/production staging, bead-resolved reporters and deuteration, the sibling
package [openmmnqe](https://github.com/LouieSlocombe/openmmnqe) drives the same potential —
see {doc}`nqe_interop`.
