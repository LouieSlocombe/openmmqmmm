# Experimental multiple-time-step QM/MM

{func}`~openmmqmmm.export_mts_potential` exports an opt-in NVE prototype using

```{math}
U = U_{\mathrm{MM,modified}} + U_{\mathrm{QM,reference}}
    + (U_{\mathrm{QM,high}} - U_{\mathrm{QM,reference}}).
```

Native MM forces and the inexpensive embedded QM reference act every inner step. The
high-minus-reference correction acts every outer step. Both QM evaluations receive the same
QM region, electronic charge and multiplicity, link-atom placements and MM charge field.
Their gradients pass through the same boundary projection before subtraction. OpenMM
distributes native virtual-site forces after adding the contributions.

## Running the prototype

Build the high-level {class}`~openmmqmmm.QMMMTheory` normally, then supply an independent
QM backend for the reference. Two ORCA backends must use distinct `filename` values so
their outputs and orbital guesses do not overwrite each other.

```python
import openmm
from openmm import app
from openmmqmmm import ORCATheory, export_mts_potential

# qm_mm is a fresh QMMMTheory whose ORCA backend uses filename="high".
reference = ORCATheory(orcasimpleinput="! HF STO-3G tightscf", filename="reference")
export = export_mts_potential(
    theory=qm_mm,
    reference_qm_theory=reference,
    inner_timestep=0.00025,  # ps: 0.25 fs
    inner_steps=2,  # outer timestep: 0.5 fs
)
simulation = app.Simulation(
    export.modeller.topology,
    export.system,
    export.integrator,
    openmm.Platform.getPlatformByName("Reference"),
)
simulation.context.setPositions(export.modeller.positions)
simulation.context.computeVirtualSites()
simulation.context.setVelocitiesToTemperature(300)
simulation.step(100)  # 100 outer steps, 200 inner steps
state = simulation.context.getState(getEnergy=True)
print(state.getPotentialEnergy())  # full high-level QM/MM potential
```

The illustrated HF/STO-3G reference is an API example, not a recommended reference for
reactive dynamics. A useful reference must capture fast bond motion and the chemistry
visited by the trajectory. Slowing the entire QM force would leave the removed internal
QM bonded MM terms without a fast replacement; this split retains the reference at every
inner step.

The exporter uses OpenMM's
[rRESPA MTSIntegrator](https://docs.openmm.org/latest/api-python/generated/openmm.mtsintegrator.MTSIntegrator.html).
Every existing native force group, including a separate reciprocal-space group, keeps the
inner cadence. The reference and correction require two unused force groups. The returned
`fast_group` and `slow_group` identify these additions; existing force-group assignments are
preserved.

Successful export attaches forces to `qm_mm.mm_theory.system` and switches `qm_mm` into
external-force mode. Build a fresh theory for standalone single points or optimization.
A rejected export leaves the original force list and standalone mode intact. The reference
QM/MM wrapper owns independent boundary and charge bookkeeping while sharing the prepared
native MM system; it does not apply QM/MM force removal a second time.

## Accuracy and cost

At any geometry, requesting all force groups gives the high-level potential and force:
the two reference contributions cancel. The integrator approximates the trajectory through
its outer-step force schedule. Full-energy reporting can request a fresh high-level job at
a geometry the propagator has not evaluated, so include reporting in performance measurements.
The high and reference providers expose `evaluation_count` and `cache_hits`; the shared
reference cache avoids repeating an electronic calculation when both forces request the
same endpoint.

Start with `inner_steps=1`, then compare ratios 2 and 4 at identical simulated time.
Reduce the inner and outer timesteps independently and inspect NVE energy drift and
the observables needed by the application. Keep mass repartitioning, constraints and
centre-of-mass removal identical across comparisons. Measure elapsed time for the complete
simulation, including reference evaluations and reporting, rather than inferring speed from
the number of high-level calls. No production speedup or universal outer timestep is claimed.

The deterministic tests verify the complete energy/force identity at displaced geometries
with covalent caps, shifted charges, dipole corrections, RCD sites and native virtual sites.
They check force-call reductions at ratios 1/2/4, and compare a harmonic NVE trajectory and
energy conservation with an independent analytic solution as the timestep is reduced.

## Scope

This prototype supports fixed-potential NVE dynamics. It rejects truncated point-charge
corrections, changing QM population charges, disabled link-atom force projection, reused
MD/export systems, Drude particles, thermostats and barostats. Do not add a thermostat or
barostat after export: NVT/NPT accuracy and coupling protocols have not been validated.
Constraints and masses retain the native System's configuration. Periodic embedding keeps
the finite, molecule-imaged charge-field approximation described in {doc}`periodic-validation`.
The native `CMMotionRemover` is also retained. Strict energy-conservation tests require
disabling centre-of-mass momentum removal and any other user forces that modify the state
or exchange energy, particularly for externally restrained systems.
Real reactive-system stability, reference selection and end-to-end ORCA speedups remain
application-specific validation work.
