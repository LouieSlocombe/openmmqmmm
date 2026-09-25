# Validating finite periodic embedding

Periodic electrostatic embedding gives the QM Hamiltonian **one selected image of each
MM molecule**. It contains no QM–MM Ewald/PME sum and no QM-image sum. MM retains its
separately configured periodic treatment. Changing a molecule's selected image can change
the finite QM energy and forces abruptly. The {doc}`qmmm` image convention is reproducible,
but image invariance and accurate derivatives within one image branch do not establish
cell-size convergence or energy conservation across an image change.

Use the following baseline before assessing changes to an embedding approximation or a
timestep. Record the current behavior; do not require an existing finite-field jump to
vanish as a condition for passing validation.

## Run the analytic baseline

From a checkout with the package and test dependencies installed, run:

```bash
python examples/periodic_embedding_validation.py --output /tmp/finite-periodic-baseline.json
python -m pytest tests/test_qmmm_periodic_switching.py
```

The [runner source](https://github.com/LouieSlocombe/openmmqmmm/blob/main/examples/periodic_embedding_validation.py)
defines the coordinates, charges, molecule membership and integration settings. Compare its
JSON output with the checked-in
[baseline](https://github.com/LouieSlocombe/openmmqmmm/blob/main/docs/audits/embedding-repros/finite-periodic-baseline.json).
Use numerical tolerances, not a byte-for-byte comparison: package versions and floating-point
results can differ. Keep the old baseline alongside results from a proposed change.

The runner uses an analytic Coulomb QM surrogate with unequal QM charges and neutral MM
dipoles. It exercises the production periodic imaging and QM/MM force path without ORCA.
It measures relative A/B energies and QM/MM force differences against the largest finite
reference and total-energy drift during short NVE trajectories. The separate switching
tests scan both sides of orthorhombic and triclinic image boundaries and check derivatives
within each branch. MM PME remains separate from the finite QM field in both checks.

| Check | Default sampling | What it establishes |
| --- | --- | --- |
| Box sensitivity | Box edges 24, 30, 40, 60 Å; the same three MM dipoles | Sensitivity to image selection as the box changes; molecule density changes |
| Embedding extent | Nested complete neutral dipoles on an 8 Å lattice; 1, 2, 3 shells | Sensitivity to adding a deterministic outer environment at fixed lattice spacing |
| Image-switch tests | A nonsymmetric environment at `+/- epsilon`, with epsilon 1e-3, 1e-4, 1e-5 Å | A finite energy/force jump alongside accurate derivatives within each branch |
| NVE | Smooth and crossing trajectories, 0.5 and 0.25 fs timesteps, each lasting 12 fs | Integration error in a smooth branch and the residual associated with image changes |

The runner JSON stores static energies in kJ/mol and forces in kJ/mol/nm, with separate
QM embedding, MM and total components. Its force RMS averages per-atom vector errors over
both A and B. The NVE trace uses an explicit velocity-Verlet `CustomIntegrator` with
synchronized physical velocities, making initial kinetic energies comparable across
timesteps. Events include the stepwise total-energy change and the QM energy difference
between the old and new image at the same event geometry.

These are small deterministic regression examples. The dilute box scan is not a
fixed-density convergence study; the lattice shell scan is not a liquid-solvent ensemble.
The largest finite case is a comparison reference, not an infinite periodic QM solution.
Twelve femtoseconds of surrogate dynamics does not validate a production QM/MM timestep.

The checked-in baseline records the following maximum absolute total-energy deviations:

| NVE case | 0.5 fs timestep | 0.25 fs timestep |
| --- | --- | --- |
| Smooth branch | 5.75e-8 kJ/mol | 1.44e-8 kJ/mol |
| One image switch | 0.254291 kJ/mol | 0.254294 kJ/mol |

The smooth error falls by approximately fourfold. The crossing trajectories each lose
about 0.25429 kJ/mol, consistent with their independently evaluated change of QM image at
the event geometry. The nested-shell QM relative-energy errors are -0.009707 and
-0.012102 kJ/mol for the first two shells against the third: this series is nonmonotonic.
The additive, nonpolarizable probe also leaves the QM contribution to forces on common
MM atoms unchanged when only outer molecules are added; real QM polarization need not.

Run the recorded energy/force and NVE regression checks with:

```bash
python -m pytest tests/test_qmmm_periodic_switching.py tests/test_periodic_validation_protocol.py
```

## Cell-size convergence for the intended QM system

Save a manifest with the repository revision, dependency and QM backend versions,
coordinates, topology, atom and molecule IDs, charges, box vectors, QM atom IDs, QM
charge/multiplicity, boundary treatment, QM method/basis/SCF settings, MM force-field files,
MM nonbonded method/cutoff/PME tolerance, and platform/precision. Save the actual inputs;
a random seed alone does not reproduce a prepared solvent configuration across versions.

1. Choose a neutral solvated model and a fixed inner environment around the QM region.
   Preserve the Cartesian coordinates and IDs of this core at every cell size. Include
   counterions consistently if needed; enlarge the cell by adding complete neutral
   solvent molecules. Keep density and composition fixed, rather than merely expanding
   the box around the same atoms. For a concrete neutral-water study, choose total water
   counts `N = 512, 1000, 1728, 2744`, including one QM water, number density
   `rho = 0.0334 Å^-3`, and cubic edges `L = (N/rho)^(1/3)` Å. Other systems should specify
   their own count/volume series and report its actual density.
2. Prepare and save each outer environment using a recorded seed, for example `20260925`.
   Keep a common inner shell, for example all molecules originally within 8 Å of the QM
   center, fixed during outer-environment preparation. Do not scale its bond lengths or
   reoptimize it differently in each cell. Check clashes and neutrality. Repeat the series
   with additional saved outer environments to separate configuration variance from a
   cell-size trend.
3. Define two local configurations A and B identically at every size. For the water example,
   B stretches one QM O–H bond by 0.01 Å along its A bond direction, with every other
   coordinate unchanged. For a chemical application, use the two conformations or local
   displacements relevant to the observable, retaining an explicit atom mapping.
4. Evaluate A and B with the same QM settings and `truncated_pc=False`. Supply each saved
   cell through `periodic_box_vectors` in Å for standalone `QMMMTheory.run` calls. Tighten
   SCF and MM numerical settings until their noise is smaller than the differences being
   measured. Do not change MM cutoff, PME tolerance or tail settings across the series;
   ensure the cutoff is valid even in the smallest box.
5. For each size, save `DeltaE(L) = E_B(L) - E_A(L)` in Hartree and compare
   `DeltaE(L) - DeltaE(L_ref)` to the largest cell. Absolute total energies of systems with
   different molecule counts are not a cell-size error measure. Save full forces for both
   A and B. Compare corresponding QM atoms and the fixed common MM shell to the reference,
   separately; report RMS and maximum per-atom force-vector differences in Hartree/Bohr.
   Also report the change in the A-to-B force response if that is the target observable.

Choose acceptable energy and force errors for the intended observable before examining the
results. Extend the largest cell if the last two sizes disagree beyond those tolerances
or the outer-environment variance. Do not infer monotonic convergence or an infinite-system
limit from agreement between only two finite cells. The combined QM/MM result includes
both finite-QM-field sensitivity and MM finite-size effects; report a separate MM component
or MM-only control when attributing the source of a trend.

## Embedding extent at a fixed cell and geometry

Use a single saved large-cell A/B pair and the same imaged coordinates. Build nested charge
lists by adding **whole neutral molecules**, for example at molecular-center radii of
8, 12, 16 Å, then the full cell. Require these radii to fit in the selected cell and report
the actual molecule counts and total charge at every radius. Keep any charged solute and
its neutralizing group together; do not silently omit half a molecule or create a charged
outer shell. Keep the local environment, QM atoms, boundary charges and virtual-site hosts
unchanged.

This is a diagnostic series of explicitly prepared finite charge lists evaluated by the
QM backend at fixed geometry. It does not introduce a production cutoff. Keep a mapping
from each point charge to its physical atom so point-charge gradients can be compared on
common MM atoms, with any virtual-site or boundary-force projection applied consistently.
For total QM/MM comparisons, use the identical full MM energy/forces at every extent.
Record the same A/B energy and force differences as for the cell series, using the full
charge list as the finite reference. Report field-only and combined results distinctly.

`truncated_pc_radius` is not an independent extent-convergence control:
`truncated_pc_recalc_iter=1` reconstructs the full field every evaluation. Larger refresh
intervals introduce cached, history-dependent corrections and are unsuitable for
conservative force or NVE validation. QM/MM dynamics rejects `truncated_pc=True`.

## Scan an image-switching surface

Choose a neutral MM molecule near an image boundary in a nonsymmetric QM environment;
symmetry can accidentally hide a jump. Translate that whole molecule along a path
`x(s) = x(0) + s n`, where `n` is a unit Cartesian direction, preserving its internal
geometry and all other coordinates. Locate the switch coordinate `s_star` by monitoring
the molecule's selected lattice translation. For an orthorhombic box it often occurs when
its center crosses a half-box displacement from the QM center; use the actual selected
images for skew cells and multi-atom molecules.

Evaluate `s_star - epsilon` and `s_star + epsilon` for
`epsilon = 1e-2, 1e-3, 1e-4, 1e-5 Å`, saving selected images, energies, and QM/MM forces.
Report the signed energy difference and RMS/maximum force differences between the two
sides at every epsilon. A nonzero limiting difference is the finite-field image-switch
jump. Avoid using the exact tie point as the only diagnostic: its selected branch can
depend on numerical tie conventions.

At points on each side, verify derivatives with a finite-difference displacement smaller
than the distance to the switch; confirm both displaced evaluations stay on the same
branch. For example, at `s_star +/- 1e-3 Å`, try steps of `1e-4` and `5e-5 Å`.
Compare the directional energy derivative to minus the summed force on the translated
molecule, converting the force length unit consistently. A central difference whose
stencil crosses the switch is not a force check: it includes the jump divided by the
stencil width. Scan Cartesian QM and MM coordinates within each branch as well to check
the corresponding force components.

## Baseline NVE drift

Use fresh theory/engine objects for every run, fixed full embedding charges, a fixed box,
and `integrator="VerletIntegrator"`, `barostat=None`, `anderson_thermostat=False`. Ensure the
System contains no other thermostat, barostat or unaccounted energy-changing operation.
Save initial positions and physical velocities explicitly, in Å and nm/ps respectively,
and initialize the same physical state for every timestep through the engine's
`pre_dynamics_hook`. OpenMM's leapfrog Verlet stores staggered velocities: initialize
the half-step velocities consistently for each timestep and use its corresponding
kinetic-energy convention. Copying the same stored half-step velocities between different
timesteps does not give the same initial physical state. An explicit velocity-Verlet
integrator with synchronized positions and velocities is another option. Do not generate
a new thermal velocity sample for each run. Record masses, constraints and constraint
tolerance; disable center-of-mass momentum removal for this energy-conservation diagnostic.

Prepare two cases: a control that stays within one image branch and a trajectory that
crosses the scanned surface. For a real QM baseline, start with timesteps 0.5, 0.25 and
0.125 fs over the same 100 fs duration (200, 400 and 800 steps). The MD API uses ps, so
these timesteps are `0.0005`, `0.00025`, `0.000125` and the duration is `0.1`. Extend the
duration for the intended application. Verify the control has zero switches and record
the observed events in the crossing case; a requested initial velocity does not guarantee
that a trajectory will cross.

Save potential energy `U`, kinetic energy `K`, and `H = U + K` at every timestep, with a
consistent kinetic-energy convention. Include the actual QM contribution and the configured
MM potential. Track molecular lattice translations relative to the QM center, excluding
the common global anchor translation, so ordinary output wrapping is not counted as an
embedding image switch. Report:

- Endpoint drift `H(T) - H(0)` and maximum absolute deviation, in kJ/mol.
- The least-squares slope of `H(t)` over the complete run, in kJ/mol/ps, plus RMS residual
  about that line; optionally normalize by the same documented atom count for every case.
- Switch count and times, the stepwise change in `H` bracketing each event, and slopes
  within the smooth segments between events. A stepwise change includes integration error;
  compare it with the static epsilon-scan jump instead of calling it an exact jump.

Smooth-branch errors should respond to timestep and numerical convergence. An image-change
energy discontinuity can survive timestep reduction, so global drift alone is insufficient
to diagnose an integrator or force bug. Compare fixed-duration runs and event locations;
do not compare equal step counts with different elapsed times. NVE has no thermostat or
barostat to absorb the error.

## Interpret and retain the results

Archive the manifest, coordinates, velocities, raw scan/trajectory rows and summary JSON.
Use Hartree for static energies, Hartree/Bohr for forces, Å for geometry, and explicitly
label any conversion to kJ/mol and fs/ps. Package results return gradients; the physical
force is their negative. For a group of `N` common atoms, define force RMS as
`sqrt(sum_i |F_i - F_ref_i|^2 / N)` and maximum as `max_i |F_i - F_ref_i|`.

A changed image choice away from the surface, failed branch-local derivatives, lost
whole-molecule neutrality, or new smooth-trajectory drift is evidence of a regression.
A reproducible jump at the same surface is a baseline limitation of the finite embedding
model. Changes to cutoff or multipole treatments should report both smooth-region accuracy
and switch behavior against these saved baselines. Neither smaller drift nor MM PME alone
establishes periodic QM electrostatic accuracy.
