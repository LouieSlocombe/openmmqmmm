# QM/MM

{class}`~openmmqmmm.QMMMTheory` couples a QM theory and an MM theory into one theory object.
It behaves like any other theory, so every job function in {doc}`jobs` takes it.

```python
from openmmqmmm import Fragment, OpenMMTheory, ORCATheory, QMMMTheory

fragment = Fragment(pdbfile="system.pdb")

qm = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
# Single points and optimizations need an MM theory without automatic constraints.
mm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml"],
    pdbfile="system.pdb",
    periodic=True,
    autoconstraints=None,
    rigidwater=False,
)

qm_mm = QMMMTheory(
    qm_theory=qm,
    mm_theory=mm,
    fragment=fragment,
    qmatoms=[93, 94, 95, 96, 97, 133, 134, 135],
    qm_charge=-1,
    qm_mult=6,
)
```

## The QM region

`qmatoms` is a list of 0-based atom indices into the fragment. Everything else is MM.

`qm_charge` and `qm_mult` are the charge and multiplicity **of the QM region**, not of the
whole system, and they are not the fragment's. Set them on the theory; the job functions then
need no charge or multiplicity of their own.

Two helpers build the list rather than counting atoms by hand:

```python
from openmmqmmm import define_active_region, expand_qm_region

# Whole residues within 10 Å of atom 93.
active = define_active_region(fragment=fragment, mmtheory=mm, radius=10, originatom=93)

# Grow a QM region outwards, keeping whole molecules.
qmatoms = expand_qm_region(fragment=fragment, initial_atoms=[93, 94, 95], radius=4)
```

{func}`~openmmqmmm.expand_qm_pc_region` grows the region by a different criterion: it runs a
QM/MM gradient and adds every molecule whose point-charge gradient exceeds `thresh`.

## Embedding

`embedding="elstat"` (the default) is electrostatic embedding: the MM point charges enter the
QM Hamiltonian, so the QM region is polarized by its environment. `embedding="mech"` is
mechanical embedding, where the two regions interact only through the MM force field.

For electrostatically embedded ML/MM, pass {class}`~openmmqmmm.MACESCFTheory` as
`qm_theory`. Its FixedPoint or QEq model receives the same MM point charges and returns
both QM forces and forces on those charges. Set `qm_mult=1`; the adapter passes the QM
total charge but has no spin input. The model's total energy already includes the
electrostatic coupling. The usual force-field partition, link atoms and MM virtual-site
handling apply, while periodic systems retain the finite embedding model below. See
{doc}`theories` for checkpoint and calculator setup.

## Polarizable MM force fields

Both Drude and AMOEBA force fields can be used as the MM model, with different QM/MM
couplings:

| MM model | Electrostatic embedding | Mechanical embedding |
| --- | --- | --- |
| Drude oscillators | Explicit core and shell charges enter the QM Hamiltonian; QM forces act on the shells. | Native Drude polarization and classical QM–MM interactions remain. |
| AMOEBA multipoles | Unsupported: the QM backend interface accepts point charges, not multipoles or induced dipoles. | Native AMOEBA multipoles and polarization remain in the MM coupling. |

### Drude embedding

For an MM system containing an OpenMM `DrudeForce`, `embedding="drude"` and
`embedding="polembed_drude"` are aliases for `embedding="elstat"`. The explicit names
require a Drude force field. `embedding="mech"` is also supported.

```python
qm_mm = QMMMTheory(
    qm_theory=qm,
    mm_theory=mm,
    fragment=fragment,
    qmatoms=qmatoms,
    qm_charge=0,
    qm_mult=1,
    embedding="drude",
)
```

The fragment must contain **every OpenMM particle**, in topology order, including Drude
shells and native virtual sites. Use element `"M"` for particles without a chemical
element; see the setup example in {doc}`system_setup`. Drude shells are independent
coordinates, while native virtual sites follow their hosts. The QM point-charge gradients
therefore contribute directly to the shell forces as well as to the other MM forces.

Select complete QM molecules that have no Drude particles or dependencies. Every Drude
shell, its parent, and its anisotropy reference atoms must remain MM. Native virtual sites
and their hosts must also remain MM. Covalent QM/MM cuts are unsupported for Drude
systems. These checks prevent a split oscillator or an undefined polarization boundary.
Prepare the QM solute with compatible non-Drude parameters and the MM environment with
Drude parameters; selecting physical atoms of a Drude molecule does not convert it into
a QM molecule.
Custom `charges=` may not change a Drude shell's force-field charge, since that charge also
sets its spring and screening model.

A standalone energy or gradient evaluates the **supplied shell coordinates**; it does not
relax them internally. Changing only the QM coordinates leaves the MM polarization
coordinates at their supplied positions. A separate shell minimization must use the
complete QM/MM potential, including the QM forces. For the supported dynamics route,
choose `DrudeLangevinIntegrator` or `DrudeSCFIntegrator` explicitly as described in
{doc}`dynamics`. The former propagates the shells with a cold thermostat; the latter
relaxes them during integration, including the QM contribution to their forces.

### AMOEBA mechanical embedding

Build `OpenMMTheory` with an AMOEBA force field, then choose `embedding="mech"` and
complete QM molecules:

```python
qm_mm = QMMMTheory(
    qm_theory=qm,
    mm_theory=mm,
    fragment=fragment,
    qmatoms=qmatoms,
    qm_charge=0,
    qm_mult=1,
    embedding="mech",
)
```

The native AMOEBA force field determines permanent multipoles and induced polarization.
Internal QM bonded and van der Waals terms are removed. The multipole contribution is the
full-system AMOEBA multipole energy minus a separate AMOEBA multipole calculation for the
QM molecules alone, using the same electrostatic method and periodic box. This retains
the MM polarization and the classical interaction between the regions without counting
the isolated QM multipole model twice. QM–QM van der Waals pair exclusions follow
OpenMM's native policy; periodic dispersion-tail contributions can remain even for an
entirely QM system.

The QM calculation receives no MM point charges in this mode. AMOEBA polarization
responds to the force-field multipoles on the QM atoms; it is not coupled to the QM
electron density. Local multipole frames, polarization groups and reduced van der Waals
sites must stay within one region. Electrostatic embedding, covalent cuts, `charges=`
overrides and population-charge updates are rejected rather than replacing AMOEBA by its
monopoles. Generalized Kirkwood implicit solvent, WCA dispersion and unrecognized forces
are unsupported for this QM/MM partition.
See {doc}`system_setup` for an AMOEBA construction example.

## Fixed-charge force-field partition

Electrostatic embedding removes MM charges and Coulomb exception charge products involving
QM atoms; QM–MM Lennard-Jones interactions remain. Internal QM bonded terms and direct
QM–QM nonbonded pairs are removed, including their nonbonded parameter offsets. Periodic
MM image and tail contributions follow the policy described below. At a covalent boundary,
the force-field policy removes angles with at least two QM atoms and ordinary, custom,
and RB torsions with at least
three. A CMAP spanning the boundary remains in MM; a CMAP whose unique atoms are all QM is
removed. Crossing bond terms remain unless `delete_qm1_mm1_bonded=True` was set on the MM
theory.

Native OpenMM virtual charge sites and all their hosts must remain in the MM region.
Standalone calculations recompute the sites from their hosts before evaluating the QM
field and return forces on the independent host coordinates, with zero virtual-site rows.
The native OpenMM callback performs its own force redistribution during dynamics. Two- and
three-particle average, out-of-plane, and local-coordinate sites are supported. A QM region
containing a native site or any of its hosts is rejected because a charge-conserving
partition of those dependencies is not defined.

## Periodic embedding

For a periodic OpenMM system, the force-field topology defines bonded molecules and the
covalent QM/MM boundary. Every QM/MM evaluation unwraps these molecules, brings separate
QM molecules together, and images other whole molecules around the QM region. OpenMM
virtual charge sites follow their hosts. Equivalent lattice images therefore produce the
same finite embedding cluster, including caps and charge-shift sites. A covalent network
that winds around the cell is rejected because it cannot be unwrapped without cutting a
bond.

Standalone calls use the OpenMM System's default box. `QMMMTheory.run` also accepts
`periodic_box_vectors`, a 3 × 3 array in Å, and uses that cell for both QM imaging and the MM
energy. Classical and RPMD force callbacks supply the instantaneous OpenMM box, including
barostat trial boxes.

This remains **finite-field electrostatic embedding**: the QM Hamiltonian contains one
image of each MM molecule, without a periodic/Ewald sum for QM–MM or QM-image interactions.
The MM subsystem uses its configured periodic force field. Molecule image changes can
introduce discontinuities in the finite QM field. Check cell-size convergence and energy
conservation for the intended calculation; image consistency alone does not establish
periodic electrostatic accuracy. `embedding="pbcmm-elstat"` remains unsupported.

Use the {doc}`periodic-validation` protocol to compare cell sizes, complete-molecule
embedding extents, image-switch jumps and NVE energy drift. It includes a reproducible
analytic baseline and a separate protocol for convergence of the intended QM system.

For standard point-charge force fields, QM–QM nonbonded exclusions remove direct pairs
in the primary cell. The configured MM
periodic image interactions and dispersion-tail corrections remain. Thus an entirely QM
periodic system can still have MM energy, forces, and pressure contributions: mechanical
embedding retains periodic Coulomb image terms, and either embedding can retain LJ image
or tail terms. This is the defined finite-QM/periodic-MM model. Converge cell size and
long-range MM settings as well as the finite QM charge field.

## The QM/MM boundary

A boundary cutting through a covalent bond is capped with a link atom. The QM calculation
sees a QM–H bond where the real system has a QM–MM bond, and the charge of the MM host atom
is redistributed so the cap does not sit on top of a full point charge.

OpenMM's covalent topology determines both the boundary and the charge-redistribution
neighbours for periodic and nonperiodic systems. Distance inference is used only when no
topology is available. A stretched bond or close nonbonded contact therefore does not
change a topology-defined cut.

`linkatom_method`
: `"simple"` (default) places the link atom at a fixed distance along the broken bond, set by
  `linkatom_simple_distance`. `"ratio"` places it at a fixed fraction of the bond, set by
  `linkatom_ratio` (0.723 by default). The force projection differentiates this placement
  rule, including both bond stretching and changes in bond direction.

`linkatom_type`
: The capping element, `"H"` by default.

`linkatom_forceproj_method`
: `"adv"` (default), `"lever"`, and `"chain"` are compatibility aliases for the exact
  derivative of the chosen `linkatom_method`. For `"ratio"`, the QM and MM hosts receive
  fractions `1-linkatom_ratio` and `linkatom_ratio` of the cap force. For `"simple"`, the
  projection accounts for the fixed cap distance as the host bond changes direction.
  `"none"` or `None` deliberately omits the cap force: use it only for diagnostics, since
  the resulting forces generally are not derivatives of the reported energy.

`chargeboundary_method`
: `"shift"` (default) moves the host charge onto its neighbours, with `dipole_correction`
  restoring the dipole those shifts would otherwise change. `"rcd"` is the redistributed
  charge and dipole scheme. Other capped boundary hosts cannot receive redistributed
  charge. An electrostatic partition without a non-boundary MM neighbour is rejected;
  expand the QM region or choose a different cut.

`excludeboundaryatomlist` and `unusualboundary`
: Escape hatches for boundaries the automatic detection declines to cut — an unusual bond
  order or a boundary you have deliberately chosen. Without `unusualboundary=True`, an
  unexpected boundary is an error rather than a silent approximation.

Cut boundaries across non-polar single bonds, as far from the chemistry as the budget allows,
and never across a bond in a conjugated or charged group.

## Point charges

Every MM atom is a point charge in the QM calculation by default. Truncation is disabled
unless requested with `truncated_pc=True`.

`truncated_pc=True`
: With the default `truncated_pc_recalc_iter=1`, performs **one full-field QM calculation**
  per evaluation. It does not calculate a subset or reuse a correction, and returns the
  full energy and all QM, cap, and point-charge gradients. This has no truncation speedup.
  Cached energy calculations require an explicit refresh interval greater than 1.

`truncated_pc_selection="nearest"`
: For cached calculations, selects a charge group if any of its sites lies strictly within
  `truncated_pc_radius` (55 Å by default) of any **real QM atom**. Link atoms are excluded
  from this distance calculation. `truncated_pc_selection="centroid"` instead uses the
  arithmetic centroid of the QM coordinates including link atoms, retaining the historical
  distance definition. A centroid radius can omit charges near the edge of an extended
  QM region. Both modes preserve complete charge groups; neither uses smooth switching.

  Groups follow complete OpenMM topology molecules, including connectivity through QM
  atoms. Drude particles, native virtual sites, and boundary dipole/RCD sites stay with
  their parents and charge-redistribution dependencies. Thus the MM portion of a large
  QM-containing macromolecule may all be selected. When OpenMM topology is unavailable,
  grouping uses `Fragment.connectivity` and known boundary dependencies. Sites without
  connectivity metadata are independent; provide explicit groups when they belong together.

`truncated_pc_groups=`
: Optional groups of **full-system atom indices** that merge automatic groups, without
  splitting existing molecules or site dependencies. For example, groups can keep a charged
  solute and chosen counterions together. Complete molecules do not guarantee a neutral
  selected field; inspect its net charge and converge the extent for the intended observable.

With a refresh interval greater than 1, the first evaluation and scheduled refreshes return
the full-field energy. Other calls reuse the selected site indices, update their coordinates,
and return `E_subset(current) + E_full(refresh) - E_subset(refresh)`. This scalar correction
does not restore the full-field electronic state between refreshes. These energies depend on
calculation history and have no guaranteed error bound.

At a scheduled refresh, the old subset is evaluated at the current geometry to measure
`E_old_subset(current) + old_correction - E_full(current)`. If the new selection differs,
its energy is also calculated to initialize the new correction. The full field is evaluated
last, so backend properties correspond to the returned full-field energy. Refreshes normally
cost two QM calls, or three when the selection changes; cached calls cost one. If complete
groups select every charge site, the subset is already the full field and its calculation
is reused, avoiding a duplicate call. These evaluations report mode `"full"`.

`theory.truncated_pc_diagnostics` reports `mode` (`"full"`, `"refresh"`, or `"cached"`),
`selected_count`, `full_count`, `selected_charge`, `full_charge`, `last_refresh_error`, and
`fallback`. Charges are in elementary-charge units. `last_refresh_error` is the signed
old-prediction-minus-full-field error in Eh at the last checked refresh, or `None` before
such a comparison. It measures the actual cached prediction at that geometry, even when
selection changes; it is not merely the change in correction values. Counts and charges in
`"refresh"` mode describe the subset selected for subsequent cached calls.

Set `truncated_pc_error_tolerance` to a positive finite energy in Eh to stop reusing cached
corrections when the absolute observed refresh error exceeds that tolerance. The triggering
call returns the full energy, and subsequent calls each perform one full-field calculation.
This fallback stays active until `theory.reset_truncated_pc_cache()` or a configuration
change. The tolerance is a checkpoint test: it does **not** bound errors between refreshes
or retroactively correct earlier energies. Its default is `None`.

Changing selection settings, groups, refresh interval, error tolerance, QM charge/multiplicity,
or the QM backend object resets the cache and fallback. An evaluation with truncation disabled
also clears them. Call `theory.reset_truncated_pc_cache()` after changing settings inside the
same QM backend object, since those changes cannot be detected automatically. Failed QM
evaluations cannot reuse a partially refreshed correction.

For example, an explicitly cached energy calculation can use:

```python
theory = QMMMTheory(
    fragment=fragment,
    qm_theory=qm,
    mm_theory=mm,
    qmatoms=qmatoms,
    qm_charge=0,
    qm_mult=1,
    truncated_pc=True,
    truncated_pc_radius=20.0,  # Å from the nearest real QM atom
    truncated_pc_recalc_iter=10,  # explicitly enable cached energies
    truncated_pc_error_tolerance=1e-4,  # Eh; choose for the intended observable
)
energy = theory.run(current_coords=fragment.coords)
print(theory.truncated_pc_diagnostics)
```

The radius must be finite and positive, and the refresh interval a positive integer.
Gradients, optimization, and numerical frequencies require interval 1 or
`truncated_pc=False`; independently cached energy and gradient corrections are not
consistent derivatives. This restriction includes `NumGrad`,
`optimize_geometry(num_grad=True)`, and numerical frequencies of wrapped theories.
The fallback does not relax these restrictions. Classical and ring-polymer QM/MM dynamics
reject `truncated_pc=True` even at interval 1 ({doc}`dynamics`, {doc}`rpmd`).

**Migration:** the refresh-interval default changed from 50 to 1, and the distance default
changed from centroid to nearest real QM atom. Existing `truncated_pc=True` setups without
an explicit interval now use the full field on every call. Choose an interval greater than 1
to request cached energies, and `"centroid"` if the historical distance definition is needed.
Complete-group selection applies in both distance modes.

`charges=`
: Supply the MM charges yourself instead of taking them from the MM theory.
  {func}`~openmmqmmm.read_charges_from_psf` reads them from a CHARMM PSF file.

`update_qm_region_charges=True`
: For uncapped mechanical embedding, recompute the QM population charges for each
  energy-only evaluation. ORCA uses Mulliken populations regardless of population logging.
  Other backends must implement `get_atomic_charges()` or initialize a legacy `charges`
  attribute. Charges must be finite, match the QM atom count and order, and preserve the
  specified QM-region charge within output rounding. Gradients, optimization, frequencies,
  and dynamics reject this option because population-charge response derivatives are not
  implemented. Capped regions are rejected until a conserving map of cap populations is
  defined. Use fixed charges for these workflows.

Fixed charge assignment preserves known Coulomb exception scaling, including a charge
passing through zero. Existing zero-product exceptions are treated as exclusions unless
a previous update established their scaling. An ambiguous nonzero custom exception or
independent charge offset is rejected before mutation when it cannot be rescaled to the
new fixed charges. QM–MM Lennard-Jones exception parameters are preserved.

## The active region

Optimizing every atom of a solvated protein is neither possible nor interesting. Pass
`actatoms=` to {func}`~openmmqmmm.optimize_geometry` and everything else stays frozen:

```python
from openmmqmmm import optimize_geometry

optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
```

A larger active region than the QM region is the usual choice: the QM atoms plus the residues
around them, from {func}`~openmmqmmm.define_active_region`. Note that
`define_active_region` takes whole residues, so no residue is ever half-frozen.

## Decomposing the energy

```python
from openmmqmmm import compute_decomposed_qm_mm_energy

compute_decomposed_qm_mm_energy(theory=qm_mm, fragment=fragment)
```

Logs the QM, MM and QM–MM coupling contributions separately, which is how you find out
whether a surprising energy came from the chemistry or from the force field.

Use a standalone electrostatically embedded theory before attaching an MD callback. LJ
decomposition evaluates cloned forces, preserving the live MM System and the original cap
settings. It includes standard nonbonded exceptions and the supported CHARMM/GROMACS LJ
custom forces. An unrecognized custom pair potential raises an error instead of being
silently omitted. Covalent coupling is not decomposed separately: the function logs this
limitation and retains those terms in the MM contribution.

The LJ decomposition helper does not support AMOEBA's buffered van der Waals potential
and raises an error for it. Use OpenMM's native force-group decomposition to inspect the
AMOEBA force contributions.

## Running it

Anything that takes a theory takes this one:

```python
from openmmqmmm import numerical_frequencies, openmm_md, optimize_geometry, single_point

single_point(theory=qm_mm, fragment=fragment)
optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
numerical_frequencies(theory=qm_mm, fragment=fragment, hessatoms=qmatoms)
openmm_md(theory=qm_mm, fragment=fragment, timestep=0.0005, simulation_time=2)
```

For QM/MM MD see {doc}`dynamics`; for ring-polymer QM/MM MD see {doc}`rpmd`.
