# QM/MM

{class}`~openmmqmmm.QMMMTheory` couples a QM theory and an MM theory into one theory object.
It behaves like any other theory, so every job function in {doc}`jobs` takes it.

```python
from openmmqmmm import Fragment, OpenMMTheory, ORCATheory, QMMMTheory

fragment = Fragment(pdbfile="system.pdb")

qm = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
mm = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml"], pdbfile="system.pdb", periodic=True)

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

{func}`~openmmqmmm.expand_qm_pc_region` is the diagnostic counterpart: it reports which point
charges actually matter, by the size of their contribution.

## Embedding

`embedding="elstat"` (the default) is electrostatic embedding: the MM point charges enter the
QM Hamiltonian, so the QM region is polarized by its environment. `embedding="mech"` is
mechanical embedding, where the two regions interact only through the MM force field.
Polarizable Drude embedding (`"polembed_drude"`) is not implemented and raises an error.

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

QM–QM nonbonded exclusions remove direct pairs in the primary cell. The configured MM
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

Every MM atom is a point charge in the QM calculation by default. For a large periodic system
that is a lot of charges:

`truncated_pc=True`
: Uses only the charges within `truncated_pc_radius` (55 Å by default) for most steps,
  recomputing the full set every `truncated_pc_recalc_iter` calls. Cached corrections are
  an energy-only, history-dependent approximation. Gradient evaluations, optimization,
  and numerical frequencies require `truncated_pc_recalc_iter=1`, or `truncated_pc=False`:
  independently cached energy and gradient corrections are not consistent derivatives.
  With interval 1, every evaluation restores the full-field energy and all QM, cap, and
  point-charge gradients. This mode provides no truncation speedup. The radius must be
  finite and positive, and the refresh interval a positive integer. Classical and
  ring-polymer QM/MM dynamics reject `truncated_pc` ({doc}`dynamics`, {doc}`rpmd`).

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

## Running it

Anything that takes a theory takes this one:

```python
from openmmqmmm import numerical_frequencies, openmm_md, optimize_geometry, single_point

single_point(theory=qm_mm, fragment=fragment)
optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
numerical_frequencies(theory=qm_mm, fragment=fragment, hessatoms=qmatoms)
openmm_md(theory=qm_mm, fragment=fragment, timestep=0.001, simulation_time=2)
```

For QM/MM MD see {doc}`dynamics`; for ring-polymer QM/MM MD see {doc}`rpmd`.
