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
`"polembed_drude"` covers polarizable Drude force fields.

## The QM/MM boundary

A boundary cutting through a covalent bond is capped with a link atom. The QM calculation
sees a QM–H bond where the real system has a QM–MM bond, and the charge of the MM host atom
is redistributed so the cap does not sit on top of a full point charge.

`linkatom_method`
: `"simple"` (default) places the link atom at a fixed distance along the broken bond, set by
  `linkatom_simple_distance`. `"ratio"` places it at a fixed fraction of the bond, set by
  `linkatom_ratio` (0.723 by default).

`linkatom_type`
: The capping element, `"H"` by default.

`linkatom_forceproj_method`
: How the link atom's force is projected back onto the two real atoms: `"adv"` (default),
  `"lever"`, `"chain"` or `"none"`.

`chargeboundary_method`
: `"shift"` (default) moves the host charge onto its neighbours, with `dipole_correction`
  restoring the dipole those shifts would otherwise change. `"rcd"` is the redistributed
  charge and dipole scheme.

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
  recomputing the full set every `truncated_pc_recalc_iter` steps. An approximation — check
  its effect on your system before relying on it, and note that it is rejected outright for
  ring-polymer dynamics ({doc}`rpmd`).

`charges=`
: Supply the MM charges yourself instead of taking them from the MM theory.
  {func}`~openmmqmmm.read_charges_from_psf` reads them from a CHARMM PSF file.

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
