# Preparing a system

Going from a PDB file downloaded this morning to something an MD engine will accept:
missing atoms added, protonation decided, a box of water and ions around it, and a force
field that covers every residue.

## openmm_modeller

{func}`~openmmqmmm.openmm_modeller` is the whole path in one call. It returns a configured
{class}`~openmmqmmm.OpenMMTheory` and the matching {class}`~openmmqmmm.Fragment`:

```python
from openmmqmmm import openmm_modeller

mm, fragment = openmm_modeller(
    pdbfile="protein.pdb",
    forcefield="CHARMM36",
    watermodel="tip3p",
    solvent_padding=10.0,  # Angstrom
    ionicstrength=0.1,  # molar
    ph=7.0,
)
```

`forcefield=` takes a name — `"Amber14"`, `"Amber99sb"`, `"Amber99sb-ildn"`, `"Amber96"`,
`"Amber03"`, `"Amber10"`, `"CHARMM36"`, `"CHARMM2013"`, `"Amoeba2013"` or `"Amoeba2009"` —
and maps it to the OpenMM XML that ships with OpenMM. `xmlfile=` names one directly instead,
`extraxmlfile=` adds one on top (this is where a ligand force field goes), and
`forcefield_object=` takes an OpenMM `ForceField` you built yourself.

By default PDBFixer repairs the structure first: missing heavy atoms and hydrogens are added
at the requested pH, and non-standard residues are handled where it can. `use_pdbfixer=False`
skips that when the input is already complete.

Other things it does:

`residue_variants=`
: Overrides the protonation state PDBFixer chose, per chain and residue — `HIE` versus
  `HID`, a protonated aspartate, a deprotonated cysteine. This is the argument you reach for
  when the active site is wrong.

`membrane=True`
: Builds a lipid bilayer (`membrane_lipidtype`, `"POPC"` by default) instead of a plain water
  box.

`implicit=True`
: Implicit solvent rather than an explicit box.

`use_higher_occupancy=True`
: Picks the highest-occupancy alternate location where a crystal structure has several.
  {func}`~openmmqmmm.openmm.find_alternate_locations_residues` lists them first.

Each stage writes its own PDB file — `system_afterfixes.pdb`, `system_afterH.pdb`,
`system_aftersolvent.pdb`, `system_aftersolvent_ions.pdb` and `finalsystem.pdb` — so when the
result is wrong you can see which step made it wrong, rather than only what came out at the
end.

## Ligands and other non-standard residues

A biomolecular force field covers amino acids, nucleic acids, water and ions — and nothing
else. Anything else in the file needs parameters of its own, which is what
[forcefill](https://github.com/LouieSlocombe/forcefill) produces.

The direct route builds the XML first and passes it in:

```python
from forcefill import build_ligand_xml

result = build_ligand_xml({"LIG": "ligand.sdf"}, "lig_ff.xml")  # or LigandSpec(smiles=...)

openmm_modeller(pdbfile="complex.pdb", forcefield="Amber14", extraxmlfile=result.forcefield_xml)
```

The same XML works anywhere a force field does:

```python
OpenMMTheory(xmlfiles=["amber14-all.xml", "amber14/tip3p.xml", "lig_ff.xml"], pdbfile="complex.pdb", periodic=True)
solvate_small_molecule(fragment=fragment, xmlfile=result.forcefield_xml, watermodel="tip3p")
```

The one-step route lets `openmm_modeller` do it: every residue the chosen force field cannot
match is parameterized through `forcefill.build_forcefield_xml`, and the generated
`nonstandard_ff.xml` is loaded automatically.

```python
openmm_modeller(
    pdbfile="complex.pdb",
    forcefield="Amber14",
    parameterize_nonstandard=True,
    net_charges={"LIG": 0},
)
```

Non-standard residues must carry explicit hydrogens and CONECT records in the PDB file.
`ligand_files={"LIG": "ligand.sdf"}` supplies bond orders from a file rather than perceiving
them from PDB geometry, and `ligand_backend` selects `"gaff"` (the default, via
antechamber/AM1-BCC), `"smirnoff"` (OpenFF) or `"charmm"` (CGenFF). For finer control —
charge methods, per-ligand `LigandSpec`, minimization checks — call forcefill directly and
pass the XML through `extraxmlfile=`.

:::{note}
forcefill is not on PyPI. Without it, `parameterize_nonstandard=True` raises
{exc}`~openmmqmmm.MissingDependencyError`; nothing else is affected. `conda_install.sh`
installs it — see {doc}`../install`.
:::

XYZ-only input has no bond orders, so convert to SDF first (RDKit's `rdDetermineBonds`, or
OpenBabel).

## A small molecule in water

{func}`~openmmqmmm.solvate_small_molecule` is the non-protein path: it takes a fragment,
builds a water box around it and returns the force field, topology and solvated fragment.

```python
from openmmqmmm import solvate_small_molecule

forcefield, topology, solvated = solvate_small_molecule(fragment=fragment, xmlfile="lig_ff.xml", watermodel="tip3p")
```

{func}`~openmmqmmm.insert_solute_into_solvent` is the more manual alternative, for an
existing solvent box of something other than water.

## Merging structures

{func}`~openmmqmmm.merge_pdb_files` concatenates two PDB files into one, renumbering as it
goes — a protein and a separately prepared ligand, say.

## Minimization

```python
from openmmqmmm import openmm_minimize

fragment = openmm_minimize(fragment=fragment, theory=mm, maxiter=1000, tolerance=1)
```

Pure MM, using OpenMM's own minimizer, and it updates the fragment. Run it before any
dynamics: a freshly solvated system almost always contains a clash that will otherwise blow
up in the first few steps. {func}`~openmmqmmm.check_gradient_for_bad_atoms` names the atoms
carrying an implausible force when it does.

## Equilibration

{func}`~openmmqmmm.openmm_box_equilibration` runs NPT in cycles until the box volume and
density stop changing:

```python
from openmmqmmm import openmm_box_equilibration

openmm_box_equilibration(
    fragment=fragment,
    theory=mm,
    numsteps_per_npt=10000,
    max_npt_cycles=10,
    volume_threshold=1.0,
    density_threshold=0.001,
)
```

{func}`~openmmqmmm.gentle_warmup_md` raises the temperature in stages, each with its own
timestep, which a system that will not survive going straight to 300 K needs:

```python
from openmmqmmm import gentle_warmup_md

gentle_warmup_md(theory=mm, fragment=fragment, initial_opt=True, check_gradient_first=True)
```

With those two done, the system is ready for {doc}`dynamics` — or for a QM region and
{doc}`qmmm`.
