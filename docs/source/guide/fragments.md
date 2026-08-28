# Fragments

A {class}`~openmmqmmm.Fragment` is the molecular system: elements, coordinates in Angstrom,
charge, multiplicity, connectivity and — when it came from a PDB file — an OpenMM topology.
Every job function takes one.

## Building a fragment

`Fragment` takes keyword arguments only, and each input route is its own keyword:

```python
from openmmqmmm import Fragment

Fragment(pdbfile="system.pdb")  # also pdbxfile= for mmCIF
Fragment(xyzfile="molecule.xyz", charge=0, mult=1)
Fragment(coordsstring="H 0.0 0.0 0.0\nF 0.0 0.0 1.0", charge=0, mult=1)
Fragment(elems=["O", "H", "H"], coords=coordinates)  # coords: (natoms, 3) array or nested list
Fragment(smiles="CCO")  # via OpenBabel, with 3D coordinates generated
Fragment(grofile="conf.gro")  # GROMACS
Fragment(amber_prmtopfile="system.prmtop", amber_inpcrdfile="system.inpcrd")
Fragment(chemshellfile="system.c")
Fragment(fragfile="system.frag")  # written earlier by print_system
```

Two more build a fragment without a file: `atom="Fe"` for a single atom, and
`diatomic="HF"` with `diatomic_bondlength=0.92` (in Angstrom) for a diatomic. Passing
`fragments=[...]` concatenates several fragments into one.

## Charge and multiplicity

Set `charge=` and `mult=` at construction where you can. A fragment carrying them lets the
job functions and theories take them from the fragment, so they need not be repeated at
every call. Where they are genuinely unknown, most job functions accept `charge=` and
`mult=` of their own, which override the fragment.

`readchargemult=True` reads them from the second line of an XYZ file, which is where
{func}`~openmmqmmm.write_xyzfile` puts them.

## Connectivity

Connectivity is a list of atom-index lists, one per connected molecule. It is computed on
request rather than at construction:

```python
fragment = Fragment(pdbfile="system.pdb", conncalc=True)
# or later
fragment.calc_connectivity(scale=1.0, tol=0.1)
```

`scale` and `tol` set the covalent-radius criterion for a bond. The QM/MM link-atom
machinery and {func}`~openmmqmmm.expand_qm_region` both need connectivity; they compute it
if it is missing.

## Inspecting

```python
fragment.info()  # formula, charge, multiplicity, number of atoms
fragment.print_coords()  # the full coordinate table
fragment.print_coords_for_atoms(qmatoms)
fragment.get_coords_for_atoms(qmatoms)
fragment.get_non_h_atomindices()
fragment.get_centroid()
```

## Writing

```python
fragment.write_xyzfile("out.xyz")
fragment.write_pdbfile("out")  # writes out.pdb
fragment.print_system("system.frag")  # everything, including connectivity, for fragfile=
```

A fragment built from a PDB file keeps its atom, residue, chain and segment names, and
`write_pdbfile` reuses them. A fragment built any other way has none, so the PDB it writes
carries placeholder names.

The `.frag` format is the package's own: it round-trips coordinates, charge, multiplicity
and connectivity, which no coordinate format does.

## Measuring geometry

```python
from openmmqmmm import angle_between_atoms, dihedral_between_atoms, distance_between_atoms

distance_between_atoms(fragment=fragment, atoms=[93, 94])  # Angstrom
angle_between_atoms(fragment=fragment, atoms=[93, 94, 95])  # degrees
dihedral_between_atoms(fragment=fragment, atoms=[93, 94, 95, 96])  # degrees
```

Each takes the atom indices as one `atoms=` sequence, of the length that measurement needs.

{func}`~openmmqmmm.print_internal_coordinate_table` logs all of the bonds, angles and
dihedrals at once, and {func}`~openmmqmmm.calculate_rmsd` compares two structures.

## Aligning and combining

{func}`~openmmqmmm.flexible_align` aligns two fragments; `flexible_align_xyz` and
`flexible_align_pdb` do the same for files. With `reordering=True` they use the `rmsd`
package to find the atom mapping first, which matters when two structures describe the same
molecule with different atom ordering.

{func}`~openmmqmmm.insert_solute_into_solvent` places a solute in a solvent box and removes
the overlapping solvent molecules — the manual counterpart to
{func}`~openmmqmmm.solvate_small_molecule` in {doc}`system_setup`.

## Reading coordinates without a fragment

{func}`~openmmqmmm.read_xyzfile` and friends return elements and coordinates rather than a
fragment, which is what you want when reading many structures:

```python
from openmmqmmm import read_xyzfile, read_xyzfiles, split_multimolxyzfile

elems, coords = read_xyzfile("molecule.xyz")  # elements and coordinates
fragments = read_xyzfiles("structures/")  # every .xyz in a directory
split_multimolxyzfile("trajectory.xyz", writexyz=True)  # one file per frame
```

{func}`~openmmqmmm.get_molecules_from_trajectory` pulls whole molecules out of an MD
trajectory, using connectivity rather than residue records.
