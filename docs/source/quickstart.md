# Quick start

A gas-phase calculation needs nothing but ORCA: a molecule, a theory, and one job function.
This is [examples/gasphase_hf.py](https://github.com/LouieSlocombe/openmmqmmm/blob/main/examples/gasphase_hf.py),
which you can run as it stands.

## Turning on output

The package is silent by default, as a library should be. One call gives you the classic
calculation output:

```python
import openmmqmmm

openmmqmmm.configure_logging()
```

Without it you get results but no running commentary. See {doc}`guide/output`.

## A molecule and a theory

```python
from openmmqmmm import Fragment, ORCATheory, configure_logging, single_point

configure_logging()

coords = """
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
hf = Fragment(coordsstring=coords, charge=0, mult=1)

orca = ORCATheory(orcasimpleinput="! r2SCAN def2-SVP def2/J tightscf", orcablocks="%scf maxiter 200 end")
```

{class}`~openmmqmmm.Fragment` holds elements, coordinates in Angstrom, charge and
multiplicity. It reads PDB, XYZ, GROMACS, Amber and ChemShell files too, or builds a
structure from a SMILES string — see {doc}`guide/fragments`.

{class}`~openmmqmmm.ORCATheory` writes ORCA input files and parses what comes back.
`orcasimpleinput` is the `!` line and `orcablocks` the `%` blocks, passed through verbatim,
so anything ORCA accepts works here.

## Running jobs

```python
from openmmqmmm import numerical_frequencies, optimize_geometry, single_point

energy = single_point(theory=orca, fragment=hf)
print(f"Single-point energy: {energy.energy:.8f} Eh")

# Updates the fragment's coordinates in place.
optimized = optimize_geometry(theory=orca, fragment=hf)
print(f"Optimized energy:    {optimized.energy:.8f} Eh")

frequencies = numerical_frequencies(theory=orca, fragment=hf)
print(f"Frequencies (cm-1):  {frequencies.frequencies}")
```

Every job function takes `theory=` and `fragment=`, returns a {class}`~openmmqmmm.Results`
object, and writes that object to a JSON file next to your script —
`results_singlepoint.json`, `results_optimizer.json`, `results_numfreq.json`. Energies are
in Hartree, gradients in Hartree/Bohr, frequencies in cm⁻¹. See {doc}`guide/jobs`.

## Where next

- {doc}`guide/system_setup` — turning a raw PDB file into a solvated, parameterized system.
- {doc}`guide/qmmm` — combining an ORCA theory and an OpenMM theory into a QM/MM one.
- {doc}`examples` — the runnable scripts shipped with the repository.
