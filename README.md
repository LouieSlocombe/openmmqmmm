# openmmqmmm — ORCA + OpenMM QM/MM

A trimmed distribution of the [ASH](https://github.com/RagnarB83/ash) multiscale modelling program, reduced to the
**ORCA + OpenMM QM/MM stack for biomolecular calculations**. The Python package is named `openmmqmmm`; the API keeps the
upstream ASH class and function names, so existing ORCA/OpenMM QM/MM scripts only need their import changed from `ash`
to `openmmqmmm`.

## What is included

- `ORCATheory` — interface to the [ORCA](https://www.faccts.de/orca/) quantum chemistry program
- `OpenMMTheory` — interface to the [OpenMM](https://openmm.org) MM library, plus
  `OpenMM_MD` (also aliased `MolecularDynamics`), `OpenMM_Modeller` (pdbfixer-based protein setup),
  `OpenMM_Opt`, `OpenMM_box_equilibration`, `Gentle_warm_up_MD`, metadynamics,
  `solvate_small_molecule` and `small_molecule_parameterizer`
- `QMMMTheory` — electrostatically embedded QM/MM with link atoms and charge-shifting
- `Singlepoint` (+ fragment/theory/reaction variants), `Job_parallel`
- `Optimizer` / `geomeTRICOptimizer` — geometry optimization via geomeTRIC, including frozen/active-region optimizations
  of large biomolecular systems (`ActiveRegion`)
- `NumFreq` / `AnFreq` — numerical/analytical frequencies with partial Hessians and thermochemistry
- `Fragment` — coordinates/topology handling incl. PDB, Amber, GROMACS file reading
- Helper interfaces genuinely used by the above: mdtraj (trajectory processing), openbabel (ligand conversion), Multiwfn
  (density/charge analysis), plotting

Everything else from upstream ASH (other QM-code interfaces, NEB/knarr, molcrys, PES, high-level workflows, ONIOM,
machine-learning tools, …) has been removed.

## Citation

This package is derived from ASH. If it is useful in your research please cite:
[ASH: a Multi-scale, Multi-theory Modeling program](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70359), R.
Bjornsson, *J. Comput. Chem* **2026**, 47, e70359.

## Installation

**Requirements**

- Linux or macOS, Python ≥ 3.10 (developed and tested on 3.13)
- [OpenMM](https://openmm.org) ≥ 8, [PDBFixer](https://github.com/openmm/pdbfixer) and
  [mdtraj](https://www.mdtraj.org) — installed from conda-forge (PDBFixer is not on PyPI, so a
  conda/mamba environment is the recommended route)
- [geomeTRIC](https://github.com/leeping/geomeTRIC), numpy, packaging — pulled in automatically by pip
- [ORCA](https://www.faccts.de/orca/) — installed separately (free for academic use); required for
  `ORCATheory` and QM/MM, not for the pure-MM/OpenMM functionality

**Conda environment (recommended)**

From the repository root:

```sh
conda env create -f environment.yml
conda activate openmmqmmm
pip install -e .
```

`pip install -e .` is an editable (development) install: changes to the source tree take effect without
reinstalling. Use `pip install .` for a regular install. The environment also provides `pytest` and
`python-build` for testing and building.

**Installing into an existing environment**

```sh
conda install -c conda-forge "openmm>=8" pdbfixer mdtraj
pip install .
```

**Optional dependencies**

Some functionality uses extra packages, all available on conda-forge: `openbabel` (ligand format
conversion in `small_molecule_parameterizer`), `matplotlib`/`scipy` (plotting), `parmed`
(Amber/GROMACS file handling in `OpenMMTheory`). They are listed, commented out, in
`environment.yml`.

**Configuring ORCA**

Either make sure the `orca` binary is in `PATH`, or point the package at your ORCA installation in one
of two ways:

- pass `orcadir="/path/to/orca_directory"` to `ORCATheory`, or
- create `~/ash_user_settings.ini`:

  ```ini
  [Settings]
  orcadir = /path/to/orca_directory
  ```

For parallel ORCA runs (`numcores` > 1) the matching OpenMPI version must also be set up, as for any
ORCA installation.

**Verifying the installation**

```sh
python -c "import openmmqmmm"
cd openmmqmmm/tests && pytest -q
```

The fragment/OpenMM/optimizer tests run without ORCA; the QM/MM tests are skipped automatically when
no `orca` binary is found in `PATH`.

## Building distributions

```sh
python -m build
```

This produces an sdist and a wheel under `dist/`. Wheels ship only the runtime data files
(`log.ini`, the Multiwfn `settings.ini`); the test suite and its ~32 MB of reference data stay in the
source repository, so run the tests from a checkout, not from an installed wheel.

## Basic example

```py
from openmmqmmm import *

coords="""
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
#Create fragment from multi-line string
HF_frag=Fragment(coordsstring=coords, charge=0, mult=1)

#Create ORCATheory object
input="! r2SCAN def2-SVP def2/J tightscf"
blocks="%scf maxiter 200 end"
ORCAcalc = ORCATheory(orcasimpleinput=input, orcablocks=blocks)

#Singlepoint calculation
Singlepoint(theory=ORCAcalc,fragment=HF_frag)

#Call optimizer
Optimizer(theory=ORCAcalc,fragment=HF_frag)

#Numerical frequencies
NumFreq(theory=ORCAcalc,fragment=HF_frag)

#DFT Molecular dynamics simulation for 2 ps with a 0.001 ps (1 fs) timestep
MolecularDynamics(fragment=HF_frag, theory=ORCAcalc, timestep=0.001, simulation_time=2)
```

## QM/MM example

```py
from openmmqmmm import *

# Defining a fragment
fragment = Fragment(pdbfile="system.pdb")
# QM-method and QM-region
qm_orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
# MM Theory
omm  = OpenMMTheory(xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"],
                    pdbfile="system.pdb", periodic=True)

# QM/MM Theory
qmatoms = [93,94,95,96,97,133,134,135, 2001,2002]
qm_mm = QMMMTheory(qm_theory= qm_orca, mm_theory= omm, fragment=fragment,
                    qm_charge=-1, qm_mult=6,  qmatoms= qmatoms, printlevel=1)

# Geometry optimization
Optimizer(theory=qm_mm,fragment=fragment, actatoms=qmatoms)
# or Molecular dynamics
MolecularDynamics(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=2)
```

## Documentation

Upstream ASH documentation (applies to the retained functionality): https://ash.readthedocs.io
