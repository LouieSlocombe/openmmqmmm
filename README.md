# openmmqmmm — ORCA + OpenMM QM/MM

Electrostatically embedded QM/MM for biomolecular systems, combining the
[ORCA](https://www.faccts.de/orca/) quantum chemistry program with the
[OpenMM](https://openmm.org) molecular-mechanics library. Derived from the
[ASH](https://github.com/RagnarB83/ash) multiscale modelling program and reduced to the
ORCA + OpenMM QM/MM stack, with a modernized, PEP8-style Python API.

> **Compatibility note:** version 1.0 renamed the public API (snake_case functions, no import-time
> side effects, logging instead of print). Scripts written for the 0.x releases need updating; the
> Conventions section below describes the naming now in force.

`openmmqmmm.__all__` is the full public API: `ORCATheory`, `OpenMMTheory`, `QMMMTheory` and
`Fragment`, the job functions `single_point`, `optimize_geometry` (via
[geomeTRIC](https://github.com/leeping/geomeTRIC)), `numerical_frequencies`,
`analytic_frequencies`, `openmm_md` and `job_parallel`, plus the OpenMM setup helpers
(`openmm_modeller`, `openmm_minimize`, `openmm_box_equilibration`, `gentle_warmup_md`,
`openmm_md_plumed`, `solvate_small_molecule`, `small_molecule_parameterizer`).

## Installation

**Requirements**

- Linux or macOS, Python ≥ 3.10
- Every Python dependency is required — there are no feature-gated extras. `pip install .` pulls
  the full set (ASE, OpenMM, PDBFixer, mdtraj, ParmEd, RDKit, openmmforcefields, OpenBabel,
  geomeTRIC, rmsd, multiprocess, numpy, scipy, packaging)
- Two of them are not on PyPI and must come from conda-forge, which is why the conda route below
  is the recommended one: **openff-toolkit** (needed by `small_molecule_parameterizer`) and
  **openmm-plumed** (needed by `openmm_md_plumed`). A pip-only install leaves those two
  entry points raising `MissingDependencyError`; everything else works
- [ORCA](https://www.faccts.de/orca/) — installed separately (free for academic use); required for
  `ORCATheory` and QM/MM, not for the pure-MM/OpenMM functionality

The full environment is large (~5 GB): openff-toolkit depends on AmberTools, which depends on
PyTorch and CUDA.

**Conda environment (recommended)**

From the repository root:

```sh
conda env create -f environment.yml
conda activate openmmqmmm
pip install -e .
```

`pip install -e .` is an editable (development) install: changes to the source tree take effect
without reinstalling. Use `pip install .` for a regular install.

**Installing into an existing environment**

```sh
conda install -c conda-forge ase "openmm>=8" pdbfixer mdtraj parmed rdkit openmmforcefields openff-toolkit openmm-plumed multiprocess rmsd
pip install .
```

`pip install .` then adds OpenBabel and geomeTRIC. OpenBabel has to come from PyPI: conda-forge
has no build for Python 3.13 or newer.

**Configuring ORCA**

ORCA is located in this order, and every candidate is validated (the directory must contain the
`orca` binary and its `orca_*` helper binaries):

1. the `orcadir` argument to `ORCATheory`,
2. the `OPENMMQMMM_ORCADIR` environment variable, e.g. `export OPENMMQMMM_ORCADIR=~/orca_6_1_1`,
3. an `orca` binary found in `PATH`.

For parallel ORCA runs (`numcores` > 1) the matching OpenMPI version must also be set up, as for
any ORCA installation.

## Output

The package is silent by default (standard library behavior). To get the classic calculation
output on the console — and optionally into a file — configure logging once at the top of a run
script:

```py
import openmmqmmm

openmmqmmm.configure_logging()  # INFO to console
# openmmqmmm.configure_logging(level="DEBUG", file="calc.log")
```

Step timings are logged at DEBUG level on the `openmmqmmm.timings` logger. The
`OPENMMQMMM_LOGLEVEL` environment variable overrides the level.

## QM/MM example

```py
from openmmqmmm import Fragment, ORCATheory, OpenMMTheory, QMMMTheory, configure_logging, openmm_md, optimize_geometry

configure_logging()

fragment = Fragment(pdbfile="system.pdb")

qm_orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
omm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"], pdbfile="system.pdb", periodic=True
)

qmatoms = [93, 94, 95, 96, 97, 133, 134, 135, 2001, 2002]
qm_mm = QMMMTheory(qm_theory=qm_orca, mm_theory=omm, fragment=fragment, qm_charge=-1, qm_mult=6, qmatoms=qmatoms)

# Geometry optimization of the QM region
optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
# or QM/MM molecular dynamics (timestep in ps, simulation_time in ps)
openmm_md(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=2)
```

Runnable scripts, including a gas-phase ORCA example, live in [examples/](examples/):

```sh
python examples/gasphase_hf.py
python examples/qmmm_optimization.py system.pdb
```

## Errors

All package errors derive from `openmmqmmm.OpenMMQMMMError`, with specific subclasses
`InputError`, `MissingDependencyError`, `ExternalProgramError`, `FileFormatError` and
`InternalError` (each also inherits the closest builtin, so `except ValueError` etc. keep
working).

## Conventions

Importing the package is silent and side-effect free, and errors raise exceptions rather than
exiting the interpreter. Job functions are snake_case (`single_point`, `optimize_geometry`,
`numerical_frequencies`, `openmm_md`), classes are CapWords (`ORCATheory`, `OpenMMTheory`,
`QMMMTheory`, `Fragment`, `Results`), and keyword arguments are snake_case (`grad=`,
`active_region=`, `num_grad=`). Fragment files use the `.frag` extension, and each job function
writes its `Results` object to a `results_*.json` file — for example `results_singlepoint.json`,
`results_optimizer.json`, `results_numfreq.json`.

## Testing

```sh
pytest
```

from the repository root, which takes about four minutes. The fragment/OpenMM/optimizer tests
run without ORCA, and so do the ORCA input-writing and output-parsing tests (they use a fake
ORCA installation and committed reference output); the two end-to-end QM/MM tests are skipped
automatically when no ORCA installation is found. Set `OPENMMQMMM_ORCADIR` to run those too.

Coverage is measured with `pytest --cov` (needs the `test` extra: `pip install -e ".[test]"`).
Tests run in isolated temporary directories, so no output files are left behind. The test data
(~2.5 MB) lives in the source repository and is not shipped in wheels.

`python -m build` produces an sdist and a wheel under `dist/`.

## Citation

This package is derived from ASH. If it is useful in your research please cite:
[ASH: a Multi-scale, Multi-theory Modeling program](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70359),
R. Bjornsson, *J. Comput. Chem* **2026**, 47, e70359.
