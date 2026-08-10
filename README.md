# openmmqmmm — ORCA + OpenMM QM/MM

Electrostatically embedded QM/MM for biomolecular systems, combining the
[ORCA](https://www.faccts.de/orca/) quantum chemistry program with the
[OpenMM](https://openmm.org) molecular-mechanics library. Derived from the
[ASH](https://github.com/RagnarB83/ash) multiscale modelling program and reduced to the
ORCA + OpenMM QM/MM stack, with a modernized, PEP8-style Python API.

> **Compatibility note:** version 1.0 renamed the public API (snake_case functions, no import-time
> side effects, logging instead of print). Scripts written for upstream ASH or for the 0.x
> `openmmqmmm` releases need the [migration table](#migrating-from-ash--0x) below.

## What is included

- `ORCATheory` — interface to ORCA (input generation, parallel runs, output parsing)
- `OpenMMTheory` — interface to OpenMM, plus `openmm_md` (molecular dynamics, also for QM/MM),
  `openmm_modeller` (pdbfixer-based protein setup), `openmm_minimize`,
  `openmm_box_equilibration`, `gentle_warmup_md`, metadynamics
  (`openmm_metadynamics`, `openmm_md_plumed`), `solvate_small_molecule` and
  `small_molecule_parameterizer`
- `QMMMTheory` — electrostatically embedded QM/MM with link atoms and charge-shifting
- `single_point` (+ fragment/theory/reaction variants), `job_parallel`
- `optimize_geometry` — geometry optimization via [geomeTRIC](https://github.com/leeping/geomeTRIC),
  including frozen/active-region optimizations of large systems (`active_region=True`)
- `numerical_frequencies` / `analytic_frequencies` — frequencies with partial Hessians and
  thermochemistry
- `Fragment` — coordinates/topology handling incl. XYZ, PDB, Amber and GROMACS file reading
- Helper interfaces: mdtraj (trajectory processing), OpenBabel (ligand conversion) and a simple
  matplotlib plotting object (`Plot`)

## Installation

**Requirements**

- Linux or macOS, Python ≥ 3.10
- [OpenMM](https://openmm.org) ≥ 8, [PDBFixer](https://github.com/openmm/pdbfixer) and
  [mdtraj](https://www.mdtraj.org) — installed from conda-forge (PDBFixer is not on PyPI, so a
  conda/mamba environment is the recommended route)
- geomeTRIC, numpy, packaging — pulled in automatically by pip
- [ORCA](https://www.faccts.de/orca/) — installed separately (free for academic use); required for
  `ORCATheory` and QM/MM, not for the pure-MM/OpenMM functionality

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
conda install -c conda-forge "openmm>=8" pdbfixer mdtraj
pip install .
```

**Optional dependencies** (all conda-forge; commented out in `environment.yml`)

| Package | Needed for |
|---|---|
| `matplotlib` | `Plot`, `metadynamics_plot_data` |
| `scipy` | electronic-entropy analysis in `ORCATheory` |
| `parmed` | Amber/GROMACS file handling in `OpenMMTheory` |
| `openbabel` | ligand format conversion (`mol_to_pdb`, `small_molecule_parameterizer`, ...) |
| `openmmforcefields`, `openff-toolkit`, `rdkit` | `small_molecule_parameterizer` |
| `openmm-plumed` | `openmm_md_plumed` (PLUMED-biased MD) |
| `multiprocess` | alternative multiprocessing backend for `job_parallel` |

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

## Basic example

```py
from openmmqmmm import Fragment, ORCATheory, configure_logging, numerical_frequencies, optimize_geometry, single_point

configure_logging()

coords = """
H 0.0 0.0 0.0
F 0.0 0.0 1.0
"""
hf_frag = Fragment(coordsstring=coords, charge=0, mult=1)

orca_calc = ORCATheory(orcasimpleinput="! r2SCAN def2-SVP def2/J tightscf", orcablocks="%scf maxiter 200 end")

single_point(theory=orca_calc, fragment=hf_frag)
optimize_geometry(theory=orca_calc, fragment=hf_frag)
numerical_frequencies(theory=orca_calc, fragment=hf_frag)
```

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

## Errors

All package errors derive from `openmmqmmm.OpenMMQMMMError`, with specific subclasses
`InputError`, `MissingDependencyError`, `ExternalProgramError`, `FileFormatError` and
`InternalError` (each also inherits the closest builtin, so `except ValueError` etc. keep
working).

## Migrating from ASH / 0.x

Module-level behavior: importing the package is silent and side-effect free — no banner, no
`~/ash_user_settings.ini` reading (use `OPENMMQMMM_ORCADIR` or `orcadir=`), no `printlevel`
keywords (use `configure_logging(level=...)` / per-module logger levels), and errors raise
exceptions instead of exiting the interpreter. Fragment files use the `.frag` extension (was
`.ygg`) and each job function writes its `Results` object to a `results_*.json` file — for
example `results_singlepoint.json`, `results_optimizer.json`, `results_numfreq.json` (was
`ASH_SP.result`, `ASH_Optimizer.result`, `ASH_NumFreq.result`).

| ASH / 0.x name | New name |
|---|---|
| `Singlepoint` (+`_fragments`, `_theories`, ...) | `single_point` (+`single_point_fragments`, ...) |
| `geomeTRICOptimizer` / `Optimizer` / `Opt` | `optimize_geometry` |
| `NumFreq` / `AnFreq` | `numerical_frequencies` / `analytic_frequencies` |
| `OpenMM_MD` / `MolecularDynamics` | `openmm_md` |
| `OpenMM_Opt` | `openmm_minimize` |
| `OpenMM_Modeller` | `openmm_modeller` |
| `OpenMM_metadynamics` / `MetaDynamics` | `openmm_metadynamics` |
| `OpenMM_box_equilibration`, `Gentle_warm_up_MD` | `openmm_box_equilibration`, `gentle_warmup_md` |
| `Job_parallel` / `Simple_parallel` | `job_parallel` / `simple_parallel` |
| `ReactionEnergy` | `reaction_energy` |
| `ASH_Results` / `ASH_plot` | `Results` / `Plot` |
| `OpenMM_MDclass` / `NumGradclass` | `MolecularDynamicsEngine` / `NumGrad` |
| `MDtraj_imagetraj`, `MDtraj_RMSD`, ... | `mdtraj_image_trajectory`, `mdtraj_rmsd`, ... |
| `actregiondefine` | `define_active_region` |
| `QMregionfragexpand` / `QMPC_fragexpand` | `expand_qm_region` / `expand_qm_pc_region` |
| `ORCA_External_Optimizer` | `orca_external_optimizer` |
| kwargs `Grad=`, `Hessian=`, `PC=`, `MMcharges=` | `grad=`, `hessian=`, `pc=`, `mm_charges=` |
| kwargs `ActiveRegion=`, `NumGrad=`, `TruncatedPC=` | `active_region=`, `num_grad=`, `truncated_pc=` |
| kwargs `TDDFT=`, `HSmult=`, `QRRHO=`, `pH=` | `tddft=`, `hs_mult=`, `qrrho=`, `ph=` |

Class names that were already CapWords (`ORCATheory`, `OpenMMTheory`, `QMMMTheory`, `Fragment`,
`Reaction`, `ZeroTheory`) are unchanged.

## Testing

```sh
pytest
```

from the repository root. The fragment/OpenMM/optimizer tests run without ORCA; the QM/MM tests
are skipped automatically when no ORCA installation is found. Tests run in isolated temporary
directories, so no output files are left behind. The test data (~2 MB) lives in the source
repository and is not shipped in wheels.

## Building distributions

```sh
python -m build
```

This produces an sdist and a wheel under `dist/`. Wheels ship the runtime data file `log.ini`
(geomeTRIC logging configuration) and the `py.typed` marker.

## Citation

This package is derived from ASH. If it is useful in your research please cite:
[ASH: a Multi-scale, Multi-theory Modeling program](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70359),
R. Bjornsson, *J. Comput. Chem* **2026**, 47, e70359.
