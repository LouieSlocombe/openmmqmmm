# openmmqmmm — ORCA + OpenMM QM/MM

[![Documentation](https://readthedocs.org/projects/openmmqmmm/badge/?version=latest)](https://openmmqmmm.readthedocs.io/en/latest/)
[![CI](https://github.com/LouieSlocombe/openmmqmmm/actions/workflows/ci.yml/badge.svg)](https://github.com/LouieSlocombe/openmmqmmm/actions/workflows/ci.yml)

Electrostatically embedded QM/MM for biomolecular systems, combining the
[ORCA](https://www.faccts.de/orca/) quantum chemistry program with the
[OpenMM](https://openmm.org) molecular-mechanics library. Derived from the
[ASH](https://github.com/RagnarB83/ash) multiscale modelling program and reduced to the
ORCA + OpenMM QM/MM stack, with a modernized, PEP8-style Python API.

**📖 Full documentation: [openmmqmmm.readthedocs.io](https://openmmqmmm.readthedocs.io)**

`openmmqmmm.__all__` is the public API. Its core is the theory classes `ORCATheory`,
`OpenMMTheory`, `QMMMTheory`, `OpenMMQMMMCalculator` and `Fragment`, the job functions
`single_point`, `optimize_geometry` (via [geomeTRIC](https://github.com/leeping/geomeTRIC)),
`numerical_frequencies`, `analytic_frequencies`, `openmm_md` and `job_parallel`, and the OpenMM
setup helpers `openmm_modeller`, `openmm_minimize`, `openmm_box_equilibration`,
`gentle_warmup_md`, `openmm_md_plumed` and `solvate_small_molecule`.

> **Compatibility note:** two releases broke the API. Version 1.0 renamed it (snake_case
> functions, no import-time side effects, logging instead of print). Version 2.0 removed the
> in-house ligand parameterization in favour of
> [forcefill](https://github.com/LouieSlocombe/forcefill). Both are described under
> [Conventions](https://openmmqmmm.readthedocs.io/en/latest/conventions.html).

## Installation

```sh
bash build_tools/conda_install.sh
```

One command: it creates the `openmmqmmm` conda environment from
`build_tools/environment.yml` with OpenMM 8.6.1, compiles PLUMED 2.10.1 (with the `opes` module), the
OpenMM-PLUMED plugin and the PLUMED Python bindings into it, installs this package and
forcefill in editable mode, and verifies each piece by importing it. The full environment is
large (~5 GB).

For OpenMM-ML 1.8 alongside OpenMM 8.6.1, add the optional ML dependencies with
`conda env update -n openmmqmmm -f build_tools/environment_ml.yml` after the base install.
This option requires Python 3.11 or higher; model backends and weights are installed
separately. See the installation guide for details.

[ORCA](https://www.faccts.de/orca/) is installed separately (free for academic use) and found
through the `orcadir` argument, the `OPENMMQMMM_ORCADIR` environment variable, or `PATH` — in
that order.

The [installation guide](https://openmmqmmm.readthedocs.io/en/latest/install.html) covers the
pip-only route and what it leaves out, and
[build_tools/README.md](https://github.com/LouieSlocombe/openmmqmmm/blob/main/build_tools/README.md)
covers the Sol cluster and source-built-OpenMM routes, reusing an existing environment, and
sharing one environment with [openmmnqe](https://github.com/LouieSlocombe/openmmnqe).

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

The package is silent by default; `configure_logging()` turns on the calculation output. Each
job function returns a `Results` object and writes it to a `results_*.json` file.

Runnable scripts, including a gas-phase ORCA example, live in
[examples/](https://github.com/LouieSlocombe/openmmqmmm/tree/main/examples).

## In the documentation

- [Quick start](https://openmmqmmm.readthedocs.io/en/latest/quickstart.html) — a first
  calculation, start to finish.
- [Preparing a system](https://openmmqmmm.readthedocs.io/en/latest/guide/system_setup.html) —
  PDB repair, solvation, ions, ligand force fields through forcefill.
- [QM/MM](https://openmmqmmm.readthedocs.io/en/latest/guide/qmmm.html) — QM region, link atoms,
  embedding, active region.
- [Molecular dynamics](https://openmmqmmm.readthedocs.io/en/latest/guide/dynamics.html) —
  integrators, reporters, restarts, PLUMED metadynamics.
- [Ring-polymer dynamics](https://openmmqmmm.readthedocs.io/en/latest/guide/rpmd.html) — QM/MM
  RPMD and adQTB for nuclear quantum effects.
- [Working with openmmnqe](https://openmmqmmm.readthedocs.io/en/latest/guide/nqe_interop.html) —
  the staged-NQE interop protocol.
- [ASE calculator](https://openmmqmmm.readthedocs.io/en/latest/guide/ase.html)
- [API reference](https://openmmqmmm.readthedocs.io/en/latest/api/index.html) — every public name.

## Development

```sh
pytest
```

from the repository root, which takes about five minutes. The four end-to-end QM/MM tests skip
automatically when no ORCA installation is found; set `OPENMMQMMM_ORCADIR` to run them too.
See the
[development guide](https://openmmqmmm.readthedocs.io/en/latest/development.html) for
coverage, linting, building the documentation and building the package.

## Citation

This package is derived from ASH. If it is useful in your research please cite:
[ASH: a Multi-scale, Multi-theory Modeling program](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70359),
R. Bjornsson, *J. Comput. Chem* **2026**, 47, e70359.
