# openmmqmmm — QM/MM and ML/MM with OpenMM

[![Documentation](https://readthedocs.org/projects/openmmqmmm/badge/?version=latest)](https://openmmqmmm.readthedocs.io/en/latest/)
[![CI](https://github.com/LouieSlocombe/openmmqmmm/actions/workflows/ci.yml/badge.svg)](https://github.com/LouieSlocombe/openmmqmmm/actions/workflows/ci.yml)

Electrostatically embedded QM/MM for biomolecular systems, combining the
[ORCA](https://www.faccts.de/orca/) quantum chemistry program with the
[OpenMM](https://openmm.org) molecular-mechanics library. Derived from the
[ASH](https://github.com/RagnarB83/ash) multiscale modelling program and reduced to the
ORCA + OpenMM QM/MM stack, with a modernized, PEP8-style Python API. Optional
[OpenMM-ML](https://github.com/openmm/openmm-ml) potentials provide mechanically embedded
ML/MM calculations and dynamics.

**📖 Full documentation: [openmmqmmm.readthedocs.io](https://openmmqmmm.readthedocs.io)**

`openmmqmmm.__all__` is the public API. Its core is the theory classes `ORCATheory`,
`OpenMMTheory`, `QMMMTheory`, `MLMMTheory`, `OpenMMQMMMCalculator` and `Fragment`, the job functions
`single_point`, `optimize_geometry` (via [geomeTRIC](https://github.com/leeping/geomeTRIC)),
`numerical_frequencies`, `analytic_frequencies`, `openmm_md` and `job_parallel`, and the OpenMM
setup helpers `openmm_modeller`, `openmm_minimize`, `openmm_box_equilibration`,
`gentle_warmup_md`, `openmm_md_plumed` and `solvate_small_molecule`.

> **Coming from ASH?** This is not a drop-in replacement. The API was renamed (snake_case
> functions, no import-time side effects, logging instead of print), and the in-house ligand
> parameterization was replaced by
> [forcefill](https://github.com/LouieSlocombe/forcefill). Both are described under
> [Conventions](https://openmmqmmm.readthedocs.io/en/latest/conventions.html).

## Installation

```sh
bash build_tools/conda_install.sh
```

One command: it creates the `openmmqmmm` conda environment from
`build_tools/environment.yml` with OpenMM 8.6, compiles PLUMED 2.10.1 (with the `opes` module), the
OpenMM-PLUMED plugin and the PLUMED Python bindings into it, installs this package and
forcefill in editable mode, and verifies each piece by importing it. The full environment is
large (~5 GB).

For OpenMM-ML 1.8 alongside OpenMM 8.6.1, add the optional ML dependencies with
`conda env update -n openmmqmmm -f build_tools/environment_ml.yml` after the base install.
OpenMM-ML's own floor is Python 3.11, below this package's 3.12 requirement; model backends
and weights are installed separately. See the installation guide for details.
ML/MM does not require ORCA.

For MACE-POLAR, the
`conda env update -n openmmqmmm -f build_tools/environment_mace_polar.yml` overlay also
installs a compatible MACE and graph-electrostatics pair. Model weights are fetched on
first use or supplied as a local checkpoint.

Python 3.12 or newer is required. For a pip-only install from a checkout, every runtime
dependency resolves from PyPI, but forcefill and PLUMED are left out, so
`openmm_modeller(parameterize_nonstandard=True)` and `openmm_md_plumed` are unavailable:

```sh
git clone https://github.com/LouieSlocombe/openmmqmmm.git
cd openmmqmmm
pip install .
```

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
# Single points and optimizations need an MM theory without automatic constraints.
omm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"],
    pdbfile="system.pdb",
    periodic=True,
    autoconstraints=None,
    rigidwater=False,
)

qmatoms = [93, 94, 95, 96, 97, 133, 134, 135, 2001, 2002]
qm_mm = QMMMTheory(qm_theory=qm_orca, mm_theory=omm, fragment=fragment, qm_charge=-1, qm_mult=6, qmatoms=qmatoms)

# Geometry optimization of the QM region
optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
# or QM/MM molecular dynamics (timestep in ps, simulation_time in ps)
openmm_md(fragment=fragment, theory=qm_mm, timestep=0.0005, simulation_time=2)
```

The package is silent by default; `configure_logging()` turns on the calculation output. The
single-point, optimization and frequency functions return a `Results` object and write it to
a `results_*.json` file.

Polarizable MM force fields are supported: Drude QM/MM uses explicit core and shell
charges with `embedding="drude"` (or `"elstat"`), and AMOEBA QM/MM uses mechanical
embedding with its native multipoles and polarization. Both require complete QM molecules;
Drude particles and their parent/reference atoms remain MM. Use `DrudeLangevinIntegrator`
or `DrudeSCFIntegrator` for Drude dynamics. AMOEBA coupling to the QM electron density is
not implemented. See the
[polarizable QM/MM guide](https://openmmqmmm.readthedocs.io/en/latest/guide/qmmm.html#polarizable-mm-force-fields)
and [system preparation](https://openmmqmmm.readthedocs.io/en/latest/guide/system_setup.html#polarizable-force-fields).

## ML/MM example

Build a fully parameterized `OpenMMTheory` named `mm` and a matching `Fragment`, then select
complete molecules by zero-based topology atom indices:

```py
from openmmqmmm import MLMMTheory, openmm_md

mlatoms = [atom.index for atom in mm.topology.atoms() if atom.residue.name == "LIG"]
mlmm = MLMMTheory(mm_theory=mm, mlatoms=mlatoms, potential="ani2x")
openmm_md(fragment=fragment, theory=mlmm, timestep=0.0005, simulation_time=1)
```

The source MM system is preserved. ML/MM uses classical interactions across the boundary;
it does not polarize the ML model with MM charges. Covalent bonds across the boundary and
ML virtual sites are unsupported. Choose a potential trained for your system's chemistry.
For single points, optimization or frequencies, build `mm` with `autoconstraints=None`
and `rigidwater=False`. See the
[ML/MM guide](https://openmmqmmm.readthedocs.io/en/latest/guide/mlmm.html) and the runnable
[ML/MM example](https://github.com/LouieSlocombe/openmmqmmm/blob/main/examples/mlmm_dynamics.py).

For MACE-POLAR, use `potential="mace-polar"` (the medium model) and set the ML region's
state with `system_kwargs={"charge": -1, "multiplicity": 1, "device": "cuda"}`. Explicit
small, medium and large model names and local checkpoints are supported. The wrapper uses
full model energy and defaults to double precision; MM coupling remains mechanical.
See the [MACE-POLAR guide](https://openmmqmmm.readthedocs.io/en/latest/guide/mlmm.html#mace-polar)
for dependencies, electronic-state settings and periodicity requirements.

Runnable scripts, including a gas-phase ORCA example, live in
[examples/](https://github.com/LouieSlocombe/openmmqmmm/tree/main/examples).

## In the documentation

- [Quick start](https://openmmqmmm.readthedocs.io/en/latest/quickstart.html) — a first
  calculation, start to finish.
- [Preparing a system](https://openmmqmmm.readthedocs.io/en/latest/guide/system_setup.html) —
  PDB repair, solvation, ions, ligand force fields through forcefill.
- [QM/MM](https://openmmqmmm.readthedocs.io/en/latest/guide/qmmm.html) — QM region, link atoms,
  embedding, active region.
- [ML/MM](https://openmmqmmm.readthedocs.io/en/latest/guide/mlmm.html) — OpenMM-ML potentials,
  complete-molecule regions and mechanical embedding.
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

from the repository root, which takes about five minutes. The end-to-end QM/MM tests that need
ORCA skip automatically when no ORCA installation is found; set `OPENMMQMMM_ORCADIR` to run
them too.
See the
[development guide](https://openmmqmmm.readthedocs.io/en/latest/development.html) for
coverage, linting, building the documentation and building the package.

## Citation

This package is derived from ASH. If it is useful in your research please cite:
[ASH: a Multi-scale, Multi-theory Modeling program](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70359),
R. Bjornsson, *J. Comput. Chem* **2026**, 47, e70359.

## License

GNU General Public License v2.0 or later, inherited from
[ASH](https://github.com/RagnarB83/ash). See [LICENSE](LICENSE).
