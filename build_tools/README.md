# Installation guide

There are three ways to install `openmmqmmm`, depending on what you need:

| Route | Use when | Script |
|---|---|---|
| Conda environment | Normal use. Everything from conda-forge except PLUMED and the editable checkout. | `conda_install.sh` |
| Sol cluster | Running on Sol. Same split, plus the module loads and SLURM wrappers. | `custom_install_sol.sh` |
| Source build | You need an unreleased OpenMM. | `custom_install.sh` |

Every route compiles PLUMED, the `openmm-plumed` plugin and the PLUMED Python bindings
(py-plumed), because there is no prebuilt combination that works: conda-forge's
`openmm-plumed` requires `openmm <8.5`, while this package pins `openmm == 8.5.2` —
`openmm.PythonForce`, which QM/MM RPMD evaluates the per-bead force through, arrived in
8.5. Building from source also gets you PLUMED's `opes` module, which the conda-forge
build omits and which `openmm_md_plumed` needs, since a PLUMED bias is the only
enhanced-sampling route in this package.

None of the routes install ORCA — see [Configuring ORCA](#configuring-orca) below.

## Prerequisites

- A compatible operating system: Linux, macOS, or Windows via WSL.
- Python 3.10 or higher.
- Conda or Mamba.
- Git, to clone the PLUMED sources and the editable dependency. The compiler, `cmake` and
  `make` come from the environment; git does not.
- About 5 GB of disk for the environment: forcefill's openff-toolkit dependency pulls
  AmberTools, which pulls PyTorch and CUDA.

## Conda environment

From this directory:

```bash
bash conda_install.sh
```

The `openmmqmmm` environment is recreated **from scratch on every run** — any existing
environment with that name is removed first. Set `ENV_NAME` to install into a
differently named environment instead:

```bash
ENV_NAME=openmmqmmm2 bash conda_install.sh
```

The script creates the environment from `environment.yml`, compiles PLUMED, the
`openmm-plumed` plugin and py-plumed into it (sources are cloned into the gitignored
`build_tools/sources/`, wiped on each run), installs `openmmqmmm` and forcefill in
editable mode so changes to either source tree are picked up without reinstalling, and
finishes with import checks. It is equivalent to running, from this directory:

```bash
conda env create -f environment.yml
conda activate openmmqmmm
# Both build functions leave the shell in ${src_dir}, so hold on to this directory.
here="${PWD}"
src_dir="$(mktemp -d)"
source build_plumed.sh && build_plumed "${src_dir}" && build_py_plumed "${src_dir}"
pip install -e "${here}/.." --no-deps
source "${here}/editable_repos.sh" && install_editable_repos "${here}/../.."
```

(`build_plumed.sh` and `editable_repos.sh` are function libraries rather than scripts.
`build_py_plumed` reuses the plumed2 checkout that `build_plumed` leaves behind, so both
take the same working directory, and the PLUMED version is pinned there in one place. The
installers are immune to that `cd` — they resolve everything from `${BASH_SOURCE[0]}`.)

`--no-deps` on the editable installs is deliberate: `environment.yml` is the authority on
the dependency set, and letting pip re-resolve risks pulling PyPI wheels over the conda
OpenMM stack.

### Editable dependencies

[forcefill](https://github.com/LouieSlocombe/forcefill) gets edited alongside this
package, so every installer clones it **next to the repository** and installs it editable
rather than pulling it from GitHub on each install:

```
skunkworks/
├── openmmqmmm/
└── forcefill/
```

Set `SRC_DIR` to keep it elsewhere. A checkout that is already there is used exactly as it
is — the installer never pulls, resets or removes one, so uncommitted work is safe across a
rebuild. Only a missing one is cloned. Every installer finishes by checking that it imports
from the checkout rather than from `site-packages`.

`editable_repos.sh` is where that list lives; `openmmnqe` ships the same file with the
longer list its workflows need, and a single set of checkouts serves both.

CI is the exception: a GitHub runner has only this repository checked out, so the workflow
installs forcefill straight from GitHub.

### Into an environment that already exists

Install the dependencies, then run the same two build functions against the active
environment:

```bash
conda install -c conda-forge ase "openmm=8.5.2" cmake make swig cxx-compiler doxygen cython \
  pdbfixer mdtraj parmed rdkit openmmforcefields openff-toolkit multiprocess rmsd
pip install .
source build_plumed.sh && build_plumed "$(mktemp -d)"
```

`pip install .` adds OpenBabel and geomeTRIC from PyPI; conda-forge has no OpenBabel
build past Python 3.12. Add forcefill and py-plumed the same way `conda_install.sh` does.
Do **not** install conda-forge's `plumed` or `openmm-plumed` packages here — the first
has no `opes` module and would be overwritten in place by `build_plumed`'s `make install`,
and the second requires `openmm <8.5`.

## Sol cluster

`custom_install_sol.sh` builds the `openmmqmmm` environment on Sol. Most dependencies
come from conda-forge, but PLUMED is compiled from source because the conda-forge build
does not include the `opes` module. PLUMED sources are cloned into
`$SCRATCH/openmmqmmm_sources`, and both the environment and those sources are recreated
from scratch on each run.

`openmmqmmm` itself and forcefill are cloned into `$HOME/openmmqmmm_src` instead — outside
the build area, since that is wiped — and installed editable, so `git pull` in a checkout
is enough to update it. Set `SRC_DIR` to put them somewhere else.

Submit it as a batch job from this directory:

```bash
sbatch sub_sol_install.sh
```

Or run it directly from an interactive session:

```bash
interactive -t 60 -p htc -c 12 --mem=128G
```

Once installed, `sub_sol_run.sh` runs a single calculation script inside that
environment. A SLURM job starts in the directory it was submitted from, so submit it from
the repository root:

```bash
sbatch build_tools/sub_sol_run.sh examples/qmmm_optimization.py
```

It defaults to `examples/gasphase_hf.py` if no script is given. Uncomment the
`OPENMMQMMM_ORCADIR` export in that file before running anything that uses `ORCATheory`.

## Source build

`custom_install.sh` compiles OpenMM from source into a separate `openmmqmmm_custom`
environment, leaving any existing `openmmqmmm` environment untouched, then builds PLUMED,
`openmm-plumed` and py-plumed on top of it. The case this exists for is an OpenMM fix
that has not been released: QM/MM RPMD goes through `openmm.PythonForce`
(`openmmqmmm/openmm/rpmd_force.py`). Pin the version at the top of the script, then:

```bash
bash custom_install.sh
```

Sources are cloned into `../../openmmqmmm_sources`, a sibling of the repository. Both the
environment and the sources are recreated from scratch on each run, so a full build takes
a while. The editable checkout is shared with the other two routes and is not wiped.

All three installers share `build_plumed.sh`, which is where the PLUMED and OpenMM-PLUMED
versions are pinned, and `editable_repos.sh`, which is where the git dependencies are
listed.

## Configuring ORCA

[ORCA](https://www.faccts.de/orca/) is licensed separately (free for academic use) and has
to be installed by hand. It is required for `ORCATheory` and QM/MM, but not for the pure-MM
OpenMM functionality. It is located in this order, and every candidate is validated (the
directory must contain the `orca` binary and its `orca_*` helper binaries):

1. the `orcadir` argument to `ORCATheory`,
2. the `OPENMMQMMM_ORCADIR` environment variable, e.g. `export OPENMMQMMM_ORCADIR=~/orca_6_1_1`,
3. an `orca` binary found in `PATH`.

For parallel ORCA runs (`numcores` > 1) the matching OpenMPI version must also be set up,
as for any ORCA installation.

## Next steps

Runnable scripts live in `examples/`: `gasphase_hf.py` (a gas-phase ORCA calculation) and
`qmmm_optimization.py` (a QM/MM geometry optimization). The main [README](../README.md)
covers the API, ligand force fields and the QM/MM examples.
