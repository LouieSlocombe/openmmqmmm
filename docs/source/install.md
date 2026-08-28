# Installation

## Requirements

- Linux or macOS, Python ≥ 3.10.
- Every Python dependency is required — there are no feature-gated extras. `pip install .`
  pulls the full set: ASE, OpenMM, PDBFixer, mdtraj, ParmEd, OpenBabel, geomeTRIC, rmsd,
  multiprocess, numpy, scipy and packaging.
- [ORCA](https://www.faccts.de/orca/), installed separately (free for academic use). It is
  needed for {class}`~openmmqmmm.ORCATheory` and therefore for QM/MM, but not for the
  pure-MM functionality.

Two pieces cannot come from PyPI, and both are optional in the sense that everything else
works without them:

`forcefill`
: Parameterizes residues a biomolecular force field does not cover. Without it,
  `openmm_modeller(parameterize_nonstandard=True)` raises
  {exc}`~openmmqmmm.MissingDependencyError` and nothing else changes. See
  {doc}`guide/system_setup`.

PLUMED and openmm-plumed
: Needed by {func}`~openmmqmmm.openmm_md_plumed`, and by nothing else. Neither can come
  from conda-forge: that `openmm-plumed` build requires OpenMM `< 8.5`, and that PLUMED
  build omits the `opes` module. `conda_install.sh` compiles both from source (PLUMED
  2.10.1, OpenMM-PLUMED v2.1).

The full environment is large — about 5 GB. forcefill's openff-toolkit dependency pulls
AmberTools, which pulls PyTorch and CUDA.

## Conda environment (recommended)

From the repository root:

```bash
bash build_tools/conda_install.sh
```

One command: it creates the `openmmqmmm` conda environment from
`build_tools/environment.yml`, compiles PLUMED 2.10.1 (with the `opes` module), the
OpenMM-PLUMED plugin and the PLUMED Python bindings into it, installs this package and
forcefill in editable mode, and verifies each piece by importing it.

The environment is removed and recreated on every run; set `ENV_NAME` to build into a
different one. The forcefill checkout it clones alongside this repository is left alone —
set `SRC_DIR` to keep it elsewhere, or `FORCEFILL_REF=main` before a first install to follow
forcefill's development branch instead of the pinned commit.

[build_tools/README.md](https://github.com/LouieSlocombe/openmmqmmm/blob/main/build_tools/README.md)
is the full installation guide: the other two routes (Sol cluster, source-built OpenMM),
what to do with an environment that already exists, how to put openmmqmmm and openmmnqe in
one environment, and the equivalent commands run by hand.

## pip

The package itself is not published on PyPI, but every one of its runtime dependencies is,
so a plain pip install from a checkout resolves cleanly on Linux and macOS:

```bash
git clone https://github.com/LouieSlocombe/openmmqmmm.git
cd openmmqmmm
pip install .
```

This is the route the documentation build uses. It leaves out forcefill and PLUMED, so
`openmm_modeller(parameterize_nonstandard=True)` and
{func}`~openmmqmmm.openmm_md_plumed` are unavailable; everything else works. openff-toolkit,
which forcefill needs, is not on PyPI at all, which is why the conda route above is the
recommended one for a working environment.

## Configuring ORCA

ORCA is located in this order, and every candidate is validated — the directory must contain
the `orca` binary and its `orca_*` helper binaries:

1. the `orcadir` argument to {class}`~openmmqmmm.ORCATheory`,
2. the `OPENMMQMMM_ORCADIR` environment variable, for example
   `export OPENMMQMMM_ORCADIR=~/orca_6_1_1`,
3. an `orca` binary found on `PATH`.

For parallel ORCA runs (`numcores` > 1) the matching OpenMPI version must also be set up, as
for any ORCA installation.

:::{note}
ORCA is always launched by absolute path. Appending its directory to `PATH` is not enough,
and on a Linux desktop it is actively harmful: the GNOME screen reader is also called
`orca`.
:::

## Checking the installation

```bash
python -c "import openmmqmmm; print(openmmqmmm.__version__)"
```

Then run the test suite from a checkout — see {doc}`development`.
