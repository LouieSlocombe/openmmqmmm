# openmmqmmm — ORCA + OpenMM QM/MM

Electrostatically embedded QM/MM for biomolecular systems, combining the
[ORCA](https://www.faccts.de/orca/) quantum chemistry program with the
[OpenMM](https://openmm.org) molecular-mechanics library. Derived from the
[ASH](https://github.com/RagnarB83/ash) multiscale modelling program and reduced to the
ORCA + OpenMM QM/MM stack, with a modernized, PEP8-style Python API.

> **Compatibility note:** version 1.0 renamed the public API (snake_case functions, no import-time
> side effects, logging instead of print). Scripts written for the 0.x releases need updating; the
> Conventions section below describes the naming now in force.

`openmmqmmm.__all__` is the full public API: `ORCATheory`, `OpenMMTheory`, `QMMMTheory`,
`OpenMMQMMMCalculator` and `Fragment`, the job functions `single_point`, `optimize_geometry` (via
[geomeTRIC](https://github.com/leeping/geomeTRIC)), `numerical_frequencies`,
`analytic_frequencies`, `openmm_md` and `job_parallel`, plus the OpenMM setup helpers
(`openmm_modeller`, `openmm_minimize`, `openmm_box_equilibration`, `gentle_warmup_md`,
`openmm_md_plumed`, `solvate_small_molecule`).

## Installation

**Requirements**

- Linux or macOS, Python ≥ 3.10
- Every Python dependency is required — there are no feature-gated extras. `pip install .` pulls
  the full set (ASE, OpenMM, PDBFixer, mdtraj, ParmEd, OpenBabel, geomeTRIC, rmsd, multiprocess,
  numpy, scipy, packaging)
- **[forcefill](https://github.com/LouieSlocombe/forcefill)** is not on PyPI: `conda_install.sh`
  adds it with
  `pip install --no-deps "forcefill @ git+https://github.com/LouieSlocombe/forcefill.git"`.
  Without it `openmm_modeller(parameterize_nonstandard=True)` raises `MissingDependencyError`;
  everything else except the PLUMED integration works. Its dependency stack (openff-toolkit,
  openmmforcefields, RDKit, AmberTools) comes from conda-forge via
  `build_tools/environment.yml` — openff-toolkit is not on PyPI, which is why the conda route
  below is recommended
- `openmm_md_plumed` also requires **PLUMED** and **openmm-plumed**, neither of which can come
  from conda-forge: that `openmm-plumed` binary requires OpenMM `<8.5`, and that PLUMED build
  omits the `opes` module. `conda_install.sh` compiles both from source (PLUMED 2.10.1, plugin
  `master`) into the environment
- [ORCA](https://www.faccts.de/orca/) — installed separately (free for academic use); required for
  `ORCATheory` and QM/MM, not for the pure-MM/OpenMM functionality

The full environment is large (~5 GB): forcefill's openff-toolkit dependency pulls AmberTools,
which pulls PyTorch and CUDA.

**Conda environment (recommended)**

From the repository root:

```sh
bash build_tools/conda_install.sh
```

One command: it creates the `openmmqmmm` conda environment from
`build_tools/environment.yml`, compiles PLUMED 2.10.1 (with the `opes` module), the
OpenMM-PLUMED plugin and the PLUMED Python bindings into it, installs this package in
editable mode along with forcefill, and verifies each piece by importing it. The
environment is removed and recreated on every run; set `ENV_NAME` to build into a
different one.

[build_tools/README.md](build_tools/README.md) is the full installation guide — the other
two routes (Sol cluster, source-built OpenMM), what to do with an environment that already
exists, and the equivalent commands run by hand.

**Configuring ORCA**

ORCA is located in this order, and every candidate is validated (the directory must contain the
`orca` binary and its `orca_*` helper binaries):

1. the `orcadir` argument to `ORCATheory`,
2. the `OPENMMQMMM_ORCADIR` environment variable, e.g. `export OPENMMQMMM_ORCADIR=~/orca_6_1_1`,
3. an `orca` binary found in `PATH`.

For parallel ORCA runs (`numcores` > 1) the matching OpenMPI version must also be set up, as for
any ORCA installation.

## Ligand force fields (forcefill)

Version 2.0 removed the in-house ligand parameterization (`small_molecule_parameterizer`,
`write_xmlfile_parmed`, `create_sys_and_check_14_scaling_nonbonding`,
`calc_nonbonding_energy_exceptions`). [forcefill](https://github.com/LouieSlocombe/forcefill)
replaces them: `build_ligand_xml` / `build_forcefield_xml` produce an OpenMM force-field XML
(GAFF via antechamber/AM1-BCC, OpenFF SMIRNOFF, or CHARMM CGenFF), and `assemble_openmm_ffxml` /
`validate_forcefield_xml` cover the lower-level XML writing and checking.

Build the XML from an SDF/MOL2/PDB file or a SMILES (XYZ-only inputs: convert to SDF first, e.g.
with RDKit's `rdDetermineBonds` or OpenBabel), then feed it to any of the setup helpers:

```py
from forcefill import build_ligand_xml

result = build_ligand_xml({"LIG": "ligand.sdf"}, "lig_ff.xml")  # or LigandSpec(smiles=...)

openmm_modeller(pdbfile="complex.pdb", forcefield="Amber14", extraxmlfile=result.forcefield_xml)
# or
OpenMMTheory(xmlfiles=["amber14-all.xml", "amber14/tip3p.xml", "lig_ff.xml"], pdbfile="complex.pdb", periodic=True)
# or
solvate_small_molecule(fragment=fragment, xmlfile=result.forcefield_xml, watermodel="tip3p")
```

`openmm_modeller` can also do this in one step: with `parameterize_nonstandard=True`, every
residue the chosen forcefield cannot match is parameterized through
`forcefill.build_forcefield_xml` and the generated `nonstandard_ff.xml` is loaded automatically:

```py
openmm_modeller(pdbfile="complex.pdb", forcefield="Amber14", parameterize_nonstandard=True, net_charges={"LIG": 0})
```

Non-standard residues must carry explicit hydrogens and CONECT records in the PDB-file.
`ligand_files={"LIG": "ligand.sdf"}` supplies bond orders from a file instead of PDB geometry
perception, and `ligand_backend` selects `"gaff"` (default), `"smirnoff"` or `"charmm"`. For finer
control (charge methods, per-ligand `LigandSpec`, minimization checks) call forcefill directly and
pass the XML via `extraxmlfile=`.

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

### QM/MM ring-polymer molecular dynamics

OpenMM 8.5.2's `PythonForce` lets the RPMD integrator request the QM/MM energy and gradient for
the bead it is currently propagating. Create the MM theory without constraints—OpenMM's
`RPMDIntegrator` does not support them—and select the RPMD integrator normally:

```py
omm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml", "specialresidue.xml"],
    pdbfile="system.pdb",
    periodic=True,
    autoconstraints=None,
    rigidwater=False,
    hydrogenmass=None,
)
qm_mm = QMMMTheory(
    qm_theory=qm_orca,
    mm_theory=omm,
    fragment=fragment,
    qm_charge=-1,
    qm_mult=6,
    qmatoms=qmatoms,
)
openmm_md(
    fragment=fragment,
    theory=qm_mm,
    integrator="RPMDIntegrator",
    rpmd_num_copies=32,
    timestep=0.0005,
    simulation_steps=100,
)
```

By default the QM force is evaluated independently on every bead. `RPMDIntegrator` evaluates
the potential twice per step, so this example performs 64 QM/MM evaluations per MD step. To use
OpenMM's ring-polymer contraction approximation for only the QM force, set—for example—
`rpmd_qm_num_copies=1` for a centroid calculation or another value no larger than
`rpmd_num_copies`. Final bead evaluations are cached, so ordinary exact-RPMD state and force
reporting does not relaunch identical QM jobs. RPMD restart files contain positions and velocities
for every bead.

`truncated_pc`, `update_qm_region_charges`, `special_wrapping` and `dummyatomrestraint` are rejected
for QM/MM RPMD because their current state is shared across beads. Standard OpenMM periodic
wrapping remains available through the `PythonForce` state.

RPMD and the adaptive quantum thermal bath require physical nuclear masses. Selecting either
`RPMDIntegrator` or `QTBIntegrator` therefore disables OpenMM's automatic hydrogen-mass
repartitioning and restores the mass transferred from each bonded heavy atom. For adQTB dynamics,
select `integrator="QTBIntegrator"`; it uses the same temperature, coupling-frequency and timestep
options as the Langevin integrators.

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
`QMMMTheory`, `OpenMMQMMMCalculator`, `Fragment`, `Results`), and keyword arguments are snake_case (`grad=`,
`active_region=`, `num_grad=`). Fragment files use the `.frag` extension, and each job function
writes its `Results` object to a `results_*.json` file — for example `results_singlepoint.json`,
`results_optimizer.json`, `results_numfreq.json`.

## ASE calculator

`OpenMMQMMMCalculator` exposes a configured `QMMMTheory` to ASE with energies in eV and forces
in eV/Å. The ASE atoms must retain the atom count, elements and ordering of the `Fragment` used
to create the QM/MM theory. Cell changes and stress are not supported.

```py
from ase import Atoms
from ase.optimize import BFGS
from openmmqmmm import OpenMMQMMMCalculator

atoms = Atoms(fragment.elems, positions=fragment.coords)
atoms.calc = OpenMMQMMMCalculator(qm_mm, directory="ase-qmmm")
BFGS(atoms).run(fmax=0.05)
```

The QM-region charge and multiplicity are taken from `QMMMTheory.qm_charge` and `qm_mult`. If
they were not set on the theory, pass `charge=` and `mult=` to the calculator. Use a separate
theory instance, process and calculation directory for every concurrent ASE calculation.

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
