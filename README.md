# openmmqmmm — ORCA + OpenMM QM/MM

A trimmed distribution of the [ASH](https://github.com/RagnarB83/ash) multiscale modelling program, reduced to the
**ORCA + OpenMM QM/MM stack for biomolecular calculations**. The Python package and API keep the upstream `ash` name, so
existing ORCA/OpenMM QM/MM scripts work unchanged.

**What is included**

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

**Citation**

This package is derived from ASH. If it is useful in your research please cite:
[ASH: a Multi-scale, Multi-theory Modeling program](https://onlinelibrary.wiley.com/doi/10.1002/jcc.70359), R.
Bjornsson, *J. Comput. Chem* **2026**, 47, e70359.

**Installation**

Use a conda/mamba environment providing OpenMM, then install the package with pip:

```sh
conda env create -f environment.yml
conda activate ash
pip install -e .
```

ORCA must be installed separately and available in `PATH` (or via `orcadir`/`~/ash_user_settings.ini`).

**Basic example**

```py
from ash import *

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

**QM/MM example**

```py
from ash import *

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

**Tests**

```sh
cd ash/tests && pytest -q
```

The OpenMM/fragment/optimizer tests run without ORCA; the QM/MM tests are skipped automatically when no `orca` binary is
found in `PATH`.

**Documentation**

Upstream ASH documentation (applies to the retained functionality): https://ash.readthedocs.io
