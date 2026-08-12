"""Electrostatically embedded QM/MM: optimize a QM region inside an MM protein.

Run with:
    python examples/qmmm_optimization.py system.pdb

Requires an ORCA installation and a solvated, parameterized PDB file. The QM-region
indices below are placeholders — replace them with the atoms of your own active site.
"""

import sys

from openmmqmmm import (
    Fragment,
    OpenMMTheory,
    ORCATheory,
    QMMMTheory,
    configure_logging,
    optimize_geometry,
)

configure_logging()

pdbfile = sys.argv[1] if len(sys.argv) > 1 else "system.pdb"
fragment = Fragment(pdbfile=pdbfile)

qm_orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
mm_openmm = OpenMMTheory(
    xmlfiles=["charmm36.xml", "charmm36/water.xml"],
    pdbfile=pdbfile,
    periodic=True,
)

# Atoms treated quantum-mechanically. Everything else is MM point charges.
qmatoms = [93, 94, 95, 96, 97, 133, 134, 135]

qm_mm = QMMMTheory(
    qm_theory=qm_orca,
    mm_theory=mm_openmm,
    fragment=fragment,
    qm_charge=-1,
    qm_mult=6,
    qmatoms=qmatoms,
)

# Optimizing only the QM region keeps the problem tractable for a large system.
result = optimize_geometry(theory=qm_mm, fragment=fragment, actatoms=qmatoms)
print(f"Optimized QM/MM energy: {result.energy:.8f} Eh")

# For QM/MM molecular dynamics instead (timestep and simulation_time in ps):
#   from openmmqmmm import openmm_md
#   openmm_md(fragment=fragment, theory=qm_mm, timestep=0.001, simulation_time=2)
