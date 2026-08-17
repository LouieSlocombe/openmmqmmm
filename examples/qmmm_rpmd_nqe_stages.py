"""QM/MM ring-polymer MD driven by openmmnqe's staged RPMD workflow.

Requires ORCA and an environment holding both packages (see
build_tools/README.md, "One environment for openmmqmmm + openmmnqe").
The same pattern with an analytic QM stand-in instead of ORCA is tested in
tests/test_nqe_interop.py.
"""

import sys

import openmmnqe

from openmmqmmm import (
    Fragment,
    OpenMMTheory,
    ORCATheory,
    QMMMTheory,
    configure_logging,
    export_rpmd_potential,
)

if __name__ == "__main__":
    configure_logging()

    pdbfile = sys.argv[1] if len(sys.argv) > 1 else "system.pdb"
    n_beads = 32
    fragment = Fragment(pdbfile=pdbfile)

    qm_orca = ORCATheory(orcasimpleinput="! r2SCAN-3c tightscf", numcores=8)
    # RPMD needs unconstrained, physical-mass dynamics.
    mm_openmm = OpenMMTheory(
        xmlfiles=["charmm36.xml", "charmm36/water.xml"],
        pdbfile=pdbfile,
        periodic=True,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )

    qmatoms = [93, 94, 95, 96, 97, 133, 134, 135]
    qm_mm = QMMMTheory(
        qm_theory=qm_orca,
        mm_theory=mm_openmm,
        fragment=fragment,
        qm_charge=-1,
        qm_mult=6,
        qmatoms=qmatoms,
    )

    # The export is the seam: a plain OpenMM System carrying the bead-specific
    # QM/MM PythonForce, and a Modeller for openmmnqe's stages to start from.
    export = export_rpmd_potential(theory=qm_mm, num_beads=n_beads)
    prepared = openmmnqe.PreparedSystem(export.system)

    openmmnqe.run_openmm_rpmd_equilibration(
        export.modeller,
        prepared,
        n_beads=n_beads,
        n_1=100,
        n_2=500,
        n_report=50,
    )
    # barostat_freq=None: a barostat cannot be combined with QM/MM RPMD.
    openmmnqe.run_openmm_rpmd_prod(
        export.modeller,
        prepared,
        checkpoint_file="rpmd_ready.chk",
        n_beads=n_beads,
        steps=5000,
        n_report=100,
        barostat_freq=None,
    )
    print(f"QM/MM evaluations: {export.provider.evaluation_count} ({export.provider.cache_hits} cache hits)")
