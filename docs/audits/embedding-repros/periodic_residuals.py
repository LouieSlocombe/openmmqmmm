"""Reproduce the QM/MM embedding audit probe: periodic_residuals.py.

Run with a Python environment containing the project's dependencies. ORCA probes
also require ORCA on PATH or OPENMMQMMM_ORCADIR pointing to its installation.
All generated calculation files are isolated in a temporary directory.
"""

import contextlib
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

with (
    tempfile.TemporaryDirectory(prefix="openmmqmmm-embedding-periodic_residuals-") as scratch,
    contextlib.chdir(scratch),
):
    import logging

    logging.disable(logging.CRITICAL)
    import numpy as np
    import openmm

    from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory

    class ZeroQM:
        numcores = 1
        theorytype = "QM"

        def run(self, *, current_coords, current_mm_coords=None, grad=False, pc=False, **kwargs):
            if not grad:
                return 0.0
            if pc:
                return 0.0, np.zeros_like(current_coords), np.zeros_like(current_mm_coords)
            return 0.0, np.zeros_like(current_coords)

    def setup(embedding, method, tail, charges):
        f = Fragment(elems=["He", "He"], coords=[[1, 1, 1], [6, 1, 1]], conncalc=False)
        mm = OpenMMTheory(
            fragment=f,
            dummysystem=True,
            platform="Reference",
            autoconstraints=None,
            rigidwater=False,
            hydrogenmass=None,
        )
        mm.periodic = True
        mm.system.setDefaultPeriodicBoxVectors(*np.eye(3) * 3.0)
        force = mm.nonbonded_force
        force.setNonbondedMethod(method)
        force.setCutoffDistance(1.0)
        force.setUseDispersionCorrection(tail)
        for i, c in enumerate(charges):
            force.setParticleParameters(i, c, 0.3, 0.0 if any(charges) else 1.0)
        mm.charges = charges
        q = QMMMTheory(
            fragment=f,
            qm_theory=ZeroQM(),
            mm_theory=mm,
            qmatoms=[0, 1],
            qm_charge=0,
            qm_mult=1,
            embedding=embedding,
        )
        _energy, g = q.run(current_coords=f.coords, grad=True)
        first = q.MMenergy
        q.run(current_coords=f.coords, periodic_box_vectors=np.diag([25.0] * 3))
        print(
            embedding,
            method,
            "LRC",
            tail,
            "charges",
            charges,
            "MM E30A",
            first,
            "E25A",
            q.MMenergy,
            "max gradient",
            abs(g).max(),
        )

    for embedding in ["elstat", "mech"]:
        for method in [openmm.NonbondedForce.PME, openmm.NonbondedForce.LJPME]:
            for tail in [False, True]:
                setup(embedding, method, tail, [0, 0])
    setup("mech", openmm.NonbondedForce.PME, False, [1.0, -1.0])
    setup("mech", openmm.NonbondedForce.NoCutoff, False, [1.0, -1.0])
