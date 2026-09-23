"""Reproduce the QM/MM embedding audit probe: boundary.py.

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

with tempfile.TemporaryDirectory(prefix="openmmqmmm-embedding-boundary-") as scratch, contextlib.chdir(scratch):
    import json
    import logging

    import numpy as np

    from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
    from openmmqmmm.coords import get_boundary_atoms
    from openmmqmmm.exceptions import InputError

    logging.disable(logging.CRITICAL)

    class QM:
        numcores = 1
        theorytype = "QM"

    class MM:
        def __init__(self, n):
            self.numatoms = n

        def update_charges(self, *args):
            pass

    def make(coords, qmatoms, charges, **kw):
        f = Fragment(elems=["C"] * len(coords), coords=coords, conncalc=False)
        t = QMMMTheory(
            fragment=f,
            qm_theory=QM(),
            mm_theory=MM(len(coords)),
            qmatoms=qmatoms,
            charges=charges,
            qm_charge=0,
            qm_mult=1,
            **kw,
        )
        t.runprep(f.coords)
        if t.chargeboundary_method == "rcd":
            p = t.rcd_shifting_update(f.coords[t.mmatoms], f.coords)
        elif t.dipole_correction:
            t.set_dipole_charges(f.coords)
            p = np.vstack((f.coords[t.mmatoms], t.dipole_coords))
        else:
            p = f.coords[t.mmatoms]
        return f, t, p

    results = {}
    for method in ["shift", "rcd"]:
        f, t, p = make(
            [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0]],
            [0, 3],
            [0, 0.3, 0.4, 0],
            chargeboundary_method=method,
        )
        results["adjacent_mm1_" + method] = {
            "boundary": t.boundaryatoms,
            "recipients": t.MMboundarydict,
            "charges": np.asarray(t.pointcharges).tolist(),
            "positions": p.tolist(),
            "original_dipole": (np.asarray(t.charges) @ f.coords).tolist(),
            "modified_dipole": (np.asarray(t.pointcharges) @ p).tolist(),
        }

    for method in ["shift", "rcd"]:
        f, t, p = make(
            [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [1.4, 1.4, 0]],
            [0],
            [0, 0.4, -0.1, -0.3],
            chargeboundary_method=method,
        )
        results["isolated_mm1_" + method] = {
            "total_charge": sum(t.pointcharges),
            "original_dipole": (np.asarray(t.charges) @ f.coords).tolist(),
            "modified_dipole": (np.asarray(t.pointcharges) @ p).tolist(),
        }

    for neighbors, coords, elems in [
        (1, [[0, 0, 0], [1.4, 0, 0]], ["O", "C"]),
        (2, [[0, 0, 0], [1.4, 0, 0], [-1.4, 0, 0]], ["O", "C", "C"]),
    ]:
        try:
            out = get_boundary_atoms([0], np.asarray(coords), elems, 1.0, 0.3, unusualboundary=False)
        except InputError as e:
            out = type(e).__name__
        results["nonpolar_guard_" + str(neighbors)] = out

    for name, coords, bonds in [
        ("missed_topology_bond", [[0, 0, 0], [2.0, 0, 0], [3.4, 0, 0]], [(0, 1), (1, 2)]),
        ("invented_topology_bond", [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0]], [(1, 2)]),
    ]:
        f = Fragment(elems=["C"] * 3, coords=coords, conncalc=False)
        mm = OpenMMTheory(
            fragment=f,
            dummysystem=True,
            platform="Reference",
            autoconstraints=None,
            rigidwater=False,
            hydrogenmass=None,
        )
        atoms = list(mm.topology.atoms())
        for i, j in bonds:
            mm.topology.addBond(atoms[i], atoms[j])
        t = QMMMTheory(fragment=f, qm_theory=QM(), mm_theory=mm, qmatoms=[0], qm_charge=0, qm_mult=1)
        results[name] = {
            "periodic": mm.periodic,
            "topology_bonds": [[b[0].index, b[1].index] for b in mm.topology.bonds()],
            "boundary": t.boundaryatoms,
            "linkatoms": t.linkatoms,
        }
    print(json.dumps(results, indent=2))
