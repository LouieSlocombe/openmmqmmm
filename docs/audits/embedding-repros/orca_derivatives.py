"""Reproduce the QM/MM embedding audit probe: orca_derivatives.py.

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

with tempfile.TemporaryDirectory(prefix="openmmqmmm-embedding-orca_derivatives-") as scratch, contextlib.chdir(scratch):
    import logging
    from pathlib import Path

    from openmmqmmm import Fragment, ORCATheory, QMMMTheory
    from openmmqmmm.constants import ANG_TO_BOHR

    logging.disable(logging.CRITICAL)
    fragment = Fragment(
        elems=["H", "H", "He", "He"], coords=[[0, 0, 0], [0.74, 0, 0], [4.0, 1, 0], [-3.0, -1, 0]], conncalc=False
    )
    qm = ORCATheory(orcasimpleinput="! HF STO-3G verytightscf", filename="field")
    theory = QMMMTheory(
        qm_theory=qm, fragment=fragment, qmatoms=[0, 1], charges=[0.0, 0.0, 0.2, -0.2], qm_charge=0, qm_mult=1
    )
    energy, analytic = theory.run(current_coords=fragment.coords, grad=True)
    analytic = analytic.copy()
    print("embedded energy", energy)
    print("net gradient", analytic.sum(axis=0).tolist())
    h = 1.0e-3
    for atom, axis in [(0, 0), (2, 0), (3, 1)]:
        plus, minus = fragment.coords.copy(), fragment.coords.copy()
        plus[atom, axis] += h
        minus[atom, axis] -= h
        numerical = (theory.run(current_coords=plus) - theory.run(current_coords=minus)) / (2 * h * ANG_TO_BOHR)
        print(
            "derivative",
            atom,
            axis,
            "analytic",
            analytic[atom, axis],
            "numeric",
            numerical,
            "difference",
            analytic[atom, axis] - numerical,
        )
