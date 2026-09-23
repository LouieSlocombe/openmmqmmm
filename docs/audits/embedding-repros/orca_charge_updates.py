"""Reproduce the QM/MM embedding audit probe: orca_charge_updates.py.

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
    tempfile.TemporaryDirectory(prefix="openmmqmmm-embedding-orca_charge_updates-") as scratch,
    contextlib.chdir(scratch),
):
    from openmmqmmm import Fragment, OpenMMTheory, ORCATheory, QMMMTheory
    from openmmqmmm.exceptions import InputError

    frag = Fragment(elems=["H", "H", "He"], coords=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74], [8.0, 0.0, 0.0]], conncalc=False)
    mm = OpenMMTheory(
        fragment=frag,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    mm.update_charges([0, 1, 2], [0.1, -0.1, 0.5])
    qm = ORCATheory(
        orcasimpleinput="! HF STO-3G TightSCF", filename="charge_test", numcores=1, print_population_analysis=True
    )
    qmmm = QMMMTheory(
        fragment=frag,
        qm_theory=qm,
        mm_theory=mm,
        qmatoms=[0, 1],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
        update_qm_region_charges=True,
    )
    try:
        print("QM/MM result:", qmmm.run(current_coords=frag.coords, grad=True))
    except InputError as exc:
        print(type(exc).__name__ + ":", str(exc))
        print("QM energy:", getattr(qm, "energy", None))
        print("has charges attribute:", hasattr(qm, "charges"))
        print("Mulliken charges property:", qm.properties.get("Mulliken_charges"))
