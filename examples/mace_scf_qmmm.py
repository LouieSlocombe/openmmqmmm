"""Evaluate an electrostatically embedded MACE-SCF/TIP3P water dimer.

Run ``python examples/mace_scf_qmmm.py /path/to/fixedpoint.model`` from an
installed checkout. The checkpoint must support neutral water; no model is downloaded.
Use ``--model-type qeq`` for a QEq checkpoint. Requires the optional mace-scf backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from openmm import app

from openmmqmmm import Fragment, MACESCFTheory, OpenMMTheory, QMMMTheory, configure_logging, single_point


def water_dimer():
    """Build two complete waters, with coordinates in Angstrom and topology order."""
    topology = app.Topology()
    chain = topology.addChain()
    for _ in range(2):
        residue = topology.addResidue("HOH", chain)
        oxygen = topology.addAtom("O", app.element.oxygen, residue)
        hydrogen1 = topology.addAtom("H1", app.element.hydrogen, residue)
        hydrogen2 = topology.addAtom("H2", app.element.hydrogen, residue)
        topology.addBond(oxygen, hydrogen1)
        topology.addBond(oxygen, hydrogen2)
    coordinates = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2400, 0.9266, 0.0000],
            [2.9000, 0.0000, 0.0000],
            [3.4859, 0.7569, 0.0000],
            [3.4859, -0.7569, 0.0000],
        ]
    )
    fragment = Fragment(elems=["O", "H", "H"] * 2, coords=coordinates)
    return fragment, topology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="MACE-SCF checkpoint supporting neutral water")
    parser.add_argument("--model-type", choices=("fixedpoint", "qeq"), default="fixedpoint")
    parser.add_argument("--device", default="cpu", help="MACE-SCF device, e.g. cpu or cuda")
    parser.add_argument("--platform", default="CPU", help="OpenMM platform, independent of --device")
    args = parser.parse_args()
    if not args.model_path.is_file():
        parser.error(f"Checkpoint does not exist: {args.model_path}")

    configure_logging()
    fragment, topology = water_dimer()
    mm = OpenMMTheory(
        topoforce=True,
        topology=topology,
        forcefield=app.ForceField("tip3p.xml"),
        periodic=False,
        platform=args.platform,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    backend = MACESCFTheory(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        default_dtype="float64",
    )
    qm_mm = QMMMTheory(
        qm_theory=backend,
        mm_theory=mm,
        fragment=fragment,
        qmatoms=[0, 1, 2],
        qm_charge=0,
        qm_mult=1,
        embedding="elstat",
    )
    result = single_point(theory=qm_mm, fragment=fragment, grad=True)
    print(f"MACE-SCF/MM energy: {result.energy:.10f} Eh")
    print(f"Maximum atomic gradient: {np.linalg.norm(result.gradient, axis=1).max():.6e} Eh/Bohr")


if __name__ == "__main__":
    main()
