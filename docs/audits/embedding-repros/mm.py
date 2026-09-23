"""Reproduce the QM/MM embedding audit probe: mm.py.

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

with tempfile.TemporaryDirectory(prefix="openmmqmmm-embedding-mm-") as scratch, contextlib.chdir(scratch):
    import numpy as np
    import openmm
    from openmm import unit

    from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
    from openmmqmmm.constants import ANG_TO_BOHR, HARTREE_TO_KJ_PER_MOL

    class QM:
        numcores = 1
        theorytype = "QM"

        def run(self, *, current_coords, grad=False, **kw):
            self.charges = [0.8] * len(current_coords)
            return (0.0, np.zeros_like(current_coords)) if grad else 0.0

    def evaluate(system, positions):
        it = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(system, it, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(positions)
        s = ctx.getState(getEnergy=True, getForces=True)
        return s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole), s.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )

    def minimal(system, n, charges):
        mm = OpenMMTheory.__new__(OpenMMTheory)
        mm.system = system
        mm.numatoms = n
        mm.charges = list(charges)
        mm.delete_qm1_mm1_bonded = False
        mm.nonbonded_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
        return mm

    # 1. QM-QM exception offsets survive stripping.
    for embedding in ("elstat", "mech"):
        system = openmm.System()
        for _ in range(2):
            system.addParticle(12)
        nb = openmm.NonbondedForce()
        for _ in range(2):
            nb.addParticle(0.5, 0.3, 0.2)
        exc = nb.addException(0, 1, 0.125, 0.3, 0.1)
        nb.addGlobalParameter("lambda", 1.0)
        nb.addExceptionParameterOffset("lambda", exc, 0.15, 0.3, 1.0)
        system.addForce(nb)
        mm = minimal(system, 2, [0.5, 0.5])
        frag = Fragment(elems=["C", "C"], coords=[[0, 0, 0], [4, 0, 0]], conncalc=False)
        qmmm = QMMMTheory(
            fragment=frag, qm_theory=QM(), mm_theory=mm, qmatoms=[0, 1], embedding=embedding, qm_charge=0, qm_mult=1
        )
        print(
            "offsets",
            embedding,
            "base",
            nb.getExceptionParameters(0),
            "offset",
            nb.getExceptionParameterOffset(0),
            "MM energy/force (expected all zero)",
            evaluate(system, frag.coords * 0.1),
        )

    # 2. Updating a fixed QM charge does not update Coulomb exception products.
    frag = Fragment(elems=["C", "C"], coords=[[0, 0, 0], [4, 0, 0]], conncalc=False)
    mm = OpenMMTheory(
        fragment=frag,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    mm.update_charges([0, 1], [0.2, -0.5])
    mm.nonbonded_force.addException(0, 1, -0.05, 0.3, 0.0)
    qmmm = QMMMTheory(
        fragment=frag,
        qm_theory=QM(),
        mm_theory=mm,
        qmatoms=[0],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
        update_qm_region_charges=True,
    )
    actual = qmmm.run(current_coords=frag.coords, grad=False) * HARTREE_TO_KJ_PER_MOL
    print(
        "charge_update",
        "q",
        mm.nonbonded_force.getParticleParameters(0)[0],
        "exception",
        mm.nonbonded_force.getExceptionParameters(0)[2],
        "E",
        actual,
        "expected",
        138.93545764438198 * (-0.2) / 0.4,
    )

    # 3. Library's custom torsion has 0 per-torsion parameters but stripping adds 2.
    system = openmm.System()
    for _ in range(4):
        system.addParticle(12)
    nb = openmm.NonbondedForce()
    for _ in range(4):
        nb.addParticle(0, 0.3, 0.0)
    system.addForce(nb)
    mm = minimal(system, 4, [0] * 4)
    mm.add_custom_torsion_force(0, 1, 2, 3, 0.0, 1.0)
    mm.modify_bonded_forces([0, 1, 2, 3])
    try:
        evaluate(system, np.array([[0, 0, 0], [0.15, 0, 0], [0.2, 0.1, 0], [0.3, 0.1, 0.1]]))
    except openmm.OpenMMException as exc:
        print("custom_torsion", type(exc).__name__, str(exc))

    # 4. Charge-response part of dE/dR is missing with geometry-dependent QM populations.
    class VariableChargesQM(QM):
        def run(self, *, current_coords, grad=False, **kw):
            coords = np.asarray(current_coords)
            q = 0.1 * np.linalg.norm(coords[1] - coords[0])
            self.charges = [q, -q]
            return (0.0, np.zeros_like(coords)) if grad else 0.0

    frag = Fragment(elems=["C"] * 3, coords=[[0, 0, 0], [3, 0, 0], [8, 0, 0]], conncalc=False)
    mm = OpenMMTheory(
        fragment=frag,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    mm.update_charges([0, 1, 2], [0, 0, 1])
    qmmm = QMMMTheory(
        fragment=frag,
        qm_theory=VariableChargesQM(),
        mm_theory=mm,
        qmatoms=[0, 1],
        embedding="mech",
        qm_charge=0,
        qm_mult=1,
        update_qm_region_charges=True,
    )
    _, g = qmmm.run(current_coords=frag.coords, grad=True)
    g = g.copy()
    h = 1e-5
    plus = np.array(frag.coords, dtype=float)
    plus[0, 0] += h
    minus = np.array(frag.coords, dtype=float)
    minus[0, 0] -= h
    fd = (qmmm.run(current_coords=plus) - qmmm.run(current_coords=minus)) / (2 * h * ANG_TO_BOHR)
    print(
        "charge_response",
        "reported gradient atom 0 x",
        g[0, 0],
        "finite_difference",
        fd,
        "difference",
        fd - g[0, 0],
    )
