"""Reproduce the QM/MM embedding audit probe: virtual_sites.py.

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

with tempfile.TemporaryDirectory(prefix="openmmqmmm-embedding-virtual_sites-") as scratch, contextlib.chdir(scratch):
    import logging

    logging.disable(logging.CRITICAL)
    import importlib.util

    import numpy as np
    import openmm

    from openmmqmmm import Fragment, OpenMMTheory, QMMMTheory
    from openmmqmmm.constants import ANG_TO_BOHR

    spec = importlib.util.spec_from_file_location(
        "periodic_tests", str(REPO_ROOT / "tests/test_qmmm_periodic_accuracy.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    coords = np.array([[1.0, 1, 1], [6.0, 1, 1], [8.0, 1, 1], [7.0, 1, 1]])
    f = Fragment(elems=["He"] * 4, coords=coords, conncalc=False)
    mm = OpenMMTheory(
        fragment=f,
        dummysystem=True,
        platform="Reference",
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    atoms = list(mm.topology.atoms())
    mm.topology.addBond(atoms[1], atoms[2])
    mm.system.setParticleMass(3, 0)
    mm.system.setVirtualSite(3, openmm.TwoParticleAverageSite(1, 2, 0.5, 0.5))
    mm.periodic = True
    mm.system.setDefaultPeriodicBoxVectors(*np.eye(3) * 3.0)
    mm.nonbonded_force.setNonbondedMethod(openmm.NonbondedForce.PME)
    mm.nonbonded_force.setCutoffDistance(1.0)
    charges = [0, 0, 0, -1]
    for i, q in enumerate(charges):
        mm.nonbonded_force.setParticleParameters(i, q, 0.1, 0)
    mm.charges = charges
    q = QMMMTheory(fragment=f, qm_theory=mod._CoulombQM(), mm_theory=mm, qmatoms=[0], qm_charge=0, qm_mult=1)
    e, g = q.run(current_coords=coords, grad=True)
    g = g.copy()
    h = 1e-5
    p = coords.copy()
    m = coords.copy()
    p[1, 0] += h
    p[3, 0] += 0.5 * h
    m[1, 0] -= h
    m[3, 0] -= 0.5 * h
    fd = (q.run(current_coords=p) - q.run(current_coords=m)) / (2 * h * ANG_TO_BOHR)
    print("gradient x", g[:, 0])
    print("host analytic", g[1, 0], "host numeric chain", fd, "relative error", abs(g[1, 0] - fd) / abs(fd))
    # stale virtual coordinate is not updated even if real host moves
    shifted = coords.copy()
    shifted[1, 0] += 1
    q.run(current_coords=shifted)
    print("stale site E", q.QMenergy, "PC coords", q.pointchargecoords.tolist())
    shifted[3, 0] += 0.5
    q.run(current_coords=shifted)
    print("correct site E", q.QMenergy)
    from openmmqmmm.constants import HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM, HARTREE_TO_KJ_PER_MOL
    from openmmqmmm.openmm.rpmd_force import RPMDQMMMForceProvider, add_rpmd_python_force

    q.openmm_externalforce = True
    provider = RPMDQMMMForceProvider(q, f.elems, 0, 1, periodic=True)
    _force, group = add_rpmd_python_force(mm.system, provider, periodic=True)
    integ = openmm.VerletIntegrator(0.001)
    context = openmm.Context(mm.system, integ, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(coords * 0.1)
    context.computeVirtualSites()
    state = context.getState(getEnergy=True, getForces=True, groups={group})
    md_grad = (
        -state.getForces(asNumpy=True).value_in_unit(openmm.unit.kilojoule_per_mole / openmm.unit.nanometer)
        / HARTREE_PER_BOHR_TO_KJ_PER_MOL_NM
    )
    values = []
    for delta in [h, -h]:
        probe = coords.copy()
        probe[1, 0] += delta
        context.setPositions(probe * 0.1)
        context.computeVirtualSites()
        values.append(
            context.getState(getEnergy=True, groups={group})
            .getPotentialEnergy()
            .value_in_unit(openmm.unit.kilojoule_per_mole)
            / HARTREE_TO_KJ_PER_MOL
        )
    md_fd = (values[0] - values[1]) / (2 * h * ANG_TO_BOHR)
    print("MD QM gradient x", md_grad[:, 0])
    print("MD host analytic", md_grad[1, 0], "MD host numeric", md_fd)
    print("last raw QM PC host/site rows", q.QM_PC_gradient[:, 0])
