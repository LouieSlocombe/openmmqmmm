"""Cross-package interop: openmmqmmm QM/MM potentials under openmmnqe drivers.

These tests exercise the OpenMM-native seam between the two sibling packages:
openmmqmmm exports plain OpenMM objects (System + PythonForce + Modeller) and
openmmnqe consumes them through its PreparedSystem force-field stand-in;
openmmnqe's bead seeding and reporters plug into openmmqmmm's engine through
run(pre_dynamics_hook=..., extra_reporters=...). Neither package imports the
other; the whole file skips when openmmnqe is not installed alongside.
"""

from pathlib import Path

import numpy as np
import openmm
import openmm.app
import pytest
from conftest import _make_analytic_qmmm

openmmnqe = pytest.importorskip("openmmnqe")

from openmmqmmm import (  # noqa: E402 - imports valid only after the importorskip gate
    MolecularDynamicsEngine,
    export_rpmd_potential,
    modeller_from_topology,
)

NUM_BEADS = 4


def test_nqe_rpmd_stages_run_on_exported_qmmm_potential():
    qmmm, fragment, qm = _make_analytic_qmmm()
    export = export_rpmd_potential(theory=qmmm, num_beads=NUM_BEADS)
    prepared = openmmnqe.PreparedSystem(export.system)

    openmmnqe.run_openmm_rpmd_equilibration(
        export.modeller,
        prepared,
        n_beads=NUM_BEADS,
        n_1=4,
        n_2=6,
        n_report=5,
        platform_name="Reference",
        seed=11,
    )

    assert qm.calls, "The QM theory must have been evaluated through the PythonForce seam"
    assert export.provider.evaluation_count > 0
    assert Path("rpmd_ready_final.pdb").exists()
    assert Path("rpmd_ready_centroid.pdb").exists()
    with np.load("rpmd_ready.chk", allow_pickle=False) as archive:
        assert archive["kind"].item() == "openmmnqe-rpmd-restart"
        assert archive["num_beads"].item() == NUM_BEADS
        assert archive["num_particles"].item() == fragment.numatoms
        assert archive["step_count"].item() == 10
        assert np.isfinite(archive["positions_nm"]).all()

    evaluations_after_equilibration = export.provider.evaluation_count
    openmmnqe.run_openmm_rpmd_prod(
        export.modeller,
        prepared,
        checkpoint_file="rpmd_ready.chk",
        n_beads=NUM_BEADS,
        steps=10,
        n_report=5,
        barostat_freq=None,
        platform_name="Reference",
    )

    assert export.provider.evaluation_count > evaluations_after_equilibration
    with np.load("rpmd_prod.chk", allow_pickle=False) as archive:
        assert archive["num_beads"].item() == NUM_BEADS
        assert archive["step_count"].item() == 20, "Production must continue from the equilibration bead archive"
        assert np.isfinite(archive["positions_nm"]).all()
        assert np.isfinite(archive["velocities_nm_per_ps"]).all()


def test_engine_rpmd_uses_nqe_bead_seeding_and_reporters():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        timestep=0.000001,
        integrator="RPMDIntegrator",
        rpmd_num_copies=NUM_BEADS,
        traj_frequency=2,
    )
    modeller = modeller_from_topology(topology=qmmm.mm_theory.topology, coords_angstrom=fragment.coords)
    centroid_reporter = openmmnqe.RPMDCentroidReporter(
        topology=modeller.topology,
        file_name="interop_centroid.pdb",
        reportInterval=2,
        num_beads=NUM_BEADS,
    )
    seeded_positions = []

    def seed_beads(md):
        openmmnqe.init_beads(modeller, md.simulation, NUM_BEADS, seed=3)
        for copy in range(NUM_BEADS):
            state = md.simulation.integrator.getState(copy, getPositions=True)
            seeded_positions.append(state.getPositions(asNumpy=True).value_in_unit(openmm.unit.nanometer))

    engine.run(simulation_steps=4, extra_reporters=[centroid_reporter], pre_dynamics_hook=seed_beads)

    seeded = np.stack(seeded_positions)
    assert seeded.shape == (NUM_BEADS, fragment.numatoms, 3)
    assert not np.allclose(seeded[0], seeded[1]), "Bead seeding must displace the copies from each other"
    assert seeded.mean(axis=0) == pytest.approx(np.asarray(fragment.coords) * 0.1, abs=1e-9), (
        "init_beads pins the ring-polymer centroid at the Modeller positions"
    )
    assert engine.rpmd_force_provider.evaluation_count > 0
    centroid_reporter._out.flush()
    assert "MODEL" in Path("interop_centroid.pdb").read_text(), "The openmmnqe reporter must have written frames"


def test_nqe_deuteration_applies_to_qmmm_system():
    qmmm, fragment, _qm = _make_analytic_qmmm()
    modeller = modeller_from_topology(topology=qmmm.mm_theory.topology, coords_angstrom=fragment.coords)
    openmmnqe.deuterate_system(modeller, qmmm.mm_theory.system, option="all")

    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        charge=0,
        mult=1,
        timestep=0.000001,
        integrator="RPMDIntegrator",
        rpmd_num_copies=2,
    )
    masses = [
        qmmm.mm_theory.system.getParticleMass(index).value_in_unit(openmm.unit.dalton)
        for index in range(qmmm.mm_theory.system.getNumParticles())
    ]
    assert masses == pytest.approx([2.014, 2.014], abs=0.01), (
        "Deuterated masses must survive engine construction (no hydrogen-mass repartitioning for NQE runs)"
    )

    engine.run(simulation_steps=2)
    assert engine.simulation.currentStep == 2


def test_prepared_system_rejects_foreign_topology():
    qmmm, _fragment, _qm = _make_analytic_qmmm()
    export = export_rpmd_potential(theory=qmmm, num_beads=2)
    prepared = openmmnqe.PreparedSystem(export.system)

    foreign = openmm.app.Topology()
    residue = foreign.addResidue("AR", foreign.addChain())
    foreign.addAtom("Ar", openmm.app.Element.getBySymbol("Ar"), residue)

    with pytest.raises(ValueError, match="prepared for a different structure"):
        prepared.createSystem(foreign)
