"""Optional profiling preserves the potential and distinguishes fresh work from cache hits."""

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import _make_analytic_qmmm

from openmmqmmm import MolecularDynamicsEngine
from openmmqmmm.profiling import _MDProfile


def _records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_force_profile_measures_phases_without_changing_energy_or_gradient():
    qmmm, fragment, _ = _make_analytic_qmmm(embedding="elstat")
    engine = MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator")
    simulation = qmmm.mm_theory.create_simulation()
    qmmm.mm_theory.set_positions(fragment.coords, simulation)
    state = simulation.context.getState(getPositions=True)
    provider = engine.rpmd_force_provider
    expected_energy, expected_forces = provider(state)
    provider.clear_cache()
    profile = _MDProfile("profile.jsonl", 1, {"platform": "Reference"})
    provider.profile = profile
    first_energy, first_forces = provider(state)
    second_energy, second_forces = provider(state)
    assert first_energy == second_energy == expected_energy
    np.testing.assert_array_equal(first_forces, expected_forces)
    np.testing.assert_array_equal(second_forces, expected_forces)
    fresh, cached = _records("profile.jsonl")
    assert fresh["status"] == cached["status"] == "success"
    assert not fresh["cache_hit"] and cached["cache_hit"]
    assert fresh["qm_atom_count"] == 1 and fresh["charge_site_count"] == 1
    assert fresh["wrapper_phases_s"].keys() >= {
        "periodic_imaging",
        "embedding_preparation",
        "charge_sites",
        "qm_backend",
        "gradient_projection",
    }
    assert all(value >= 0 for value in fresh["wrapper_phases_s"].values())
    assert sum(fresh["wrapper_phases_s"].values()) <= fresh["wall_time_s"]
    assert cached["wrapper_phases_s"] == {} and cached["backend"] is None
    assert profile.evaluations == profile.cache_hits == 1


def test_force_profile_failure_does_not_reuse_stale_backend_statistics(monkeypatch):
    qmmm, fragment, qm = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(fragment=fragment, theory=qmmm, integrator="VerletIntegrator")
    simulation = qmmm.mm_theory.create_simulation()
    qmmm.mm_theory.set_positions(fragment.coords, simulation)
    provider = engine.rpmd_force_provider
    provider.profile = _MDProfile("failure.jsonl", 1, {})
    qm.last_evaluation_stats = {"status": "success", "scf_iterations": 8}

    def fail(**kwargs):
        raise ValueError("SCF failed")

    monkeypatch.setattr(qm, "run", fail)
    with pytest.raises(RuntimeError, match="SCF failed"):
        provider(simulation.context.getState(getPositions=True))
    (record,) = _records("failure.jsonl")
    assert record["status"] == "error" and record["backend"] is None
    assert provider.profile.evaluations == 0


@pytest.mark.parametrize("rpmd", [False, True])
@pytest.mark.parametrize("interval", [1, 2])
def test_md_profile_reports_each_interval_and_final_partial_without_energy_reads(rpmd, interval, monkeypatch):
    qmmm, fragment, _ = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        integrator="RPMDIntegrator" if rpmd else "VerletIntegrator",
        timestep=1e-7,
        traj_frequency=100,
        restartfile_frequency=100,
        profile_file="md.jsonl",
        profile_interval=interval,
        **({"rpmd_num_copies": 2, "rpmd_energy_frequency": 0} if rpmd else {}),
    )

    def instrument(current_engine):
        receiver = current_engine.simulation.integrator if rpmd else current_engine.simulation.context
        original = receiver.getState

        def state(*args, **kwargs):
            assert not (kwargs.get("getEnergy") or kwargs.get("energy"))
            assert not (kwargs.get("getForces") or kwargs.get("forces"))
            return original(*args, **kwargs)

        monkeypatch.setattr(receiver, "getState", state)

    engine.run(simulation_steps=3, pre_dynamics_hook=instrument)
    engine.run(simulation_steps=2, restart=True)
    records = _records("md.jsonl")
    steps = [record for record in records if record["event"] == "md_step"]
    assert [row["step"] for row in steps] == ([1, 2, 3, 4, 5] if interval == 1 else [2, 3, 4, 5])
    assert sum(row["steps"] for row in steps) == 5
    assert sum(row["fresh_evaluations"] for row in steps) == engine.rpmd_force_provider.evaluation_count
    assert all(row["force_callback_wall_s"] <= row["wall_time_s"] for row in steps)
    assert all(row["other_wall_s"] >= 0 for row in steps)
    engine.close()


def test_provider_restore_disables_profile_and_accepts_older_serialized_state():
    from test_rpmd_force_provider import _provider, _state, _Theory

    provider = _provider(_Theory(result=(0.0, np.zeros((2, 3)))))
    provider.profile = _MDProfile("original.jsonl", 1, {})
    saved = provider.__getstate__()
    assert saved["profile"] is None
    del saved["profile"]
    restored = type(provider).__new__(type(provider))
    restored.__setstate__(saved)
    energy, _ = restored(_state([[0, 0, 0], [0.1, 0, 0]]))
    assert energy == 0
    assert not Path("original.jsonl").exists()


@pytest.mark.parametrize("rpmd", [False, True])
def test_profile_includes_final_extra_reporter_evaluations(rpmd):
    qmmm, fragment, _ = _make_analytic_qmmm()
    engine = MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        integrator="RPMDIntegrator" if rpmd else "VerletIntegrator",
        timestep=1e-7,
        traj_frequency=2,
        restartfile_frequency=100,
        profile_file="extra.jsonl",
        **({"rpmd_num_copies": 4, "rpmd_qm_num_copies": 1, "rpmd_energy_frequency": 0} if rpmd else {}),
    )

    class ExtraReporter:
        def describeNextReport(self, simulation):  # noqa: N802
            return {"steps": 1, "include": []}

        def report(self, simulation, state):
            if rpmd:
                simulation.integrator.getTotalEnergy()
            else:
                simulation.context.getState(getEnergy=True)

    engine.run(simulation_steps=2, extra_reporters=[ExtraReporter()])
    records = _records("extra.jsonl")
    steps = [row for row in records if row["event"] == "md_step"]
    assert sum(row["fresh_evaluations"] for row in steps) == engine.rpmd_force_provider.evaluation_count
    assert sum(row["cache_hits"] for row in steps) == engine.rpmd_force_provider.cache_hits
    assert records[-1]["event"] == "md_step" and records[-1]["step"] == 2
    engine.close()
