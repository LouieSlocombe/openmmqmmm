"""Numerical baselines for the runnable finite periodic embedding diagnostic."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def validation():
    runner = runpy.run_path(str(ROOT / "examples" / "periodic_embedding_validation.py"))
    return runner["run_validation"]()


@pytest.mark.parametrize("scan", ["box_scan", "extent_scan"])
def test_relative_energies_and_qm_mm_probe_forces_match_recorded_baseline(validation, scan):
    baseline = json.loads((ROOT / "docs/audits/embedding-repros/finite-periodic-baseline.json").read_text())
    actual_rows = validation[scan]["rows"]
    expected_rows = baseline[scan]["rows"]
    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        assert (actual["edge_angstrom"], actual["molecules"]) == (expected["edge_angstrom"], expected["molecules"])
        # Compare paired energies, not extensive absolute energies across boxes.
        # Allow Reference-platform PME roundoff across dependency versions.
        assert actual["relative_energy_B_minus_A_kj_mol"] == pytest.approx(
            expected["relative_energy_B_minus_A_kj_mol"], rel=0, abs=1e-7
        )
        for state in ("A", "B"):
            for component in ("qm_embedding", "mm", "total"):
                np.testing.assert_allclose(
                    actual["states"][state]["probe_forces_kj_mol_nm"][component],
                    expected["states"][state]["probe_forces_kj_mol_nm"][component],
                    rtol=0,
                    atol=1e-6,
                )


def test_box_dilation_plateau_is_distinct_from_adding_outer_molecules(validation):
    small, medium, _large, reference = validation["box_scan"]["rows"]
    assert small["relative_energy_error_kj_mol"]["qm"] > 0.005
    assert medium["relative_energy_error_kj_mol"]["qm"] == pytest.approx(0, abs=1e-10)
    assert medium["force_errors_vs_largest"]["qm_embedding"]["qm_atoms"]["max_vector_kj_mol_nm"] < 1e-10
    # Identical finite QM clusters coexist with different MM periodic fields.
    assert medium["force_errors_vs_largest"]["mm"]["mm_atoms"]["max_vector_kj_mol_nm"] > 0.1
    assert reference["relative_energy_error_kj_mol"]["total"] == 0

    inner, middle, outer = validation["extent_scan"]["rows"]
    assert [row["molecules"] for row in (inner, middle, outer)] == [26, 124, 342]
    # This particular nested neutral environment is nonmonotonic. A plateau in
    # the dilute-box scan above cannot certify convergence of its finite field.
    assert abs(middle["relative_energy_error_kj_mol"]["qm"]) > abs(inner["relative_energy_error_kj_mol"]["qm"])
    assert middle["force_errors_vs_largest"]["total"]["qm_atoms"]["rms_vector_kj_mol_nm"] > 1
    assert middle["force_errors_vs_largest"]["total"]["mm_atoms"]["rms_vector_kj_mol_nm"] > 0.5


def test_nve_timestep_reduction_resolves_smooth_error_but_preserves_image_jump(validation):
    runs = {(row["case"], row["timestep_fs"]): row for row in validation["nve"]}
    for case in ("smooth", "switch"):
        coarse, fine = (runs[case, timestep] for timestep in (0.5, 0.25))
        assert coarse["duration_fs"] == fine["duration_fs"] == 12
        assert fine["steps"] == 2 * coarse["steps"]
        np.testing.assert_array_equal(coarse["initial_coords_angstrom"], fine["initial_coords_angstrom"])
        np.testing.assert_array_equal(coarse["initial_velocities_nm_ps"], fine["initial_velocities_nm_ps"])
        assert coarse["trace"][0]["total_kj_mol"] == pytest.approx(fine["trace"][0]["total_kj_mol"], abs=1e-12)
        for run in (coarse, fine):
            # Both QM and MM coordinates participate in the actual dynamics.
            displacement = np.linalg.norm(run["final_displacements_angstrom"], axis=1)
            assert np.all(displacement > 1e-4)
            assert len(run["trace"]) == run["steps"] + 1
            assert run["trace"][-1]["time_fs"] == run["duration_fs"]

    smooth_coarse, smooth_fine = (runs["smooth", timestep] for timestep in (0.5, 0.25))
    assert smooth_coarse["image_switch_count"] == smooth_fine["image_switch_count"] == 0
    coarse_error = smooth_coarse["max_abs_energy_deviation_kj_mol"]
    fine_error = smooth_fine["max_abs_energy_deviation_kj_mol"]
    assert coarse_error < 1e-6
    assert coarse_error / fine_error == pytest.approx(4, rel=0.1)

    switch_coarse, switch_fine = (runs["switch", timestep] for timestep in (0.5, 0.25))
    for run in (switch_coarse, switch_fine):
        assert run["image_switch_count"] == len(run["events"]) == 1
        event = run["events"][0]
        assert event["from_image"] == [0, 0, 0]
        assert event["to_image"] == [-1, 0, 0]
        assert -0.26 < event["same_geometry_qm_branch_jump_kj_mol"] < -0.25
        # The finite-field discontinuity leaves a finite energy change. The
        # event step also contains integration error; it is not an exact jump.
        assert event["step_total_energy_change_kj_mol"] == pytest.approx(
            event["same_geometry_qm_branch_jump_kj_mol"], abs=1e-5
        )
        assert run["net_energy_change_kj_mol"] == pytest.approx(event["step_total_energy_change_kj_mol"], abs=1e-6)
    assert switch_fine["net_energy_change_kj_mol"] == pytest.approx(switch_coarse["net_energy_change_kj_mol"], abs=1e-5)
