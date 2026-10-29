"""Benchmark accounting and accuracy metrics against a closed-form model."""

import copy
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

RUNNER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/benchmark_qmmm.py"))


def _config(tmp_path):
    config = copy.deepcopy(RUNNER["DEFAULT_CONFIG"])
    config["matrix"]["scratch"] = [str(tmp_path)]
    config["matrix"]["numerical"] = [{"name": "reference", "scale": 1}, {"name": "approximate", "scale": 1.1}]
    config["repeats"] = 2
    return config


def test_benchmark_measures_displaced_accuracy_and_excludes_warmup(tmp_path):
    calls = []

    def factory(case, _options):
        scale = case["numerical"]["scale"]

        class Theory:
            def run(self, *, current_coords, grad):
                assert grad is True
                calls.append((scale, current_coords.copy()))
                # Units are arbitrary here; values independently determine report metrics.
                return scale * np.sum(current_coords**2), 2 * scale * current_coords

        return Theory(), np.array([[[1.0, 0, 0]], [[2.0, 0, 0]]])

    report = RUNNER["run_benchmark"](_config(tmp_path), factory=factory)
    json.dumps(report, allow_nan=False)
    reference, approximate = report["cases"]
    assert len(calls) == 12
    assert reference["wall_seconds"]["count"] == 4
    assert len(reference["warmup_wall_seconds"]) == 2
    errors = approximate["samples"][1]["error_vs_reference"]
    assert errors["relative_energy_eh"] == pytest.approx(0.3)
    assert errors["force_max_component_eh_bohr"] == pytest.approx(0.4)
    assert errors["force_rms_component_eh_bohr"] == pytest.approx(0.4 / np.sqrt(3))
    assert errors["force_relative_l2"] == pytest.approx(0.1)
    assert reference["snapshot_sha256"] == approximate["snapshot_sha256"]
    assert all(not Path(row["scratch_directory"]).exists() for row in report["cases"])


def test_benchmark_rejects_incomparable_snapshots_and_preserves_failed_scratch(tmp_path):
    def factory(case, _options):
        class Theory:
            def run(self, *, current_coords, grad):
                return 1.0, np.zeros_like(current_coords)

        return Theory(), np.ones((2, 1, 3)) * case["numerical"]["scale"]

    report = RUNNER["run_benchmark"](_config(tmp_path), factory=factory)
    assert report["cases"][0]["status"] == "ok"
    failed = report["cases"][1]
    assert failed["status"] == "error"
    assert "identical snapshots" in failed["error"]
    assert Path(failed["scratch_directory"]).exists()
    assert "median_wall_ratio_vs_reference" not in failed


def test_benchmark_failed_reference_does_not_report_accuracy(tmp_path):
    def factory(case, _options):
        if case["numerical"]["scale"] == 1:
            raise RuntimeError("SCF did not converge")

        class Theory:
            def run(self, *, current_coords, grad):
                return 0.0, np.zeros_like(current_coords)

        return Theory(), np.zeros((2, 1, 3))

    report = RUNNER["run_benchmark"](_config(tmp_path), factory=factory)
    assert report["cases"][0]["status"] == "error"
    assert "error_vs_reference" not in report["cases"][1]["samples"][0]


def test_benchmark_loads_local_factory_and_handles_zero_reference_forces(tmp_path):
    fixture = tmp_path / "fixture.py"
    fixture.write_text(
        "import numpy as np\n"
        "class Theory:\n"
        "    def run(self, *, current_coords, grad):\n"
        "        return 0.0, np.zeros_like(current_coords)\n"
        "def build(case, options):\n"
        "    return Theory(), np.array(options['snapshots'])\n"
    )
    config = _config(tmp_path)
    config["fixture"] = {
        "factory": "fixture.py:build",
        "options": {"snapshots": [[[1.0, 0, 0]], [[2.0, 0, 0]]]},
    }
    original_cwd = Path.cwd()
    report = RUNNER["run_benchmark"](config, base_dir=tmp_path)
    assert Path.cwd() == original_cwd
    assert all(case["status"] == "ok" for case in report["cases"])
    assert report["cases"][1]["samples"][0]["error_vs_reference"]["force_relative_l2"] is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_benchmark_rejects_invalid_repetitions(tmp_path, value):
    config = _config(tmp_path)
    config["repeats"] = value
    with pytest.raises(ValueError, match="positive integer"):
        RUNNER["run_benchmark"](config)
