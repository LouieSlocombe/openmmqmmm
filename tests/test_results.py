import json
import os
from pathlib import Path

import numpy as np
import pytest

from openmmqmmm import Results, read_results_from_file


def test_results_roundtrip():
    gradient = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    result = Results(label="Singlepoint", energy=-100.5, charge=0, mult=1, gradient=gradient)

    result.write_to_disk(filename="results_test.json")
    read_back = read_results_from_file(filename="results_test.json")

    assert read_back.label == "Singlepoint"
    assert read_back.energy == -100.5
    assert read_back.charge == 0
    assert read_back.mult == 1
    assert isinstance(read_back.gradient, np.ndarray)
    assert np.allclose(read_back.gradient, gradient)
    # Fields that were never set stay None
    assert read_back.hessian is None


def test_results_ignores_unknown_fields():
    """Files written by other versions may carry fields this Results no longer has."""
    data = {"label": "Singlepoint", "energy": -1.0, "some_retired_field": 42}
    with open("results_extra.json", "w") as f:
        json.dump(data, f)

    read_back = read_results_from_file(filename="results_extra.json")

    assert read_back.energy == -1.0
    assert not hasattr(read_back, "some_retired_field")


def test_results_writes_lists_of_arrays():
    """Lists of ndarrays (e.g. polarizability derivatives) are converted elementwise."""
    result = Results(label="NumFreq", gradients=[np.zeros((2, 3)), np.ones((2, 3))])

    result.write_to_disk(filename="results_lists.json")
    read_back = read_results_from_file(filename="results_lists.json")

    assert len(read_back.gradients) == 2
    assert all(isinstance(gradient, np.ndarray) for gradient in read_back.gradients)
    assert np.allclose(np.array(read_back.gradients[1]), 1.0)


def test_results_serializes_numpy_values_nested_in_properties(tmp_path):
    filename = tmp_path / "nested.json"
    result = Results(
        properties={
            "charges": np.array([-0.4, 0.4]),
            "summary": {"iterations": np.int64(7), "converged": np.bool_(True)},
        }
    )

    result.write_to_disk(filename)

    data = json.loads(filename.read_text())
    assert data["properties"] == {
        "charges": [-0.4, 0.4],
        "summary": {"iterations": 7, "converged": True},
    }


def test_results_serialization_failure_preserves_existing_file(tmp_path):
    class Unsupported:
        pass

    filename = tmp_path / "results.json"
    filename.write_text("previous result\n")

    Results(properties={"unsupported": Unsupported()}).write_to_disk(filename)

    assert filename.read_text() == "previous result\n"


def test_results_omits_fields_containing_non_finite_values(tmp_path):
    filename = tmp_path / "results.json"

    Results(energy=-1.0, properties={"gradient": np.array([0.0, np.nan])}).write_to_disk(filename)

    data = json.loads(filename.read_text())
    assert data["energy"] == -1.0
    assert "properties" not in data
    assert "NaN" not in filename.read_text()


def test_results_omits_fields_containing_complex_arrays(tmp_path):
    filename = tmp_path / "results.json"

    Results(energy=-1.0, properties={"amplitudes": np.array([1.0 + 2.0j])}).write_to_disk(filename)

    data = json.loads(filename.read_text())
    assert data["energy"] == -1.0
    assert "properties" not in data


def test_results_write_replaces_the_destination_atomically(tmp_path, monkeypatch):
    filename = tmp_path / "results.json"
    filename.write_text("old\n")
    replacements = []
    real_replace = os.replace

    def record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr("openmmqmmm.results.os.replace", record_replace)

    Results(energy=-2.0).write_to_disk(filename)

    assert replacements and replacements[0][1] == filename
    assert json.loads(filename.read_text())["energy"] == -2.0


def test_results_serializes_extended_numpy_floats_without_recursing(tmp_path):
    filename = tmp_path / "extended-float.json"

    Results(energy=np.longdouble("1.25"), properties={"values": np.array([1.5], dtype=np.longdouble)}).write_to_disk(
        filename
    )

    data = json.loads(filename.read_text())
    assert data["energy"] == pytest.approx(1.25)
    assert data["properties"]["values"] == pytest.approx([1.5])


def test_results_restores_array_values_inside_result_mappings(tmp_path):
    filename = tmp_path / "mapped-arrays.json"
    result = Results(
        gradients_dict={"job": np.ones((2, 3))},
        displacement_dipole_dictionary={"job": np.arange(3.0)},
        displacement_polarizability_dictionary={"job": np.eye(3)},
    )

    result.write_to_disk(filename)
    restored = read_results_from_file(filename)

    assert isinstance(restored.gradients_dict["job"], np.ndarray)
    assert isinstance(restored.displacement_dipole_dictionary["job"], np.ndarray)
    assert isinstance(restored.displacement_polarizability_dictionary["job"], np.ndarray)


def test_results_omits_a_mapping_whose_keys_would_collide_in_json(tmp_path, caplog):
    filename = tmp_path / "colliding-keys.json"

    Results(energy=-1.0, energies_dict={1: 10.0, "1": 20.0}).write_to_disk(filename)

    data = json.loads(filename.read_text())
    assert data["energy"] == -1.0
    assert "energies_dict" not in data
    assert "both normalize" in caplog.text
