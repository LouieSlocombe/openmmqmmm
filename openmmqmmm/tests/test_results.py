"""Tests for the Results dataclass and its JSON serialization."""

import json

import numpy as np

from openmmqmmm import Results, read_results_from_file


def test_results_roundtrip():
    """Write a Results object to disk and read it back unchanged."""
    gradient = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])
    result = Results(label="Singlepoint", energy=-100.5, charge=0, mult=1, gradient=gradient)

    result.write_to_disk(filename="results_test.json")
    read_back = read_results_from_file(filename="results_test.json")

    assert read_back.label == "Singlepoint"
    assert read_back.energy == -100.5
    assert read_back.charge == 0
    assert read_back.mult == 1
    # ndarrays round-trip through JSON as nested lists
    assert np.allclose(np.array(read_back.gradient), gradient)
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
    assert np.allclose(np.array(read_back.gradients[1]), 1.0)
