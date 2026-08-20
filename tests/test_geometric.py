import logging.config
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import pytest

from openmmqmmm import Fragment, ZeroTheory, optimize_geometry
from openmmqmmm.exceptions import InputError
from openmmqmmm.geometric import (
    CONVERGENCE_PRESETS,
    Constraints,
    GeometricOptimizer,
    _run_optimizer_without_reconfiguring_logging,
)


def test_geometric_optimizer_preserves_application_logging(monkeypatch, tmp_path):
    file_config_calls = []

    def application_file_config(*args, **kwargs):
        file_config_calls.append((args, kwargs))

    monkeypatch.setattr(logging.config, "fileConfig", application_file_config)
    logfile = tmp_path / "geometric.log"
    geometric_logger = logging.getLogger("geometric")
    original_level = geometric_logger.level
    geometric_logger.setLevel(logging.ERROR)
    root_output = StringIO()
    root_handler = logging.StreamHandler(root_output)
    root_logger = logging.getLogger()
    root_logger.addHandler(root_handler)

    def fake_run_optimizer(**kwargs):
        logging.config.fileConfig(kwargs["logIni"], defaults={"logfilename": logfile})
        logging.getLogger("geometric.fake").info("Step 1: test")
        logging.getLogger("application.fake").warning("application warning")
        return "finished"

    try:
        result = _run_optimizer_without_reconfiguring_logging(fake_run_optimizer, {"logIni": "openmmqmmm/log.ini"})
    finally:
        root_logger.removeHandler(root_handler)
        root_handler.close()
        geometric_logger.setLevel(original_level)

    assert result == "finished"
    assert file_config_calls == []
    assert logging.config.fileConfig is application_file_config
    assert logfile.read_text() == "Step 1: test"
    assert root_output.getvalue() == "application warning\n"


def test_geometric_logging_isolation_restores_state_after_failure(monkeypatch, tmp_path):
    def application_file_config(*args, **kwargs):
        pytest.fail("geomeTRIC must not invoke the application's fileConfig")

    monkeypatch.setattr(logging.config, "fileConfig", application_file_config)
    geometric_logger = logging.getLogger("geometric")
    original_handlers = list(geometric_logger.handlers)
    original_state = (geometric_logger.level, geometric_logger.propagate, geometric_logger.disabled)
    application_output = StringIO()
    application_handler = logging.StreamHandler(application_output)
    geometric_logger.addHandler(application_handler)
    geometric_logger.setLevel(logging.ERROR)
    geometric_logger.disabled = True
    logfile = tmp_path / "failed-geometric.log"
    internal_handler = None

    def failing_run_optimizer(**kwargs):
        nonlocal internal_handler
        logging.config.fileConfig(kwargs["logIni"], defaults={"logfilename": logfile})
        internal_handler = next(
            handler
            for handler in geometric_logger.handlers
            if handler is not application_handler and handler not in original_handlers
        )
        logging.getLogger("geometric.failure").info("Step 1: before failure")
        raise RuntimeError("optimizer failed")

    try:
        with pytest.raises(RuntimeError, match="optimizer failed"):
            _run_optimizer_without_reconfiguring_logging(failing_run_optimizer, {"logIni": "openmmqmmm/log.ini"})

        assert logging.config.fileConfig is application_file_config
        assert geometric_logger.handlers == [*original_handlers, application_handler]
        assert geometric_logger.level == logging.ERROR
        assert geometric_logger.propagate == original_state[1]
        assert geometric_logger.disabled is True
        assert application_handler.filters == []
        assert application_output.getvalue() == ""
        assert logfile.read_text() == "Step 1: before failure"
        assert internal_handler is not None and internal_handler.stream is None
    finally:
        for handler in list(geometric_logger.handlers):
            if handler not in original_handlers:
                geometric_logger.removeHandler(handler)
                handler.close()
        geometric_logger.setLevel(original_state[0])
        geometric_logger.propagate = original_state[1]
        geometric_logger.disabled = original_state[2]


def test_geometric_logging_isolation_serializes_concurrent_runs(monkeypatch, tmp_path):
    def application_file_config(*args, **kwargs):
        return None

    monkeypatch.setattr(logging.config, "fileConfig", application_file_config)
    start = threading.Barrier(2)
    activity_lock = threading.Lock()
    active_runs = 0
    maximum_active_runs = 0

    def worker(label):
        logfile = tmp_path / f"{label}.log"

        def fake_run_optimizer(**kwargs):
            nonlocal active_runs, maximum_active_runs
            logging.config.fileConfig(kwargs["logIni"], defaults={"logfilename": logfile})
            with activity_lock:
                active_runs += 1
                maximum_active_runs = max(maximum_active_runs, active_runs)
            try:
                time.sleep(0.05)
                logging.getLogger("geometric.concurrent").info(label)
            finally:
                with activity_lock:
                    active_runs -= 1

        start.wait()
        _run_optimizer_without_reconfiguring_logging(fake_run_optimizer, {"logIni": "openmmqmmm/log.ini"})
        return logfile

    with ThreadPoolExecutor(max_workers=2) as executor:
        logfiles = list(executor.map(worker, ("first", "second")))

    assert maximum_active_runs == 1
    assert {logfile.read_text() for logfile in logfiles} == {"first", "second"}
    assert logging.config.fileConfig is application_file_config


def test_geometric_dummy():
    coords = """
    O       -1.377626260      0.000000000     -1.740199718
    H       -1.377626260      0.759337000     -1.144156718
    H       -1.377626260     -0.759337000     -1.144156718
    """
    H2Ofragment = Fragment(coordsstring=coords, charge=0, mult=1)

    zerotheorycalc = ZeroTheory()

    # Optimize with dummy theory: exercises the geomeTRIC coupling
    result = optimize_geometry(fragment=H2Ofragment, theory=zerotheorycalc)

    assert result.energy == 0.0, "ZeroTheory energy should be 0.0"
    assert "Step " in Path("geometric_OPTtraj.log").read_text()


# The constraints.txt format is geomeTRIC's, not ours: a $freeze or $set section header
# followed by one constraint per line, atom indices 1-based. Nothing downstream validates
# it, so a malformed file means a silently unconstrained optimization.


@pytest.fixture
def optimizer():
    """A GeometricOptimizer with no initialization run: write_constraintsfile needs no state."""
    return GeometricOptimizer.__new__(GeometricOptimizer)


def test_no_constraints_writes_no_file(optimizer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    optimizer.write_constraintsfile([], Constraints(), constrainvalue=False)

    assert optimizer.constraintsfile is None
    assert not (tmp_path / "constraints.txt").exists()


def test_frozen_atoms_are_written_one_based(optimizer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    optimizer.write_constraintsfile([0, 3], Constraints(), constrainvalue=False)

    assert optimizer.constraintsfile == "constraints.txt"
    assert (tmp_path / "constraints.txt").read_text() == "$freeze\nxyz 1\nxyz 4\n"


def test_internal_coordinates_without_values_are_frozen(optimizer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    constraints = Constraints(bond=[[0, 1]], angle=[[0, 1, 2]], dihedral=[[0, 1, 2, 3]])
    optimizer.write_constraintsfile([], constraints, constrainvalue=False)

    assert (tmp_path / "constraints.txt").read_text() == (
        "$freeze\ndistance 1 2\n$freeze\nangle 1 2 3\n$freeze\ndihedral 1 2 3 4\n"
    )


def test_internal_coordinates_with_values_are_set(optimizer, tmp_path, monkeypatch):
    """constrainvalue=True means the last element of each entry is a target value."""
    monkeypatch.chdir(tmp_path)
    constraints = Constraints(bond=[[0, 1, 1.5]], angle=[[0, 1, 2, 104.5]], dihedral=[[0, 1, 2, 3, 180.0]])
    optimizer.write_constraintsfile([], constraints, constrainvalue=True)

    assert (tmp_path / "constraints.txt").read_text() == (
        "$set\ndistance 1 2 1.5\n$set\nangle 1 2 3 104.5\n$set\ndihedral 1 2 3 4 180.0\n"
    )


def test_cartesian_freezes_never_take_a_value(optimizer, tmp_path, monkeypatch):
    """x/y/z freezes stay under $freeze even when constrainvalue is set: there is no value."""
    monkeypatch.chdir(tmp_path)
    constraints = Constraints(x=[0], y=[1], z=[2], xy=[3], xz=[4], yz=[5])
    optimizer.write_constraintsfile([], constraints, constrainvalue=True)

    assert (tmp_path / "constraints.txt").read_text() == (
        "$freeze\nx 1\n$freeze\ny 2\n$freeze\nz 3\n$freeze\nxy 4\n$freeze\nxz 5\n$freeze\nyz 6\n"
    )


def test_a_stale_constraints_file_is_replaced(optimizer, tmp_path, monkeypatch):
    """A file left by a previous run must not be appended to."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "constraints.txt").write_text("$freeze\nxyz 99\n")

    optimizer.write_constraintsfile([0], Constraints(), constrainvalue=False)

    assert "99" not in (tmp_path / "constraints.txt").read_text()


def test_define_constraints_accepts_both_dihedral_spellings():
    optimizer = GeometricOptimizer.__new__(GeometricOptimizer)
    optimizer.active_region = False

    assert optimizer.define_constraints({"dihedral": [[0, 1, 2, 3]]}).dihedral == [[0, 1, 2, 3]]
    assert optimizer.define_constraints({"torsion": [[0, 1, 2, 3]]}).dihedral == [[0, 1, 2, 3]]
    assert optimizer.define_constraints(None).dihedral is None


def test_convergence_presets_are_complete():
    """Every preset must set all six thresholds; a missing key means geomeTRIC's default."""
    for name, criteria in CONVERGENCE_PRESETS.items():
        assert set(criteria) == {
            "convergence_energy",
            "convergence_grms",
            "convergence_gmax",
            "convergence_drms",
            "convergence_dmax",
            "convergence_cmax",
        }, f"{name} is missing thresholds"
        assert all(value > 0 for value in criteria.values())

    # Tighter presets must actually be tighter than the ones they refine.
    assert CONVERGENCE_PRESETS["ORCA_TIGHT"]["convergence_grms"] < CONVERGENCE_PRESETS["ORCA"]["convergence_grms"]
    assert (
        CONVERGENCE_PRESETS["GAU_VERYTIGHT"]["convergence_grms"] < CONVERGENCE_PRESETS["GAU_TIGHT"]["convergence_grms"]
    )


def test_unknown_convergence_setting_is_rejected():
    optimizer = GeometricOptimizer.__new__(GeometricOptimizer)
    with pytest.raises(InputError):
        optimizer.convergence_criteria("NotAPreset", None)


def test_user_criteria_override_the_preset():
    optimizer = GeometricOptimizer.__new__(GeometricOptimizer)
    optimizer.convergence_criteria("ORCA", {"convergence_grms": 1e-9})

    assert optimizer.conv_criteria["convergence_grms"] == 1e-9
    assert optimizer.conv_criteria["convergence_energy"] == CONVERGENCE_PRESETS["ORCA"]["convergence_energy"]
    assert CONVERGENCE_PRESETS["ORCA"]["convergence_grms"] != 1e-9, "The preset itself must not be mutated"
