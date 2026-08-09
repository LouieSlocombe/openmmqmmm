import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path, monkeypatch):
    """Run every test in its own temporary directory.

    The ORCA/OpenMM/geomeTRIC runs scatter output files (orca.*, trajectories,
    fragment files, ...) into the current working directory. An isolated cwd per
    test keeps the repository clean regardless of where pytest is invoked from,
    and prevents runs from picking up each other's files (e.g. ORCA autostart
    reading a stale .gbw from a previous test).
    """
    monkeypatch.chdir(tmp_path)
