import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_generated_files():
    """Remove files and directories generated in the working directory during the test session.

    The ORCA/OpenMM/geomeTRIC runs scatter output files (orca.*, finalsystem.*, trajectories,
    fragment files, ...) into the current working directory. Snapshot the directory before the
    session and delete anything new afterwards, so only pre-existing files survive.
    """
    cwd = Path.cwd()
    before = set(os.listdir(cwd))
    yield
    for name in set(os.listdir(cwd)) - before:
        path = cwd / name
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
