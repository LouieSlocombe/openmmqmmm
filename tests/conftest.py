import stat

import pytest

# What a real ORCA binary prints when invoked with no arguments; find_orca probes
# for this to tell the quantum chemistry program from unrelated `orca` binaries.
ORCA_PROBE_OUTPUT = "This program requires the name of a parameterfile"


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


@pytest.fixture
def make_fake_orca_install():
    """Factory for a directory that passes find_orca's validation.

    Lets the input-writing and output-parsing paths be tested without a real ORCA
    installation, so they are covered in CI too.
    """

    def _make(directory, with_helpers=True, output=ORCA_PROBE_OUTPUT):
        directory.mkdir(parents=True, exist_ok=True)
        orca = directory / "orca"
        orca.write_text(f"#!/bin/sh\necho '{output}'\nexit 2\n")
        orca.chmod(orca.stat().st_mode | stat.S_IXUSR)
        if with_helpers:
            for helper in ("orca_scf", "orca_gtoint"):
                (directory / helper).write_text("")
        return directory

    return _make


@pytest.fixture
def fake_orca_dir(tmp_path, monkeypatch, make_fake_orca_install):
    """A fake ORCA installation wired up as OPENMMQMMM_ORCADIR.

    ORCATheory validates an ORCA installation in __init__, so constructing one at
    all requires this even for tests that never launch a calculation.
    """
    orca_dir = make_fake_orca_install(tmp_path / "fake_orca")
    monkeypatch.setenv("OPENMMQMMM_ORCADIR", str(orca_dir))
    return orca_dir
