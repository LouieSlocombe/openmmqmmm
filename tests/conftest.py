import stat

import pytest

# What a real ORCA binary prints when invoked with no arguments; find_orca probes
# for this to tell the quantum chemistry program from unrelated `orca` binaries.
ORCA_PROBE_OUTPUT = "This program requires the name of a parameterfile"


@pytest.fixture(autouse=True)
def run_in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def make_fake_orca_install():
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
    orca_dir = make_fake_orca_install(tmp_path / "fake_orca")
    monkeypatch.setenv("OPENMMQMMM_ORCADIR", str(orca_dir))
    return orca_dir
