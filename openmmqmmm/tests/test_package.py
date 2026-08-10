import subprocess
import sys

import openmmqmmm
from openmmqmmm.orca import find_orca


def test_version_is_exposed():
    """The package exposes __version__, sourced from the installed metadata."""
    assert isinstance(openmmqmmm.__version__, str)
    assert openmmqmmm.__version__
    assert "__version__" in openmmqmmm.__all__


def test_import_silent():
    """Importing the package must produce no output (no banner, no settings dump)."""
    result = subprocess.run(
        [sys.executable, "-c", "import openmmqmmm"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""


def test_find_orca_explicit_dir(tmp_path, monkeypatch, make_fake_orca_install):
    monkeypatch.delenv("OPENMMQMMM_ORCADIR", raising=False)
    orca_dir = make_fake_orca_install(tmp_path / "orca_install")
    assert find_orca(orcadir=str(orca_dir)) == str(orca_dir)


def test_find_orca_env_var(tmp_path, monkeypatch, make_fake_orca_install):
    orca_dir = make_fake_orca_install(tmp_path / "orca_install")
    monkeypatch.setenv("OPENMMQMMM_ORCADIR", str(orca_dir))
    assert find_orca() == str(orca_dir)


def test_find_orca_rejects_impostor_in_path(tmp_path, monkeypatch, make_fake_orca_install):
    """A lone orca binary in PATH without orca_* helpers (e.g. the GNOME
    screen reader) must not be mistaken for the quantum chemistry program."""
    impostor_dir = make_fake_orca_install(tmp_path / "usr_bin", with_helpers=False, output="not the qc program")
    monkeypatch.delenv("OPENMMQMMM_ORCADIR", raising=False)
    monkeypatch.setenv("PATH", str(impostor_dir))
    assert find_orca(required=False) is None


def test_find_orca_accepts_valid_path_install(tmp_path, monkeypatch, make_fake_orca_install):
    orca_dir = make_fake_orca_install(tmp_path / "orca_install")
    monkeypatch.delenv("OPENMMQMMM_ORCADIR", raising=False)
    monkeypatch.setenv("PATH", str(orca_dir))
    assert find_orca(required=False) == str(orca_dir)


def test_find_orca_nothing_found(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENMMQMMM_ORCADIR", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert find_orca(required=False) is None


def test_find_orca_invalid_explicit_dir_not_required(tmp_path, monkeypatch, make_fake_orca_install):
    """An explicit location that fails validation must not fall back to PATH."""
    valid_dir = make_fake_orca_install(tmp_path / "valid")
    monkeypatch.delenv("OPENMMQMMM_ORCADIR", raising=False)
    monkeypatch.setenv("PATH", str(valid_dir))
    assert find_orca(orcadir=str(tmp_path / "missing"), required=False) is None
