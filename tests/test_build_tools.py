import re
import subprocess
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILD_TOOLS = REPOSITORY_ROOT / "build_tools"
FORCEFILL_COMMIT = "bff20e2498474b3b3aa965b090fa722f762a49ae"


def _call_sourced(script, name, *args, check=True):
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && shift && "$@"',
            "bash",
            str(BUILD_TOOLS / script),
            name,
            *(str(arg) for arg in args),
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def test_build_job_count_is_available_and_positive():
    result = _call_sourced("build_plumed.sh", "build_job_count")

    assert int(result.stdout) >= 1


def test_source_builds_use_stable_release_tags():
    script = (BUILD_TOOLS / "build_plumed.sh").read_text()

    assert 'PLUMED_VERSION="v2.10.1"' in script
    assert 'OPENMM_PLUMED_VERSION="v2.1"' in script
    assert 'VERSION="master"' not in script


def test_openmm_plumed_build_tolerates_cmake_4():
    script = (BUILD_TOOLS / "build_plumed.sh").read_text()

    assert "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" in script, (
        "openmm-plumed v2.1 declares CMAKE_MINIMUM_REQUIRED(VERSION 2.8), and CMake 4 removed "
        "compatibility with anything below 3.5"
    )


def test_forcefill_install_sources_are_pinned():
    editable_repos = (BUILD_TOOLS / "editable_repos.sh").read_text()
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert f'FORCEFILL_REF="${{FORCEFILL_REF:-{FORCEFILL_COMMIT}}}"' in editable_repos
    assert f"forcefill.git@{FORCEFILL_COMMIT}" in workflow

    unpinned_url = re.compile(r"git\+https://github\.com/LouieSlocombe/forcefill\.git(?=[\"'\s])")
    checked_files = [
        *BUILD_TOOLS.glob("*.sh"),
        REPOSITORY_ROOT / "README.md",
        BUILD_TOOLS / "README.md",
        REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml",
        *(REPOSITORY_ROOT / "docs" / "source").rglob("*.md"),
    ]
    offenders = [
        str(path.relative_to(REPOSITORY_ROOT)) for path in checked_files if unpinned_url.search(path.read_text())
    ]
    assert not offenders, f"forcefill installs must name an immutable ref: {offenders}"


def test_linker_flags_are_valid_for_darwin_and_linux():
    darwin_flags = _call_sourced(
        "build_plumed.sh", "build_linker_flags", "Darwin", "/opt/environment/lib"
    ).stdout.strip()
    linux_flags = _call_sourced("build_plumed.sh", "build_linker_flags", "Linux", "/opt/environment/lib").stdout.strip()

    assert darwin_flags == "-Wl,-rpath,/opt/environment/lib"
    assert "rpath-link" not in darwin_flags
    assert linux_flags == "-Wl,-rpath-link,/opt/environment/lib -Wl,-rpath,/opt/environment/lib"


def test_plumed_kernel_discovery_prefers_the_platform_extension(tmp_path):
    shared_object = tmp_path / "libplumedKernel.so"
    dynamic_library = tmp_path / "libplumedKernel.dylib"
    shared_object.touch()
    dynamic_library.touch()

    darwin_kernel = _call_sourced("build_plumed.sh", "find_plumed_kernel", tmp_path, "Darwin").stdout.strip()
    linux_kernel = _call_sourced("build_plumed.sh", "find_plumed_kernel", tmp_path, "Linux").stdout.strip()

    assert darwin_kernel == str(dynamic_library)
    assert linux_kernel == str(shared_object)


def test_build_scripts_do_not_require_gnu_nproc():
    offenders = [script.name for script in BUILD_TOOLS.glob("*.sh") if "$(nproc)" in script.read_text()]

    assert not offenders, f"GNU nproc is not available on every supported platform: {offenders}"


def test_build_scripts_have_valid_bash_syntax():
    for script in BUILD_TOOLS.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True, text=True)


def test_sdist_prunes_generated_build_source_checkouts():
    manifest_lines = (REPOSITORY_ROOT / "MANIFEST.in").read_text().splitlines()
    include_index = manifest_lines.index("recursive-include build_tools *.md *.sh *.yml")
    prune_index = manifest_lines.index("prune build_tools/sources")

    assert prune_index > include_index


def test_pinned_clone_is_published_only_after_the_checkout_succeeds(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "README.md").write_text("fixture\n")
    subprocess.run(["git", "-C", str(source), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    destination = tmp_path / "dependency"

    failed = _call_sourced("editable_repos.sh", "clone_repo", source, destination, "0" * 40, check=False)

    assert failed.returncode != 0
    assert not destination.exists(), "A failed pinned checkout must never become an accepted final clone"

    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _call_sourced("editable_repos.sh", "clone_repo", source, destination, commit)

    assert (destination / ".git").is_dir()
    assert not list(tmp_path.glob(".dependency.clone.*/checkout"))


def test_conda_and_project_dependency_bounds_agree():
    from packaging.requirements import Requirement

    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())["project"]
    pip_requirements = {
        Requirement(s).name: Requirement(s).specifier
        for s in project["dependencies"] + project["optional-dependencies"]["test"]
    }
    environment = (BUILD_TOOLS / "environment.yml").read_text()
    for name in ("openmm", "pytest"):
        line = re.search(rf"^  - ({name} .+)$", environment, re.MULTILINE).group(1)
        assert Requirement(line).specifier == pip_requirements[name]


def test_verify_install_stops_at_the_first_failed_component(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name, body in {"plumed": "exit 0", "fake-python": "echo invoked; exit 7"}.items():
        path = fake_bin / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; PATH="$2:$PATH"; verify_install "$3" fake-python',
            "bash",
            str(BUILD_TOOLS / "editable_repos.sh"),
            str(fake_bin),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 7
    assert result.stdout.count("invoked") == 1
    assert "openmmqmmm: OK" not in result.stdout
