#!/bin/bash
# Shared handling of the dependencies that come from git rather than conda-forge,
# sourced by conda_install.sh, custom_install.sh and custom_install_sol.sh. forcefill
# gets edited alongside openmmqmmm, so it is cloned once and installed in editable mode
# instead of being pulled fresh from GitHub on every install -- a `git pull` in the
# checkout is then all it takes to update it.
#
# openmmnqe ships the same file, with the longer list of repositories its workflows need.

# name=url pairs, in install order. The name is both the directory the repo is cloned
# into and the module the install is checked against.
EDITABLE_REPOS=(
    "forcefill=https://github.com/LouieSlocombe/forcefill.git"
)

# clone_repo <url> <path>
# Clones <url> into <path> unless a checkout is already there, which is left
# exactly as it is -- these hold work in progress, so nothing here pulls,
# resets or removes them.
clone_repo() {
    local url="$1"
    local path="$2"

    if [ -d "${path}/.git" ]; then
        echo "=== Using existing checkout: ${path} ==="
    elif [ -e "${path}" ]; then
        echo "${path} exists but is not a git checkout; move it aside and re-run." >&2
        return 1
    else
        echo "=== Cloning $(basename "${path}") into ${path} ==="
        git clone "${url}" "${path}"
    fi
}

# install_editable_repos <src_dir>
# Clones each git dependency into <src_dir> and installs it editable. --no-deps for the
# reason the installers pass it for openmmqmmm itself: the environment file is the
# authority on the dependency set -- forcefill's stack (openff-toolkit,
# openmmforcefields, rdkit) comes from conda-forge, and letting pip re-resolve would
# risk pulling PyPI wheels over the conda OpenMM stack.
install_editable_repos() {
    local src_dir="$1"
    local entry name url

    mkdir -p "${src_dir}"
    for entry in "${EDITABLE_REPOS[@]}"; do
        name="${entry%%=*}"
        url="${entry#*=}"
        clone_repo "${url}" "${src_dir}/${name}"
        echo "=== Installing ${name} (editable) ==="
        pip install -e "${src_dir}/${name}" --no-deps
    done
}

# check_editable_repos <src_dir>
# Fails if any of them import from site-packages rather than the checkout.
check_editable_repos() {
    local src_dir="$1"

    python -c "
import importlib, pathlib, sys

src = pathlib.Path('${src_dir}').resolve()
for name in '${EDITABLE_REPOS[*]%%=*}'.split():
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    if src not in path.parents:
        sys.exit(f'{name} is not editable: imported from {path.parent}')
    print(f'{name}: {path.parent}')
"
}
