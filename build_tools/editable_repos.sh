#!/bin/bash
# Shared handling of the dependencies that come from git rather than conda-forge,
# sourced by conda_install.sh, custom_install.sh and custom_install_sol.sh. forcefill
# gets edited alongside openmmqmmm, so it is cloned once and installed in editable mode.
# New clones use a reviewed commit by default; set FORCEFILL_REF=main to opt into the
# latest development branch. Existing checkouts are always left exactly as they are.
#
# openmmnqe ships the same file, with the longer list of repositories its workflows need.

# name=url pairs, in install order. The name is both the directory the repo is cloned
# into and the module the install is checked against.
EDITABLE_REPOS=(
    "forcefill=https://github.com/LouieSlocombe/forcefill.git"
)
FORCEFILL_REF="${FORCEFILL_REF:-bff20e2498474b3b3aa965b090fa722f762a49ae}"

# clone_repo <url> <path> [ref]
# Clones <url> into <path> unless a checkout is already there, which is left
# exactly as it is -- these hold work in progress, so nothing here pulls,
# resets or removes them.
clone_repo() {
    local url="$1"
    local path="$2"
    local ref="${3:-}"
    local parent_dir
    local checkout_name
    local staging_dir
    local staging_checkout

    if [ -e "${path}" ]; then
        if [ -e "${path}/.git" ] \
            && [ "$(git -C "${path}" rev-parse --is-inside-work-tree 2>/dev/null)" = "true" ]; then
            echo "=== Using existing checkout: ${path} ==="
            return 0
        fi
        echo "${path} exists but is not a git checkout; move it aside and re-run." >&2
        return 1
    else
        echo "=== Cloning $(basename "${path}") into ${path} ==="
        # Prepare the checkout under a unique sibling and publish it with one
        # atomic rename only after the requested ref is checked out. An interrupted
        # clone can therefore never masquerade as an accepted existing checkout on
        # the next installer run.
        parent_dir="$(dirname "${path}")"
        checkout_name="$(basename "${path}")"
        mkdir -p "${parent_dir}" || return 1
        staging_dir="$(mktemp -d "${parent_dir}/.${checkout_name}.clone.XXXXXX")" || return 1
        staging_checkout="${staging_dir}/checkout"
        if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
            if ! git clone "${url}" "${staging_checkout}"; then
                rm -rf "${staging_dir:?}"
                return 1
            fi
            if ! git -C "${staging_checkout}" checkout --detach "${ref}"; then
                rm -rf "${staging_dir:?}"
                return 1
            fi
        elif [ -n "${ref}" ]; then
            if ! git clone --branch "${ref}" --single-branch "${url}" "${staging_checkout}"; then
                rm -rf "${staging_dir:?}"
                return 1
            fi
        else
            if ! git clone "${url}" "${staging_checkout}"; then
                rm -rf "${staging_dir:?}"
                return 1
            fi
        fi
        if ! python -c 'import os, sys; os.rename(sys.argv[1], sys.argv[2])' "${staging_checkout}" "${path}"; then
            rm -rf "${staging_dir:?}"
            return 1
        fi
        rmdir "${staging_dir}"
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
    local entry name url ref

    mkdir -p "${src_dir}"
    for entry in "${EDITABLE_REPOS[@]}"; do
        name="${entry%%=*}"
        url="${entry#*=}"
        case "${name}" in
            forcefill) ref="${FORCEFILL_REF}" ;;
            *) ref="" ;;
        esac
        clone_repo "${url}" "${src_dir}/${name}" "${ref}"
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
