#!/bin/bash
# Builds the openmmqmmm environment on the Sol cluster, using conda-forge packages for
# everything except PLUMED, which has to be compiled with the opes module, and geomeTRIC
# and Open Babel, which come from PyPI.
#
#   sbatch sub_sol_install.sh          # batch
#   ./custom_install_sol.sh            # from an interactive session, e.g.
#                                      # interactive -t 60 -p htc -c 12 --mem=128G
#
# The environment is recreated from scratch on every run.
#
# openmmqmmm and forcefill are cloned into $SRC_DIR and installed editable. Existing
# checkouts are used as they are, never wiped; new forcefill clones use the reviewed
# commit from editable_repos.sh unless FORCEFILL_REF selects a development branch.

set -eo pipefail

# === Configuration ===
ENV_NAME="openmmqmmm"

# Sources are built under $SCRATCH; refuse to run rather than risk rm -rf'ing / below.
WORK_DIR="${SCRATCH:?SCRATCH is not set - run this on a Sol node, or set it manually}/${ENV_NAME}_sources"
# Editable checkouts live outside the build area: WORK_DIR is wiped on every run.
SRC_DIR="${SRC_DIR:-${HOME}/${ENV_NAME}_src}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"
# Pulls in clone_repo(), install_editable_repos() and check_editable_repos().
source "${SCRIPT_DIR}/editable_repos.sh"

# === Environment Setup ===
module purge
module load mamba/latest

echo "=== Cleaning previous installations ==="
rm -rf "${WORK_DIR}"
mamba env remove -n "${ENV_NAME}" -y 2>/dev/null || true

echo "=== Initializing Conda Environment ==="
mamba env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
source activate "${ENV_NAME}"
# Sol's compiler/module stack is tested with Python 3.12.
mamba install -n "${ENV_NAME}" -c conda-forge -y python=3.12

echo "=== Preparing Build Directory ==="
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

echo "=== Installing openmmqmmm (editable) ==="
clone_repo "https://github.com/LouieSlocombe/openmmqmmm.git" "${SRC_DIR}/${ENV_NAME}"
pip3 install -e "${SRC_DIR}/${ENV_NAME}" --no-deps

install_editable_repos "${SRC_DIR}"

echo "=== Verifying Installation ==="
verify_install "${SRC_DIR}" python3

conda deactivate
echo "=== Build Complete! ==="
echo "Checkouts: ${SRC_DIR}"
echo
echo "ORCA is licensed separately and is not installed by this script. Put it on the"
echo "cluster by hand, then export OPENMMQMMM_ORCADIR (and load the matching OpenMPI"
echo "module for ORCATheory(numcores > 1)) in the job script that uses it."
