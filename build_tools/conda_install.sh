#!/bin/bash
# One-command install of the openmmqmmm conda environment: creates the environment
# from environment.yml, compiles PLUMED (with the opes module), the OpenMM-PLUMED
# plugin and the PLUMED Python bindings (py-plumed) into it, then installs
# openmmqmmm and forcefill and verifies the result.
#
#   bash conda_install.sh
#
# WARNING: the target environment (default: openmmqmmm) is REMOVED and recreated
# from scratch on every run, as are the sources cloned into build_tools/sources/.
# Set ENV_NAME to install into a differently named environment instead:
#
#   ENV_NAME=openmmqmmm2 bash conda_install.sh
#
# ORCA is licensed separately and is not installed here; see the end of this script.

# Exit immediately on error and fail pipelines cleanly, so a broken build does not
# fall through to the later steps and report success.
set -eo pipefail

ENV_NAME="${ENV_NAME:-openmmqmmm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${SCRIPT_DIR}/sources"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"

echo "=== Initializing Conda Environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda refuses to remove the active environment, so drop back to base first
# (covers running this script from inside an activated ${ENV_NAME}).
conda activate base
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
# -n overrides the name pinned inside environment.yml, so ENV_NAME works.
conda env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
conda activate "${ENV_NAME}"

echo "=== Preparing Build Directory ==="
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

# --no-deps: environment.yml is the authority on the dependency set and was created
# moments ago, so letting pip re-resolve would only risk pulling PyPI wheels over the
# conda OpenMM stack.
echo "=== Installing openmmqmmm (editable) ==="
pip install -e "${REPO_DIR}" --no-deps

# Not on PyPI or conda-forge, so it cannot go in environment.yml; without it
# openmm_modeller(parameterize_nonstandard=True) raises MissingDependencyError. Its
# dependency stack (openff-toolkit, openmmforcefields, rdkit) came from environment.yml.
echo "=== Installing forcefill ==="
pip install --no-deps "forcefill @ git+https://github.com/LouieSlocombe/forcefill.git"

echo "=== Verifying Installation ==="
cd "${REPO_DIR}"
plumed --no-mpi config -q module opes
echo "PLUMED opes module: OK"
python -c "import plumed; plumed.Plumed()"
echo "py-plumed kernel load: OK"
python -c "from openmmplumed import PlumedForce"
echo "openmm-plumed: OK"
python -c "import forcefill"
echo "forcefill: OK"
python -c "import openmmqmmm"
echo "openmmqmmm: OK"

echo "=== Build Complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo
echo "ORCA is not installed by this script. Install it separately (free for academic"
echo "use) and point openmmqmmm at it, for example:"
echo "  export OPENMMQMMM_ORCADIR=~/orca_6_1_1"
echo "ORCATheory and QM/MM need it; the pure-MM functionality does not."
