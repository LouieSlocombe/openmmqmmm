#!/bin/bash
# Builds OpenMM from source into a dedicated conda environment, then PLUMED, the
# OpenMM-PLUMED plugin and py-plumed on top of it.
#
#   bash custom_install.sh
#
# Use this route when you need an OpenMM that has not been released yet — QM/MM RPMD
# goes through openmm.PythonForce (see openmmqmmm/openmm/rpmd_force.py), so a fix
# landing on OpenMM master is the case this exists for. The conda route is otherwise
# the one to use.
#
# The environment is recreated from scratch on every run, and sources are cloned
# into ../../openmmqmmm_sources (a sibling of the repo). A full build takes a while.

# Exit immediately on error and fail pipelines cleanly, so a broken build does not
# fall through to the later steps and report success.
set -eo pipefail

ENV_NAME="openmmqmmm_custom"
OPENMM_VERSION="master"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${SCRIPT_DIR}/../../openmmqmmm_sources"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"

echo "=== Initializing Conda Environment ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda refuses to remove the active environment, so drop back to base first.
conda activate base
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda env create -f "${SCRIPT_DIR}/environment_custom.yml"
conda activate "${ENV_NAME}"

echo "=== Preparing Build Directory ==="
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

# Overwrites the conda-forge openmm that environment_custom.yml pulled in as a
# dependency of pdbfixer and openff-toolkit. CMake finds a CUDA toolkit itself if one
# is installed; there is no CUDA pin in the environment to point it at.
echo "=== Compiling OpenMM ${OPENMM_VERSION} ==="
git clone --branch "${OPENMM_VERSION}" --depth 1 --filter=blob:none https://github.com/openmm/openmm.git
cd openmm
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="${CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(which python)"
make -j"$(nproc)"
make install
make PythonInstall

cd "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

# --no-deps matters more here than in the conda route: pyproject.toml pins
# openmm == 8.5.2, which a master build does not report, and OpenMM is not on PyPI for
# pip to fall back to. environment_custom.yml is the authority on the dependency set.
echo "=== Installing openmmqmmm (editable) ==="
pip install -e "${REPO_DIR}" --no-deps

# Not on PyPI or conda-forge, so it cannot go in environment_custom.yml; without it
# openmm_modeller(parameterize_nonstandard=True) raises MissingDependencyError.
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
python -c "import openmm; print(openmm.__version__)"
python -c "import openmmqmmm"
echo "openmmqmmm: OK"

echo "=== Build Complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo
echo "ORCA is not installed by this script. Install it separately (free for academic"
echo "use) and point openmmqmmm at it, for example:"
echo "  export OPENMMQMMM_ORCADIR=~/orca_6_1_1"
