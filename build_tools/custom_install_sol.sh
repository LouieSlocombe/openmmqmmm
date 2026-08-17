#!/bin/bash
# Builds the openmmqmmm environment on the Sol cluster, using conda-forge packages for
# everything except PLUMED, which has to be compiled with the opes module.
#
#   sbatch sub_sol_install.sh          # batch
#   ./custom_install_sol.sh            # from an interactive session, e.g.
#                                      # interactive -t 60 -p htc -c 12 --mem=128G
#
# The environment is recreated from scratch on every run.
#
# openmmqmmm and forcefill are cloned into $SRC_DIR and installed editable, so a
# `git pull` there is all it takes to update them. Existing checkouts are used as they
# are, never wiped.

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
mamba create -n "${ENV_NAME}" -c conda-forge python=3.12 -y
source activate "${ENV_NAME}"

# The same set as environment.yml, which is where the reasons for each package are
# recorded. PLUMED is absent for the same reason it is absent there: build_plumed
# compiles it into this prefix below.
echo "=== Installing Dependencies ==="
mamba install -c conda-forge -y \
    ase \
    openmm=8.5.2 \
    cmake \
    make \
    swig \
    cxx-compiler \
    doxygen \
    cython \
    pdbfixer \
    mdtraj \
    numpy \
    scipy \
    packaging \
    parmed \
    openmmforcefields \
    openff-toolkit \
    rdkit \
    multiprocess \
    rmsd \
    pytest \
    pytest-cov
pip3 install "geometric>=1.0.1" openbabel

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
plumed --no-mpi config -q module opes
echo "PLUMED opes module: OK"
python3 -c "import plumed; plumed.Plumed()"
echo "py-plumed kernel load: OK"
python3 -c "from openmmplumed import PlumedForce"
echo "openmm-plumed: OK"
check_editable_repos "${SRC_DIR}"
echo "editable dependencies: OK"
python3 -c "import openmmqmmm"
echo "openmmqmmm: OK"

conda deactivate
echo "=== Build Complete! ==="
echo "Checkouts: ${SRC_DIR}"
echo
echo "ORCA is licensed separately and is not installed by this script. Put it on the"
echo "cluster by hand, then export OPENMMQMMM_ORCADIR (and load the matching OpenMPI"
echo "module for ORCATheory(numcores > 1)) in the job script that uses it."
