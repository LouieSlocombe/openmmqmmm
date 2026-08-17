#!/bin/bash
# Builds the openmmqmmm environment on the Sol cluster, using conda-forge packages for
# everything except PLUMED, which has to be compiled with the opes module.
#
#   sbatch sub_sol_install.sh          # batch
#   ./custom_install_sol.sh            # from an interactive session, e.g.
#                                      # interactive -t 60 -p htc -c 12 --mem=128G
#
# The environment is recreated from scratch on every run.

set -eo pipefail

# === Configuration ===
ENV_NAME="openmmqmmm"

# Sources are built under $SCRATCH; refuse to run rather than risk rm -rf'ing / below.
WORK_DIR="${SCRATCH:?SCRATCH is not set - run this on a Sol node, or set it manually}/${ENV_NAME}_sources"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pulls in build_plumed() and build_py_plumed(), with the PLUMED versions they pin.
source "${SCRIPT_DIR}/build_plumed.sh"

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
pip3 install --no-deps "forcefill @ git+https://github.com/LouieSlocombe/forcefill.git"

echo "=== Preparing Build Directory ==="
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

build_plumed "${WORK_DIR}"
build_py_plumed "${WORK_DIR}"

echo "=== Installing openmmqmmm ==="
pip3 install --no-deps git+https://github.com/LouieSlocombe/openmmqmmm.git

echo "=== Verifying Installation ==="
plumed --no-mpi config -q module opes
echo "PLUMED opes module: OK"
python3 -c "import plumed; plumed.Plumed()"
echo "py-plumed kernel load: OK"
python3 -c "from openmmplumed import PlumedForce"
echo "openmm-plumed: OK"
python3 -c "import openmmqmmm"
echo "openmmqmmm: OK"

conda deactivate
echo "=== Build Complete! ==="
echo
echo "ORCA is licensed separately and is not installed by this script. Put it on the"
echo "cluster by hand, then export OPENMMQMMM_ORCADIR (and load the matching OpenMPI"
echo "module for ORCATheory(numcores > 1)) in the job script that uses it."
