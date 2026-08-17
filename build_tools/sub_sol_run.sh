#!/bin/bash
#SBATCH --job-name=run
#SBATCH -N 1
#SBATCH -p htc
#SBATCH -c 128
#SBATCH -q public
#SBATCH --time=0-04:00:00
#SBATCH --mem=0
#SBATCH -o run.out
#SBATCH -e run.out
#SBATCH --export=NONE

# Runs a single calculation script in the openmmqmmm environment:
#   sbatch sub_sol_run.sh [script.py]
#
# A SLURM job starts in the directory it was submitted from, so the argument is relative
# to that; the default assumes you submit from the repository root. Anything using
# ORCATheory needs OPENMMQMMM_ORCADIR set below, and the OpenMPI module that ORCA
# installation was built against for ORCATheory(numcores > 1).

set -eo pipefail

ENV_NAME="openmmqmmm"
PY_SCRIPT="${1:-examples/gasphase_hf.py}"

module load mamba/latest
source activate "${ENV_NAME}"

# export OPENMMQMMM_ORCADIR=~/orca_6_1_1

python3 "${PY_SCRIPT}" >> py.out 2>&1
