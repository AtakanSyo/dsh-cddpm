#!/bin/bash
#SBATCH --job-name=dsh378_smoke
#SBATCH --account=mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --qos=mid_mdbf
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/smoke_test_%j.log
#
# CPU-only check; run from the run folder:  sbatch slurm/smoke_test.sh
set -eo pipefail  # no -u: conda activate reads unset variables
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${DSH_CONDA_ENV:-pyt_env}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
python smoke_test.py
