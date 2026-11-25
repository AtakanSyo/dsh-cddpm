#!/bin/bash
#SBATCH --job-name=dsh378_sample_ensemble
#SBATCH --account=mdbf
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sample_ensemble_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=atakans@sabanciuniv.edu
#
# Run from the run folder:  sbatch slurm/sample_ensemble.sh --variant C --epoch 12000
set -eo pipefail  # no -u: conda activate reads unset variables
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${DSH_CONDA_ENV:-pyt_env}"
export PYTHONUNBUFFERED=1
python sample_ensemble.py "$@"
