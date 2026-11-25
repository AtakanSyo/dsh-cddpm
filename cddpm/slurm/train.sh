#!/bin/bash
#SBATCH --job-name=dsh378_train
#SBATCH --account=mdbf
#SBATCH --partition=cuda
#SBATCH --qos=cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=atakans@sabanciuniv.edu
#
# Run from the run folder:  sbatch slurm/train.sh --variant C   (A, B or C)
set -eo pipefail  # no -u: conda activate reads unset variables
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${DSH_CONDA_ENV:-pyt_env}"
export PYTHONUNBUFFERED=1
python train.py "$@"
