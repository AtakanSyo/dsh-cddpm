#!/bin/bash
#SBATCH --job-name=sample_cond
#SBATCH --account=mdbf
#SBATCH --qos=cuda
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --output=sample_model_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atakans@sabanciuniv.edu

START=$(date +%s)

echo "Starting sampling run using sampling_config.py..."

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 save_samples.py \
    > test_output_sampling.log 2>&1

RET=$?

if [ $RET -ne 0 ]; then
    echo "Sampling FAILED with return code $RET"
else
    echo "Sampling completed successfully."
fi

END=$(date +%s)
ELAPSED=$(( END - START ))

echo "Execution time: $ELAPSED seconds"

exit 0