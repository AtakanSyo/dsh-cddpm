#!/bin/bash
#
# SLURM job submission script for running diffusion_model.py with multi-GPU distributed training
#

#SBATCH --job-name=Anscombe
#SBATCH --account=mdbf
#SBATCH --qos=cuda
#SBATCH --partition=cuda         # Use appropriate GPU partition
#SBATCH --gres=gpu:1               # Request 2 GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1                # Request 1 node
#SBATCH --ntasks=1               # Single task (torchrun will spawn multiple processes)
#SBATCH --time=1-00:00:00        # Set the time limit (3 days in this example)
#SBATCH --output=diffusion_model_%j.out  # Standard output file with job ID
#SBATCH --mail-type=ALL          # Notifications for job done or failed
#SBATCH --mail-user=atakans@sabanciuniv.edu  # Replace with your email

# Set stack size to unlimited (optional, based on your needs)
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

# Record the start time
START=$(date +%s)

# 1) Load .bashrc to initialize conda settings
source ~/.bashrc

# 2) Activate the Conda environment
echo "Activating Conda environment 'pyt_env'..."
conda activate pyt_env

# 3) Run the diffusion_model.py using torchrun for multi-GPU training
# --master_port=25001
echo "Running diffusion_model.py with torchrun on 1 GPUs..."
torchrun --nproc_per_node=1 diffusion_model.py &> output.log

# Record the end time
END=$(date +%s)

# Calculate elapsed time
ELAPSED=$(( END - START ))

# Print the elapsed time
echo "Execution time: $ELAPSED seconds"

# Log runtime to a file
echo "Job Name: Diffusion_6" > runtime.log
echo "Start Time: $(date -d @$START)" >> runtime.log
echo "End Time: $(date -d @$END)" >> runtime.log
echo "Elapsed Time: $ELAPSED seconds" >> runtime.log

# Capture return code and exit accordingly
RET=$?
echo
echo "Execution completed with return code: $RET"
exit $RET