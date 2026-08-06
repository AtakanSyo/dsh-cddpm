#!/bin/bash
#
# SLURM job submission script for running reflex_3.0_Linux_x86_64_2.17
#

#SBATCH -p hamsi
#SBATCH -A asaracyakupoglu
#SBATCH -J RUN_TO_CHECK
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH -C weka
#SBATCH --time=3-00:00:00
#SBATCH --output=reflex.out
#SBATCH --error=slurm-%j.err

# Load necessary modules if any (uncomment and modify if needed)
# module load <module-name>

# Set stack size to unlimited (optional based on your needs)
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

# Record the start time
START=$(date +%s)

# Run the executable with the parameter file
echo "Running reflex_3.0_Linux_x86_64_2.17 with params.par..."
./reflex_3.0_Linux_x86_64_2.17 params.par &> output.log

# Record the end time
END=$(date +%s)

# Calculate elapsed time
ELAPSED=$(( END - START ))

# Print the elapsed time
echo "Execution time: $ELAPSED seconds"

# Write runtime to runtime.log
echo "Execution time: $ELAPSED seconds" > runtime.log

# Capture return code and exit accordingly
RET=$?
echo
echo "Execution completed with return code: $RET"
echo "Execution completed with return code: $RET" >> runtime.log
exit $RET
