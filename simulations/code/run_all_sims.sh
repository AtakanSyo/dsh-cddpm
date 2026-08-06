#!/bin/bash
#SBATCH -p hamsi
#SBATCH -A asaracyakupoglu
#SBATCH -J dsh_grid
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 56
#SBATCH -C weka
#SBATCH --time=24:00:00
#SBATCH --output=simulation_logs/grid_%A_%a.out
#SBATCH --error=simulation_logs/grid_%A_%a.err

# ===========================================================================
# Packed RefleX runner.
#
# Each job gets 56 cores and runs (56 / THREADS_PER_SIM) simulations IN
# PARALLEL, each pinned to THREADS_PER_SIM threads.
#
#   THREADS_PER_SIM=2  -> 28 sims per job
#   THREADS_PER_SIM=4  -> 14 sims per job
#   THREADS_PER_SIM=8  ->  7 sims per job
#
# Submit with:
#   sbatch --export=ALL,THREADS_PER_SIM=2 --array=0-N%9 run_grid.sbatch
#
# where N+1 = number of array tasks = ceil(total_dirs / sims_per_job).
# The %9 caps simultaneous jobs at 9 (your HPC limit).
# ===========================================================================

set -uo pipefail

# --- inputs / defaults ---
THREADS_PER_SIM="${THREADS_PER_SIM:-2}"      # default 2 if not exported
CORES_PER_JOB=56
SIMS_PER_JOB=$(( CORES_PER_JOB / THREADS_PER_SIM ))
LIST="${LIST:-remaining_dirs.txt}"

mkdir -p simulation_logs

if [[ ! -f "$LIST" ]]; then
  echo "ERROR: $LIST not found. Build it first (one sim directory per line)."
  exit 1
fi

if (( SIMS_PER_JOB < 1 )); then
  echo "ERROR: THREADS_PER_SIM=$THREADS_PER_SIM exceeds CORES_PER_JOB=$CORES_PER_JOB"
  exit 1
fi

TASK=${SLURM_ARRAY_TASK_ID:-0}

# This task handles the slice of lines:
#   [ TASK*SIMS_PER_JOB + 1 , TASK*SIMS_PER_JOB + SIMS_PER_JOB ]   (1-based)
START_LINE=$(( TASK * SIMS_PER_JOB + 1 ))
END_LINE=$(( START_LINE + SIMS_PER_JOB - 1 ))

echo "=========================================================="
echo "Job $SLURM_JOB_ID  array task $TASK"
echo "THREADS_PER_SIM = $THREADS_PER_SIM   SIMS_PER_JOB = $SIMS_PER_JOB"
echo "Handling list lines $START_LINE..$END_LINE of $LIST"
echo "=========================================================="

# --- launch one RefleX per assigned directory, in parallel ---
declare -a PIDS=()
declare -a PDIR=()

run_one () {
  local dir="$1"
  local nthreads="$2"
  if [[ -z "$dir" ]]; then return 0; fi
  if [[ ! -d "$dir" ]]; then
    echo "[skip] not a directory: $dir"
    return 0
  fi
  # idempotent restart: skip if already completed
  if [[ -f "$dir/.complete" ]]; then
    echo "[skip] already complete: $dir"
    return 0
  fi
  (
    cd "$dir" || exit 11
    export HDF5_USE_FILE_LOCKING=FALSE
    export OMP_NUM_THREADS="$nthreads"        # belt-and-suspenders thread cap
    # force the THREAD directive in params.par to match the packing choice
    sed -i "s/^THREAD .*/THREAD $nthreads/" params.par
    s=$(date +%s)
    ./reflex_3.0_Linux_x86_64_2.17 params.par > output.log 2>&1
    rc=$?
    e=$(date +%s)
    echo "runtime_s=$((e-s)) return=$rc" > runtime.log
    if [[ $rc -eq 0 ]]; then
      touch .complete
    fi
    exit $rc
  ) &
  PIDS+=("$!")
  PDIR+=("$dir")
}

idx=0
for (( ln=START_LINE; ln<=END_LINE; ln++ )); do
  DIR=$(sed -n "${ln}p" "$LIST" | tr -d '\r')
  [[ -z "${DIR:-}" ]] && continue          # past end of list
  echo "[launch] ($((idx+1))/$SIMS_PER_JOB) thread=$THREADS_PER_SIM  $DIR"
  run_one "$DIR" "$THREADS_PER_SIM"
  idx=$((idx+1))
done

if (( idx == 0 )); then
  echo "No directories assigned to this task (list shorter than array). Exiting 0."
  exit 0
fi

# --- wait for all, collect exit codes ---
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[ok]   ${PDIR[$i]}"
  else
    rc=$?
    echo "[FAIL rc=$rc] ${PDIR[$i]}"
    FAIL=$((FAIL+1))
  fi
done

echo "Task $TASK done. $((idx-FAIL))/$idx succeeded, $FAIL failed."
exit $(( FAIL > 0 ? 1 : 0 ))
