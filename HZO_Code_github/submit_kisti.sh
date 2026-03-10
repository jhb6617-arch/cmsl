#!/bin/bash
#SBATCH -p eme_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@120
#SBATCH --job-name=hzo_gen
#SBATCH --output=slurm-%A_%a.out
#SBATCH --comment=inhouse
#SBATCH --array=0-99          # 100 jobs × 500 sims = 50,000 total

echo "CWD            = $(pwd)"
echo "SLURM_JOB_ID   = $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"

module load cuda/12.4.1

# Each job handles 500 simulations; offset shifts sim_index globally
SIMS_PER_JOB=500
OFFSET=$(( SLURM_ARRAY_TASK_ID * SIMS_PER_JOB ))

echo "offset_sims = $OFFSET  (sims $((OFFSET+1)) to $((OFFSET+SIMS_PER_JOB)))"

srun ./Pfmhzo.exe \
    $OFFSET \
    inputs/landau_pool.txt \
    inputs/phase_pool.txt \
    $SIMS_PER_JOB
