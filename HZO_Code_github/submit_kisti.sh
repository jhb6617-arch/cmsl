#!/bin/bash
#SBATCH -p eme_h200nv_8
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@120
#SBATCH --job-name=hzo_gen
#SBATCH --output=slurm-%j.out
#SBATCH --comment=inhouse

# -------------------------------------------------------
# 제출 전 이 값만 수정하세요 (0, 500, 1000, ..., 49500)
OFFSET=0
# -------------------------------------------------------

echo "CWD          = $(pwd)"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "offset_sims  = $OFFSET  (sims $((OFFSET+1)) to $((OFFSET+500)))"

module load cuda/12.4.1

srun ./Pfmhzo.exe \
    $OFFSET \
    inputs/landau_pool.txt \
    inputs/phase_pool.txt \
    500
