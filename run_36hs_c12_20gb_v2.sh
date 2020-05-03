#!/usr/bin/env bash
#
#SBATCH --job-name=gs_36hs
#SBATCH --array=1-5
#SBATCH --time=36:00:00
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=20GB
#SBATCH -o ./../../datum/reddit/output/slurm/slurm-%A_%a.out

echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}

python3 run.py --job_array_task_id=${SLURM_ARRAY_TASK_ID} --run_version_number=2 --toy=False

echo 'Finished.'

# tesla-k20:2
# GEFORCEGTX1080TI:4
# --gres=gpu:1

# https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html
