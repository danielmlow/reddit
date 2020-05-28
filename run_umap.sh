#!/usr/bin/env bash
#
#SBATCH --job-name=reddit_umap
#SBATCH --array=1-50
#SBATCH --time=03:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=10GB
#SBATCH -o ./../../datum/reddit/output/slurm/slurm-%A_%a.out

echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}

python3 reddit_cluster.py --job_array_task_id=${SLURM_ARRAY_TASK_ID} --toy=False

echo 'Finished.'

# python3 -i run.py --job_array_task_id=1 --run_version_number=6 --toy=True --dim_reduction=False --run_modelN=0
# tesla-k20:2
# GEFORCEGTX1080TI:4
# --gres=gpu:1

# https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html
