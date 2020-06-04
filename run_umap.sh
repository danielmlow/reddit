#!/usr/bin/env bash
#
#SBATCH --job-name=reddit_umap
#SBATCH --array=1-50
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=5GB
#SBATCH -o ./../../datum/reddit/output/slurm/slurm-%A_%a.out

echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}

python3 reddit_cluster.py --job_array_task_id=${SLURM_ARRAY_TASK_ID} --toy=False --pre_or_post='pre'

echo 'Finished.'

# python3 -i reddit_cluster.py --job_array_task_id=1 --toy=True --pre_or_post='pre'
# tesla-k20:2
# GEFORCEGTX1080TI:4
# --gres=gpu:1

# https://hpc-uit.readthedocs.io/en/latest/jobs/examples.html
