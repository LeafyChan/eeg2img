#!/usr/bin/env bash
#SBATCH --job-name=reve_aj16P
#SBATCH --partition=cpu_student
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/path/to/home/eeg2img_new
#SBATCH --output=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.err
set -uo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img_new
export NEURALBENCH_CONFIG=/path/to/home/eeg2img_new/config.json
export ALLJOINED_ACCEPT_LICENCE=1
echo "CONFIG=$NEURALBENCH_CONFIG"
neuralbench eeg image -m reve --dataset xu2025alljoined_reve_ho_eval --max-neuro-hours-per-epoch 100 -c /path/to/home/eeg2img_new/checkpoints/auto_353a43de_1015_1917.ckpt
echo "exit: $?"
