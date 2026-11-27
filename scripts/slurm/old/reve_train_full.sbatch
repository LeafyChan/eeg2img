#!/usr/bin/env bash
#SBATCH --job-name=reve_train
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%1
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%A_%a.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%A_%a.err

set -euo pipefail
mkdir -p scripts/slurm/outputs

source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

echo "================ JOB INFO ================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Array Task  : ${SLURM_ARRAY_TASK_ID}"
echo "Node List   : ${SLURM_NODELIST}"
echo "Working Dir : $(pwd)"
echo "=========================================="

nvidia-smi

neuralbench eeg image --dataset xu2025alljoined -m reve --force

echo "REVE full training finished successfully."
