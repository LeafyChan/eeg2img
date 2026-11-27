#!/usr/bin/env bash
#SBATCH --job-name=reve_lora
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err

set -euo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

neuralbench eeg image --dataset xu2025alljoined -m reve -w lora_r4_flatten --force

echo "REVE LoRA training finished."
