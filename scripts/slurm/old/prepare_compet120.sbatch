#!/usr/bin/env bash
#SBATCH --job-name=prep_c120
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err
set -euo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img
neuralbench eeg image -m reve --dataset compet120 --prepare
