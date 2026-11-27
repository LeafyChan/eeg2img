#!/usr/bin/env bash
#SBATCH --job-name=prep_aj16j
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --chdir=/path/to/home/eeg2img_new
#SBATCH --output=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.err
set -uo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img_new
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NEURALBENCH_CONFIG=/path/to/home/eeg2img_new/config_prep.json
export ALLJOINED_ACCEPT_LICENCE=1
neuralbench eeg image --dataset xu2025alljoined_bs8 --prepare
echo "exit: $?"
du -sh /path/to/home/eeg2img_new/cache/*
