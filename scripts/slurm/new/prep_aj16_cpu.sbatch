#!/usr/bin/env bash
#SBATCH --job-name=prep_aj16c
#SBATCH --partition=cpu_student
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --chdir=/path/to/home/eeg2img_new
#SBATCH --output=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.err
set -euo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img_new
export NEURALBENCH_CONFIG=/path/to/home/eeg2img_new/config.json
echo "CONFIG=$NEURALBENCH_CONFIG"
export ALLJOINED_ACCEPT_LICENCE=1
neuralbench eeg image --dataset xu2025alljoined_prep --prepare
echo "Prepare finished."
du -sh /path/to/home/eeg2img_new/cache/*
