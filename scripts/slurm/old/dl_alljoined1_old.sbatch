#!/usr/bin/env bash
#SBATCH --job-name=dl_aj1_old
#SBATCH --partition=cpu_student
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err
set -uo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img
export NEURALBENCH_CONFIG=/path/to/home/.neuralbench/config.json
export ALLJOINED_ACCEPT_LICENCE=1
echo "CONFIG=$NEURALBENCH_CONFIG"
neuralbench eeg image --dataset xu2024alljoined --download
echo "exit: $?"
du -sh /path/to/home/eeg2img/data/*
