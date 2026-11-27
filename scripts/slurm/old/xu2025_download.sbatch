#!/usr/bin/env bash
#SBATCH --job-name=xu2025_dl
#SBATCH --partition=cpu_student
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err

set -euo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img
export ALLJOINED_ACCEPT_LICENCE=1

echo "=== START ===" && date && hostname
neuralbench eeg image --dataset xu2025alljoined --download
echo "=== END ===" && date
du -sh data/Xu2025Alljoined
export HF_TOKEN=<REDACTED>
