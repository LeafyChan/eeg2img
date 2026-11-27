#!/usr/bin/env bash
#SBATCH --job-name=download_eeg2
#SBATCH --partition=cpu_student
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err

set -euo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

export NEURALFETCH_THINGS_PASSWORD=<REDACTED>
neuralbench eeg image --download

echo "Download finished."
