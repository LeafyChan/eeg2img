#!/usr/bin/env bash
#SBATCH --job-name=xu2025_prepare
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err

set -euo pipefail
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img
export ALLJOINED_ACCEPT_LICENCE=1

echo "=== START ===" && date && hostname
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"

neuralbench eeg image --dataset xu2025alljoined --prepare

echo "=== END ===" && date
du -sh cache/
