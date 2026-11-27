#!/usr/bin/env bash
#SBATCH --job-name=reve_smoke
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --chdir=/path/to/home/eeg2img
#SBATCH --output=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img/scripts/slurm/outputs/%x-%j.err

set -euo pipefail
mkdir -p scripts/slurm/outputs

source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

echo "================ JOB INFO ================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node List   : ${SLURM_NODELIST}"
echo "Working Dir : $(pwd)"
echo "=========================================="

nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

neuralbench eeg image --dataset xu2025alljoined -m reve --debug

echo "REVE smoke test finished successfully."
