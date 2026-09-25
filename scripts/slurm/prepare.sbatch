#!/usr/bin/env bash
#SBATCH --job-name=eeg2img_prepare
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --chdir=/dgxa_home/se25uari118/eeg2img
#SBATCH --output=/dgxa_home/se25uari118/eeg2img/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/dgxa_home/se25uari118/eeg2img/scripts/slurm/outputs/%x-%j.err

set -euo pipefail
mkdir -p scripts/slurm/outputs

source /dgxa_home/se25uari118/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

echo "================ JOB INFO ================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Job Name    : ${SLURM_JOB_NAME}"
echo "Node List   : ${SLURM_NODELIST}"
echo "Working Dir : $(pwd)"
echo "=========================================="

nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"

echo "--- Starting --prepare ---"
neuralbench eeg image --dataset xu2024alljoined --prepare

echo "Prepare finished successfully."
