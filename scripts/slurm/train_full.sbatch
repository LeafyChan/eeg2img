#!/usr/bin/env bash
#SBATCH --job-name=eeg2img_train
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/dgxa_home/se25uari118/eeg2img
#SBATCH --output=/dgxa_home/se25uari118/eeg2img/logs/train_full-%j.out
#SBATCH --error=/dgxa_home/se25uari118/eeg2img/logs/train_full-%j.err

set -euo pipefail
source /dgxa_home/se25uari118/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img
export ALLJOINED_ACCEPT_LICENCE=1
export HF_TOKEN=REDACTED

echo "=== START ===" && date && hostname
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"

neuralbench eeg image --dataset xu2025alljoined -m eegnet

echo "=== DONE ===" && date
