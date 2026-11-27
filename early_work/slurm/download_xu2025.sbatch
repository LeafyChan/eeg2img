#!/bin/bash
#SBATCH --job-name=xu2025_download
#SBATCH --partition=cpu_student
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dgxa_home/se25uari118/eeg2img/logs/xu2025_download_%j.out
#SBATCH --error=/dgxa_home/se25uari118/eeg2img/logs/xu2025_download_%j.err

set -euo pipefail
source /dgxa_home/se25uari118/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img
export ALLJOINED_ACCEPT_LICENCE=1

cd /dgxa_home/se25uari118/eeg2img
echo "=== START ==="
date
hostname

neuralbench eeg image --dataset xu2025alljoined --download

echo "=== END ==="
date
du -sh data/Xu2025Alljoined
