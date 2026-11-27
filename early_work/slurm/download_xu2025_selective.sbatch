#!/bin/bash
#SBATCH --job-name=xu2025_selective
#SBATCH --partition=cpu_student
#SBATCH --account=student
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dgxa_home/se25uari118/eeg2img/logs/xu2025_selective_%j.out
#SBATCH --error=/dgxa_home/se25uari118/eeg2img/logs/xu2025_selective_%j.err

set -euo pipefail

source /dgxa_home/se25uari118/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

cd /dgxa_home/se25uari118/eeg2img

echo "=== START ==="
date
hostname
df -h "$HOME/eeg2img"

hf download \
  Alljoined/Alljoined-1.6M \
  --repo-type dataset \
  --local-dir "$HOME/eeg2img/data/Xu2025Alljoined/download" \
  --include "raw_eeg/**" \
  --include "stimuli.zip"

echo "=== DOWNLOAD COMPLETE ==="
date

du -sh "$HOME/eeg2img/data/Xu2025Alljoined"
du -sh "$HOME/eeg2img/data/Xu2025Alljoined/download/raw_eeg" 2>/dev/null || true
du -sh "$HOME/eeg2img/data/Xu2025Alljoined/download/stimuli.zip" 2>/dev/null || true
find "$HOME/eeg2img/data/Xu2025Alljoined/download/raw_eeg" -type f 2>/dev/null | wc -l
