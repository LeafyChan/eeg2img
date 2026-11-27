#!/bin/bash
#SBATCH --job-name=extract_nsd
#SBATCH --partition=cpu_student
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --output=scripts/slurm/outputs/extract_nsd_%j.out
#SBATCH --error=scripts/slurm/outputs/extract_nsd_%j.err

set -euo pipefail

source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

python - <<'PY'
import h5py
from pathlib import Path
from PIL import Image
from tqdm import tqdm

h5_path = Path.home() / "eeg2img/data/nsd_common/nsd_stimuli.hdf5"
out_dir = Path.home() / "eeg2img/data/nsd_common/nsd_stimuli"
out_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(h5_path, "r") as f:
    stim = f["imgBrick"]
    print("Dataset:", stim.shape, stim.dtype)

    for i in tqdm(range(len(stim)), desc="Extracting NSD images"):
        out = out_dir / f"{i}.png"
        if not out.exists():
            Image.fromarray(stim[i]).save(out)

print("PNG count:", len(list(out_dir.glob("*.png"))))
PY
