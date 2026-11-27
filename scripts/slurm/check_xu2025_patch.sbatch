#!/bin/bash
#SBATCH --job-name=check_xu2025
#SBATCH --partition=gpu_student
#SBATCH --account=student
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=10:00
#SBATCH --output=/dgxa_home/se25uari118/eeg2img/logs/check_xu2025_%j.out
#SBATCH --error=/dgxa_home/se25uari118/eeg2img/logs/check_xu2025_%j.err

set -euo pipefail
source /dgxa_home/se25uari118/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

python - <<'PY'
import inspect
import neuralfetch
from neuralfetch.studies.xu2025alljoined import Xu2025Alljoined

print("neuralfetch:", neuralfetch.__file__)
print("loader:", inspect.getfile(Xu2025Alljoined))
print()
print(inspect.getsource(Xu2025Alljoined._load_timeline_events))
PY
