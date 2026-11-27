#!/usr/bin/env bash
# Replay a packaged sealed submission through the real competition harness.
# Submit from anywhere (the job sets its own environment and config):
#
#   # fast plumbing check (about 1 h, mostly prepare; stops after 3 batches):
#   sbatch --export=ALL,SUB=$HOME/eeg2img/submissions/sealed_1m,TAP_ABORT=1 --time=03:00:00 replay_sealed.sbatch
#   # full score replay of one member (hours):
#   sbatch --export=ALL,SUB=$HOME/eeg2img/submissions/sealed_1m replay_sealed.sbatch
#   # if the guard says the GPU slice is too small, resubmit with a full A100:
#   sbatch --gres=gpu:a100:1 --export=ALL,SUB=... replay_sealed.sbatch
#
#SBATCH --job-name=replay_sealed
#SBATCH --partition=gpu_student
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --chdir=/path/to/home/eeg2img_new
#SBATCH --output=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.out
#SBATCH --error=/path/to/home/eeg2img_new/scripts/slurm/outputs/%x-%j.err

set -uo pipefail
H=/path/to/home/eeg2img_new
WORK=$H/replay_work
SUB=${SUB:?set SUB=/path/to/package_dir via --export=ALL,SUB=...}
REPO=${REPO:-$H/2026-competition-main}
SCRIPTS=${SCRIPTS:-$H/repro/eeg2img_repro/scripts}

source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img_new

# --- isolate from every other run (Section 54: never trust the submitting shell)
unset NEURALBENCH_DATA_DIR NEURALBENCH_CACHE_DIR NEURALBENCH_SAVE_DIR
mkdir -p "$WORK/cache" "$WORK/save" "$WORK/bdata/neural_compet"
python - <<'PYEOF'
import json, os
H = "/path/to/home/eeg2img_new"; W = H + "/replay_work"
c = json.load(open(H + "/config.json"))
c["CACHE_DIR"] = W + "/cache"; c["SAVE_DIR"] = W + "/save"
c["DATA_DIR"] = W + "/bdata/neural_compet"
c["CLUSTER"] = None   # extraction in-process, as the working prep jobs (no child arrays)
json.dump(c, open(W + "/config_replay.json", "w"), indent=1)
PYEOF
export NEURALBENCH_CONFIG=$WORK/config_replay.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Accept the Alljoined licence yourself, then uncomment: export ALLJOINED_ACCEPT_LICENCE=1

# private COPY of the downloaded Alljoined files (7.9 GB): a re-download or lock file can never touch the shared folder
LINK=$WORK/bdata/neural_compet/Xu2025Alljoined
if [ ! -e "$LINK" ] && [ -d "$H/data/Xu2025Alljoined" ]; then cp -r "$H/data/Xu2025Alljoined" "$LINK"; fi
# benchopt resolves the data path as <benchmark_dir>/data/neural_compet when no data_home is
# set (benchopt/config.py get_data_path; the competition adapter calls it at import).
# Point <benchmark_dir>/data at our private copy, so the harness's data AND caches stay in replay_work.
BD=$REPO/tracks/image_decoding
[ -d "$BD" ] || { echo "GUARD FAILED: $BD not found (set REPO=...)"; exit 3; }
if [ -e "$BD/data" ] && [ "$(readlink -f "$BD/data")" != "$(readlink -f "$WORK/bdata")" ]; then
  echo "GUARD FAILED: $BD/data exists and is not our private copy"; exit 3
fi
[ -e "$BD/data" ] || ln -s "$WORK/bdata" "$BD/data"
touch "$WORK/.start_marker"

echo "NEURALBENCH_CONFIG=$NEURALBENCH_CONFIG"
echo "SUB=$SUB REPO=$REPO TAP_ABORT=${TAP_ABORT:-0}"
nvidia-smi -L

export WORK
bash "$SCRIPTS/10_harness_replay.sh" "$REPO" "$SUB"
rc=$?
echo "replay exit: $rc"

# the shared folder should be untouched (expect no files listed)
if [ -d "$H/data/Xu2025Alljoined" ]; then
  echo "files in shared Alljoined folder newer than job start:"
  find "$H/data/Xu2025Alljoined" -newer "$WORK/.start_marker" -type f 2>/dev/null | head -20
fi
du -sh "$WORK" 2>/dev/null
exit $rc
