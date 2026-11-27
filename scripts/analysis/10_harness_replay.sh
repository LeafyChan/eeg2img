#!/usr/bin/env bash
# Inner replay script. Normally started by replay_sealed.sbatch, which sets
# WORK, NEURALBENCH_CONFIG, BENCHOPT_CONFIG and activates eeg2img_new.
#
#   10_harness_replay.sh <competition_repo> <submission_dir>
#
# What a pass looks like (stim_test = held-out trials of the 20 TRAINING
# subjects; same 324,294 windows NeuralBench scored): a single 20-subject
# member should score about 0.099-0.101 single-window and 0.36-0.37
# subject-agg top-5 (members a and d, Sections 131/134 of the notes). Pass =
# within ~0.02. Far below (chance is 0.02) = preprocessing mismatch.
# TAP_ABORT=1 only checks what the harness feeds the model (no score).
set -uo pipefail
REPO=$(realpath "$1"); SUB=$(realpath "$2")
STUDY="${STUDY:-xu2025alljoined}"
: "${WORK:?WORK must be set by replay_sealed.sbatch}"
HERE=$(dirname "$(realpath "$0")")
TMP=$(mktemp -d "$WORK/run_XXXX"); export TAP_LOG="$TMP/tap.jsonl"
fail() { echo "GUARD FAILED: $*" >&2; exit 3; }

# 1. config isolation (Section 54)
case "${NEURALBENCH_CONFIG:-}" in "$WORK"/*) ;; *) fail "NEURALBENCH_CONFIG is '${NEURALBENCH_CONFIG:-}', not under $WORK";; esac
[ -z "${NEURALBENCH_DATA_DIR:-}${NEURALBENCH_CACHE_DIR:-}${NEURALBENCH_SAVE_DIR:-}" ] || fail "NEURALBENCH_*_DIR variables are set"

# 2. imports the harness needs (no installs here)
python -c "import benchopt, neuralbench, transformers, torch, sklearn, torchvision, torchaudio" \
  >/dev/null 2>"$TMP/import.err" \
  || { tail -5 "$TMP/import.err" >&2; fail "a required module is missing in this environment; do not pip install into eeg2img_new, ask first"; }
# benchopt's data path = <benchmark_dir>/data/neural_compet; it must be our private copy
BD="$REPO/tracks/image_decoding"
[ "$(readlink -f "$BD/data" 2>/dev/null)" = "$(readlink -f "$WORK/bdata")" ] \
  || fail "$BD/data is not linked to $WORK/bdata (run via replay_sealed.sbatch)"
[ -d "$WORK/bdata/neural_compet/Xu2025Alljoined/download" ] || fail "private Alljoined copy missing under $WORK/bdata/neural_compet"
echo "data path: $(readlink -f "$BD/data")/neural_compet"
# a ZIP download flattens symlinks into small text files: benchmark_utils must be a real folder or link
[ -f "$BD/benchmark_utils/__init__.py" ] || fail "$BD/benchmark_utils is not a folder (flattened symlink?): ln -s ../../benchmark_utils"

# 3. GPU memory (Section 52: DINOv2-giant at batch 32 ran out of memory on a 9.6 GiB slice)
# torch sees the MIG slice this job was given; nvidia-smi may report the parent GPU
MIB=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory/2**20))" 2>/dev/null || echo 0)
echo "GPU memory: ${MIB} MiB"
[ "${MIB:-0}" -ge 15000 ] || fail "GPU has ${MIB} MiB; resubmit with: sbatch --gres=gpu:a100:1 ..."

# 4. work on a copy of the package, with the tap appended
mkdir -p "$TMP/sub"; cp -r "$SUB"/. "$TMP/sub/"
[ -f "$TMP/sub/submission.py" ] || fail "no submission.py in $SUB"
cat "$HERE/tap.py" >> "$TMP/sub/submission.py"
python -I -m py_compile "$TMP/sub/submission.py" || fail "tap does not compile"

# 5. prepare + run through the real harness
cd "$REPO" || fail "cannot cd to $REPO"
benchopt prepare tracks/image_decoding -d "Image[study=$STUDY]" 2>&1 | tee "$TMP/prepare.log" | grep -v -i warn
prc=${PIPESTATUS[0]}; [ "$prc" -eq 0 ] || fail "benchopt prepare exited $prc"
# benchopt prepare exits 0 even when it skipped the dataset (import error): check the evidence
if grep -q -E "not installed|0/0 datasets ready|Failed to import" "$TMP/prepare.log"; then
  fail "benchopt prepare skipped the dataset (see the import error above)"
fi
[ -d "$WORK/bdata/neural_compet/cache" ] || fail "no cache folder under $WORK/bdata/neural_compet after prepare"
du -sh "$WORK/bdata/neural_compet"/* 2>/dev/null
COMPET_SUBMISSION_DIR="$TMP/sub" \
  benchopt run tracks/image_decoding -d "Image[study=$STUDY]" \
  -s "$TMP/sub/submission.py" --no-plot 2>&1 | tee "$TMP/run.log" | grep -v -i warn
rrc=${PIPESTATUS[0]}; echo "benchopt run exit: $rrc"

echo; echo "===== what the harness handed the model ====="
cat "$TAP_LOG" 2>/dev/null || echo "(no tap log: submission never loaded?)"
echo; echo "===== scores ====="
python "$HERE/show_result.py" "$REPO" "$TMP/run.log" || true
echo; echo "tap log: $TAP_LOG   run log: $TMP/run.log"
exit "$rrc"
