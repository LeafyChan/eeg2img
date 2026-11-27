#!/usr/bin/env bash
# Snapshot everything the audit needs into ./snapshot/ . Safe on the login
# node: it never imports torch/neuralbench (it reads files and package
# metadata only). Run from ~/eeg2img_new :
#   bash repro/eeg2img_repro/scripts/00_collect_state.sh
set -uo pipefail
H=/path/to/home/eeg2img_new
ENVD=/path/to/home/miniconda3/envs/eeg2img_new
PY=$ENVD/bin/python
HERE=$(dirname "$(realpath "$0")")
OUT=${OUT:-snapshot}; mkdir -p "$OUT"
cd "$H" || exit 1

# 1. Exact environment (pip metadata only)
$PY --version > "$OUT/python_version.txt" 2>&1
$PY -m pip freeze > "$OUT/pip_freeze.txt" 2>/dev/null
/path/to/home/miniconda3/bin/conda env export -n eeg2img_new > "$OUT/conda_env.yml" 2>/dev/null
nvidia-smi -L > "$OUT/gpu_login.txt" 2>&1 || true

# 2. NeuralBench version and the task yamls that live in site-packages
SP=$(ls -d "$ENVD"/lib/python3*/site-packages | head -1)
ls -d "$SP"/neuralbench*dist-info "$SP"/neuralset*dist-info "$SP"/neuraltrain*dist-info \
      "$SP"/neuralfetch*dist-info "$SP"/benchopt*dist-info 2>/dev/null > "$OUT/library_versions.txt"
mkdir -p "$OUT/neuralbench_yamls"
cp -r "$SP/neuralbench/tasks/eeg/image" "$OUT/neuralbench_yamls/image_task" 2>/dev/null
cp "$SP/neuralbench/defaults/config.yaml" "$OUT/neuralbench_yamls/defaults_config.yaml" 2>/dev/null
# the pristine wheel for diffing (pip download needs the network; a login node has it)
mkdir -p "$OUT/wheel_check"
echo "diff against pristine: $PY -m pip download neuralbench==0.4.0 --no-deps -d $OUT/wheel_check" > "$OUT/wheel_check/HOWTO.txt"

# 3. Our configs, variants and scripts
mkdir -p "$OUT/our_files"
cp config.json config_prep.json "$OUT/our_files/" 2>/dev/null
cp -r scripts "$OUT/our_files/" 2>/dev/null
cp -r frozen_1013/variants_new "$OUT/our_files/" 2>/dev/null
cp ~/ens_eval.py "$OUT/our_files/" 2>/dev/null
ls -la scripts/slurm/*.sbatch > "$OUT/sbatch_list.txt" 2>/dev/null

# 4. Model sources: revision hashes of the pretrained weights actually cached
for hub in "${HF_HOME:-$HOME/.cache/huggingface}/hub" "$HOME/.cache/huggingface/hub"; do
  for m in models--brain-bzh--reve-base models--facebook--dinov2-giant; do
    [ -d "$hub/$m/snapshots" ] && { echo "$hub/$m"; ls "$hub/$m/snapshots"; } >> "$OUT/hf_revisions.txt"
  done
done

# 5. Run table: seed / hash / hyper-parameters / score, one row per run
R="outputs/neuralbench.main.Experiment.run,1"
ls -d "$R"/*/ 2>/dev/null > "$OUT/run_dirs.txt"
$PY "$HERE/build_run_table.py" "$H" "$OUT/run_table.tsv" 2>&1 | tail -5

# 6. Code revisions of anything under git
: > "$OUT/git_revs.txt"
for d in "$H" "$HOME/2026-competition" "$HOME/eeg2img"; do
  [ -d "$d/.git" ] && (cd "$d" && echo "$d $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)") >> "$OUT/git_revs.txt"
done
echo "snapshot written to $OUT/ :"; du -sh "$OUT"; ls "$OUT"
