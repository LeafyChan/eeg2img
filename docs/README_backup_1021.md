# EEG-to-Image (Track 1) — reproducibility repo

Status: draft for the audit. Items marked **TODO(fill)** need facts only the cluster has;
`scripts/00_collect_state.sh` gathers most of them into `snapshot/`. Facts marked
*(measured)* were read from logs or scorer output; the rest are recorded settings.

## Method (one paragraph)
Fine-tune REVE end to end on Alljoined-1.6M (NeuralBench `xu2025alljoined`
task, 200 Hz native preprocessing, RobustScaler) with a CLIP contrastive loss
against frozen DINOv2-giant embeddings (1536-d). Final submission: seed
ensemble (3 or 6 members), raw outputs averaged per (subject, image) by the
scorer, so members return unit-norm embeddings. The solver upsamples the
harness' 120 Hz / 120-sample input to the training rate before the encoder.

## Data declaration
- Training: Alljoined-1.6M (public), all 20 subjects, `stim_train` trials only (train 1,065,150 / val 270,466 windows, the 20% val split by timeline). The `stim_test` trials (324,294 windows) are the held-out test split and were never trained on. No warm-up (THINGS-EEG2) data, and no sealed-cohort data, was used by the submitted models.
- Pretrained weights: `brain-bzh/reve-base`, an external pretrained EEG model. About 70M parameters in our setting: `Total params: 70,042,626`
  (REVE + head, 32-channel Alljoined training logs; 70,106,114 for the 63-channel THINGS run; the difference is 31 x 2048). Revision `dc2a075c287bb2f6c04ee5875bd79535a0f7dba6` *(measured: `snapshot/hf_revisions.txt`; HF cache
  `~/eeg2img/cache/huggingface`, the only copy found)*.
- REVE pretraining corpus (external, not ours): masked-autoencoder pretraining on over 60,000 hours of EEG from 92
  datasets and about 25,000 subjects (Elouahidi et al., NeurIPS 2025, arXiv 2510.21585). **TODO(fill)**: read the paper's
  dataset list / appendix and the model card, and state here whether THINGS-EEG, Alljoined or any other image-viewing EEG
  dataset is in it. Not known to us; do not assume either way.
- Target embeddings: `facebook/dinov2-giant`, frozen, as fixed by the competition; revision `611a9d42f2335e0f921f1e313ad3c1b7178d206d`
  *(measured, same file)*.
- Other data touched during the project, NOT used by any sealed candidate: THINGS-EEG2 (warm-up models), THINGS-EEG1 and
  Alljoined-1 (cache preparation only, no training run). `reve_full` (Codabench submissions 976279 and 984923) was trained on
  37 of the 40 THINGS-EEG2 test-kind recordings; it exists only as a warm-up upload and is in no sealed candidate and no
  ensemble.

## Compute declaration
DGX A100 nodes (`dgxa`, `dgxb`), partition `gpu_student`. GPU model *(measured, replay job 28477)*:
`NVIDIA A100-SXM4-40GB`, 40,326 MiB. The MIG profiles offered on both nodes (1g.5gb, 2g.10gb, 3g.20gb, 4g.20gb) are the
40 GB A100 profile set. Node *(measured, `sacct`)*: every job behind members a-f and the `sealed_6m` replay (28477) ran on `dgxa`: a and c on
`a100_4g.20gb`, b and the seed-33 rerun on `a100_3g.20gb`, d-f and the replay on a full A100.
Final b256 members: ~0:59 h each on a full A100; held-out folds ~2:48 h each on a slice. Replay of the packaged submissions (below) is inference only.
Total GPU allocation time since 2026-09-20 *(measured, `sacct -X`, D-HH:MM:SS-aware)*: **319.8 GPU-hours** over every GPU job
of the project, including failed, cancelled, test, preparation and idle-orchestrator jobs, with a MIG slice counted as one GPU:
3g.20gb 164.0 h, 4g.20gb 52.7 h, full A100 20.3 h, jobs with an untyped GPU request 82.9 h (slice type not recorded).
Compute to reproduce the six packaged members: **8:45:10 = 8.75 GPU-hours** (`sacct`, job elapsed incl. startup and test step):
a 1:47:29, b 2:13:58, c 1:47:18 (slices, see below); d 0:59:16, e 0:59:10, f 0:57:59 (array 28386, full A100, `dgxa`).
Most of the 319.8 h is exploration that no sealed candidate
depends on: THINGS-EEG2 warm-up models, the failed Alljoined runs with the old loader, held-out folds and sweeps, replays.
Slice drift: members a and c trained on 4g.20gb slices and member b on 3g.20gb; the seed-33 rerun on a different slice
differed by 0.0025 subj-agg, so an audit rerun of any member on another slice should be expected to differ by about that much.

## Seeds and runs
| member | seed | recipe | batch | NeuralBench hash | checkpoint file |
|---|---|---|---|---|---|
| a | 33 | b64, 100 h/epoch, 6 ep | 64 | 146a009b | auto_146a009b_1012_1944 |
| b | 34 | b64, 100 h/epoch, 6 ep | 64 | 632062be | auto_632062be_1013_0358 |
| c | 35 | b64, 100 h/epoch, 6 ep | 64 | 36cb4ac5 | auto_36cb4ac5_1012_2135 |
| d | 33 | b256, lr 3e-4, 6 ep (variant xu2025alljoined_reve_b256) | 256 | 5b389c23 | auto_5b389c23_1019_0043 |
| e | 34 | same | 256 | 0b4610ff | auto_0b4610ff_1019_0142 |
| f | 35 | same | 256 | 14ac7b3d | auto_14ac7b3d_1019_0240 |

Run history (so the table is not read as one uniform array): member a = job 27265_0 (wrapper 27264, with `-f`,
4g.20gb slice, 1:47); a second run of the same seed 33 (27281_2, 3g.20gb, 2:14) has the same hash 146a009b and differs by
0.0025 subj-agg; the packaged checkpoint is the first run. Member c = 27281_1 (wrapper 27280, 4g.20gb, 1:47). Member b
(seed 34) = job 27324 from wrapper 27321, submitted WITHOUT `-f` after the experiment folder of a cancelled task (27281_0)
was moved to `stale_outputs_1013/` (a stale `job.pkl` made NeuralBench report the seed as still running); 3g.20gb, 2:14.
Members d-f = array 28386 (wrapper 28385), full A100, about 0:59 each.

Packages: sealed_1m = a; sealed_3m = a,b,c; sealed_6m = a-f (equal weights).
md5: sealed_1m 1a3ff4f5c14cf383e5d1a28755907219, sealed_3m 8db46169252c1437a67bb00d02b2a90e,
sealed_6m 1677e2fbb3a45dec44f47465c913a7ac.
Variant yaml for a-c: `xu2025alljoined_reve`.

## Layout
- `configs/` — copy of the NeuralBench task yamls actually used (from
  `snapshot/neuralbench_yamls/`; they live in site-packages on the
  cluster, which is not auditable — commit them here and diff against the wheel).
- `scripts/00_collect_state.sh` — login-node-safe snapshot (env, yamls from
  site-packages, library versions, HF revisions, run table, git revisions).
  Never imports torch/neuralbench.
- `scripts/build_run_table.py` — seed / hash / batch / lr / epochs / scores per run
  (`snapshot/run_table.tsv`). Matches the 0.4.0 box-table log format; checked
  against the six sealed members (*measured*: values equal the logs).
- `scripts/replay_sealed.sbatch` + `10_harness_replay.sh` + `tap.py` +
  `show_result.py` — replay a packaged submission through the real competition
  harness. The sbatch exports its own `NEURALBENCH_CONFIG` (isolated work
  folder `replay_work/`), unsets `NEURALBENCH_*_DIR`, symlinks
  `tracks/image_decoding/data` to `replay_work/bdata` (benchopt's data path is
  `<benchmark_dir>/data/neural_compet`), and the inner script refuses to run on a
  GPU with <15 GB (DINOv2 OOM on 2g.10gb), when `benchmark_utils` is not a real
  folder, or when `prepare` skipped the dataset (benchopt exits 0 in that case).
- `results/replay_sealed_1m/`, `results/replay_sealed_6m/` — tap log, run log and benchopt
  result parquet of the harness replays (below).
- `scripts/ens_eval.py` — offline seed-ensemble scorer (matches the public
  `objective.py`: average raw predictions per subject-image, then normalise).

## Known facts about the harness (public repo, ZIP download of 2026-10-20)
- Scorer: `tracks/image_decoding/objective.py`; subject-agg averages RAW
  predictions per (subject, image), then L2-normalises and ranks vs the unique
  target embeddings. Per-prediction normalisation is not applied first.
- Pins `neuralbench`/`neuralset`/`neuralfetch`/`neuraltrain` 0.4.0 (the version used here).
- Input handed to `predict` *(measured, tap log)*: shape (64, 32, 120), `sfreq` 120,
  `n_times` 120 (FAQ is right, the adapter docstring "1.2 s" is stale), 32 channels
  in the order `Cz, FCz, AFz, Fp1, F5, F1, CP5, CP3, CP1, P1, P3, P5, P7, PO7, PO3, O1,
  Pz, POz, Oz, O2, PO4, PO8, P8, P6, P4, P2, CP2, CP4, CP6, F2, F6, Fp2`; robust-scaled
  (std about 0.8, max abs about 6.7, no NaN).
- Sealed cohort: 11 subjects, 256 Hz native, 32-ch Emotiv; gallery size and
  repeat count are private.
- A ZIP download of the repo flattens symlinks into small text files. Fixed by hand
  for `tracks/image_decoding/benchmark_utils` (`ln -s ../../benchmark_utils`).
  A `git clone` does not need this.

## Harness replay (*measured*, Alljoined-1.6M `stim_test`, 324,294 windows, 200 candidates)
Seen subjects, unseen images: this validates the packaging path, not unseen-subject performance.

| package | top-5 | top-1 | top-5 subj-agg | top-1 subj-agg | job time |
|---|---|---|---|---|---|
| sealed_1m (member a) | 0.0999 | 0.0261 | 0.3573 | 0.1305 | 15:05 |
| sealed_6m (a-f) | 0.1062 | 0.0282 | 0.3783 | 0.1400 | 25:28 |

Job time includes about 2-3 min of fixed loading; about 2.1 min per extra member per
324k windows. NeuralBench test of the single members: top-5 0.0986-0.1009, subj-agg
0.3595-0.3717. The zip files equal the replayed folders (md5 of `submission.py` and all
six `weights_*.pt` compared).

Unseen-subject estimate (4 held-out subjects per fold, mean of 3 seeds, subj-agg top-5, full ~81 repeats per subject-image
pair): fold 1 0.31, fold 2 0.17, fold 3 0.23, about 0.24 on average (fold spread is much larger than seed spread).
Repeat sweep, **fold 1 only** (held-out subjects 3/8/13/18, batch-256 recipe, 3 seeds, k windows averaged per (subject, image),
mean of 5 random draws; *measured*, `scripts/ens_eval.py`): the other folds sit lower, so scale these down.

| k | single-seed mean top-5 / top-1 | 3-seed ensemble top-5 / top-1 |
|---|---|---|
| 1 | 0.096 / 0.024 | 0.102 / 0.023 |
| 5 | 0.181 / 0.048 | 0.190 / 0.055 |
| 10 | 0.228 / 0.073 | 0.238 / 0.076 |
| 20 | 0.272 / 0.091 | 0.286 / 0.094 |
| 40 | 0.301 / 0.106 | 0.312 / 0.110 |
| all (~81) | 0.322 / 0.111 | 0.338 / 0.113 |

## Reproduce a member
Environment: `eeg2img_new` (library 0.4.0; `snapshot/conda_env.yml`, `snapshot/pip_freeze.txt`),
`NEURALBENCH_CONFIG=$HOME/eeg2img_new/config.json`. The NeuralBench task yamls live in
site-packages; commit the copies from `snapshot/neuralbench_yamls/` and diff them against the wheel.
Local edit: `neuralbench/defaults/config.yaml` has `timeout_min: 4320` (in `snapshot/`).

Wrappers (CPU jobs that submit the training children; copies in `snapshot/our_files/scripts/slurm/`):
- members a-c: `reve_aj16L.sbatch`, i.e. `neuralbench eeg image -m reve --dataset xu2025alljoined_reve --max-neuro-hours-per-epoch 100 -f`
- members d-f: `reve_aj16F.sbatch`, dataset `xu2025alljoined_reve_b256`, `--max-neuro-hours-per-epoch 100`, no `-f`
  *(measured: `grep -n neuralbench` on both wrappers, line 16)*. `reve_aj16L` has `-f`; `reve_aj16F` does not.
Seeds 33, 34, 35 are a grid inside the wrapper. At 100 h/epoch the array index does NOT map
to a fixed seed: read `seed:` (top level) from each run's `config.yaml`.

Expected single-seed test metrics (seen subjects, unseen images): a-c single-window top-5
0.0986-0.0997, subj-agg 0.3595-0.3665; d-f 0.0999-0.1009 and 0.3640-0.3717. The same seed rerun
on a different slice differed by 0.0025 (subj-agg). Final best checkpoint = the last copy
(best epoch 4-5 of 6). Packaging: `make_weights32.py` and the `sealed_*` folders
(**TODO(fill)**: copy them from `~/eeg2img/submissions/` into this repo).