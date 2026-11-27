#!/bin/bash
#SBATCH --job-name=nsd_stimuli
#SBATCH --partition=cpu_student
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=scripts/slurm/outputs/nsd_stimuli_%j.out
#SBATCH --error=scripts/slurm/outputs/nsd_stimuli_%j.err

set -euo pipefail

mkdir -p ~/eeg2img/data/nsd_common

aws s3 cp \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  ~/eeg2img/data/nsd_common/nsd_stimuli.hdf5 \
  --no-sign-request

ls -lh ~/eeg2img/data/nsd_common/nsd_stimuli.hdf5
