#!/usr/bin/env bash
#SBATCH --job-name=alljoined_download
#SBATCH --partition=cpu_student
#SBATCH --account=student
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=07:00:00
#SBATCH --chdir=/dgxa_home/se25uari118/eeg2img
#SBATCH --output=/dgxa_home/se25uari118/eeg2img/logs/slurm-%x-%j.out
#SBATCH --error=/dgxa_home/se25uari118/eeg2img/logs/slurm-%x-%j.err

set -euo pipefail

mkdir -p logs

echo "=========================================="
echo "ALLJOINED-1 DOWNLOAD"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : ${SLURM_NODELIST}"
echo "Account     : ${SLURM_JOB_ACCOUNT}"
echo "Partition   : ${SLURM_JOB_PARTITION}"
echo "Start       : $(date)"
echo "=========================================="

# Initialize Conda
source /dgxa_home/se25uari118/miniconda3/etc/profile.d/conda.sh
conda activate eeg2img

echo "Python:"
which python
python --version

echo "NeuralBench:"
which neuralbench

echo "Starting Alljoined-1 download..."

srun neuralbench eeg image \
    --dataset xu2024alljoined \
    --download

echo "=========================================="
echo "DOWNLOAD FINISHED: $(date)"
echo "=========================================="
