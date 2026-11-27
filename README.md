# SleepNet — EEG/EMG Foundation Challenge 2026, Track 3

Sleep onset prediction from wearable EEG (Muse headband).
Cross-user generalisation: predict seconds to first stable N2 epoch.

## Track
- Task: Sleep onset prediction (bMAE lower is better)
- Device: Muse 4-channel frontal EEG
- Competition: https://neural-interfaces26.github.io
- Benchmark: https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track3_sleep_onset.html

## Setup
    python3 -m venv sleepnet
    source sleepnet/bin/activate
    pip install neuralbench

## Baselines
    neuralbench eeg sleep_onset --download
    neuralbench eeg sleep_onset -m eegnet --debug
    neuralbench eeg sleep_onset -m eegnet

## Structure
    src/models/     custom model architectures
    src/data/       data loading and preprocessing
    src/utils/      metrics, logging, helpers
    experiments/    configs and results per run
    notebooks/      EDA and analysis
