#!/usr/bin/env bash
CK="$(find /path/to/home/eeg2img/outputs -name best.ckpt -newer /path/to/home/eeg2img/scripts/slurm/reve_thingseeg2_train.sbatch | head -1)"
last=""
while true; do
  m=$(stat -c %Y "$CK" 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    sleep 60
    cp "$CK" /path/to/home/eeg2img/things_reve_best_$(date +%m%d_%H%M).ckpt && last=$m
  fi
  sleep 300
done
