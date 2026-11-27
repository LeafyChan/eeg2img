#!/usr/bin/env bash
declare -A seen
while true; do
  while IFS= read -r CK; do
    m=$(stat -c %Y "$CK" 2>/dev/null) || continue
    if [ "${seen[$CK]:-}" != "$m" ]; then
      sleep 60
      d=$(basename "$(dirname "$CK")")
      s=$(echo "$d" | grep -o '^seed=[0-9]*' | cut -d= -f2)
      cp "$CK" /path/to/home/eeg2img/checkpoints/auto_s${s}_${d: -8}_$(date +%m%d_%H%M).ckpt && seen[$CK]=$m
    fi
  done < <(find /path/to/home/eeg2img/outputs -name best.ckpt 2>/dev/null)
  sleep 300
done
