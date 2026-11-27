#!/usr/bin/env bash
CK="$(find /path/to/home/eeg2img/outputs -path '*f5887f3d*' -name best.ckpt | head -1)"
echo "watching: $CK"
last=""
while true; do
  m=$(stat -c %Y "$CK" 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    sleep 60
    cp "$CK" /path/to/home/eeg2img/checkpoints/reve_flat_s33_$(date +%m%d_%H%M).ckpt && last=$m
  fi
  sleep 300
done
