#!/usr/bin/env bash
CK="$(find /path/to/home/eeg2img/outputs -path '*8c4c5d67*' -name best.ckpt | head -1)"
echo "watching: $CK"
last=""
while true; do
  m=$(stat -c %Y "$CK" 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$last" ]; then
    sleep 60
    cp "$CK" /path/to/home/eeg2img/checkpoints/reve_c120_s34_$(date +%m%d_%H%M).ckpt && last=$m
  fi
  sleep 300
done
