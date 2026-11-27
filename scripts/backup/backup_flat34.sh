#!/usr/bin/env bash
last=""
while true; do
  CK="$(find /path/to/home/eeg2img/outputs -path '*f00f85e7*' -name best.ckpt 2>/dev/null | head -1)"
  if [ -n "$CK" ]; then
    m=$(stat -c %Y "$CK" 2>/dev/null)
    if [ -n "$m" ] && [ "$m" != "$last" ]; then
      sleep 60
      cp "$CK" /path/to/home/eeg2img/checkpoints/reve_flat_s34_$(date +%m%d_%H%M).ckpt && last=$m
    fi
  fi
  sleep 300
done
