#!/bin/bash
SRC=$HOME/eeg2img_new/outputs
DST=$HOME/eeg2img_new/checkpoints
declare -A seen
while true; do
  while IFS= read -r f; do
    m=$(stat -c %Y "$f" 2>/dev/null) || continue
    if [ "${seen[$f]}" != "$m" ] && [ $(( $(date +%s) - m )) -gt 60 ]; then
      h=$(echo "$f" | grep -o '[0-9a-f]\{8\}/best.ckpt' | head -1 | cut -c1-8)
      cp "$f" "$DST/auto_${h}_$(date -d @$m +%m%d_%H%M).ckpt" && seen[$f]=$m
    fi
  done < <(find "$SRC" -name best.ckpt 2>/dev/null)
  sleep 30
done
