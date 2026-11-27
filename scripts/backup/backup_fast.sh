#!/bin/bash
SRC=$HOME/eeg2img_new/outputs
DST=$HOME/eeg2img_new/checkpoints
declare -A seen
while true; do
  while IFS= read -r f; do
    m=$(stat -c %Y "$f" 2>/dev/null) || continue
    [ "${seen[$f]}" = "$m" ] && continue
    s1=$(stat -c %s "$f" 2>/dev/null) || continue
    sleep 3
    s2=$(stat -c %s "$f" 2>/dev/null) || continue
    if [ "$s1" = "$s2" ] && [ "$s1" -gt 200000000 ]; then
      h=$(echo "$f" | grep -o '[0-9a-f]\{8\}/best.ckpt' | head -1 | cut -c1-8)
      cp "$f" "$DST/fast_${h}_$(date -d @$m +%m%d_%H%M%S).ckpt" && seen[$f]=$m
    fi
  done < <(find "$SRC" -name best.ckpt 2>/dev/null)
  sleep 5
done

