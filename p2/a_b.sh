#!/usr/bin/env bash
# A/B/C kernel-time test for TILE = 8192 / 16384 / 32768 at 100M.
# Round-robin interleave cancels the node's ~20% timing drift.
set -euo pipefail
cd /home/hice1/dnguyen487/gpu-labs/p2

SIZE=100000000
ROUNDS=5            # round 1 is treated as warmup and dropped
TILES=(8192 16384 32768)

# --- build one binary per TILE into /tmp (uses dynamic-shared bitonic.cu) ---
cp bitonic.cu /tmp/b.cu && cp main.cu /tmp/main.cu && cp main.h /tmp/main.h
for T in "${TILES[@]}"; do
  sed "s/#define TILE [0-9]\+/#define TILE $T/" student.h > /tmp/student.h
  ( cd /tmp && nvcc -O3 -x cu b.cu main.cu -o "a_$T" ) || { echo "build TILE=$T FAILED"; exit 1; }
done
echo "built: ${TILES[*]}"

# --- correctness gate at 10M before timing ---
for T in "${TILES[@]}"; do
  ok=$(/tmp/"a_$T" 10000000 | grep -c "FUNCTIONAL SUCCESS" || true)
  [[ "$ok" == "1" ]] && echo "TILE=$T: FUNCTIONAL SUCCESS" || { echo "TILE=$T: FUNCTIONAL FAIL"; exit 1; }
done

# --- interleaved timing ---
declare -A sum cnt
printf '\n%-6s' "round"; for T in "${TILES[@]}"; do printf '%12s' "TILE=$T"; done; echo
for r in $(seq 1 "$ROUNDS"); do
  printf '%-6s' "$r"
  for T in "${TILES[@]}"; do
    k=$(/tmp/"a_$T" "$SIZE" | grep "Kernel Time" | grep -oE '[0-9]+\.[0-9]+')
    printf '%12s' "$k"
    if [[ "$r" -gt 1 ]]; then                       # drop warmup round
      sum[$T]=$(awk -v a="${sum[$T]:-0}" -v b="$k" 'BEGIN{print a+b}')
      cnt[$T]=$(( ${cnt[$T]:-0} + 1 ))
    fi
  done
  echo
done

# --- trimmed means (rounds 2..N) ---
echo; echo "mean kernel time (rounds 2-$ROUNDS):"
for T in "${TILES[@]}"; do
  awk -v s="${sum[$T]}" -v n="${cnt[$T]}" -v t="$T" 'BEGIN{printf "  TILE=%-6s %.2f ms\n", t, s/n}'
done