#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$DIR/eemicrogpt.c"
BIN=/tmp/fgpt_sweep
CSV="$DIR/sweep_results.csv"

echo "sweep,value,backend,params,final_loss,us_per_step,fwd_us,bwd_us,total_ms" > "$CSV"

# Table header
printf "%-10s %6s  %-7s %7s  %6s  %8s  %8s  %8s  %9s\n" \
    "sweep" "value" "backend" "params" "loss" "us/step" "fwd" "bwd" "total_ms"
printf '%s\n' "$(printf '=%.0s' {1..82})"

run() {
    local sweep=$1 val=$2 defs=$3

    for backend in scalar sme2; do
        if [ "$backend" = "sme2" ]; then
            clang -O3 -mcpu=native+sme2 -ffast-math $defs -o $BIN $SRC -lm 2>/dev/null
        else
            clang -O3 -ffast-math $defs -o $BIN $SRC -lm 2>/dev/null
        fi

        out=$($BIN 2>&1)

        params=$(echo "$out" | grep -oE '[0-9]+ parameters' | grep -oE '[0-9]+')
        loss=$(echo "$out" | grep '^Step' | tail -1 | sed 's/.*loss=\([0-9.]*\).*/\1/')
        us=$(echo "$out" | grep 'Average step' | sed 's/.*: \([0-9.]*\) us/\1/')
        fwd=$(echo "$out" | grep 'Forward:' | grep -oE '[0-9.]+ us/step' | grep -oE '[0-9.]+')
        bwd=$(echo "$out" | grep 'Backward:' | grep -oE '[0-9.]+ us/step' | grep -oE '[0-9.]+')
        total=$(echo "$out" | grep 'Training complete' | grep -oE '[0-9.]+ ms' | head -1 | grep -oE '[0-9.]+')

        printf "%-10s %6s  %-7s %7s  %6s  %8s  %8s  %8s  %9s\n" \
            "$sweep" "$val" "$backend" "$params" "$loss" "$us" "$fwd" "$bwd" "$total"
        echo "$sweep,$val,$backend,$params,$loss,$us,$fwd,$bwd,$total" >> "$CSV"
    done
}

# D_MODEL sweep
for dm in 16 32 64; do
    run "d_model" "$dm" "-DD_MODEL=$dm"
done
run "d_model" "128" "-DD_MODEL=128 -DN_HEADS=8"

printf '%s\n' "$(printf '-%.0s' {1..82})"

# N_HEADS sweep (D_MODEL=64)
for nh in 1 2 4 8; do
    run "n_heads" "$nh" "-DN_HEADS=$nh"
done

printf '%s\n' "$(printf '-%.0s' {1..82})"

# LR sweep
for lr in 0.001 0.003 0.01 0.03; do
    run "lr" "$lr" "-DLR_INIT=${lr}f"
done

printf '%s\n' "$(printf '-%.0s' {1..82})"

# N_STEPS sweep
for ns in 500 1000 2000 5000; do
    run "n_steps" "$ns" "-DN_STEPS=$ns"
done

printf '%s\n' "$(printf '-%.0s' {1..82})"

# BATCH sweep
for bs in 8 16 32 64; do
    run "batch" "$bs" "-DBATCH=$bs"
done

printf '%s\n' "$(printf '=%.0s' {1..82})"
echo ""
echo "Results saved to $CSV"
