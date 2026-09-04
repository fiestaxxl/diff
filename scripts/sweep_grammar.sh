#!/usr/bin/env bash
#
# lambda_grammar sweep. Instead of environment variables (as before) it uses
# CLI overrides on top of the yaml, so each run is fully recoverable from
# runs/<run_name>/config.resolved.yaml.
#
# Usage:
#   bash scripts/sweep_grammar.sh
set -euo pipefail

CONFIG="${CONFIG:-configs/diffusion_chebi.yaml}"
GEN_CONFIG="${GEN_CONFIG:-configs/generate_chebi.yaml}"
NPROC="${NPROC:-3}"
LAMBDAS=(${LAMBDAS:-0.0 0.001 0.003 0.01 0.03 0.1})   # 0.0 = control run without the grammar loss
DURATION="${DURATION:-250ep}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EVAL_BATCH="${EVAL_BATCH:-500}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
SUMMARY="${SUMMARY:-results/grammar_sweep_summary.txt}"

mkdir -p "$(dirname "$SUMMARY")"
echo "# grammar sweep $(date -u +%Y-%m-%dT%H:%M:%SZ)  duration=$DURATION nsamples=$EVAL_SAMPLES nproc=$NPROC" >> "$SUMMARY"

i=0
for lg in "${LAMBDAS[@]}"; do
    i=$((i + 1))
    run_name="gram_lg${lg}"
    port=$((29500 + i))

    echo ""
    echo "=================================================================="
    echo ">>> TRAIN  lambda_grammar=$lg  run=$run_name  duration=$DURATION  port=$port"
    echo "=================================================================="

    torchrun --nproc_per_node="$NPROC" --nnodes=1 --node_rank=0 --master_port="$port" \
        scripts/train.py "$CONFIG" \
        "run_name=$run_name" \
        "loss.lambda_grammar=$lg" \
        "max_duration=$DURATION"

    run_dir="${CKPT_ROOT}/${run_name}"
    ckpt=$(ls -dv "${run_dir}"/ep*-ba*/ 2>/dev/null | tail -1 | sed 's:/*$::' || true)
    if [ -z "${ckpt:-}" ]; then
        echo "!! no checkpoint under ${run_dir}; skipping eval"
        echo "lg=${lg}  NO_CHECKPOINT" >> "$SUMMARY"
        continue
    fi

    echo ""
    echo ">>> EVAL   $ckpt"
    python3 scripts/generate.py "$GEN_CONFIG" \
        "generate.checkpoint=$ckpt" \
        "generate.num_samples=$EVAL_SAMPLES" \
        "generate.batch_size=$EVAL_BATCH"

    report="${ckpt}/report.txt"
    if [ -f "$report" ]; then
        val=$(grep -m1 'validity:'   "$report" | sed 's/^[[:space:]]*//')
        uniq=$(grep -m1 'uniqueness:' "$report" | sed 's/^[[:space:]]*//')
        div=$(grep -m1 'diversity:'  "$report" | sed 's/^[[:space:]]*//')
        mlen=$(grep -m1 'mean_len:'  "$report" | sed 's/^[[:space:]]*//')
        echo "lg=${lg}  ${val}  ${uniq}  ${div}  ${mlen}  [${ckpt}]" >> "$SUMMARY"
    else
        echo "lg=${lg}  NO_REPORT  [${ckpt}]" >> "$SUMMARY"
    fi
done

echo ""
echo "=================== SWEEP SUMMARY ==================="
cat "$SUMMARY"
