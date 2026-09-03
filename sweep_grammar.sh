#!/usr/bin/env bash
#
# lambda_grammar sweep (DDP training + single-GPU eval).
#
# Requires the two env hooks described in chat:
#   train.py    reads RUN_NAME, NUM_EPOCHS, LAMBDA_GRAMMAR
#   evaluate.py reads EVAL_CHECKPOINT, EVAL_REGIME, EVAL_NUM_SAMPLES, EVAL_BATCH_SIZE
#
# Usage:
#   bash sweep_grammar.sh
#
set -euo pipefail

# ---------------- GPUs / DDP ----------------
export CUDA_VISIBLE_DEVICES="GPU-dc48e1f4-b5a9-76e3-488f-64573ffede9b,GPU-fee0dc35-8889-3f90-cd80-238315957336,GPU-c624b4e0-f383-df3a-3cc1-4c3b797558e5"
NPROC=3                                   # must match number of GPUs above

# ---------------- sweep knobs ---------------
LAMBDAS=(0.0 0.001 0.003 0.01 0.03 0.1)       # 0.0 = no-grammar control
EPOCHS=250                                # shorter than full 500 for faster signal
REGIME="epsilon"
EVAL_SAMPLES=2000
EVAL_BATCH=500
CKPT_ROOT="checkpoints"
SUMMARY="results/grammar_sweep_summary.txt"
# --------------------------------------------

mkdir -p "$(dirname "$SUMMARY")"
echo "# grammar sweep $(date -u +%Y-%m-%dT%H:%M:%SZ)  epochs=$EPOCHS regime=$REGIME nsamples=$EVAL_SAMPLES nproc=$NPROC" >> "$SUMMARY"

i=0
for lg in "${LAMBDAS[@]}"; do
    i=$((i + 1))
    run_name="gram_lg${lg}_e${EPOCHS}"
    port=$((29500 + i))                   # distinct rendezvous port per run

    echo ""
    echo "=================================================================="
    echo ">>> TRAIN  lambda_grammar=$lg  run=$run_name  epochs=$EPOCHS  port=$port"
    echo "=================================================================="

    LAMBDA_GRAMMAR="$lg" RUN_NAME="$run_name" NUM_EPOCHS="$EPOCHS" \
        torchrun --nproc_per_node="$NPROC" --nnodes=1 --node_rank=0 \
                 --master_port="$port" train.py

    # latest (highest-numbered) epoch checkpoint for this run
    run_dir="${CKPT_ROOT}/${run_name}"
    ckpt=$(ls -dv "${run_dir}"/*/ 2>/dev/null | tail -1 | sed 's:/*$::' || true)
    if [ -z "${ckpt:-}" ]; then
        echo "!! no checkpoint under ${run_dir}; skipping eval"
        echo "lg=${lg}  NO_CHECKPOINT" >> "$SUMMARY"
        continue
    fi

    echo ""
    echo ">>> EVAL   $ckpt"
    EVAL_CHECKPOINT="$ckpt" \
    EVAL_REGIME="$REGIME" \
    EVAL_NUM_SAMPLES="$EVAL_SAMPLES" \
    EVAL_BATCH_SIZE="$EVAL_BATCH" \
        python3 generate.py

    # collect headline metrics from the per-run report.txt
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
