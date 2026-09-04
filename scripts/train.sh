#!/usr/bin/env bash
#
# Training entrypoint: takes a config path, launches training, tees the log next to
# the run.
#
#   bash scripts/train.sh configs/diffusion_chebi.yaml
#   bash scripts/train.sh configs/diffusion_chebi.yaml --gpus 8
#   bash scripts/train.sh configs/diffusion_chebi.yaml --gpus 8 run_name=bf16 precision=amp_bf16
#   bash scripts/train.sh configs/diffusion_chebi.yaml --dry-run
#
# Everything after the config path that is not a flag of this script is passed to
# scripts/train.py as a config override.
#
# Flags:
#   --gpus N        number of processes (default: all visible GPUs, 1 on CPU)
#   --port P        rendezvous port for torchrun (default: 29500 + a per-run offset)
#   --no-log        do not tee into a log file
#   --dry-run       print the command instead of running it
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

if [ $# -lt 1 ]; then
    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' | sed -n '2,20p'
    exit 1
fi

CONFIG="$1"; shift
if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

GPUS=""
PORT=""
DO_LOG=1
DRY_RUN=0
OVERRIDES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)    GPUS="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --no-log)  DO_LOG=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *)         OVERRIDES+=("$1"); shift ;;
    esac
done

# How many processes: what the user asked, else every visible GPU, else one.
if [ -z "$GPUS" ]; then
    GPUS="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)"
fi

# Ask the config (with the same overrides) where this run stores its files, so the
# log lands next to the checkpoints and the resolved config.
RUN_DIR="$("$PYTHON" - "$ROOT" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"} <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from dimol.config import load_config

cfg = load_config(sys.argv[2], sys.argv[3:])
print(Path(cfg.save_folder or "runs") / str(cfg.run_name))
PY
)"

# One thread per worker process by default: oversubscribing CPU threads slows the
# dataloader down on a node with many ranks.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if [ "$GPUS" -gt 1 ]; then
    if [ -z "$PORT" ]; then
        PORT=$((29500 + $(echo "$RUN_DIR" | cksum | cut -d' ' -f1) % 500))
    fi
    CMD=(torchrun --standalone --nproc_per_node="$GPUS" --master_port="$PORT"
         "$ROOT/scripts/train.py" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
else
    CMD=("$PYTHON" "$ROOT/scripts/train.py" "$CONFIG" ${OVERRIDES[@]+"${OVERRIDES[@]}"})
fi

echo "config:    $CONFIG"
echo "run dir:   $RUN_DIR"
echo "processes: $GPUS"
[ ${#OVERRIDES[@]} -gt 0 ] && echo "overrides: ${OVERRIDES[*]}" || true
echo "command:   ${CMD[*]}"

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

cd "$ROOT"
if [ "$DO_LOG" -eq 1 ]; then
    mkdir -p "$RUN_DIR"
    LOG="$RUN_DIR/train-$(date +%Y%m%d-%H%M%S).log"
    echo "log:       $LOG"
    echo
    "${CMD[@]}" 2>&1 | tee "$LOG"
    exit "${PIPESTATUS[0]}"
fi

echo
exec "${CMD[@]}"
