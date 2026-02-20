#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh — Launch the GPU benchmark
# ============================================================================
# Usage:
#   ./run_benchmark.sh              # All GPUs (auto-detected), default settings
#   ./run_benchmark.sh --single     # Single GPU mode
#   ./run_benchmark.sh --full       # Full ~30GB dataset (slower)
#   ./run_benchmark.sh --disk       # Write data to disk, benchmark with I/O
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK="$SCRIPT_DIR/benchmark.py"
RESULTS_DIR="$SCRIPT_DIR/results"

# Defaults
NUM_GPUS=""  # auto-detect
BATCH_SIZE=128
SEQ_LEN=256
NUM_SAMPLES=3750000   # ~14GB — fast enough for benchmarking
EPOCHS=3
WARMUP=1
EXTRA_ARGS=""
MODE="multi"
DATA_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)     MODE="single"; NUM_GPUS=1; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --full)       NUM_SAMPLES=7500000; shift ;;  # ~28GB — closer to real 30GB
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --no-amp)     EXTRA_ARGS="$EXTRA_ARGS --no-amp"; shift ;;
        --disk)       DATA_DIR="$SCRIPT_DIR/data"; shift ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--single] [--batch-size N] [--full] [--epochs N] [--no-amp] [--disk] [--data-dir PATH]"
            echo "Extra flags (e.g. --d-model, --nhead, --num-encoder-layers) are passed through to benchmark.py"
            exit 0 ;;
        *)
            # Pass unrecognised flags through to benchmark.py
            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# Detect available GPUs
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Is nvidia-smi available?"
    exit 1
fi

# Use all available GPUs unless --single was specified
if [ -z "$NUM_GPUS" ]; then
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Print system info
echo "============================================================================"
echo "  GPU BENCHMARK LAUNCHER"
echo "============================================================================"
echo "  Host          : $(hostname)"
echo "  Date          : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  GPUs detected : $AVAILABLE_GPUS"
echo "  GPUs to use   : $NUM_GPUS"
nvidia-smi -L 2>/dev/null | head -1 | sed 's/^/  GPU model     : /'
echo "  CUDA version  : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) (driver)"
echo "  PyTorch       : $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "============================================================================"
echo ""

# Ensure output dir
mkdir -p "$RESULTS_DIR"

# Disk dataset flag
if [ -n "$DATA_DIR" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --data-dir $DATA_DIR"
fi

# Run
if [ "$MODE" = "single" ] || [ "$NUM_GPUS" -eq 1 ]; then
    echo ">>> Running single-GPU benchmark..."
    python3 "$BENCHMARK" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --num-samples "$NUM_SAMPLES" \
        --epochs "$EPOCHS" \
        --warmup-epochs "$WARMUP" \
        --output-dir "$RESULTS_DIR" \
        $EXTRA_ARGS
else
    echo ">>> Running ${NUM_GPUS}-GPU DDP benchmark via torchrun..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --standalone \
        "$BENCHMARK" \
        --batch-size "$BATCH_SIZE" \
        --seq-len "$SEQ_LEN" \
        --num-samples "$NUM_SAMPLES" \
        --epochs "$EPOCHS" \
        --warmup-epochs "$WARMUP" \
        --output-dir "$RESULTS_DIR" \
        $EXTRA_ARGS
fi

echo ""
echo ">>> Done. Results saved in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null | tail -5
