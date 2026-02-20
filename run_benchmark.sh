#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh — Launch the GPU benchmark
# ============================================================================
# Usage:
#   ./run_benchmark.sh              # All GPUs, default settings
#   ./run_benchmark.sh --single     # Single GPU mode
#   ./run_benchmark.sh --full       # Full ~30GB dataset
#   ./run_benchmark.sh --disk       # Benchmark with disk I/O
#
# All other flags are passed directly to benchmark.py:
#   ./run_benchmark.sh --disk --batch-size 64 --d-model 1024 --nhead 16
#
# Run 'python3 benchmark.py --help' for the full list of benchmark flags.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARK="$SCRIPT_DIR/benchmark.py"
RESULTS_DIR="$SCRIPT_DIR/results"

# Shell-level flags
MODE="multi"
NUM_GPUS=""
DATA_DIR=""
FULL=false

# Collect all benchmark.py arguments (passed through as-is)
BENCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --single)   MODE="single"; NUM_GPUS=1; shift ;;
        --disk)     DATA_DIR="$SCRIPT_DIR/data"; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --full)     FULL=true; shift ;;
        --help)
            echo "Usage: $0 [--single] [--full] [--disk] [--data-dir PATH] [benchmark.py flags...]"
            echo ""
            echo "Shell flags:"
            echo "  --single          Force single-GPU mode"
            echo "  --full            Use ~30GB dataset (7.5M samples)"
            echo "  --disk            Write dataset to disk for I/O benchmarking (uses ./data/)"
            echo "  --data-dir PATH   Like --disk but with a custom path"
            echo ""
            echo "All other flags are passed through to benchmark.py."
            echo "Run 'python3 benchmark.py --help' for the full list."
            exit 0 ;;
        *)
            BENCH_ARGS+=("$1"); shift ;;
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

# Append shell-managed flags to benchmark args
if [ "$FULL" = true ]; then
    BENCH_ARGS+=(--num-samples 7500000)
fi
if [ -n "$DATA_DIR" ]; then
    BENCH_ARGS+=(--data-dir "$DATA_DIR")
fi
BENCH_ARGS+=(--output-dir "$RESULTS_DIR")

# Run
if [ "$MODE" = "single" ] || [ "$NUM_GPUS" -eq 1 ]; then
    echo ">>> Running single-GPU benchmark..."
    python3 "$BENCHMARK" "${BENCH_ARGS[@]}"
else
    echo ">>> Running ${NUM_GPUS}-GPU DDP benchmark via torchrun..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --standalone \
        "$BENCHMARK" "${BENCH_ARGS[@]}"
fi

echo ""
echo ">>> Done. Results saved in: $RESULTS_DIR/"
ls -lh "$RESULTS_DIR"/*.json 2>/dev/null | tail -5
