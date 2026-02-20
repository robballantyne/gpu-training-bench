# GPU Benchmark — Encoder-Decoder Transformer

Benchmarks a configurable Transformer encoder-decoder on synthetic data to
compare GPU performance across different hardware configurations.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPUs with `nvidia-smi`

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
chmod +x run_benchmark.sh

# All GPUs auto-detected, default ~3.5M param model, in-memory data
./run_benchmark.sh

# Single GPU only
./run_benchmark.sh --single

# Full ~30GB dataset
./run_benchmark.sh --full

# Include disk I/O in the benchmark (writes sharded dataset, reads during training)
./run_benchmark.sh --disk

# Custom batch size and model architecture
./run_benchmark.sh --disk --batch-size 64 --d-model 1024 --nhead 16

# Direct torchrun (bypasses shell script for full control)
torchrun --nproc_per_node=8 benchmark.py --batch-size 128 --epochs 5
```

All flags not handled by `run_benchmark.sh` are passed directly to
`benchmark.py`. Run `python3 benchmark.py --help` for the full list.

## Scaling the Model

The default model is ~3.5M parameters. Use architecture flags to scale up:

```bash
# Default ~3.5M params (quick sanity check)
./run_benchmark.sh

# ~60M params
./run_benchmark.sh --d-model 512 --nhead 8 --num-encoder-layers 6 --num-decoder-layers 6 --dim-feedforward 2048

# ~220M params
./run_benchmark.sh --d-model 768 --nhead 12 --num-encoder-layers 12 --num-decoder-layers 12 --dim-feedforward 3072

# ~2.9B params (requires significant VRAM — calibrate batch size to your GPU)
./run_benchmark.sh --disk --batch-size 8 --seq-len 512 \
  --d-model 2048 --nhead 32 \
  --num-encoder-layers 24 --num-decoder-layers 24 \
  --dim-feedforward 8192
```

| Flag | Default | Description |
|------|---------|-------------|
| `--d-model` | 128 | Model embedding dimension |
| `--nhead` | 4 | Number of attention heads (must divide d-model) |
| `--num-encoder-layers` | 3 | Transformer encoder layers |
| `--num-decoder-layers` | 3 | Transformer decoder layers |
| `--dim-feedforward` | 512 | FFN inner dimension |

## Data Source: In-Memory vs Disk

By default, synthetic data is generated **on the fly in CPU memory** — no disk
I/O is involved. This isolates GPU compute from storage performance.

To include **real disk I/O** in the benchmark (representative of production
training pipelines that read from local or network storage), use `--disk` or
`--data-dir`:

```bash
# Writes sharded dataset to ./data/, then reads from shards during training
./run_benchmark.sh --disk

# Test a specific storage path (e.g., NVMe, NFS, or network mount)
python benchmark.py --data-dir /mnt/ssd/data
python benchmark.py --data-dir /mnt/nfs/data
```

The dataset is split across numbered binary shard files (`shard_0000.bin`,
`shard_0001.bin`, ...), each targeting ~128 MB. This mirrors production
training pipelines that use sharded data formats (WebDataset, TFRecord,
Parquet shards). Use `--num-shards` to override the auto-calculated count.

The dataset is cached on disk — re-runs with the same parameters skip
generation automatically.

## Comparing Results

After running on multiple machines, collect the JSON files from `results/`
and compare:

```bash
# Compare all results in the directory
python compare_results.py results/

# With cost analysis (provide $/GPU/hr for each result, in order)
python compare_results.py results/ --cost 3.00 5.50 8.00

# Compare specific files
python compare_results.py results/bench_H100_8gpu.json results/bench_H200_8gpu.json --cost 3.00 5.50
```

## What It Measures

| Metric | Description |
|--------|-------------|
| Wall time per epoch | Real clock time to process the entire dataset once |
| Throughput | Samples processed per second (aggregated across all GPUs) |
| Peak GPU memory | Maximum VRAM allocated by the training process |
| GPU utilisation | Average GPU compute utilisation (sampled via nvidia-smi) |

## Output

Results are saved as JSON in `results/` with the naming pattern:

```
bench_{GPU_MODEL}_{N}gpu_{TIMESTAMP}.json
```

Each file contains full metadata (hardware, software versions), the exact
configuration used, per-epoch metrics, and an overall summary.

## Notes

- **GPU auto-detection**: All available NVIDIA GPUs are detected and used
  automatically. Pass `--single` to force single-GPU mode.
- **Warmup**: 1 warmup epoch runs before measurement to let CUDA kernels
  JIT-compile and caches warm up. Adjust with `--warmup-epochs`.
- **Mixed precision**: fp16 is on by default for realistic performance
  numbers. Use `--no-amp` to benchmark in full fp32.
- **Synthetic tokens**: The dataset uses random integer tokens. This is
  intentional — it isolates GPU compute and memory bandwidth from data
  preprocessing complexity.
- **Fair multi-GPU comparison**: When comparing different GPU counts (e.g.,
  8x GPU-A vs 4x GPU-B), scale `--batch-size` proportionally so the
  effective batch size (per-GPU batch * GPU count) stays the same.
- **VRAM calibration**: Start with a small batch size and increase until
  you reach your target VRAM utilisation. Run a quick 1-epoch test
  (`--epochs 1 --warmup-epochs 0`) to check before committing to a full run.
