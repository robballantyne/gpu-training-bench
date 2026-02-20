# GPU Benchmark — Encoder-Decoder Transformer

Benchmarks a configurable Transformer encoder-decoder on synthetic data to
compare GPU performance across different hardware configurations.

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- NVIDIA GPUs with nvidia-smi

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

# Full ~30GB dataset (closer to client workload, takes longer)
./run_benchmark.sh --full

# Benchmark with disk I/O (writes sharded dataset to disk, reads during training)
./run_benchmark.sh --disk

# Custom batch size
./run_benchmark.sh --batch-size 256

# Direct torchrun (full control)
torchrun --nproc_per_node=8 benchmark.py --batch-size 128 --epochs 5 --num-samples 7500000
```

## Scaling the Model

The default model is ~3.5M parameters (d_model=128, 4 heads, 3+3 layers, ffn=512).
Use architecture flags to scale up to production-relevant sizes:

```bash
# Default ~3.5M params (quick benchmark)
python benchmark.py

# ~60M params (≈ T5-Small)
python benchmark.py --d-model 512 --nhead 8 --num-encoder-layers 6 --num-decoder-layers 6 --dim-feedforward 2048

# ~220M params (≈ T5-Base)
python benchmark.py --d-model 768 --nhead 12 --num-encoder-layers 12 --num-decoder-layers 12 --dim-feedforward 3072
```

| Flag | Default | Description |
|------|---------|-------------|
| `--d-model` | 128 | Model embedding dimension |
| `--nhead` | 4 | Number of attention heads (must divide d-model) |
| `--num-encoder-layers` | 3 | Transformer encoder layers |
| `--num-decoder-layers` | 3 | Transformer decoder layers |
| `--dim-feedforward` | 512 | FFN inner dimension |

## Data Source: In-Memory vs Disk

By default, synthetic data is generated **on-the-fly in CPU memory** — no disk I/O
is involved. This isolates GPU compute from storage performance.

To include **real disk I/O** in the benchmark (representative of production training
pipelines that read from local/network storage), use `--data-dir`:

```bash
# Writes sharded dataset to disk, then reads from shards during training
./run_benchmark.sh --disk                      # uses ./data/ by default
python benchmark.py --data-dir /mnt/ssd/data   # test specific storage
python benchmark.py --data-dir /mnt/nfs/data   # test network storage

# The dataset is cached — re-runs with the same config skip generation
```

The dataset is split across numbered binary shard files (`shard_0000.bin`,
`shard_0001.bin`, ...), each targeting ~128 MB. This mirrors production training
pipelines that use sharded data formats (WebDataset, TFRecord, Parquet shards).
Use `--num-shards` to override the auto-calculated shard count.

At default settings (seq_len=256, 3.75M samples), the total dataset is ~14 GB
spread across ~114 shards.

## Comparing Results

After running on multiple machines, collect the JSON files from `results/` and compare:

```bash
# Compare all results in the directory
python compare_results.py results/

# With cost analysis (provide $/GPU/hr for each result in order)
python compare_results.py results/ --cost 3.00 5.50 8.00

# Compare specific files
python compare_results.py results/bench_H100_8gpu.json results/bench_H200_8gpu.json --cost 3.00 5.50
```

## What It Measures

| Metric | Description |
|--------|-------------|
| Wall time per epoch | Real clock time to process entire dataset once |
| Throughput | Samples processed per second (across all GPUs) |
| Peak GPU memory | Maximum VRAM used by the process |
| GPU utilisation | Average GPU compute utilisation (via nvidia-smi) |

## Output

Results are saved as JSON in `results/` with the naming pattern:
```
bench_{GPU_MODEL}_{N}gpu_{TIMESTAMP}.json
```

## Notes

- The script auto-detects all available NVIDIA GPUs and uses all of them.
  Use `--single` to force single-GPU mode.
- The default dataset (~14GB) is smaller than the client's 30GB to keep benchmark
  runs quick. Use `--full` for a closer approximation.
- 1 warmup epoch is run before measurement to let CUDA kernels compile and caches warm up.
- Mixed precision (fp16) is on by default — use `--no-amp` to disable.
- The synthetic data uses random integer tokens — this is intentional to isolate
  GPU compute/bandwidth from data loading complexity.
- When using `--data-dir`, the first run generates the sharded dataset (one-time cost).
  Subsequent runs with the same parameters reuse the cached shards.
