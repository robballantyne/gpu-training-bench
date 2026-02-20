#!/usr/bin/env python3
"""
GPU Benchmark: Configurable Encoder-Decoder on Synthetic Dataset
=====================================================================
Measures wall time per epoch, throughput (samples/sec), GPU memory usage,
and GPU utilisation across different hardware configurations.

Usage:
    Single GPU:   python benchmark.py
    Multi-GPU:    torchrun --nproc_per_node=8 benchmark.py
    Custom args:  torchrun --nproc_per_node=8 benchmark.py --batch-size 256 --epochs 5
"""

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ---------------------------------------------------------------------------
# Model: ~3M parameter Transformer encoder-decoder
# ---------------------------------------------------------------------------
class EncoderDecoder3M(nn.Module):
    """
    Transformer encoder-decoder sized to ~3M parameters.
    d_model=128, nhead=4, 3 enc layers, 3 dec layers, ffn=512.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        seq_len_src = src.size(1)
        seq_len_tgt = tgt.size(1)
        device = src.device

        pos_src = torch.arange(seq_len_src, device=device).unsqueeze(0)
        pos_tgt = torch.arange(seq_len_tgt, device=device).unsqueeze(0)

        src_emb = self.embedding(src) + self.pos_embedding(pos_src)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(pos_tgt)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len_tgt, device=device)

        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.output_proj(out)


# ---------------------------------------------------------------------------
# Synthetic dataset — generates random token sequences
# ---------------------------------------------------------------------------
class SyntheticSeq2SeqDataset(Dataset):
    """
    Generates a fixed-size synthetic dataset of random token pairs.
    Configured to approximate a 30GB dataset by default.
    Each sample: (src_tokens, tgt_tokens) as int64 tensors.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Deterministic per-index for reproducibility across ranks
        g = torch.Generator().manual_seed(idx)
        src = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
        tgt = torch.randint(0, self.vocab_size, (self.seq_len,), generator=g)
        return src, tgt


class DiskSyntheticSeq2SeqDataset(Dataset):
    """
    Reads pre-generated synthetic data from sharded binary files on disk.

    This exercises real disk I/O during training, unlike SyntheticSeq2SeqDataset
    which generates data on-the-fly in CPU memory.  Data is split across numbered
    shard files (shard_0000.bin, shard_0001.bin, ...), each containing a contiguous
    block of samples.  Each sample is 2 * seq_len int64 values (src then tgt).
    """

    def __init__(self, data_dir: Path, num_samples: int, seq_len: int, num_shards: int):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.sample_bytes = 2 * seq_len * 8  # int64 = 8 bytes
        self.samples_per_shard = math.ceil(num_samples / num_shards)
        self.num_shards = num_shards
        self._fh_cache: dict[int, object] = {}  # shard_idx -> file handle

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        shard_idx = idx // self.samples_per_shard
        offset_in_shard = idx % self.samples_per_shard

        if shard_idx not in self._fh_cache:
            path = self.data_dir / f"shard_{shard_idx:04d}.bin"
            self._fh_cache[shard_idx] = open(path, "rb")

        fh = self._fh_cache[shard_idx]
        fh.seek(offset_in_shard * self.sample_bytes)
        data = fh.read(self.sample_bytes)
        buf = torch.frombuffer(bytearray(data), dtype=torch.int64)
        return buf[: self.seq_len].clone(), buf[self.seq_len :].clone()


def generate_disk_dataset(
    data_dir: Path, num_samples: int, seq_len: int, vocab_size: int, num_shards: int
) -> bool:
    """Write synthetic dataset as numbered binary shard files.

    Each shard (shard_NNNN.bin) holds a contiguous block of samples, where each
    sample is 2 * seq_len int64 values (src then tgt).  A _manifest.json file
    records the generation parameters for cache validation.

    Returns True if generated, False if a matching cached dataset was reused.
    """
    manifest_path = data_dir / "_manifest.json"
    manifest = {
        "num_samples": num_samples,
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "num_shards": num_shards,
    }

    # Reuse existing dataset if manifest matches
    if manifest_path.exists():
        with open(manifest_path) as f:
            if json.load(f) == manifest:
                return False

    data_dir.mkdir(parents=True, exist_ok=True)

    samples_per_shard = math.ceil(num_samples / num_shards)
    chunk_size = 10_000  # samples per write chunk within a shard

    for shard_idx in range(num_shards):
        shard_start = shard_idx * samples_per_shard
        shard_end = min(shard_start + samples_per_shard, num_samples)
        shard_n = shard_end - shard_start

        shard_path = data_dir / f"shard_{shard_idx:04d}.bin"
        with open(shard_path, "wb") as f:
            for offset in range(0, shard_n, chunk_size):
                n = min(chunk_size, shard_n - offset)
                g = torch.Generator().manual_seed(shard_start + offset)
                src = torch.randint(0, vocab_size, (n, seq_len), generator=g)
                tgt = torch.randint(0, vocab_size, (n, seq_len), generator=g)
                combined = torch.stack([src, tgt], dim=1)
                f.write(combined.numpy().tobytes())

        if shard_idx % max(1, num_shards // 20) == 0:
            pct = shard_idx / num_shards * 100
            print(f"    {pct:5.1f}% ({shard_idx}/{num_shards} shards)", flush=True)

    # Write manifest for cache validation
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return True


# ---------------------------------------------------------------------------
# GPU monitoring utilities
# ---------------------------------------------------------------------------
class GPUMonitor:
    """Collects GPU memory and utilisation stats via nvidia-smi and torch."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.peak_mem_mb = 0.0
        self.utilisation_samples = []

    def snapshot(self):
        """Record current GPU stats."""
        # Memory from torch (more accurate for our process)
        mem_mb = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 2)
        self.peak_mem_mb = max(self.peak_mem_mb, mem_mb)

        # Utilisation from nvidia-smi (optional, best-effort)
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device_id}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                util = float(result.stdout.strip().split("\n")[0])
                self.utilisation_samples.append(util)
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass

    def summary(self) -> dict:
        avg_util = (
            sum(self.utilisation_samples) / len(self.utilisation_samples)
            if self.utilisation_samples
            else -1
        )
        return {
            "peak_memory_mb": round(self.peak_mem_mb, 1),
            "avg_gpu_utilisation_pct": round(avg_util, 1),
            "utilisation_samples": len(self.utilisation_samples),
        }


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def setup_distributed():
    """Initialise distributed process group if running under torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        torch.cuda.set_device(0)
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(args):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if is_main(rank):
        print("=" * 70)
        print("  GPU BENCHMARK — Encoder-Decoder Transformer")
        print("=" * 70)
        gpu_name = torch.cuda.get_device_name(local_rank)
        print(f"  GPU           : {gpu_name}")
        print(f"  GPUs in use   : {world_size}")
        print(f"  Batch size    : {args.batch_size} per GPU ({args.batch_size * world_size} effective)")
        print(f"  Seq length    : {args.seq_len}")
        print(f"  Model arch    : d={args.d_model} h={args.nhead} enc={args.num_encoder_layers} dec={args.num_decoder_layers} ffn={args.dim_feedforward}")
        print(f"  Dataset       : {args.num_samples:,} samples (~{estimate_dataset_gb(args):.1f} GB)")
        data_source = f"disk ({args.data_dir}, {args.num_shards} shards)" if args.data_dir else "in-memory (on-the-fly)"
        print(f"  Data source   : {data_source}")
        print(f"  Epochs        : {args.warmup_epochs} warmup + {args.epochs} measured")
        print(f"  Mixed prec    : {'ON (fp16)' if args.amp else 'OFF (fp32)'}")
        print(f"  Num workers   : {args.num_workers}")
        print("=" * 70)

    # --- Model ---
    model = EncoderDecoder3M(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_seq_len=args.seq_len,
    ).to(device)

    if is_main(rank):
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model params  : {param_count:,} ({param_count/1e6:.2f}M)")
        print("=" * 70)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # --- Data ---
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if is_main(rank):
            size_gb = estimate_dataset_gb(args)
            print(f"\n  Preparing disk dataset ({size_gb:.1f} GB, {args.num_shards} shards)...")
            t0 = time.perf_counter()
            generated = generate_disk_dataset(
                data_dir, args.num_samples, args.seq_len, args.vocab_size, args.num_shards
            )
            elapsed = time.perf_counter() - t0
            if generated:
                print(f"  Written to {data_dir} in {elapsed:.1f}s")
            else:
                print(f"  Using cached: {data_dir}")
        if world_size > 1:
            dist.barrier()
        dataset = DiskSyntheticSeq2SeqDataset(
            data_dir, args.num_samples, args.seq_len, args.num_shards
        )
    else:
        dataset = SyntheticSeq2SeqDataset(args.num_samples, args.seq_len, args.vocab_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Optimiser & loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=args.amp)

    # --- GPU monitor ---
    monitor = GPUMonitor(device_id=local_rank)
    torch.cuda.reset_peak_memory_stats(device)

    # --- Training loop ---
    epoch_results = []
    total_epochs = args.warmup_epochs + args.epochs

    for epoch in range(total_epochs):
        is_warmup = epoch < args.warmup_epochs
        label = "warmup" if is_warmup else f"epoch {epoch - args.warmup_epochs + 1}/{args.epochs}"

        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_samples = 0
        epoch_loss = 0.0

        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        for batch_idx, (src, tgt) in enumerate(loader):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=args.amp):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, args.vocab_size), tgt_label.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size_actual = src.size(0)
            epoch_samples += batch_size_actual
            epoch_loss += loss.item() * batch_size_actual

            # Sample GPU stats periodically
            if batch_idx % 50 == 0 and not is_warmup:
                monitor.snapshot()

        torch.cuda.synchronize(device)
        t_end = time.perf_counter()
        wall_time = t_end - t_start

        # Aggregate across ranks
        if world_size > 1:
            samples_tensor = torch.tensor([epoch_samples], device=device)
            dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
            total_samples = samples_tensor.item()
        else:
            total_samples = epoch_samples

        throughput = total_samples / wall_time

        if is_main(rank):
            status = "WARMUP" if is_warmup else "BENCH "
            print(
                f"  [{status}] {label:<16s}  "
                f"wall={wall_time:7.2f}s  "
                f"samples={total_samples:>8,}  "
                f"throughput={throughput:>10,.1f} samples/s  "
                f"loss={epoch_loss / epoch_samples:.4f}"
            )

        if not is_warmup:
            epoch_results.append({
                "epoch": epoch - args.warmup_epochs + 1,
                "wall_time_s": round(wall_time, 3),
                "total_samples": total_samples,
                "throughput_samples_per_s": round(throughput, 1),
                "avg_loss": round(epoch_loss / epoch_samples, 5),
            })

    # --- Final GPU snapshot ---
    monitor.snapshot()
    gpu_stats = monitor.summary()

    # --- Compile results ---
    if is_main(rank):
        avg_wall = sum(r["wall_time_s"] for r in epoch_results) / len(epoch_results)
        avg_throughput = sum(r["throughput_samples_per_s"] for r in epoch_results) / len(epoch_results)

        results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hostname": socket.gethostname(),
                "gpu_name": torch.cuda.get_device_name(local_rank),
                "gpu_count": world_size,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "config": {
                "model_params": sum(p.numel() for p in (model.module if hasattr(model, 'module') else model).parameters()),
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "dim_feedforward": args.dim_feedforward,
                "batch_size_per_gpu": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
                "seq_len": args.seq_len,
                "vocab_size": args.vocab_size,
                "num_samples": args.num_samples,
                "dataset_size_gb": round(estimate_dataset_gb(args), 2),
                "data_source": "disk" if args.data_dir else "memory",
                "num_shards": args.num_shards if args.data_dir else None,
                "amp": args.amp,
                "num_workers": args.num_workers,
                "warmup_epochs": args.warmup_epochs,
                "measured_epochs": args.epochs,
            },
            "summary": {
                "avg_wall_time_per_epoch_s": round(avg_wall, 3),
                "avg_throughput_samples_per_s": round(avg_throughput, 1),
                "peak_gpu_memory_mb": gpu_stats["peak_memory_mb"],
                "avg_gpu_utilisation_pct": gpu_stats["avg_gpu_utilisation_pct"],
            },
            "epochs": epoch_results,
        }

        # Save results
        gpu_tag = torch.cuda.get_device_name(local_rank).replace(" ", "_")
        outfile = Path(args.output_dir) / f"bench_{gpu_tag}_{world_size}gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)

        print()
        print("=" * 70)
        print("  RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Avg wall time / epoch : {avg_wall:.2f} s")
        print(f"  Avg throughput        : {avg_throughput:,.1f} samples/s")
        print(f"  Peak GPU memory       : {gpu_stats['peak_memory_mb']:,.1f} MB")
        print(f"  Avg GPU utilisation   : {gpu_stats['avg_gpu_utilisation_pct']:.1f}%")
        print(f"  Results saved to      : {outfile}")
        print("=" * 70)

    cleanup_distributed()


def estimate_dataset_gb(args) -> float:
    """Estimate dataset size: 2 tensors × seq_len × int64 (8 bytes) per sample."""
    bytes_per_sample = 2 * args.seq_len * 8
    return (args.num_samples * bytes_per_sample) / (1024 ** 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GPU Benchmark: Encoder-Decoder Transformer")

    # Training
    p.add_argument("--batch-size", type=int, default=128, help="Batch size per GPU (default: 128)")
    p.add_argument("--seq-len", type=int, default=256, help="Sequence length (default: 256)")
    p.add_argument("--vocab-size", type=int, default=8192, help="Vocabulary size (default: 8192)")
    p.add_argument("--num-samples", type=int, default=3_750_000,
                   help="Number of samples in synthetic dataset (default: 3.75M ≈ 14GB at seq_len=256)")
    p.add_argument("--epochs", type=int, default=3, help="Number of measured epochs (default: 3)")
    p.add_argument("--warmup-epochs", type=int, default=1, help="Warmup epochs before measurement (default: 1)")
    p.add_argument("--amp", action="store_true", default=True, help="Enable mixed precision (default: on)")
    p.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per GPU (default: 4)")

    # Model architecture (scale the transformer up or down)
    p.add_argument("--d-model", type=int, default=128, help="Model embedding dimension (default: 128)")
    p.add_argument("--nhead", type=int, default=4, help="Number of attention heads (default: 4)")
    p.add_argument("--num-encoder-layers", type=int, default=3, help="Encoder layers (default: 3)")
    p.add_argument("--num-decoder-layers", type=int, default=3, help="Decoder layers (default: 3)")
    p.add_argument("--dim-feedforward", type=int, default=512, help="FFN inner dimension (default: 512)")

    # Data source
    p.add_argument("--data-dir", type=str, default=None,
                   help="Write synthetic data to disk and read during training (exercises I/O). "
                        "Omit for in-memory generation (default: in-memory)")
    p.add_argument("--num-shards", type=int, default=None,
                   help="Number of shard files when using --data-dir (default: auto, ~128 MB per shard)")

    # Output
    p.add_argument("--output-dir", type=str, default="./results", help="Directory for result JSON files")

    return p.parse_args()


def validate_args(args):
    """Validate CLI arguments before running the benchmark."""
    for name, val in [
        ("--batch-size", args.batch_size),
        ("--seq-len", args.seq_len),
        ("--epochs", args.epochs),
        ("--num-samples", args.num_samples),
        ("--d-model", args.d_model),
        ("--nhead", args.nhead),
        ("--num-encoder-layers", args.num_encoder_layers),
        ("--num-decoder-layers", args.num_decoder_layers),
        ("--dim-feedforward", args.dim_feedforward),
    ]:
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    if args.d_model % args.nhead != 0:
        raise ValueError(
            f"--d-model ({args.d_model}) must be divisible by --nhead ({args.nhead})"
        )


if __name__ == "__main__":
    args = parse_args()

    # Auto-calculate shard count: target ~128 MB per shard
    if args.data_dir and args.num_shards is None:
        target_shard_bytes = 128 * 1024 * 1024
        sample_bytes = 2 * args.seq_len * 8
        samples_per_shard = max(1, target_shard_bytes // sample_bytes)
        args.num_shards = max(1, math.ceil(args.num_samples / samples_per_shard))

    validate_args(args)
    run_benchmark(args)
