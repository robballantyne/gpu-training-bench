#!/usr/bin/env python3
"""
compare_results.py — Compare benchmark results across GPU configurations.

Usage:
    python compare_results.py results/              # all JSON files in dir
    python compare_results.py file1.json file2.json  # specific files
    python compare_results.py results/ --cost 3.00 5.50 8.00  # add $/GPU/hr for cost calc
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(paths: list[str]) -> list[dict]:
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("bench_*.json")))
        elif pp.is_file() and pp.suffix == ".json":
            files.append(pp)
    
    results = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
                data["_file"] = f.name
                results.append(data)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"WARNING: Skipping {f} — {e}", file=sys.stderr)
    return results


def print_comparison(results: list[dict], costs: list[float] | None = None):
    if not results:
        print("No results found.")
        return

    # Header
    print()
    print("=" * 90)
    print("  BENCHMARK COMPARISON")
    print("=" * 90)

    # Table header
    col_w = 22
    label_w = 28
    n = len(results)
    
    # Build labels
    labels = []
    for r in results:
        gpu = r["metadata"]["gpu_name"]
        count = r["metadata"]["gpu_count"]
        # Shorten GPU name
        short = gpu.replace("NVIDIA ", "").replace("Tesla ", "")
        labels.append(f"{count}x {short}")

    header = f"{'Metric':<{label_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * (label_w + col_w * n))

    # Rows
    def row(label, values, fmt="{}", highlight_min=False, highlight_max=False):
        formatted = [fmt.format(v) if v is not None else "n/a" for v in values]
        
        # Find best value for highlighting
        numeric = [v for v in values if v is not None and v > 0]
        best = None
        if highlight_min and numeric:
            best = min(numeric)
        elif highlight_max and numeric:
            best = max(numeric)
        
        cells = []
        for v, fv in zip(values, formatted):
            if best is not None and v == best:
                cells.append(f"{'★ ' + fv:>{col_w}}")
            else:
                cells.append(f"{fv:>{col_w}}")
        
        print(f"{label:<{label_w}}" + "".join(cells))

    # Extract metrics
    row("GPU Model",
        [r["metadata"]["gpu_name"].replace("NVIDIA ", "") for r in results],
        fmt="{}")
    row("GPU Count",
        [r["metadata"]["gpu_count"] for r in results],
        fmt="{}")
    row("Effective Batch Size",
        [r["config"]["effective_batch_size"] for r in results],
        fmt="{:,}")
    row("Mixed Precision",
        ["ON" if r["config"]["amp"] else "OFF" for r in results])
    
    print("-" * (label_w + col_w * n))
    
    row("Avg Wall Time / Epoch",
        [r["summary"]["avg_wall_time_per_epoch_s"] for r in results],
        fmt="{:.2f} s", highlight_min=True)
    row("Avg Throughput",
        [r["summary"]["avg_throughput_samples_per_s"] for r in results],
        fmt="{:,.0f} samp/s", highlight_max=True)
    row("Peak GPU Memory",
        [r["summary"]["peak_gpu_memory_mb"] for r in results],
        fmt="{:,.0f} MB")
    row("Avg GPU Utilisation",
        [r["summary"]["avg_gpu_utilisation_pct"] for r in results],
        fmt="{:.1f}%", highlight_max=True)

    # Relative performance (first result = baseline)
    base_throughput = results[0]["summary"]["avg_throughput_samples_per_s"]
    if base_throughput > 0:
        print("-" * (label_w + col_w * n))
        row("Relative Throughput",
            [
                r["summary"]["avg_throughput_samples_per_s"] / base_throughput
                if r["summary"]["avg_throughput_samples_per_s"] > 0 else None
                for r in results
            ],
            fmt="{:.2f}x", highlight_max=True)
        row("Relative Wall Time",
            [
                base_throughput / r["summary"]["avg_throughput_samples_per_s"]
                if r["summary"]["avg_throughput_samples_per_s"] > 0 else None
                for r in results
            ],
            fmt="{:.2f}x", highlight_min=True)

    # Cost analysis if provided
    if costs and len(costs) >= len(results):
        print("-" * (label_w + col_w * n))
        print(f"{'COST ANALYSIS':<{label_w}}" + "".join(f"{'─' * (col_w - 2):>{col_w}}" for _ in results))
        
        cost_per_node = [c * r["metadata"]["gpu_count"] for c, r in zip(costs, results)]
        row("$/GPU/hr (input)",
            costs[:n], fmt="${:.2f}")
        row("$/hr (full node)",
            cost_per_node, fmt="${:.2f}")
        
        # Cost per epoch
        cost_per_epoch = [
            (c * r["summary"]["avg_wall_time_per_epoch_s"] / 3600)
            for c, r in zip(cost_per_node, results)
        ]
        row("Cost per Epoch",
            cost_per_epoch, fmt="${:.3f}", highlight_min=True)

        # Estimate full training cost (all epochs including warmup)
        total_epochs = results[0]["config"]["measured_epochs"]  # assume same for all
        est_total = [c * total_epochs for c in cost_per_epoch]
        row(f"Est. Total ({total_epochs} epochs)",
            est_total, fmt="${:.2f}", highlight_min=True)

    print("=" * (label_w + col_w * n))
    print(f"  ★ = best in row")
    
    # File references
    print()
    print("  Source files:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r['_file']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare GPU benchmark results")
    parser.add_argument("paths", nargs="+", help="JSON files or directories containing them")
    parser.add_argument("--cost", nargs="*", type=float, default=None,
                       help="$/GPU/hr for each result (in order) to calculate cost comparison")
    args = parser.parse_args()

    results = load_results(args.paths)
    if not results:
        print("No benchmark result files found.", file=sys.stderr)
        sys.exit(1)

    print_comparison(results, args.cost)


if __name__ == "__main__":
    main()
