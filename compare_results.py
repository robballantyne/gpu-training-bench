#!/usr/bin/env python3
"""
compare_results.py — Compare benchmark results across GPU configurations.

Reads one or more JSON result files produced by benchmark.py and prints a
side-by-side comparison table.  Optionally calculates cost-per-epoch and
total training cost when $/GPU/hr pricing is provided.

Usage:
    python compare_results.py results/                                  # all JSON files in dir
    python compare_results.py file1.json file2.json                     # specific files
    python compare_results.py results/ --cost 3.00 5.50 8.00           # with $/GPU/hr pricing
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(paths: list[str]) -> list[dict]:
    """
    Load benchmark JSON files from the given paths.

    Accepts a mix of directories (scans for bench_*.json) and individual
    JSON files.  Malformed files are skipped with a warning.
    """
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
                data["_file"] = f.name  # Track source filename for reference
                results.append(data)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            print(f"WARNING: Skipping {f} — {e}", file=sys.stderr)
    return results


def print_comparison(results: list[dict], costs: list[float] | None = None):
    """
    Print a formatted comparison table for all loaded results.

    The first result is used as the baseline for relative performance
    calculations.  Best values in each row are marked with a star.
    """
    if not results:
        print("No results found.")
        return

    print()
    print("=" * 90)
    print("  BENCHMARK COMPARISON")
    print("=" * 90)

    # Column layout
    col_w = 22    # Width per result column
    label_w = 28  # Width for the metric label column
    n = len(results)

    # Build short labels from GPU name + count (e.g., "8x RTX PRO 6000")
    labels = []
    for r in results:
        gpu = r["metadata"]["gpu_name"]
        count = r["metadata"]["gpu_count"]
        short = gpu.replace("NVIDIA ", "").replace("Tesla ", "")
        labels.append(f"{count}x {short}")

    header = f"{'Metric':<{label_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    print("-" * (label_w + col_w * n))

    def row(label, values, fmt="{}", highlight_min=False, highlight_max=False):
        """Print one row of the comparison table, optionally highlighting the best value."""
        formatted = [fmt.format(v) if v is not None else "n/a" for v in values]

        # Determine the best value for star highlighting
        numeric = [v for v in values if v is not None and v > 0]
        best = None
        if highlight_min and numeric:
            best = min(numeric)  # Lower is better (e.g., wall time, cost)
        elif highlight_max and numeric:
            best = max(numeric)  # Higher is better (e.g., throughput)

        cells = []
        for v, fv in zip(values, formatted):
            if best is not None and v == best:
                cells.append(f"{'★ ' + fv:>{col_w}}")
            else:
                cells.append(f"{fv:>{col_w}}")

        print(f"{label:<{label_w}}" + "".join(cells))

    # --- Configuration rows ---
    row("GPU Model",
        [r["metadata"]["gpu_name"].replace("NVIDIA ", "") for r in results])
    row("GPU Count",
        [r["metadata"]["gpu_count"] for r in results])
    row("Effective Batch Size",
        [r["config"]["effective_batch_size"] for r in results],
        fmt="{:,}")
    row("Mixed Precision",
        ["ON" if r["config"]["amp"] else "OFF" for r in results])

    print("-" * (label_w + col_w * n))

    # --- Performance rows ---
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

    # --- Relative performance (first result = 1.00x baseline) ---
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

    # --- Cost analysis (optional, requires --cost flag) ---
    if costs and len(costs) >= len(results):
        print("-" * (label_w + col_w * n))
        print(f"{'COST ANALYSIS':<{label_w}}" + "".join(
            f"{'─' * (col_w - 2):>{col_w}}" for _ in results
        ))

        # Calculate hourly cost for the full node (price * GPU count)
        cost_per_node = [c * r["metadata"]["gpu_count"] for c, r in zip(costs, results)]
        row("$/GPU/hr (input)",
            costs[:n], fmt="${:.2f}")
        row("$/hr (full node)",
            cost_per_node, fmt="${:.2f}")

        # Cost per epoch = hourly node cost * epoch duration in hours
        cost_per_epoch = [
            (c * r["summary"]["avg_wall_time_per_epoch_s"] / 3600)
            for c, r in zip(cost_per_node, results)
        ]
        row("Cost per Epoch",
            cost_per_epoch, fmt="${:.3f}", highlight_min=True)

        # Extrapolate total training cost across all measured epochs
        total_epochs = results[0]["config"]["measured_epochs"]
        est_total = [c * total_epochs for c in cost_per_epoch]
        row(f"Est. Total ({total_epochs} epochs)",
            est_total, fmt="${:.2f}", highlight_min=True)

    print("=" * (label_w + col_w * n))
    print(f"  ★ = best in row")

    # List source files for reference
    print()
    print("  Source files:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r['_file']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare GPU benchmark results")
    parser.add_argument("paths", nargs="+",
                        help="JSON result files or directories containing them")
    parser.add_argument("--cost", nargs="*", type=float, default=None,
                        help="$/GPU/hr for each result (in order) for cost comparison")
    args = parser.parse_args()

    results = load_results(args.paths)
    if not results:
        print("No benchmark result files found.", file=sys.stderr)
        sys.exit(1)

    print_comparison(results, args.cost)


if __name__ == "__main__":
    main()
