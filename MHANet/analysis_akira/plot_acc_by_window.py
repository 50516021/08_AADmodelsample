"""
Parse accuracy from MHANet result.log, compute mean/SD per window length, and save a figure.

Log format:
    [DATE] - result - INFO - [dataset]_S[subject]_[window]s loss:xxx acc:xxx

Usage:
    python plot_acc_by_window.py [log_path]

Default log_path: ../log/result.log

20260803 for MHANet
"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(os.path.dirname(__file__), "../log/result.log")

def parse_log(log_path):
    """Return {window_label: [acc, ...]} sorted by numeric window value."""
    results = {}
    pattern = re.compile(r"INFO - \w+_S\d+_([\d.]+s) loss:[\d.]+ acc:([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                win, acc = m.group(1), float(m.group(2))
                results.setdefault(win, []).append(acc * 100)
    return dict(sorted(results.items(), key=lambda x: float(x[0].rstrip("s"))))

def plot(results, out_dir, tag):
    labels = list(results.keys())
    means  = [np.mean(v) for v in results.values()]
    sds    = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in results.values()]
    counts = [len(v) for v in results.values()]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    ax.plot(x, means, color="steelblue", alpha=0.8, marker="o", markersize=6, linewidth=2, label="Mean Accuracy")
    ax.errorbar(x, means, yerr=sds, capsize=5, fmt="o", markersize=10, ecolor="black", markeredgecolor="black", color="w")

    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(x[i], m + s + 0.5, f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"({l}, {n})" for l, n in zip(labels, counts)], rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(tag)
    ax.set_ylim(40, 105)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="chance (50%)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"acc_{tag}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    log_path = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else LOG_PATH)
    results = parse_log(log_path)
    if not results:
        print("No results found.")
        sys.exit(1)

    tag = "MHANet"
    out_dir = os.path.join(os.path.dirname(__file__), "figure")

    summary_path = os.path.join(out_dir, f"{tag}_acc_summary.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(summary_path, "w") as o:
        print(f"Results from: {log_path}", file=o)
        print(f"{'Window':<12} {'N':>4} {'Mean':>8} {'SD':>8}", file=o)
        print("-" * 36, file=o)
        for k, v in results.items():
            print(f"{k:<12} {len(v):>4} {np.mean(v):>8.2f} {np.std(v, ddof=1) if len(v)>1 else 0:>8.2f}", file=o)
    print(f"Summary: {summary_path}")

    plot(results, out_dir, tag)
