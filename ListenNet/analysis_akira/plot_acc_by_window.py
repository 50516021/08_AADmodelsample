"""
Parse accuracy from log files under a given result folder,
compute mean/SD per window-length subfolder, and save a figure.

Usage:
    python plot_acc_by_window.py <folder_path>

Example:
    python plot_acc_by_window.py ../dep/writer/Subject_dependent_AAD/DTU
    
    
20260715 for ListenNet test

"""

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_acc(log_path):
    """Return the final test accuracy from the last line of a log file."""
    with open(log_path, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        m = re.search(r"Test acc[:\s]+([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None

def collect(folder):
    """Return {window_label: [acc, ...]} sorted by numeric window value."""
    results = {}
    for entry in os.scandir(folder):
        if not entry.is_dir():
            continue
        result_dir = os.path.join(entry.path, "result")
        if not os.path.isdir(result_dir):
            continue
        accs = []
        for f in os.scandir(result_dir):
            if f.name.endswith(".log"):
                acc = parse_acc(f.path)
                if acc is not None:
                    accs.append(acc)
        if accs:
            results[entry.name] = accs
    # sort by numeric part of folder name (e.g. "0.1s" -> 0.1)
    return dict(sorted(results.items(), key=lambda x: float(re.sub(r"[^\d.]", "", x[0]) or 0)))

def plot(folder, results, out_dir):
    labels = list(results.keys())
    means  = [np.mean(v) for v in results.values()]
    sds    = [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in results.values()]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sds, capsize=5, color="steelblue", alpha=0.8, error_kw={"elinewidth": 1.5})

    # annotate mean ± SD on each bar
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(x[i], m + s + 0.5, f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(os.path.basename(folder.rstrip("/\\")))
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="chance (50%)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(folder.rstrip("/\\"))
    out_path = os.path.join(out_dir, f"acc_{tag}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    folder = os.path.abspath(folder)

    out_dir = os.path.join(os.path.dirname(__file__)) + '/figure/' + folder.split("/")[-1]  # analysis_akira/figure/<folder_name>

    results = collect(folder)
    if not results:
        print("No log files found.")
        sys.exit(1)

    print(f"\nResults for: {folder}")
    print(f"{'Window':<12} {'N':>4} {'Mean':>8} {'SD':>8}")
    print("-" * 36)
    for k, v in results.items():
        print(f"{k:<12} {len(v):>4} {np.mean(v):>8.2f} {np.std(v, ddof=1) if len(v)>1 else 0:>8.2f}")

    plot(folder, results, out_dir)
