"""
Summarize per-subject DARNet results written by main_akira.py to log/result.log
(e.g. after a SLURM array job where each task runs one subject).

For each (dataset, time_len) the latest entry per subject is used, so re-running a subject replaces its old result.

Usage (from the DARNet directory):
    .venv/bin/python summarize_results_akira.py
    .venv/bin/python summarize_results_akira.py --dataset KUL --time_len 1
    .venv/bin/python summarize_results_akira.py --since "2026-11-04 12:00"
"""
import argparse
import re
from collections import defaultdict

import numpy as np

# e.g. "2026-11-04 12:00:00,123 - result - INFO - KUL_S1_1.0s loss:0.61 acc:0.83"
LINE_RE = re.compile(r"^(?P<time>\S+ \S+) - result - INFO - "
                     r"(?P<dataset>KUL|DTU)_(?P<loso>LOSO_)?S(?P<subject>\d+)_(?P<time_len>[\d.]+)s "
                     r"loss:(?P<loss>\S+) acc:(?P<acc>\S+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="log/result.log")
    parser.add_argument("--dataset", choices=["KUL", "DTU"], default=None)
    parser.add_argument("--time_len", type=float, default=None)
    parser.add_argument("--since", default=None, help='ignore entries before this time, e.g. "2026-11-04 12:00"')
    args = parser.parse_args()

    # results[(dataset, loso, time_len)][subject] = (time, acc, loss)
    results = defaultdict(dict)
    with open(args.log) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if m is None:
                continue
            if args.since is not None and m["time"] < args.since:
                continue
            key = (m["dataset"], bool(m["loso"]), float(m["time_len"]))
            results[key][int(m["subject"])] = (m["time"], float(m["acc"]), float(m["loss"]))

    keys = sorted(k for k in results
                  if (args.dataset is None or k[0] == args.dataset)
                  and (args.time_len is None or k[2] == args.time_len))
    if not keys:
        print(f"no matching entries in {args.log}")
        return

    for dataset, loso, time_len in keys:
        subjects = results[(dataset, loso, time_len)]
        evaluation = "LOSO" if loso else "subject-dependent"
        print(f"===== {dataset} {time_len}s {evaluation} ({len(subjects)} subjects) =====")
        for s in sorted(subjects):
            t, acc, loss = subjects[s]
            print(f"  S{s:<3d} acc {acc:.4f}  loss {loss:.4f}  ({t})")
        accs = np.array([subjects[s][1] for s in subjects])
        print(f"  mean acc {accs.mean():.4f}  std {accs.std():.4f}")


if __name__ == "__main__":
    main()
