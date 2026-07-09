"""Convert the original AAD datasets into the formats expected by ListenNet.

Output layout:
- KUL:  <out>/KUL/pre_data/<subject_id>/trial_*.csv and <out>/KUL/label/S<subject_id>No.csv
- DTU:  <out>/DTU/128/data/s<subject_id>_data.npy and <out>/DTU/128/label/s<subject_id>_label.npy
- AVED: <out>/AVED/audio-video/sub<subject_id>.csv (or audio-only)

The script avoids pandas so it still works when the local pandas installation
is unavailable or binary-incompatible.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
from scipy.io import loadmat


KUL_EEG_KEEP = 64
DTU_EEG_KEEP = 64


def _is_resource_fork(path: Path) -> bool:
    return path.name.startswith("._") or path.name == ".DS_Store"


def _subject_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer subject id from {path}")
    return int(match.group(1))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv_matrix(path: Path, matrix: np.ndarray) -> None:
    _ensure_dir(path.parent)
    np.savetxt(path, matrix, delimiter=",", fmt="%.10e")


def _write_label_csv(path: Path, labels: list[int]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label"])
        for label in labels:
            writer.writerow([int(label)])


def _mat_strings(value: object) -> list[str]:
    array = np.asarray(value)
    strings: list[str] = []
    for item in array.ravel():
        if isinstance(item, bytes):
            strings.append(item.decode("utf-8", errors="ignore"))
        else:
            strings.append(str(item))
    return strings


def convert_kul(source_root: Path, output_root: Path, overwrite: bool) -> None:
    kul_root = source_root / "KULeuven"
    subject_files = sorted([path for path in kul_root.glob("S*.mat") if not _is_resource_fork(path)])
    if not subject_files:
        raise FileNotFoundError(f"No KUL .mat files found under {kul_root}")

    data_root = output_root / "KUL" / "pre_data"
    label_root = output_root / "KUL" / "label"

    for subject_file in subject_files:
        subject_id = _subject_number(subject_file)
        subject_dir = data_root / str(subject_id)
        label_path = label_root / f"S{subject_id}No.csv"

        if overwrite and subject_dir.exists():
            shutil.rmtree(subject_dir)
        if overwrite and label_path.exists():
            label_path.unlink()

        if subject_dir.exists() and any(subject_dir.iterdir()) and label_path.exists():
            print(f"[KUL] skip subject {subject_id}: already exists")
            continue

        mat = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
        trials = np.asarray(mat["trials"]).ravel()

        label_values: list[int] = []
        _ensure_dir(subject_dir)

        for trial_index, trial in enumerate(trials, start=1):
            eeg = np.asarray(trial.RawData.EegData, dtype=np.float64)
            if eeg.ndim != 2:
                raise ValueError(f"Unexpected KUL EEG shape in {subject_file}, trial {trial_index}: {eeg.shape}")
            if eeg.shape[1] < KUL_EEG_KEEP:
                raise ValueError(f"KUL trial {trial_index} in {subject_file} has only {eeg.shape[1]} channels")

            eeg = eeg[:, :KUL_EEG_KEEP]
            sample_index = np.arange(1, eeg.shape[0] + 1, dtype=np.int64)
            trial_id = np.full((eeg.shape[0], 1), trial_index, dtype=np.int64)
            sample_col = sample_index.reshape(-1, 1)
            export_matrix = np.concatenate([trial_id, sample_col, eeg], axis=1)

            _write_csv_matrix(subject_dir / f"trial_{trial_index:02d}.csv", export_matrix)
            label_values.append(int(trial.attended_track))

        _write_label_csv(label_path, label_values)
        print(f"[KUL] wrote subject {subject_id}: {len(trials)} trials -> {subject_dir}")


def convert_dtu(source_root: Path, output_root: Path, overwrite: bool) -> None:
    dtu_root = source_root / "DTU" / "DATA_preproc"
    subject_files = sorted([path for path in dtu_root.glob("S*_data_preproc.mat") if not _is_resource_fork(path)])
    if not subject_files:
        raise FileNotFoundError(f"No DTU preprocessed .mat files found under {dtu_root}")

    data_root = output_root / "DTU" / "128" / "data"
    label_root = output_root / "DTU" / "128" / "label"

    for subject_file in subject_files:
        subject_id = _subject_number(subject_file)
        data_path = data_root / f"s{subject_id}_data.npy"
        label_path = label_root / f"s{subject_id}_label.npy"

        if not overwrite and data_path.exists() and label_path.exists():
            print(f"[DTU] skip subject {subject_id}: already exists")
            continue

        if overwrite:
            if data_path.exists():
                data_path.unlink()
            if label_path.exists():
                label_path.unlink()

        mat = loadmat(subject_file, squeeze_me=True, struct_as_record=False)
        obj = mat["data"]

        raw_trials = np.asarray(obj.eeg).ravel()
        event_trials = np.asarray(obj.event.eeg).ravel()

        if len(raw_trials) != len(event_trials):
            raise ValueError(
                f"DTU trial count mismatch in {subject_file}: eeg={len(raw_trials)} event={len(event_trials)}"
            )

        channel_names = _mat_strings(obj.dim.chan.eeg[0])
        keep_indices = [index for index, name in enumerate(channel_names) if not name.upper().startswith("EXG")]
        if len(keep_indices) < DTU_EEG_KEEP:
            raise ValueError(f"DTU keep-indices too short in {subject_file}: {len(keep_indices)}")
        keep_indices = keep_indices[:DTU_EEG_KEEP]

        exported_trials = []
        labels = []

        for trial_index, (trial_data, event) in enumerate(zip(raw_trials, event_trials), start=1):
            eeg = np.asarray(trial_data, dtype=np.float64)
            if eeg.ndim != 2:
                raise ValueError(f"Unexpected DTU EEG shape in {subject_file}, trial {trial_index}: {eeg.shape}")
            if eeg.shape[1] < max(keep_indices) + 1:
                raise ValueError(f"DTU trial {trial_index} in {subject_file} has only {eeg.shape[1]} columns")

            eeg = eeg[:, keep_indices]
            exported_trials.append(eeg)
            labels.append(int(event.value))

        data_array = np.stack(exported_trials, axis=0)
        label_array = np.asarray(labels, dtype=np.int64)

        _ensure_dir(data_path.parent)
        _ensure_dir(label_path.parent)
        np.save(data_path, data_array)
        np.save(label_path, label_array)
        print(f"[DTU] wrote subject {subject_id}: {data_array.shape} -> {data_path}")


def convert_aved(source_root: Path, output_root: Path, variant: str, overwrite: bool) -> None:
    if variant not in {"audio-only", "audio-video"}:
        raise ValueError("AVED variant must be 'audio-only' or 'audio-video'")

    aved_root = source_root / "AVED" / "eeg_preproc" / variant
    subject_files = sorted([path for path in aved_root.glob("sub*.csv") if not _is_resource_fork(path)])
    if not subject_files:
        raise FileNotFoundError(f"No AVED CSV files found under {aved_root}")

    target_root = output_root / "AVED" / variant

    for subject_file in subject_files:
        subject_id = _subject_number(subject_file)
        target_path = target_root / f"sub{subject_id}.csv"
        if target_path.exists() and not overwrite:
            print(f"[AVED] skip subject {subject_id}: already exists")
            continue

        _ensure_dir(target_path.parent)
        shutil.copy2(subject_file, target_path)
        print(f"[AVED] copied subject {subject_id}: -> {target_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert original AAD MATLAB datasets into ListenNet-ready files.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("../01_OriginalData/Dataset"),
        help="Path to the original dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("../01_OriginalData/Dataset_csv"),
        help="Destination root for the converted files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["KUL", "DTU", "AVED"],
        choices=["KUL", "DTU", "AVED"],
        help="Datasets to convert.",
    )
    parser.add_argument(
        "--aved-variant",
        choices=["audio-only", "audio-video"],
        default="audio-only",
        help="AVED subfolder to copy into ListenNet format.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted files.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    print(f"Source root: {source_root}")
    print(f"Output root: {output_root}")

    if "KUL" in args.datasets:
        convert_kul(source_root, output_root, args.overwrite)
    if "DTU" in args.datasets:
        convert_dtu(source_root, output_root, args.overwrite)
    if "AVED" in args.datasets:
        convert_aved(source_root, output_root, args.aved_variant, args.overwrite)


if __name__ == "__main__":
    main()