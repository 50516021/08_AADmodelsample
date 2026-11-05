"""
Convert the original KULeuven AAD dataset (S1.mat ... S16.mat) into one .npz per subject
for data_loader_akira.get_KUL_data / get_KUL_data_correct.

Output: <out_dir>/S<n>.npz with
    eeg   : float32 (8, 46080, 64)  trials x samples x channels
    label : int64   (8,)            attended ear, L=1 / R=2 (the loader subtracts 1)

Only the 8 non-repetition trials (experiments 1 and 2) are kept, each truncated to 360 s
(46080 samples at 128 Hz) so that all trials have the same length.
Labels are taken from `attended_ear`, NOT `attended_track` (Dataset_csv/KUL/label holds the latter).

Usage (from the DARNet directory):
    .venv/bin/python data_conv_akira/convert_KUL.py
    .venv/bin/python data_conv_akira/convert_KUL.py --reref --bandpass 0.1 50
"""
import argparse
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IN = os.path.join(SCRIPT_DIR, "../../../../01_OriginalData/Dataset/KULeuven")
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "KUL_npz")

FS = 128
N_SUBJECTS = 16
N_TRIALS = 8
TRIAL_LEN = 46080  # 360 s
EAR_TO_LABEL = {"L": 1, "R": 2}


def preprocess(eeg, reref, bandpass):
    eeg = eeg.astype(np.float64)
    if reref:
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
    if bandpass is not None:
        sos = butter(4, bandpass, btype="bandpass", fs=FS, output="sos")
        eeg = sosfiltfilt(sos, eeg, axis=0)
    return eeg


def convert_subject(mat_path, reref, bandpass):
    trials = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["trials"]
    trials = [t for t in trials if t.repetition == 0]
    assert len(trials) == N_TRIALS, f"{mat_path}: expected {N_TRIALS} non-repetition trials, got {len(trials)}"

    eeg, label = [], []
    for t in trials:
        assert t.FileHeader.SampleRate == FS
        x = preprocess(t.RawData.EegData, reref, bandpass)
        assert x.shape[0] >= TRIAL_LEN and x.shape[1] == 64
        eeg.append(x[:TRIAL_LEN].astype(np.float32))
        label.append(EAR_TO_LABEL[t.attended_ear])
    return np.stack(eeg), np.array(label, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default=DEFAULT_IN)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--reref", action="store_true", help="common average re-reference")
    parser.add_argument("--bandpass", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                        help="zero-phase 4th-order Butterworth band-pass, e.g. --bandpass 0.1 50")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for n in range(1, N_SUBJECTS + 1):
        eeg, label = convert_subject(os.path.join(args.in_dir, f"S{n}.mat"), args.reref, args.bandpass)
        np.savez(os.path.join(args.out_dir, f"S{n}.npz"), eeg=eeg, label=label)
        print(f"S{n}: eeg {eeg.shape}, label {label.tolist()}")
    print(f"saved to {os.path.abspath(args.out_dir)} (reref={args.reref}, bandpass={args.bandpass})")


if __name__ == "__main__":
    main()
