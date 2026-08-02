"""
data_process_akira.py  –  copy of data_process.py with two additional functions:

  read_prepared_data_akira_kul(args)
      Reads KUL data from the Dataset_csv/KUL/ layout produced by
      ListenNet/convert_original_aad_datasets.py, instead of the original
      MHANet layout.

      Original layout expected by read_prepared_data():
          <path>/csv/S1No.csv                  (label file)
          <path>/No/S1Tra1.csv  ...            (trial files, named S<N>Tra<K>.csv)

      New layout (Dataset_csv/KUL/):
          <path>/label/S1No.csv                (label file, has a "label" header row)
          <path>/pre_data/1/trial_01.csv  ...  (trial files, subject id as folder name)

      Data facts confirmed from the actual files:
          - 16 subjects, 20 trials total; only the first 8 are the long ~390s
            trials MHANet was designed for (trials 9-20 are short 124s trials).
          - Variable row counts across the first 8 trials (49792–51072 rows).
            Truncated to args.cell_number=46080 (360s × 128Hz), safely below
            the minimum of 49792.
          - Each trial CSV: (n_rows, 66) – col0=trial_id, col1=sample_idx, col2..65=EEG
          - Labels: 1 or 2  (same convention; caller does event_data - 1 → 0/1)
      args values stay identical to original main_KUL():
          args.trail_number = 8
          args.cell_number  = 46080

  read_prepared_data_akira_dtu(args)
      Reads DTU data from the Dataset_csv/DTU/128/ layout (.npy files).

      Original layout expected by get_data_from_mat():
          <path>/S1_data_preproc.mat

      New layout (Dataset_csv/DTU/128/):
          <path>/data/s1_data.npy    shape (60, 3200, 64)
          <path>/label/s1_label.npy  shape (60,)  values 1 or 2

      Returns the same (data_list, target_list) interface as get_data_from_mat().
      Required args changes vs original main_DTU():
          args.trail_number = 60   (unchanged – already correct)
          args.cell_number  = 3200 (unchanged)

20260802    Added read_prepared_data_akira_kul and read_prepared_data_akira_dtu
"""

import math
import numpy as np
import random
from pathlib import Path
from scipy.io import loadmat

# ---------- original imports kept for the unchanged functions ----------
try:
    import pandas as pd
    from sklearn.preprocessing import scale
    from scipy.interpolate import griddata
except ImportError:
    pass  # only needed by the original functions below


# ======================================================================
# NEW: readers for Dataset_csv layout
# ======================================================================

def read_prepared_data_akira_kul(args):
    """
    Drop-in replacement for read_prepared_data() for the KUL dataset stored
    in the Dataset_csv/KUL/ layout.

    Returns
    -------
    data   : list of np.ndarray, each shape (n_samples, 64)
    target : list of int, label per trial (1 or 2; caller subtracts 1)
    """
    subject_id = int(args.name[1:])  # "S1" -> 1
    base = Path(args.data_document_path)

    # --- labels: skip "label" header, take only first trail_number entries ---
    label_path = base / "label" / f"S{subject_id}No.csv"
    label_lines = label_path.read_text().strip().splitlines()
    labels = [int(line.strip()) for line in label_lines[1:args.trail_number + 1]]

    # --- trials: first trail_number files only, truncate rows to cell_number ---
    trial_dir = base / "pre_data" / str(subject_id)
    trial_files = sorted(trial_dir.glob("trial_*.csv"))[:args.trail_number]

    data = []
    target = []
    for k, trial_file in enumerate(trial_files):
        # col0=trial_id, col1=sample_idx, col2..65=EEG (64 channels)
        raw = np.loadtxt(str(trial_file), delimiter=",")
        eeg = raw[:args.cell_number, 2:]  # truncate to cell_number rows, drop index cols
        data.append(eeg)
        target.append(labels[k])

    return data, target


def read_prepared_data_akira_dtu(args):
    """
    Drop-in replacement for get_data_from_mat() for the DTU dataset stored
    in the Dataset_csv/DTU/128/ layout.

    Returns
    -------
    data   : list of np.ndarray, each shape (n_samples, 64)  – one per trial
    target : list of int, label per trial (1 or 2; caller subtracts 1)
    """
    subject_id = int(args.name[1:])  # "S1" -> 1
    base = Path(args.data_document_path)

    data_path  = base / "data"  / f"s{subject_id}_data.npy"
    label_path = base / "label" / f"s{subject_id}_label.npy"

    # data: (n_trials, n_samples, 64)
    eeg_array   = np.load(str(data_path))
    label_array = np.load(str(label_path))  # (n_trials,)  values 1 or 2

    data   = [eeg_array[i] for i in range(eeg_array.shape[0])]
    target = [int(label_array[i]) for i in range(len(label_array))]

    return data, target


# ======================================================================
# ORIGINAL functions – unchanged
# ======================================================================

def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")
        target = []
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2:] #KUL,DTU

            data.append(eeg_data)
            target.append(label.iloc[k, args.label_col])

    return data, target

def get_data_from_mat(mat_path):
      
        mat_eeg_data = []
        mat_event_data = []
        matstruct_contents = loadmat(mat_path)
        matstruct_contents = matstruct_contents['data']
        mat_event = matstruct_contents[0, 0]['event']['eeg'].item()
        mat_event_value = mat_event[0]['value']  # 1*60 1=male, 2=female
        mat_eeg = matstruct_contents[0, 0]['eeg']  # 60 trials 3200*66
        for i in range(mat_eeg.shape[1]):
            mat_eeg_data.append(mat_eeg[0, i])
            mat_event_data.append(mat_event_value[i][0][0])

        return mat_eeg_data, mat_event_data


def sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i+window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows)*0.9)])
        test_eeg.append(np.array(windows)[int(len(windows)*0.9):])
        train_label.append(np.array(new_label)[:int(len(windows)*0.9)])
        test_label.append(np.array(new_label)[int(len(windows)*0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label

def new_sliding_window(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(128 * (1 - args.overlap))

    train_eeg = []
    test_eeg = []
    train_label = []
    test_label = []

    for m in range(len(labels)):
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i+window_size, :]
            windows.append(window)
            new_label.append(label)
        train_eeg.append(np.array(windows)[:int(len(windows)*0.9)])
        test_eeg.append(np.array(windows)[int(len(windows)*0.9):])
        train_label.append(np.array(new_label)[:int(len(windows)*0.9)])
        test_label.append(np.array(new_label)[int(len(windows)*0.9):])

    train_eeg = np.stack(train_eeg, axis=0).reshape(-1, window_size, out_channels)
    test_eeg = np.stack(test_eeg, axis=0).reshape(-1, window_size, out_channels)
    train_label = np.stack(train_label, axis=0).reshape(-1, 1)
    test_label = np.stack(test_label, axis=0).reshape(-1, 1)

    return train_eeg, test_eeg, train_label, test_label

def sliding_window_csp(eeg_datas, labels, args, out_channels):
    window_size = args.window_length
    stride = int(window_size * (1 - args.overlap))

    eeg_set = []
    label_set = []

    for m in range(len(labels)): #labels 0-19
        eeg = eeg_datas[m]
        label = labels[m]
        windows = []
        new_label = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i+window_size, :]
            windows.append(window)
            new_label.append(label)

        eeg_set.append(np.array(windows))
        label_set.append(np.array(new_label))

    eeg_set = np.stack(eeg_set, axis=0).reshape(-1, window_size, out_channels)
    label_set = np.stack(label_set, axis=0).reshape(-1, 1)

    return eeg_set, label_set

def within_data(eeg_datas, labels):
        train_datas = []
        test_datas = []
        train_labels = []
        test_labels = []

        for m in range(len(labels)): #labels 0-19
            eeg = eeg_datas[m]
            label = labels[m]

            train_datas.append(np.array(eeg)[:, :int(eeg.shape[1]*0.9)])
            test_datas.append(np.array(eeg)[:, int(eeg.shape[1]*0.9):])
            train_labels.append(np.array(label))
            test_labels.append(np.array(label))

        train_datas = np.stack(train_datas, axis=0)
        test_datas = np.stack(test_datas, axis=0)
        train_labels = np.stack(train_labels, axis=0)
        test_labels = np.stack(test_labels, axis=0)

        return train_datas, test_datas, train_labels, test_labels
