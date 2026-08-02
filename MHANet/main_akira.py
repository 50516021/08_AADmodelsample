"""
main_akira.py  –  copy of main.py adapted for the Dataset_csv/ layout.

Changes vs main.py
------------------
1. Import: `from data_process_akira import *` instead of `from data_process import *`
   so the new reader functions are available.

2. main_KUL():
   - data_document_path default changed to
       "../../01_OriginalData/Dataset_csv/KUL"
   - args.trail_number  : 8  → 20   (Dataset_csv has 20 trials per subject)
   - args.cell_number   : 46080 → 49792  (actual row count per trial CSV)
   - data loading: read_prepared_data(args)  →  read_prepared_data_akira_kul(args)
   - eeg_data stacking: np.vstack then reshape replaced by np.stack directly,
     because read_prepared_data_akira_kul already returns one ndarray per trial
     (no pandas DataFrame wrapping).

3. main_DTU():
   - data_document_path default changed to
       "../../01_OriginalData/Dataset_csv/DTU/128"
   - data loading block replaced: get_data_from_mat() + manual .mat parsing
     → read_prepared_data_akira_dtu(args)
   - The eeg_data[:, :, 0:64] slice is removed because the .npy files already
     contain exactly 64 channels.
   - Stacking logic same as main_KUL change above.

4. config_akira is imported instead of config so the paths can be set
   independently (see config_akira.py).

Everything else (CustomDatasets, initiate, train_model, __main__ block)
is identical to main.py.

------
[execution example]
- Run all KUL subjects with 1s window:
python main_akira.py --dataset KUL --time_len 1

- Run all DTU subjects with 2s window:
python main_akira.py --dataset DTU --time_len 2 --data_path ../../01_OriginalData/Dataset_csv/DTU/128 --people_number 18

- Run a single subject (S3) only:
python main_akira.py --dataset KUL --subject 3 --time_len 1

- Override the data path (e.g. on a different machine):
python main_akira.py --dataset KUL --data_path /data/Dataset_csv/KUL --time_len 2

20260721    Created
20260802    commandline option, path adoptation 
"""

from dotmap import DotMap
from tqdm import tqdm
from utils import *
from data_process_akira import *          # CHANGED: use akira variant
from model import MHANet
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from mne.decoding import CSP
import numpy as np
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import argparse
import config_akira as config             # CHANGED: use akira config; overridden by CLI args

np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CustomDatasets(Dataset):
    def __init__(self, seq_data, label_data):
        self.seq_data = seq_data
        self.label = label_data

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])
        label = torch.Tensor(self.label[index])
        return seq_data, label


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initiate(args, train_loader, valid_loader, test_loader, subject):
    model = MHANet(args)
    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=0.005, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.003 / 10)
    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model, 'optimizer': optimizer, 'criterion': criterion, 'scheduler': scheduler}
    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)


def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, scheduler):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        for i_batch, batch_data in enumerate(train_loader):
            seq_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data, train_label = seq_data.cuda(), train_label.cuda()
            batch_size = train_label.size(0)
            preds = model(seq_data)
            loss = criterion(preds, train_label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_size
            predicted = preds.data.max(1)[1]
            train_acc_sum += predicted.eq(train_label).cpu().sum()
        scheduler.step()
        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        num_batches = len(loader)
        total_loss = 0.0
        test_acc_sum = 0
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                seq_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                seq_data, test_label = seq_data.cuda(), test_label.cuda()
                preds = model(seq_data)
                optimizer.zero_grad()
                total_loss += criterion(preds, test_label.long()).item() * args.batch_size
                predicted = preds.data.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()
        avg_loss = total_loss / (num_batches * args.batch_size)
        avg_acc = test_acc_sum / (num_batches * args.batch_size)
        return avg_loss, avg_acc

    best_epoch = 1
    best_valid = float('inf')
    epochs_without_improvement = 0
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)
        print()
        print('Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f}'.format(
            epoch, args.name, train_loss, train_acc, val_loss, val_acc))
        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0
            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                break

    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")
    return test_loss, test_acc


# -----------------------------------------------------------------------
# CHANGED: main_KUL – reads from Dataset_csv/KUL/
# -----------------------------------------------------------------------
def main_KUL(name="S1", dataset="KUL",
             data_document_path="../../01_OriginalData/Dataset_csv/KUL",  # CHANGED
             time_len=1):
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 100
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8        # unchanged: reader loads only first 8 trials
    args.cell_number = 46080     # unchanged: reader truncates each trial to this
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 64
    args.log_path = "./result"
    args.frequency_resolution = args.fs / args.window_length
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)

    logger = get_logger(args.name, args.log_path, time_len)

    # CHANGED: use new reader instead of read_prepared_data(args)
    eeg_data, event_data = read_prepared_data_akira_kul(args)

    # CHANGED: each element is already an ndarray; stack directly
    eeg_data = np.stack(eeg_data, axis=0)                          # (8, 46080, 64)
    eeg_data = eeg_data.transpose(0, 2, 1)                         # (20, 64, 49792)
    event_data = np.array(event_data) - 1                          # 0/1

    train_data, test_data, train_label, test_label = within_data(eeg_data, event_data)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat',
              transform_into='csp_space', norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)
    train_eeg, train_label = sliding_window_csp(train_data, train_label, args, args.csp_comp)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args, args.csp_comp)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    del eeg_data

    np.random.seed(200); np.random.shuffle(seq_train_data)
    np.random.seed(200); np.random.shuffle(train_label)
    np.random.seed(200); np.random.shuffle(seq_test_data)
    np.random.seed(200); np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(
        seq_train_data, train_label, test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)
    print(loss, acc)
    logger.info(loss)
    logger.info(acc)
    return acc


result_logger = logging.getLogger('result')
result_logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------
# CHANGED: main_DTU – reads from Dataset_csv/DTU/128/
# -----------------------------------------------------------------------
def main_DTU(name="S1", dataset="DTU",
             data_document_path="../../01_OriginalData/Dataset_csv/DTU/128",  # CHANGED
             time_len=1):
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * time_len)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 20
    args.patience = 15
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 18
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 60
    args.cell_number = 3200
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.csp_comp = 64
    args.log_path = "./result"
    args.frequency_resolution = args.fs / args.window_length
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)

    logger = get_logger(args.name, args.log_path, time_len)

    # CHANGED: use new reader instead of get_data_from_mat() + manual .mat parsing
    # The .npy files already contain exactly 64 channels, so no [:, :, 0:64] slice needed.
    eeg_data, event_data = read_prepared_data_akira_dtu(args)

    # CHANGED: each element is already an ndarray; stack directly
    eeg_data = np.stack(eeg_data, axis=0)                          # (60, 3200, 64)
    eeg_data = eeg_data.transpose(0, 2, 1)                         # (60, 64, 3200)
    event_data = np.array(event_data) - 1                          # 0/1

    train_data, test_data, train_label, test_label = within_data(eeg_data, event_data)

    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat',
              transform_into='csp_space', norm_trace=True)
    train_data = csp.fit_transform(train_data, train_label)
    test_data = csp.transform(test_data)

    train_data = train_data.transpose(0, 2, 1)
    test_data = test_data.transpose(0, 2, 1)
    train_eeg, train_label = sliding_window_csp(train_data, train_label, args, args.csp_comp)
    test_eeg, test_label = sliding_window_csp(test_data, test_label, args, args.csp_comp)

    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    del eeg_data

    np.random.seed(200); np.random.shuffle(seq_train_data)
    np.random.seed(200); np.random.shuffle(train_label)
    np.random.seed(200); np.random.shuffle(seq_test_data)
    np.random.seed(200); np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, train_label, valid_label = train_test_split(
        seq_train_data, train_label, test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)

    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, train_label),
                              batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, valid_label),
                              batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, test_label),
                             batch_size=args.batch_size, drop_last=True)

    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)
    print(loss, acc)
    logger.info(loss)
    logger.info(acc)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHANet AAD training (akira variant)")
    parser.add_argument("--dataset",  default=config.dataset,
                        choices=["KUL", "DTU"], help="Dataset to use (default: %(default)s)")
    parser.add_argument("--data_path", default=config.data_document_path,
                        help="Path to dataset root (default: %(default)s)")
    parser.add_argument("--time_len",  type=float, default=config.time_len,
                        help="Window length in seconds (default: %(default)s)")
    parser.add_argument("--people_number", type=int, default=config.people_number,
                        help="Number of subjects to run (default: %(default)s)")
    parser.add_argument("--subject",   type=int, default=None,
                        help="Run a single subject by number, e.g. 1 for S1 (overrides --people_number)")
    cli = parser.parse_args()

    file_handler = logging.FileHandler('log/result.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    result_logger.addHandler(file_handler)

    subject_range = [cli.subject] if cli.subject else range(1, cli.people_number + 1)
    all_test_acc = []
    main_fn = main_KUL if cli.dataset == "KUL" else main_DTU

    for i in subject_range:
        acc = main_fn(name=f'S{i}', dataset=cli.dataset,
                      data_document_path=cli.data_path, time_len=cli.time_len)
        all_test_acc.append(acc)

    print(f'avg_acc: {np.mean(all_test_acc):.4f}')
    info_msg = (f'The average accuracy of {cli.dataset}_{cli.time_len}s '
                f'avg_acc:{np.mean(all_test_acc):.4f} std:{np.std(all_test_acc):.4f}')
    result_logger.info(info_msg)
