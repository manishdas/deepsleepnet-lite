#!/usr/bin/env python
"""
Data loading pipeline for SleepStageNet LSTM extension.
Loads pre-processed NPZ files from the DeepSleepNet-Lite Jupyter Notebook pipeline,
creates temporal sequences, and provides PyTorch DataLoaders.

Uses the existing data_split_v1.npz for 20-fold cross-validation.
"""

import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data',
                        'SleepEDF', 'processed', 'eeg_FpzCz_PzOz_v1')
SPLIT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data',
                          'SleepEDF', 'processed', 'data_split_v1.npz')


def load_npz(path):
    """Load a single NPZ file.

    Returns:
        x: (n_epochs, 2, 3000) float32 — channels-first for PyTorch
        y: (n_epochs,) int32 — labels 0-4
    """
    f = np.load(path, allow_pickle=True)
    x = f['x']        # (n_epochs, 3000, 2)
    y = f['y']         # (n_epochs,)
    x = x.transpose(0, 2, 1)  # -> (n_epochs, 2, 3000) channels-first
    return x.astype(np.float32), y.astype(np.int64)


def get_subject_files(data_dir):
    """Map subject index (0-19) to list of NPZ file paths.

    File naming: SC40{subj:02d}{night}E0.npz
    Subject 00 -> SC4001E0.npz, SC4002E0.npz
    Subject 13 -> SC4131E0.npz (only 1 night)
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    subjects = defaultdict(list)
    for f in files:
        subj_id = int(f[3:5])
        subjects[subj_id].append(os.path.join(data_dir, f))
    return dict(subjects)


def load_split(split_path, fold_idx):
    """Load the pre-defined 20-fold CV split.

    Returns:
        train_ids, val_ids, test_ids: arrays of subject indices
    """
    split = np.load(split_path, allow_pickle=True)
    return (split['train_files'][fold_idx],
            split['valid_files'][fold_idx],
            split['test_files'][fold_idx])


def load_subjects_data(subject_ids, subject_files):
    """Load all NPZ files for given subject IDs.

    Returns list of (x, y) tuples, one per recording (night).
    Each x is (n_epochs, 2, 3000), y is (n_epochs,).
    """
    recordings = []
    for sid in subject_ids:
        for fpath in subject_files[sid]:
            x, y = load_npz(fpath)
            recordings.append((x, y))
    return recordings


def create_sequences(x, y, seq_length):
    """Create sliding-window sequences from a single recording.

    Args:
        x: (n_epochs, 2, 3000) — one night's EEG data
        y: (n_epochs,) — labels
        seq_length: int, should be odd (3, 5, 11, 21)

    Returns:
        sequences: (n_seqs, seq_length, 2, 3000)
        labels: (n_seqs,) — center epoch label
    """
    half = seq_length // 2
    n_epochs = len(y)
    if n_epochs < seq_length:
        return np.empty((0, seq_length, x.shape[1], x.shape[2]), dtype=x.dtype), \
               np.empty((0,), dtype=y.dtype)

    sequences = []
    labels = []
    for i in range(half, n_epochs - half):
        seq = x[i - half: i + half + 1]  # (seq_length, 2, 3000)
        sequences.append(seq)
        labels.append(y[i])
    return np.array(sequences), np.array(labels)


def build_sequences_from_recordings(recordings, seq_length):
    """Create sequences from multiple recordings, keeping each recording separate.

    Returns:
        all_sequences: (total_seqs, seq_length, 2, 3000)
        all_labels: (total_seqs,)
    """
    all_seqs = []
    all_labels = []
    for x, y in recordings:
        seqs, labels = create_sequences(x, y, seq_length)
        if len(labels) > 0:
            all_seqs.append(seqs)
            all_labels.append(labels)
    return np.concatenate(all_seqs), np.concatenate(all_labels)


class SleepSequenceDataset(Dataset):
    """PyTorch Dataset for sequences of EEG epochs."""

    def __init__(self, sequences, labels):
        """
        Args:
            sequences: (N, seq_length, 2, 3000) ndarray
            labels: (N,) ndarray
        """
        self.sequences = torch.from_numpy(sequences)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SleepEpochDataset(Dataset):
    """PyTorch Dataset for individual EEG epochs (for CNN pre-training)."""

    def __init__(self, recordings):
        """
        Args:
            recordings: list of (x, y) tuples from load_subjects_data
        """
        xs = [x for x, _ in recordings]
        ys = [y for _, y in recordings]
        self.data = torch.from_numpy(np.concatenate(xs))
        self.labels = torch.from_numpy(np.concatenate(ys))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_fold_data(fold_idx, seq_length, data_dir=None, split_path=None):
    """Load data for a given fold and sequence length.

    Returns:
        train_recordings, val_recordings, test_recordings: lists of (x, y)
        train_seq_dataset, val_seq_dataset, test_seq_dataset: SleepSequenceDataset
        class_weights: tensor of balanced class weights from training set
    """
    data_dir = data_dir or DATA_DIR
    split_path = split_path or SPLIT_PATH

    subject_files = get_subject_files(data_dir)
    train_ids, val_ids, test_ids = load_split(split_path, fold_idx)

    train_recs = load_subjects_data(train_ids, subject_files)
    val_recs = load_subjects_data(val_ids, subject_files)
    test_recs = load_subjects_data(test_ids, subject_files)

    train_seqs, train_labels = build_sequences_from_recordings(train_recs, seq_length)
    val_seqs, val_labels = build_sequences_from_recordings(val_recs, seq_length)
    test_seqs, test_labels = build_sequences_from_recordings(test_recs, seq_length)

    # Compute class weights from training labels
    weights = compute_class_weight('balanced', classes=np.arange(5), y=train_labels)
    class_weights = torch.FloatTensor(weights)

    train_ds = SleepSequenceDataset(train_seqs, train_labels)
    val_ds = SleepSequenceDataset(val_seqs, val_labels)
    test_ds = SleepSequenceDataset(test_seqs, test_labels)

    return (train_recs, val_recs, test_recs,
            train_ds, val_ds, test_ds,
            class_weights)


if __name__ == '__main__':
    print('=' * 60)
    print('SleepStageNet Data Loader Verification')
    print('=' * 60)

    # Check files
    subject_files = get_subject_files(DATA_DIR)
    total_files = sum(len(v) for v in subject_files.values())
    print(f'\nSubjects: {len(subject_files)}, NPZ files: {total_files}')

    # Load one file
    first_file = subject_files[0][0]
    x, y = load_npz(first_file)
    print(f'\nSingle file ({os.path.basename(first_file)}):')
    print(f'  x shape: {x.shape} (epochs, channels, samples)')
    print(f'  y shape: {y.shape}')
    print(f'  Labels: { {STAGE_NAMES[i]: c for i, c in zip(*np.unique(y, return_counts=True))} }')

    # Check fold split
    train_ids, val_ids, test_ids = load_split(SPLIT_PATH, fold_idx=0)
    print(f'\nFold 0 split: train={len(train_ids)} subj, val={len(val_ids)}, test={len(test_ids)}')

    # Build sequences
    seq_length = 5
    print(f'\nBuilding sequences (L={seq_length}), fold 0...')
    _, _, _, train_ds, val_ds, test_ds, class_weights = get_fold_data(0, seq_length)

    print(f'  Train: {len(train_ds)} sequences')
    print(f'  Val:   {len(val_ds)} sequences')
    print(f'  Test:  {len(test_ds)} sequences')
    print(f'  Class weights: {class_weights.numpy().round(3)}')

    # Check a single item
    seq, label = train_ds[0]
    print(f'\n  Single item: seq={seq.shape}, label={label.item()} ({STAGE_NAMES[label]})')

    # DataLoader test
    loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    batch_seq, batch_labels = next(iter(loader))
    print(f'  Batch: seq={batch_seq.shape}, labels={batch_labels.shape}')
    print('\nAll checks passed.')
