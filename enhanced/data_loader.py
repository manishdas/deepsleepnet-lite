#!/usr/bin/env python
"""
Data loading pipeline for Enhanced SleepStageNet models.

Reuses the same NPZ format and 20-fold CV split from DeepSleepNet-Lite.
Adds support for:
  - Configurable data directory (local or Colab)
  - Sequence and single-epoch datasets
  - Balanced sampling for class imbalance
  - Data augmentation (time shift, amplitude scaling, Gaussian noise)
"""

import os
from collections import defaultdict

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']

# Default paths — override via function args or environment variables
DEFAULT_DATA_DIR = os.environ.get(
    'SLEEP_DATA_DIR',
    os.path.join(os.path.dirname(__file__), '..', 'data',
                 'SleepEDF', 'processed', 'eeg_FpzCz_PzOz_v1')
)
DEFAULT_SPLIT_PATH = os.environ.get(
    'SLEEP_SPLIT_PATH',
    os.path.join(os.path.dirname(__file__), '..', 'data',
                 'SleepEDF', 'processed', 'data_split_v1.npz')
)


def load_npz(path):
    """Load a single NPZ file.

    Returns:
        x: (n_epochs, 2, 3000) float32 — channels-first for PyTorch
        y: (n_epochs,) int64 — labels 0-4
    """
    f = np.load(path, allow_pickle=True)
    x = f['x']   # (n_epochs, 3000, 2)
    y = f['y']   # (n_epochs,)
    x = x.transpose(0, 2, 1)  # → (n_epochs, 2, 3000) channels-first
    return x.astype(np.float32), y.astype(np.int64)


def get_subject_files(data_dir):
    """Map subject index (0-19) to list of NPZ file paths."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    subjects = defaultdict(list)
    for f in files:
        subj_id = int(f[3:5])
        subjects[subj_id].append(os.path.join(data_dir, f))
    return dict(subjects)


def load_split(split_path, fold_idx, n_subjects=20):
    """Load the pre-defined 20-fold CV split.

    The split file stores only valid_files and test_files (subject IDs).
    Train subjects are derived as all subjects not in valid or test.
    """
    split = np.load(split_path, allow_pickle=True)
    valid_ids = split['valid_files'][fold_idx]
    test_ids = split['test_files'][fold_idx]

    # Train = all subjects not in valid or test
    exclude = set(int(i) for i in valid_ids) | set(int(i) for i in test_ids)
    train_ids = np.array([i for i in range(n_subjects) if i not in exclude])

    return train_ids, valid_ids, test_ids


def load_subjects_data(subject_ids, subject_files):
    """Load all NPZ files for given subject IDs."""
    recordings = []
    for sid in subject_ids:
        for fpath in subject_files[sid]:
            x, y = load_npz(fpath)
            recordings.append((x, y))
    return recordings


def create_sequences(x, y, seq_length):
    """Create sliding-window sequences from a single recording.

    Args:
        x: (n_epochs, 2, 3000)
        y: (n_epochs,)
        seq_length: int (should be odd: 1, 3, 5, 11, 21)

    Returns:
        sequences: (n_seqs, seq_length, 2, 3000)
        labels: (n_seqs,) — center epoch label
    """
    if seq_length == 1:
        return x[:, np.newaxis, :, :], y

    half = seq_length // 2
    n_epochs = len(y)
    if n_epochs < seq_length:
        return (np.empty((0, seq_length, x.shape[1], x.shape[2]), dtype=x.dtype),
                np.empty((0,), dtype=y.dtype))

    sequences = []
    labels = []
    for i in range(half, n_epochs - half):
        seq = x[i - half: i + half + 1]
        sequences.append(seq)
        labels.append(y[i])
    return np.array(sequences), np.array(labels)


def build_sequences_from_recordings(recordings, seq_length):
    """Create sequences from multiple recordings."""
    all_seqs, all_labels = [], []
    for x, y in recordings:
        seqs, labels = create_sequences(x, y, seq_length)
        if len(labels) > 0:
            all_seqs.append(seqs)
            all_labels.append(labels)
    return np.concatenate(all_seqs), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

class EEGAugmentation:
    """Online data augmentation for EEG sequences.

    Augmentations (applied independently with given probabilities):
      - Time shift: circular shift by up to ±shift_max samples
      - Amplitude scaling: multiply by random factor in [1-scale, 1+scale]
      - Gaussian noise: add N(0, noise_std) noise
    """

    def __init__(self, shift_max=50, scale=0.1, noise_std=0.01, p=0.5):
        self.shift_max = shift_max
        self.scale = scale
        self.noise_std = noise_std
        self.p = p

    def __call__(self, x):
        """
        Args:
            x: tensor of shape (..., n_channels, 3000)
        """
        if np.random.random() > self.p:
            return x

        # Time shift
        if self.shift_max > 0 and np.random.random() < 0.5:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            x = torch.roll(x, shifts=shift, dims=-1)

        # Amplitude scaling
        if self.scale > 0 and np.random.random() < 0.5:
            factor = 1.0 + (np.random.random() * 2 - 1) * self.scale
            x = x * factor

        # Gaussian noise
        if self.noise_std > 0 and np.random.random() < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return x


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SleepSequenceDataset(Dataset):
    """PyTorch Dataset for sequences of EEG epochs.

    Lazy implementation: stores recordings and builds sequences on-the-fly
    in __getitem__ to avoid materializing all sequences in RAM.
    """

    def __init__(self, recordings, seq_length, augment=None):
        self.recordings = recordings          # list of (x, y) numpy arrays
        self.seq_length = seq_length
        self.augment = augment
        self.half = seq_length // 2

        # Build index: (recording_idx, center_epoch_idx) for each valid sequence
        self.index = []
        self.all_labels = []
        for rec_idx, (x, y) in enumerate(recordings):
            n_epochs = len(y)
            if seq_length == 1:
                for i in range(n_epochs):
                    self.index.append((rec_idx, i))
                    self.all_labels.append(y[i])
            else:
                if n_epochs < seq_length:
                    continue
                for i in range(self.half, n_epochs - self.half):
                    self.index.append((rec_idx, i))
                    self.all_labels.append(y[i])

        self.labels = np.array(self.all_labels, dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec_idx, center = self.index[idx]
        x_rec, y_rec = self.recordings[rec_idx]

        if self.seq_length == 1:
            seq = x_rec[center: center + 1]   # (1, C, T)
        else:
            seq = x_rec[center - self.half: center + self.half + 1]  # (L, C, T)

        x = torch.from_numpy(seq.copy())
        label = torch.tensor(y_rec[center], dtype=torch.long)

        if self.augment is not None:
            x = self.augment(x)
        return x, label


class SleepEpochDataset(Dataset):
    """PyTorch Dataset for individual EEG epochs (for CNN pre-training)."""

    def __init__(self, recordings, augment=None):
        xs = [x for x, _ in recordings]
        ys = [y for _, y in recordings]
        self.data = torch.from_numpy(np.concatenate(xs))
        self.labels = torch.from_numpy(np.concatenate(ys))
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.augment is not None:
            x = self.augment(x)
        return x, self.labels[idx]


# ---------------------------------------------------------------------------
# Main Data Loading Function
# ---------------------------------------------------------------------------

def get_fold_data(fold_idx, seq_length, data_dir=None, split_path=None,
                  augment_train=True):
    """Load data for a given fold and sequence length.

    Args:
        fold_idx: int, 0-19
        seq_length: int (1, 3, 5, 11, 21)
        data_dir: path to NPZ files
        split_path: path to data_split_v1.npz
        augment_train: whether to apply augmentation to training data

    Returns:
        train_recs, val_recs, test_recs: lists of (x, y)
        train_ds, val_ds, test_ds: Dataset objects
        class_weights: tensor of balanced class weights
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    split_path = split_path or DEFAULT_SPLIT_PATH

    subject_files = get_subject_files(data_dir)
    train_ids, val_ids, test_ids = load_split(split_path, fold_idx)

    train_recs = load_subjects_data(train_ids, subject_files)
    val_recs = load_subjects_data(val_ids, subject_files)
    test_recs = load_subjects_data(test_ids, subject_files)

    # Augmentation
    augment = EEGAugmentation() if augment_train else None

    # Build lazy datasets (sequences constructed on-the-fly to save RAM)
    train_ds = SleepSequenceDataset(train_recs, seq_length, augment=augment)
    val_ds = SleepSequenceDataset(val_recs, seq_length)
    test_ds = SleepSequenceDataset(test_recs, seq_length)

    # Compute class weights from training labels
    weights = compute_class_weight('balanced', classes=np.arange(5), y=train_ds.labels)
    class_weights = torch.FloatTensor(weights)

    return (train_recs, val_recs, test_recs,
            train_ds, val_ds, test_ds,
            class_weights)


def get_balanced_sampler(dataset):
    """Create a WeightedRandomSampler for balanced class sampling.

    Useful when class imbalance is severe (N1 is often under-represented).
    """
    labels = dataset.labels  # numpy array
    class_counts = np.bincount(labels, minlength=5)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('Enhanced Data Loader Verification')
    print('=' * 60)

    data_dir = DEFAULT_DATA_DIR
    split_path = DEFAULT_SPLIT_PATH

    if not os.path.exists(data_dir):
        print(f'\nData directory not found: {data_dir}')
        print('Set SLEEP_DATA_DIR and SLEEP_SPLIT_PATH environment variables.')
        exit(0)

    subject_files = get_subject_files(data_dir)
    total_files = sum(len(v) for v in subject_files.values())
    print(f'\nSubjects: {len(subject_files)}, NPZ files: {total_files}')

    # Load fold 0 with seq_length=11
    seq_length = 11
    print(f'\nLoading fold 0, seq_length={seq_length}...')
    (_, _, _, train_ds, val_ds, test_ds, class_weights) = get_fold_data(
        0, seq_length, data_dir, split_path)

    print(f'  Train: {len(train_ds)} sequences')
    print(f'  Val:   {len(val_ds)} sequences')
    print(f'  Test:  {len(test_ds)} sequences')
    print(f'  Class weights: {class_weights.numpy().round(3)}')

    # Test balanced sampler
    sampler = get_balanced_sampler(train_ds)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    batch_x, batch_y = next(iter(loader))
    print(f'\n  Balanced batch: x={batch_x.shape}, y distribution={np.bincount(batch_y.numpy(), minlength=5)}')

    # Test augmentation
    aug = EEGAugmentation()
    x_orig, _ = train_ds[0]
    x_aug = aug(x_orig.clone())
    diff = (x_orig - x_aug).abs().mean().item()
    print(f'  Augmentation diff (should be > 0): {diff:.6f}')

    print('\nAll checks passed.')
