#!/usr/bin/env python
"""Download and preprocess Sleep-EDF data from PhysioNet for DeepSleepNet-Lite.

Produces .npz files in the format expected by deepsleeplite/data_loader.py:
  - x: float32, shape [n_epochs, 3000, 1] (Fpz-Cz channel, 30s @ 100Hz)
  - y: int32, shape [n_epochs] (W=0, N1=1, N2=2, N3=3, REM=4)
  - fs: scalar 100

Also produces data_split_v1.npz with cross-validation fold assignments.

Usage:
    # Option 1: Auto-download via MNE (slow)
    python prepare_physionet.py --output_dir data/eeg_FpzCz_PzOz_v1 --n_subjects 20

    # Option 2: Process pre-downloaded EDF files (fast)
    python prepare_physionet.py --output_dir data/eeg_FpzCz_PzOz_v1 --raw_dir raw_data/sleep-cassette
"""

import argparse
import os
import re
import glob

import numpy as np
import mne


# Sleep stage mapping (R&K / AASM annotations -> integer labels)
ANNOTATION_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,   # NREM3+4 merged per AASM
    'Sleep stage R': 4,
}

# Stages to discard
DISCARD_STAGES = {'Sleep stage ?', 'Movement time'}

EPOCH_DURATION = 30  # seconds
SAMPLING_RATE = 100  # Hz
SAMPLES_PER_EPOCH = EPOCH_DURATION * SAMPLING_RATE  # 3000


def process_recording(psg_path, hyp_path, output_path):
    """Process a single PSG recording and its hypnogram into an .npz file.

    Uses the "v1" strategy: include 30 minutes of wake before the first
    non-wake epoch and after the last non-wake epoch.
    """
    # Read PSG
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # Pick Fpz-Cz channel
    channel_name = 'EEG Fpz-Cz'
    raw.pick([channel_name])

    # Resample to 100 Hz if needed
    if raw.info['sfreq'] != SAMPLING_RATE:
        raw.resample(SAMPLING_RATE, verbose=False)

    # Read annotations from hypnogram
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot, verbose=False)

    # Extract events from annotations
    event_id = {k: v for k, v in ANNOTATION_MAP.items()
                if k in [a for a in annot.description]}
    if not event_id:
        print(f"  WARNING: No sleep stage annotations found in {hyp_path}, skipping.")
        return False

    events, _ = mne.events_from_annotations(
        raw, event_id=event_id, chunk_duration=EPOCH_DURATION, verbose=False
    )

    # Get labels
    labels = events[:, 2]

    # Apply v1 trim: keep 30 min of wake before first non-wake and after last non-wake
    non_wake_idx = np.where(labels != 0)[0]
    if len(non_wake_idx) == 0:
        print(f"  WARNING: No non-wake epochs found in {psg_path}, skipping.")
        return False

    first_non_wake = non_wake_idx[0]
    last_non_wake = non_wake_idx[-1]

    # 30 min = 60 epochs of 30s each
    trim_epochs = 60
    start_idx = max(0, first_non_wake - trim_epochs)
    end_idx = min(len(labels), last_non_wake + trim_epochs + 1)

    # Extract data for selected epochs
    data_list = []
    label_list = []

    for i in range(start_idx, end_idx):
        # Get sample indices for this epoch
        sample_start = events[i, 0]
        sample_end = sample_start + SAMPLES_PER_EPOCH

        if sample_end > raw.n_times:
            break

        epoch_data = raw.get_data(start=sample_start, stop=sample_end)
        data_list.append(epoch_data[0])  # [3000,]
        label_list.append(labels[i])

    if len(data_list) == 0:
        print(f"  WARNING: No valid epochs extracted from {psg_path}, skipping.")
        return False

    data = np.array(data_list, dtype=np.float32)     # [n_epochs, 3000]
    data = data[:, :, np.newaxis]                      # [n_epochs, 3000, 1]
    labels_out = np.array(label_list, dtype=np.int32)  # [n_epochs]

    np.savez(output_path, x=data, y=labels_out, fs=SAMPLING_RATE)
    print(f"  Saved {output_path}: {data.shape[0]} epochs, "
          f"W={np.sum(labels_out==0)}, N1={np.sum(labels_out==1)}, "
          f"N2={np.sum(labels_out==2)}, N3={np.sum(labels_out==3)}, "
          f"REM={np.sum(labels_out==4)}")
    return True


def create_data_splits(n_subjects, n_folds, output_path):
    """Create leave-one-subject-out cross-validation splits.

    For fold k: test = subject k, validation = subject (k+1) % n_subjects.
    """
    valid_files = []
    test_files = []
    for fold_idx in range(n_folds):
        # Test subject is the fold index
        test_files.append(np.array([fold_idx]))
        # Validation subject is the next one after the test subject
        valid_subj = (fold_idx + 1) % n_subjects
        valid_files.append(np.array([valid_subj]))

    # Save as object arrays to match expected format (array of arrays)
    valid_files_arr = np.empty(n_folds, dtype=object)
    test_files_arr = np.empty(n_folds, dtype=object)
    for i in range(n_folds):
        valid_files_arr[i] = valid_files[i]
        test_files_arr[i] = test_files[i]

    np.savez(output_path, valid_files=valid_files_arr, test_files=test_files_arr)
    print(f"Saved data splits to {output_path}")


def process_from_local_dir(raw_dir, output_dir, n_folds):
    """Process pre-downloaded EDF files from a local directory.

    Expects files named like SC4001E0-PSG.edf and SC4001E*-Hypnogram.edf
    in the raw_dir.
    """
    # Find all PSG files for SC (Sleep Cassette) subjects
    psg_pattern = os.path.join(raw_dir, 'SC*-PSG.edf')
    psg_files = sorted(glob.glob(psg_pattern))

    if not psg_files:
        raise FileNotFoundError(
            f"No SC*-PSG.edf files found in {raw_dir}. "
            f"Make sure you downloaded the sleep-cassette files from PhysioNet."
        )

    print(f"Found {len(psg_files)} PSG files in {raw_dir}")
    os.makedirs(output_dir, exist_ok=True)

    subject_ids = set()
    processed = 0

    for psg_path in psg_files:
        basename = os.path.basename(psg_path)
        # Extract subject info from filename like SC4001E0-PSG.edf
        match = re.match(r'SC4(\d{2})(\d)E0-PSG\.edf', basename)
        if not match:
            print(f"  Skipping unrecognized file: {basename}")
            continue

        subj_id = int(match.group(1))
        night = int(match.group(2))
        subject_ids.add(subj_id)

        # Find matching hypnogram
        hyp_pattern = os.path.join(raw_dir, f'SC4{subj_id:02d}{night}E*-Hypnogram.edf')
        hyp_files = glob.glob(hyp_pattern)
        if not hyp_files:
            # Try alternative naming
            hyp_pattern = os.path.join(raw_dir, f'SC4{subj_id:02d}{night}*Hypnogram*')
            hyp_files = glob.glob(hyp_pattern)
        if not hyp_files:
            print(f"  WARNING: No hypnogram found for {basename}, skipping.")
            continue
        hyp_path = hyp_files[0]

        out_name = f"SC4{subj_id:02d}{night}E0.npz"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            print(f"  {out_name} already exists, skipping.")
            processed += 1
            continue

        print(f"Processing {basename} -> {out_name}...")
        if process_recording(psg_path, hyp_path, out_path):
            processed += 1

    n_subjects = len(subject_ids)
    print(f"\nProcessed {processed} recordings from {n_subjects} subjects.")
    return n_subjects


def process_via_mne(n_subjects, output_dir):
    """Download and process data via MNE (slower, but no manual download needed)."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading Sleep-EDF data for {n_subjects} subjects via MNE...")
    print("This may take a while on the first run (downloads ~2GB from PhysioNet).\n")

    subjects_processed = 0

    for subj_id in range(n_subjects):
        print(f"Subject {subj_id}/{n_subjects - 1}:")

        for night in [1, 2]:
            try:
                paths = mne.datasets.sleep_physionet.age.fetch_data(
                    subjects=[subj_id], recording=[night], verbose=False
                )
            except (ValueError, FileNotFoundError):
                print(f"  Night {night}: not available, skipping.")
                continue

            if not paths:
                continue

            psg_path, hyp_path = paths[0]

            out_name = f"SC4{subj_id:02d}{night}E0.npz"
            out_path = os.path.join(output_dir, out_name)

            if os.path.exists(out_path):
                print(f"  {out_name} already exists, skipping.")
                continue

            process_recording(psg_path, hyp_path, out_path)

        subjects_processed += 1

    return subjects_processed


def main():
    parser = argparse.ArgumentParser(
        description='Download and preprocess Sleep-EDF data for DeepSleepNet-Lite')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed .npz files')
    parser.add_argument('--raw_dir', type=str, default=None,
                        help='Directory with pre-downloaded EDF files (skip MNE download)')
    parser.add_argument('--n_subjects', type=int, default=20,
                        help='Number of subjects to download via MNE (default: 20)')
    parser.add_argument('--n_folds', type=int, default=20,
                        help='Number of cross-validation folds (default: 20)')
    args = parser.parse_args()

    if args.raw_dir:
        # Fast path: process pre-downloaded files
        n_subjects = process_from_local_dir(args.raw_dir, args.output_dir, args.n_folds)
    else:
        # Slow path: download via MNE one file at a time
        n_subjects = process_via_mne(args.n_subjects, args.output_dir)

    # Create data splits file one directory up from output_dir
    split_dir = os.path.dirname(args.output_dir)
    if not split_dir:
        split_dir = '.'
    split_path = os.path.join(split_dir, 'data_split_v1.npz')
    create_data_splits(min(n_subjects, args.n_folds), args.n_folds, split_path)

    # Summary
    npz_files = sorted(glob.glob(os.path.join(args.output_dir, '*.npz')))
    print(f"\nDone! {len(npz_files)} recordings processed.")
    print(f"Data directory: {args.output_dir}")
    print(f"Data splits: {split_path}")

    # Print overall class distribution
    total_labels = []
    for f in npz_files:
        with np.load(f) as data:
            total_labels.append(data['y'])
    if total_labels:
        all_labels = np.concatenate(total_labels)
        print(f"\nOverall class distribution ({len(all_labels)} total epochs):")
        for stage, name in [(0, 'W'), (1, 'N1'), (2, 'N2'), (3, 'N3'), (4, 'REM')]:
            count = np.sum(all_labels == stage)
            pct = 100 * count / len(all_labels)
            print(f"  {name}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
