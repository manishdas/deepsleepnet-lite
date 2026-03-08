#!/usr/bin/env python
"""
Generate figures for CNN+BiLSTM temporal model results.

Two modes:
  single — Parse a single training log (default, works with fold 0)
  all    — Aggregate all fold JSON results (after running all 20 folds)

Usage:
  python plot_results.py                                    # single fold from log
  python plot_results.py --mode all --results_dir output    # aggregate 20 folds
  python plot_results.py --show                             # display instead of save
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 1. Parse training log
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Extract training curves, confusion matrix, and classification report."""
    with open(log_path) as f:
        text = f.read()

    # --- Training curves per stage ---
    stages = {}
    pattern = re.compile(
        r'\[(?P<stage>[^\]]+)\]\s+epoch\s+(?P<epoch>\d+)/\d+:\s+'
        r'train loss=(?P<train_loss>[\d.]+)\s+acc=(?P<train_acc>[\d.]+)\s+'
        r'f1=(?P<train_f1>[\d.]+)\s+\|\s+'
        r'val loss=(?P<val_loss>[\d.]+)\s+acc=(?P<val_acc>[\d.]+)\s+'
        r'f1=(?P<val_f1>[\d.]+)\s+kappa=(?P<val_kappa>[\d.]+)\s+'
        r'lr=(?P<lr>[\d.e+-]+)'
    )
    for m in pattern.finditer(text):
        stage = m.group('stage')
        if stage not in stages:
            stages[stage] = {'epoch': [], 'train_loss': [], 'val_loss': [],
                             'train_f1': [], 'val_f1': [], 'train_acc': [],
                             'val_acc': [], 'lr': []}
        stages[stage]['epoch'].append(int(m.group('epoch')))
        stages[stage]['train_loss'].append(float(m.group('train_loss')))
        stages[stage]['val_loss'].append(float(m.group('val_loss')))
        stages[stage]['train_f1'].append(float(m.group('train_f1')))
        stages[stage]['val_f1'].append(float(m.group('val_f1')))
        stages[stage]['train_acc'].append(float(m.group('train_acc')))
        stages[stage]['val_acc'].append(float(m.group('val_acc')))
        stages[stage]['lr'].append(float(m.group('lr')))

    # --- Test confusion matrix ---
    # Find the test section's confusion matrix
    test_section = text.split('Test:')[1] if 'Test:' in text else ''
    cm_pattern = re.compile(r'\[\[?\s*([\d\s]+)\]')
    cm_rows = []
    for line in test_section.split('\n'):
        nums = re.findall(r'\d+', line)
        if len(nums) == 5 and '[' in line:
            cm_rows.append([int(n) for n in nums])
    cm = np.array(cm_rows) if len(cm_rows) == 5 else None

    # --- Test classification report ---
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    report = {}
    report_pattern = re.compile(
        r'(?:Wake|N1|N2|N3|REM)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)'
    )
    # Only parse from test section
    for m in report_pattern.finditer(test_section):
        idx = len(report)
        if idx < 5:
            report[class_names[idx]] = {
                'precision': float(m.group(1)),
                'recall': float(m.group(2)),
                'f1': float(m.group(3)),
                'support': int(m.group(4)),
            }

    # --- Overall test metrics ---
    metrics = {}
    for key in ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Kappa']:
        m = re.search(rf'{re.escape(key)}:\s+([\d.]+)', test_section)
        if m:
            metrics[key] = float(m.group(1))

    return stages, cm, report, metrics


# ---------------------------------------------------------------------------
# 2. Plotting functions
# ---------------------------------------------------------------------------

STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
COLORS = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#9b59b6']

# Baseline 20-fold results from team_handoff.md
BASELINE_F1 = {'Wake': 0.802, 'N1': 0.441, 'N2': 0.865, 'N3': 0.859, 'REM': 0.795}
BASELINE_MACRO_F1 = 0.7527


def plot_class_distribution(output_dir, data_dir=None):
    """Plot class distribution across the full dataset (all subjects)."""
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data',
                                         'SleepEDF', 'processed', 'eeg_FpzCz_PzOz_v1')
    if not os.path.isdir(data_dir):
        # Fall back to known full-dataset counts from Sleep-EDF v1
        supports = [8285, 2804, 17799, 5703, 7717]
        total = 42308
        source = 'Sleep-EDF v1'
    else:
        files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
        counts = np.zeros(5, dtype=int)
        for f in files:
            y = np.load(os.path.join(data_dir, f), allow_pickle=True)['y']
            for label in range(5):
                counts[label] += np.sum(y == label)
        supports = counts.tolist()
        total = int(counts.sum())
        source = f'Sleep-EDF v1 ({len(files)} recordings)'

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(STAGE_NAMES, supports, color=COLORS, edgecolor='white', linewidth=0.5)

    for bar, count in zip(bars, supports):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Number of Epochs', fontsize=12)
    ax.set_title(f'Sleep Stage Distribution — {source} ({total:,} total epochs)',
                 fontsize=13)
    ax.set_ylim(0, max(supports) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved class_distribution.png')


def plot_confusion_matrices(cm, output_dir, suptitle='CNN+BiLSTM — Test Set (Fold 0)'):
    """Plot side-by-side confusion matrices (counts + normalized)."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, data, title, fmt in [
        (ax1, cm, 'Confusion Matrix (counts)', 'd'),
        (ax2, cm_norm, 'Confusion Matrix (normalized)', '.2f'),
    ]:
        im = ax.imshow(data, cmap='Blues', aspect='equal')
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(STAGE_NAMES)
        ax.set_yticklabels(STAGE_NAMES)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(title, fontsize=12)

        for i in range(5):
            for j in range(5):
                val = data[i, j]
                text = format(val, fmt)
                color = 'white' if data[i, j] > data.max() * 0.6 else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved confusion_matrices.png')


def plot_training_curves_and_f1(stages, report, output_dir):
    """Plot per-class F1 (left) and 3-stage training curves (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Left: Per-class F1 comparison ---
    f1_scores = [report[s]['f1'] for s in STAGE_NAMES]
    macro_f1 = np.mean(f1_scores)
    baseline_f1s = [BASELINE_F1[s] for s in STAGE_NAMES]

    x = np.arange(len(STAGE_NAMES))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, baseline_f1s, width, color='#bdc3c7',
                    edgecolor='white', label='DeepSleepNet-Lite (20-fold)')
    bars2 = ax1.bar(x + width / 2, f1_scores, width, color=COLORS,
                    edgecolor='white', label='CNN+BiLSTM (Fold 0)')

    ax1.axhline(y=BASELINE_MACRO_F1, color='gray', linestyle='--', alpha=0.7,
                label=f'Baseline Macro F1 = {BASELINE_MACRO_F1:.3f}')
    ax1.axhline(y=macro_f1, color='#2c3e50', linestyle='--', alpha=0.7,
                label=f'CNN+BiLSTM Macro F1 = {macro_f1:.3f}')

    for bar, val in zip(bars2, f1_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars1, baseline_f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8, color='gray')

    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title('Per-Class F1 — Baseline vs CNN+BiLSTM', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGE_NAMES)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Right: 3-stage training curves ---
    stage_colors = {
        'CNN pretrain': ('#3498db', '#85c1e9'),
        'LSTM': ('#e67e22', '#f0b27a'),
        'Fine-tune': ('#2ecc71', '#82e0aa'),
    }
    epoch_offset = 0
    for stage_name in ['CNN pretrain', 'LSTM', 'Fine-tune']:
        if stage_name not in stages:
            continue
        s = stages[stage_name]
        epochs = [e + epoch_offset for e in s['epoch']]
        c_train, c_val = stage_colors.get(stage_name, ('#333', '#999'))

        ax2.plot(epochs, s['train_loss'], color=c_train, linewidth=1.5,
                 label=f'{stage_name} (train)')
        ax2.plot(epochs, s['val_loss'], color=c_val, linewidth=1.5,
                 linestyle='--', label=f'{stage_name} (val)')

        # Stage boundary
        if epoch_offset > 0:
            ax2.axvline(x=epoch_offset, color='gray', linestyle=':', alpha=0.5)

        epoch_offset = max(epochs)

    ax2.set_xlabel('Epoch (cumulative across stages)', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title('Training Curves — 3-Stage Training (Fold 0)', fontsize=12)
    ax2.legend(fontsize=7, loc='upper right', ncol=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_curves_and_f1.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved training_curves_and_f1.png')


def plot_sample_eeg(output_dir, data_dir=None):
    """Plot a sample EEG epoch from the dataset."""
    data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data',
                                         'SleepEDF', 'processed', 'eeg_FpzCz_PzOz_v1')
    if not os.path.isdir(data_dir):
        print(f'  Skipping sample_eeg_epoch.png (data not found at {data_dir})')
        return

    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    if not files:
        print(f'  Skipping sample_eeg_epoch.png (no NPZ files)')
        return

    d = np.load(os.path.join(data_dir, files[0]), allow_pickle=True)
    x, y = d['x'], d['y']
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

    # Pick an N2 epoch (most common) for consistency with baseline figure
    n2_indices = np.where(y == 2)[0]
    idx = n2_indices[len(n2_indices) // 2] if len(n2_indices) > 0 else 0
    epoch = x[idx, :, 0]  # Fpz-Cz channel
    stage = stage_names[y[idx]]
    time = np.arange(len(epoch)) / 100.0  # 100 Hz

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time, epoch, color='#3498db', linewidth=0.5)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude (uV)', fontsize=11)
    ax.set_title(f'Sample EEG Epoch (Fpz-Cz) — Stage: {stage}', fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sample_eeg_epoch.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved sample_eeg_epoch.png')


# ---------------------------------------------------------------------------
# 3. 20-fold aggregation from JSON results
# ---------------------------------------------------------------------------

def load_fold_results(results_dir):
    """Load all fold*_results.json files from results_dir."""
    pattern = os.path.join(results_dir, 'fold*_results.json')
    files = sorted(glob.glob(pattern))
    results = []
    for f in files:
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def plot_aggregated_confusion_matrices(fold_results, output_dir):
    """Sum confusion matrices across all folds and plot."""
    cm = np.zeros((5, 5), dtype=int)
    for fr in fold_results:
        cm += np.array(fr['test_confusion_matrix'])
    n_folds = len(fold_results)
    plot_confusion_matrices(cm, output_dir,
                            suptitle=f'CNN+BiLSTM — Test Set ({n_folds}-fold LOSO-CV)')


def plot_aggregated_f1_and_curves(fold_results, output_dir):
    """Plot per-class F1 with error bars (left) and training curves with std shading (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    n_folds = len(fold_results)

    # --- Left: Per-class F1 with error bars ---
    per_class_f1s = {s: [] for s in STAGE_NAMES}
    for fr in fold_results:
        for s in STAGE_NAMES:
            per_class_f1s[s].append(fr['test_per_class'][s]['f1-score'])

    means = [np.mean(per_class_f1s[s]) for s in STAGE_NAMES]
    stds = [np.std(per_class_f1s[s]) for s in STAGE_NAMES]
    macro_f1_mean = np.mean(means)
    baseline_f1s = [BASELINE_F1[s] for s in STAGE_NAMES]

    x = np.arange(len(STAGE_NAMES))
    width = 0.35

    ax1.bar(x - width / 2, baseline_f1s, width, color='#bdc3c7',
            edgecolor='white', label='DeepSleepNet-Lite (20-fold)')
    bars2 = ax1.bar(x + width / 2, means, width, color=COLORS,
                    edgecolor='white', yerr=stds, capsize=4,
                    label=f'CNN+BiLSTM ({n_folds}-fold)')

    ax1.axhline(y=BASELINE_MACRO_F1, color='gray', linestyle='--', alpha=0.7,
                label=f'Baseline Macro F1 = {BASELINE_MACRO_F1:.3f}')
    ax1.axhline(y=macro_f1_mean, color='#2c3e50', linestyle='--', alpha=0.7,
                label=f'CNN+BiLSTM Macro F1 = {macro_f1_mean:.3f}')

    for bar, val, std in zip(bars2, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('F1 Score', fontsize=11)
    ax1.set_title(f'Per-Class F1 — Baseline vs CNN+BiLSTM ({n_folds}-fold)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGE_NAMES)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Right: Training curves with mean ± std ---
    stage_colors = {
        'CNN pretrain': '#3498db',
        'LSTM': '#e67e22',
        'Fine-tune': '#2ecc71',
    }

    epoch_offset = 0
    for stage_name in ['CNN pretrain', 'LSTM', 'Fine-tune']:
        # Collect curves from all folds, pad to max length
        all_train = []
        all_val = []
        for fr in fold_results:
            hist = fr.get('training_history', {}).get(stage_name)
            if hist:
                all_train.append(hist['train_loss'])
                all_val.append(hist['val_loss'])

        if not all_train:
            continue

        max_len = max(len(t) for t in all_train)
        # Pad shorter runs with their last value
        train_padded = np.array([t + [t[-1]] * (max_len - len(t)) for t in all_train])
        val_padded = np.array([v + [v[-1]] * (max_len - len(v)) for v in all_val])

        epochs = np.arange(1, max_len + 1) + epoch_offset
        train_mean = train_padded.mean(axis=0)
        train_std = train_padded.std(axis=0)
        val_mean = val_padded.mean(axis=0)
        val_std = val_padded.std(axis=0)

        color = stage_colors.get(stage_name, '#333')
        ax2.plot(epochs, train_mean, color=color, linewidth=1.5,
                 label=f'{stage_name} (train)')
        ax2.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         color=color, alpha=0.15)
        ax2.plot(epochs, val_mean, color=color, linewidth=1.5,
                 linestyle='--', label=f'{stage_name} (val)')
        ax2.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                         color=color, alpha=0.1)

        if epoch_offset > 0:
            ax2.axvline(x=epoch_offset, color='gray', linestyle=':', alpha=0.5)

        epoch_offset = int(epochs[-1])

    ax2.set_xlabel('Epoch (cumulative across stages)', fontsize=11)
    ax2.set_ylabel('Loss', fontsize=11)
    ax2.set_title(f'Training Curves — Mean ± Std ({n_folds} folds)', fontsize=12)
    ax2.legend(fontsize=7, loc='upper right', ncol=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'training_curves_and_f1.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved training_curves_and_f1.png')


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate result figures')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'all'],
                        help='single: one fold from log. all: aggregate fold JSONs')
    parser.add_argument('--log', type=str, default='output/train_fold0_L5.log',
                        help='Path to training log (single mode)')
    parser.add_argument('--results_dir', type=str, default='output',
                        help='Directory with fold*_results.json (all mode)')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory to save figures')
    parser.add_argument('--show', action='store_true',
                        help='Show figures instead of saving')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to NPZ data directory (for sample EEG + class dist)')
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    def resolve(p):
        return os.path.join(script_dir, p) if not os.path.isabs(p) else p

    output_dir = resolve(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode == 'all':
        # --- Aggregate all folds ---
        results_dir = resolve(args.results_dir)
        fold_results = load_fold_results(results_dir)
        n_folds = len(fold_results)

        if n_folds == 0:
            print(f'No fold*_results.json found in {results_dir}')
            return

        print(f'Loaded {n_folds} fold results from {results_dir}')

        # Print aggregated metrics
        for key in ['accuracy', 'f1_macro', 'f1_weighted', 'kappa']:
            vals = [fr['test_metrics'][key] for fr in fold_results]
            print(f'  Test {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

        print(f'\nGenerating aggregated figures → {output_dir}/')
        plot_class_distribution(output_dir, data_dir=args.data_dir)
        plot_aggregated_confusion_matrices(fold_results, output_dir)
        plot_aggregated_f1_and_curves(fold_results, output_dir)
        plot_sample_eeg(output_dir, data_dir=args.data_dir)

    else:
        # --- Single fold from log ---
        log_path = resolve(args.log)
        print(f'Parsing log: {log_path}')
        stages, cm, report, metrics = parse_log(log_path)

        print(f'\nTest metrics:')
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')

        print(f'\nGenerating figures → {output_dir}/')
        plot_class_distribution(output_dir, data_dir=args.data_dir)

        if cm is not None:
            plot_confusion_matrices(cm, output_dir)

        if stages and report:
            plot_training_curves_and_f1(stages, report, output_dir)

        plot_sample_eeg(output_dir, data_dir=args.data_dir)

    print('\nDone.')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
