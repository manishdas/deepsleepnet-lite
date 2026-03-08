#!/usr/bin/env python
"""
Training script for SleepStageNet CNN+BiLSTM.

Three-stage training:
  Stage 1: Pre-train CNN on individual epochs (warm up feature extractor)
  Stage 2: Freeze CNN, train BiLSTM on sequences
  Stage 3: Unfreeze, fine-tune end-to-end with low lr

Usage:
  python train_sequence.py --fold 0 --seq_len 5
  python train_sequence.py --fold 0 --seq_len 5 --skip_pretrain  # if CNN already trained
"""

import argparse
import json
import logging
import multiprocessing
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import (
    STAGE_NAMES,
    SleepEpochDataset,
    get_fold_data,
)
from models import SleepCNNBiLSTM, SleepCNNOnly
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

# Use half of available CPU cores for data loading, rest for compute
NUM_WORKERS = min(multiprocessing.cpu_count() // 2, 8)

log = logging.getLogger('train')


def setup_logging(output_dir, fold_idx, seq_length):
    """Configure logging to both console and log file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f'train_fold{fold_idx}_L{seq_length}.log')

    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)

    fh = logging.FileHandler(log_path, mode='w')
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log_path


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_one_epoch(model, loader, criterion, optimizer, device, clip_norm=None):
    model.train()
    total_loss = 0
    all_preds, all_true = [], []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        if clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    n = len(all_true)
    return (total_loss / n,
            accuracy_score(all_true, all_preds),
            f1_score(all_true, all_preds, average='macro'))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_true = [], []

    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    n = len(all_true)
    acc = accuracy_score(all_true, all_preds)
    f1_mac = f1_score(all_true, all_preds, average='macro')
    f1_wt = f1_score(all_true, all_preds, average='weighted')
    kappa = cohen_kappa_score(all_true, all_preds)

    return {
        'loss': total_loss / n,
        'accuracy': acc,
        'f1_macro': f1_mac,
        'f1_weighted': f1_wt,
        'kappa': kappa,
        'preds': np.array(all_preds),
        'true': np.array(all_true),
    }


def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler,
               device, n_epochs, patience, clip_norm=None, stage_name=''):
    """Generic training loop with early stopping on val F1-macro.

    Returns:
        best_f1: float
        history: dict with per-epoch metrics (every epoch, not just printed ones)
    """
    best_f1 = 0
    best_state = None
    wait = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': [],
               'train_f1': [], 'val_f1': []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_norm)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # Record every epoch
        history['epoch'].append(epoch)
        history['train_loss'].append(round(train_loss, 5))
        history['val_loss'].append(round(val_metrics['loss'], 5))
        history['train_f1'].append(round(train_f1, 4))
        history['val_f1'].append(round(val_metrics['f1_macro'], 4))

        scheduler.step(-val_metrics['f1_macro'])

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = ' *'
        else:
            wait += 1
            marker = ''

        if epoch % 5 == 0 or epoch == 1 or marker:
            lr = optimizer.param_groups[0]['lr']
            log.info(f'  [{stage_name}] epoch {epoch:3d}/{n_epochs}: '
                     f'train loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | '
                     f'val loss={val_metrics["loss"]:.4f} acc={val_metrics["accuracy"]:.3f} '
                     f'f1={val_metrics["f1_macro"]:.3f} kappa={val_metrics["kappa"]:.3f} '
                     f'lr={lr:.1e} ({elapsed:.1f}s){marker}')

        if wait >= patience:
            log.info(f'  Early stopping at epoch {epoch} (patience={patience})')
            break

    model.load_state_dict(best_state)
    return best_f1, history


def run_fold(fold_idx, seq_length, args):
    """Run full 3-stage training for one fold."""
    device = get_device()
    n_cpu = multiprocessing.cpu_count()
    torch.set_num_threads(n_cpu)
    log.info(f'\n{"=" * 70}')
    log.info(f'Fold {fold_idx} | seq_length={seq_length} | device={device}')
    log.info(f'CPU cores: {n_cpu} | DataLoader workers: {NUM_WORKERS} | '
             f'PyTorch threads: {torch.get_num_threads()}')
    log.info(f'{"=" * 70}')

    # Load data
    log.info('\n[1] Loading data...')
    (train_recs, val_recs, _,
     train_seq_ds, val_seq_ds, test_seq_ds,
     class_weights) = get_fold_data(fold_idx, seq_length)

    log.info(f'  Train: {len(train_seq_ds)} sequences')
    log.info(f'  Val:   {len(val_seq_ds)} sequences')
    log.info(f'  Test:  {len(test_seq_ds)} sequences')
    log.info(f'  Class weights: {class_weights.numpy().round(3)}')

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': NUM_WORKERS,
        'persistent_workers': NUM_WORKERS > 0,
        'prefetch_factor': 2 if NUM_WORKERS > 0 else None,
    }
    train_seq_loader = DataLoader(train_seq_ds, shuffle=True, **loader_kwargs)
    val_seq_loader = DataLoader(val_seq_ds, shuffle=False, **loader_kwargs)
    test_seq_loader = DataLoader(test_seq_ds, shuffle=False, **loader_kwargs)

    # ── Stage 1: Pre-train CNN on individual epochs ──
    if not args.skip_pretrain:
        log.info('\n[2] Stage 1: Pre-training CNN on individual epochs...')
        cnn_model = SleepCNNOnly(n_channels=2, n_classes=5).to(device)

        epoch_train_ds = SleepEpochDataset(train_recs)
        epoch_val_ds = SleepEpochDataset(val_recs)
        epoch_train_loader = DataLoader(epoch_train_ds, shuffle=True, **loader_kwargs)
        epoch_val_loader = DataLoader(epoch_val_ds, shuffle=False, **loader_kwargs)

        optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

        _, cnn_history = train_loop(
            cnn_model, epoch_train_loader, epoch_val_loader, criterion,
            optimizer, scheduler, device, n_epochs=args.cnn_epochs,
            patience=15, stage_name='CNN pretrain')

        # Save CNN checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        cnn_path = os.path.join(args.output_dir, f'cnn_fold{fold_idx}.pt')
        torch.save(cnn_model.cnn.state_dict(), cnn_path)
        log.info(f'  Saved CNN weights: {cnn_path}')

        # Evaluate CNN-only baseline
        cnn_val = evaluate(cnn_model, epoch_val_loader, criterion, device)
        log.info(f'  CNN-only val: acc={cnn_val["accuracy"]:.3f} '
                 f'f1_macro={cnn_val["f1_macro"]:.3f} kappa={cnn_val["kappa"]:.3f}')

        # Free epoch data
        del epoch_train_ds, epoch_val_ds, epoch_train_loader, epoch_val_loader

    # ── Build sequence model ──
    model = SleepCNNBiLSTM(
        n_channels=2, feature_dim=64,
        lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
        n_classes=5, dropout=args.dropout
    ).to(device)

    # Load pre-trained CNN weights
    if not args.skip_pretrain:
        model.cnn.load_state_dict(torch.load(cnn_path, map_location=device,
                                             weights_only=True))
        log.info('\n  Loaded pre-trained CNN into sequence model')

    # ── Stage 2: Freeze CNN, train LSTM ──
    log.info('\n[3] Stage 2: Training BiLSTM (CNN frozen)...')
    model.freeze_cnn()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  Trainable params: {trainable:,}')

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    _, lstm_history = train_loop(
        model, train_seq_loader, val_seq_loader, criterion,
        optimizer, scheduler, device, n_epochs=args.lstm_epochs,
        patience=15, clip_norm=1.0, stage_name='LSTM')

    # ── Stage 3: Unfreeze, fine-tune end-to-end ──
    log.info('\n[4] Stage 3: Fine-tuning end-to-end...')
    model.unfreeze_cnn()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'  Trainable params: {trainable:,}')

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.5)

    _, finetune_history = train_loop(
        model, train_seq_loader, val_seq_loader, criterion,
        optimizer, scheduler, device, n_epochs=args.finetune_epochs,
        patience=15, clip_norm=1.0, stage_name='Fine-tune')

    # ── Final evaluation ──
    log.info('\n[5] Final evaluation...')
    val_results = evaluate(model, val_seq_loader, criterion, device)
    test_results = evaluate(model, test_seq_loader, criterion, device)

    for name, results in [('Validation', val_results), ('Test', test_results)]:
        log.info(f'\n  {name}:')
        log.info(f'    Accuracy:    {results["accuracy"]:.4f}')
        log.info(f'    F1 (macro):  {results["f1_macro"]:.4f}')
        log.info(f'    F1 (weighted): {results["f1_weighted"]:.4f}')
        log.info(f'    Kappa:       {results["kappa"]:.4f}')
        log.info('\n    Classification Report:')
        log.info(classification_report(results['true'], results['preds'],
                                       target_names=STAGE_NAMES, digits=3))
        log.info('    Confusion Matrix:')
        log.info(confusion_matrix(results['true'], results['preds']))

    # Save model
    model_path = os.path.join(args.output_dir,
                              f'cnn_bilstm_fold{fold_idx}_L{seq_length}.pt')
    torch.save(model.state_dict(), model_path)
    log.info(f'\n  Saved model: {model_path}')

    # Save structured results as JSON for aggregation
    test_cm = confusion_matrix(test_results['true'], test_results['preds'])
    test_report = classification_report(test_results['true'], test_results['preds'],
                                        target_names=STAGE_NAMES, digits=4,
                                        output_dict=True)
    val_cm = confusion_matrix(val_results['true'], val_results['preds'])
    val_report = classification_report(val_results['true'], val_results['preds'],
                                       target_names=STAGE_NAMES, digits=4,
                                       output_dict=True)

    training_history = {
        'LSTM': lstm_history,
        'Fine-tune': finetune_history,
    }
    if not args.skip_pretrain:
        training_history['CNN pretrain'] = cnn_history

    fold_results = {
        'fold': fold_idx,
        'seq_length': seq_length,
        'test_metrics': {
            'accuracy': test_results['accuracy'],
            'f1_macro': test_results['f1_macro'],
            'f1_weighted': test_results['f1_weighted'],
            'kappa': test_results['kappa'],
        },
        'val_metrics': {
            'accuracy': val_results['accuracy'],
            'f1_macro': val_results['f1_macro'],
            'f1_weighted': val_results['f1_weighted'],
            'kappa': val_results['kappa'],
        },
        'test_confusion_matrix': test_cm.tolist(),
        'val_confusion_matrix': val_cm.tolist(),
        'test_per_class': {s: {k: test_report[s][k] for k in ['precision', 'recall', 'f1-score', 'support']}
                           for s in STAGE_NAMES},
        'val_per_class': {s: {k: val_report[s][k] for k in ['precision', 'recall', 'f1-score', 'support']}
                          for s in STAGE_NAMES},
        'training_history': training_history,
    }

    json_path = os.path.join(args.output_dir, f'fold{fold_idx}_results.json')
    with open(json_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    log.info(f'  Saved results: {json_path}')

    return {
        'fold': fold_idx,
        'seq_length': seq_length,
        'val_accuracy': val_results['accuracy'],
        'val_f1_macro': val_results['f1_macro'],
        'val_kappa': val_results['kappa'],
        'test_accuracy': test_results['accuracy'],
        'test_f1_macro': test_results['f1_macro'],
        'test_kappa': test_results['kappa'],
    }


def main():
    parser = argparse.ArgumentParser(description='SleepStageNet CNN+BiLSTM Training')
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0-19)')
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence length (3,5,11,21)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lstm_hidden', type=int, default=128)
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--cnn_epochs', type=int, default=50,
                        help='Epochs for CNN pre-training')
    parser.add_argument('--lstm_epochs', type=int, default=50,
                        help='Epochs for LSTM training (CNN frozen)')
    parser.add_argument('--finetune_epochs', type=int, default=30,
                        help='Epochs for end-to-end fine-tuning')
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip CNN pre-training (use random init)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory for checkpoints')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip fold if results JSON already exists')
    args = parser.parse_args()

    # Set up logging to console + file
    log_path = setup_logging(args.output_dir, args.fold, args.seq_len)
    log.info(f'Logging to: {log_path}')

    # Check if fold already completed
    json_path = os.path.join(args.output_dir, f'fold{args.fold}_results.json')
    if args.skip_existing and os.path.exists(json_path):
        log.info(f'Fold {args.fold} already completed ({json_path}), skipping.')
        return

    results = run_fold(args.fold, args.seq_len, args)

    log.info(f'\n{"=" * 70}')
    log.info('SUMMARY')
    log.info(f'{"=" * 70}')
    for k, v in results.items():
        if isinstance(v, float):
            log.info(f'  {k}: {v:.4f}')
        else:
            log.info(f'  {k}: {v}')


if __name__ == '__main__':
    main()
