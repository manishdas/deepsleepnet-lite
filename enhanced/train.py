#!/usr/bin/env python
"""
Training script for Enhanced SleepStageNet models.

Supports three model architectures with:
  - Three-stage training (pretrain CNN → train temporal → fine-tune end-to-end)
  - Cosine annealing with warmup
  - Focal loss + label smoothing
  - Mixup augmentation
  - Balanced sampling
  - Gradient accumulation
  - Full k-fold cross-validation

Usage:
  # Train single fold with default config:
  python train.py --model conformer --fold 0

  # Train all 20 folds:
  python train.py --model conformer --all_folds

  # Custom hyperparameters:
  python train.py --model cnn_bilstm --fold 0 --seq_len 11 --lr 3e-4 --epochs 80

  # List available models:
  python train.py --list_models
"""

import argparse
import json
import multiprocessing
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from data_loader import (
    STAGE_NAMES,
    SleepEpochDataset,
    get_balanced_sampler,
    get_fold_data,
)
from models import (
    MODEL_CONFIGS,
    MODEL_REGISTRY,
    FocalLoss,
    SleepCNNOnly,
    build_model,
    mixup_criterion,
    mixup_data,
)

NUM_WORKERS = min(multiprocessing.cpu_count() // 2, 8)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Learning Rate Schedulers
# ---------------------------------------------------------------------------

class CosineAnnealingWithWarmup:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    clip_norm=None, mixup_alpha=0.0, accumulation_steps=1):
    """Train for one epoch with optional mixup and gradient accumulation."""
    model.train()
    total_loss = 0
    all_preds, all_true = [], []

    optimizer.zero_grad()
    for step, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)

        if mixup_alpha > 0 and np.random.random() < 0.5:
            data, y_a, y_b, lam = mixup_data(data, labels, alpha=mixup_alpha)
            outputs = model(data)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(data)
            loss = criterion(outputs, labels)

        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            if clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * len(labels)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_true.extend(labels.cpu().numpy())

    # Handle remaining gradients
    if (step + 1) % accumulation_steps != 0:
        if clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad()

    n = len(all_true)
    return (total_loss / n,
            accuracy_score(all_true, all_preds),
            f1_score(all_true, all_preds, average='macro'))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader."""
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
    return {
        'loss': total_loss / n,
        'accuracy': accuracy_score(all_true, all_preds),
        'f1_macro': f1_score(all_true, all_preds, average='macro'),
        'f1_weighted': f1_score(all_true, all_preds, average='weighted'),
        'kappa': cohen_kappa_score(all_true, all_preds),
        'preds': np.array(all_preds),
        'true': np.array(all_true),
    }


def train_loop(model, train_loader, val_loader, criterion, optimizer,
               scheduler, device, n_epochs, patience, clip_norm=None,
               mixup_alpha=0.0, accumulation_steps=1, stage_name='',
               save_path=None):
    """Generic training loop with early stopping on val F1-macro."""
    best_f1 = 0
    best_state = None
    wait = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                'val_f1': [], 'val_kappa': []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Update LR
        if isinstance(scheduler, CosineAnnealingWithWarmup):
            scheduler.step(epoch - 1)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            clip_norm, mixup_alpha, accumulation_steps)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        # ReduceLROnPlateau
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-val_metrics['f1_macro'])

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['val_kappa'].append(val_metrics['kappa'])

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = ' *'
            if save_path:
                torch.save(best_state, save_path)
        else:
            wait += 1
            marker = ''

        if epoch % 5 == 0 or epoch == 1 or marker:
            lr = optimizer.param_groups[0]['lr']
            print(f'  [{stage_name}] ep {epoch:3d}/{n_epochs}: '
                  f'loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f} | '
                  f'val loss={val_metrics["loss"]:.4f} acc={val_metrics["accuracy"]:.3f} '
                  f'f1={val_metrics["f1_macro"]:.3f} κ={val_metrics["kappa"]:.3f} '
                  f'lr={lr:.1e} ({elapsed:.1f}s){marker}')

        if wait >= patience:
            print(f'  Early stopping at epoch {epoch} (patience={patience})')
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1, history


# ---------------------------------------------------------------------------
# Fold Runner
# ---------------------------------------------------------------------------

def run_fold(fold_idx, args, config):
    """Run full training for one fold."""
    device = get_device()
    model_name = args.model
    seq_length = args.seq_len or config['seq_length']
    batch_size = args.batch_size or config['batch_size']
    lr = args.lr or config['lr']
    weight_decay = config.get('weight_decay', 1e-4)
    use_focal = config.get('use_focal_loss', False)
    focal_gamma = config.get('focal_gamma', 2.0)
    label_smoothing = config.get('label_smoothing', 0.0)
    mixup_alpha = config.get('mixup_alpha', 0.0)
    n_epochs = args.epochs or config['epochs']
    patience = config.get('patience', 20)

    print(f'\n{"=" * 70}')
    print(f'Fold {fold_idx} | model={model_name} | seq_len={seq_length} | '
          f'device={device}')
    print(f'batch_size={batch_size} | lr={lr} | epochs={n_epochs} | '
          f'focal={use_focal} | mixup={mixup_alpha}')
    print(f'{"=" * 70}')

    # ── Load data ──
    print('\n[1] Loading data...')
    data_dir = args.data_dir
    split_path = args.split_path
    (train_recs, val_recs, _,
     train_ds, val_ds, test_ds,
     class_weights) = get_fold_data(fold_idx, seq_length, data_dir, split_path,
                                     augment_train=True)

    print(f'  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')
    print(f'  Class weights: {class_weights.numpy().round(3)}')

    # ── Loss function ──
    if use_focal:
        criterion = FocalLoss(weight=class_weights.to(device),
                              gamma=focal_gamma,
                              label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                        label_smoothing=label_smoothing)

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': NUM_WORKERS,
        'persistent_workers': NUM_WORKERS > 0,
        'prefetch_factor': 2 if NUM_WORKERS > 0 else None,
        'pin_memory': device.type == 'cuda',
    }

    # Balanced sampling for training
    if args.balanced_sampling:
        sampler = get_balanced_sampler(train_ds)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Pre-train CNN (if sequence model) ──
    model_kwargs = config['model_kwargs'].copy()
    is_sequence_model = model_name != 'cnn_only'

    if is_sequence_model and not args.skip_pretrain:
        print('\n[2] Stage 1: Pre-training CNN on individual epochs...')
        cnn_only = SleepCNNOnly(
            n_channels=model_kwargs.get('n_channels', 2),
            n_classes=5,
            feature_dim=model_kwargs.get('feature_dim', 128),
            cnn_type=model_kwargs.get('cnn_type', 'multiscale'),
        ).to(device)

        epoch_train_ds = SleepEpochDataset(train_recs)
        epoch_val_ds = SleepEpochDataset(val_recs)
        epoch_train_loader = DataLoader(epoch_train_ds, shuffle=True, **loader_kwargs)
        epoch_val_loader = DataLoader(epoch_val_ds, shuffle=False, **loader_kwargs)

        cnn_optimizer = optim.Adam(cnn_only.parameters(), lr=1e-3,
                                   weight_decay=weight_decay)
        cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            cnn_optimizer, patience=7, factor=0.5)

        cnn_save_path = os.path.join(output_dir, f'cnn_fold{fold_idx}.pt')
        _, cnn_history = train_loop(cnn_only, epoch_train_loader, epoch_val_loader, criterion,
                   cnn_optimizer, cnn_scheduler, device, n_epochs=args.cnn_epochs,
                   patience=15, stage_name='CNN pretrain', save_path=cnn_save_path)

        torch.save(cnn_only.cnn.state_dict(), cnn_save_path)
        print(f'  Saved CNN: {cnn_save_path}')

        cnn_val = evaluate(cnn_only, epoch_val_loader, criterion, device)
        print(f'  CNN-only val: acc={cnn_val["accuracy"]:.3f} '
              f'f1={cnn_val["f1_macro"]:.3f} κ={cnn_val["kappa"]:.3f}')

        del epoch_train_ds, epoch_val_ds, epoch_train_loader, epoch_val_loader

    # ── Build model ──
    model = build_model(model_name, **model_kwargs).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n  Model: {model_name} ({total_params:,} params)')

    # Load pre-trained CNN
    if is_sequence_model and not args.skip_pretrain:
        cnn_path = os.path.join(output_dir, f'cnn_fold{fold_idx}.pt')
        if os.path.exists(cnn_path):
            model.cnn.load_state_dict(torch.load(cnn_path, map_location=device,
                                                  weights_only=True))
            print('  Loaded pre-trained CNN weights')

    if is_sequence_model:
        # ── Stage 2: Freeze CNN, train temporal layers ──
        print('\n[3] Stage 2: Training temporal layers (CNN frozen)...')
        model.freeze_cnn()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Trainable params: {trainable:,}')

        temporal_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay)
        temporal_scheduler = CosineAnnealingWithWarmup(
            temporal_optimizer, warmup_epochs=3, total_epochs=n_epochs)

        _, temporal_history = train_loop(model, train_loader, val_loader, criterion,
                   temporal_optimizer, temporal_scheduler, device,
                   n_epochs=args.temporal_epochs, patience=patience,
                   clip_norm=1.0, mixup_alpha=mixup_alpha,
                   stage_name='Temporal')

        # ── Stage 3: Fine-tune end-to-end ──
        print('\n[4] Stage 3: Fine-tuning end-to-end...')
        model.unfreeze_cnn()
        trainable = sum(p.numel() for p in model.parameters())
        print(f'  Trainable params: {trainable:,}')

        ft_optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1,
                                    weight_decay=weight_decay)
        ft_scheduler = CosineAnnealingWithWarmup(
            ft_optimizer, warmup_epochs=2, total_epochs=args.finetune_epochs)

        model_path = os.path.join(output_dir,
                                   f'{model_name}_fold{fold_idx}_L{seq_length}.pt')
        _, finetune_history = train_loop(model, train_loader, val_loader, criterion,
                   ft_optimizer, ft_scheduler, device,
                   n_epochs=args.finetune_epochs, patience=patience,
                   clip_norm=1.0, mixup_alpha=mixup_alpha * 0.5,
                   stage_name='Fine-tune', save_path=model_path)

    else:
        # CNN-only: single-stage training
        print('\n[2] Training CNN-only model...')
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
        scheduler = CosineAnnealingWithWarmup(
            optimizer, warmup_epochs=5, total_epochs=n_epochs)

        model_path = os.path.join(output_dir, f'{model_name}_fold{fold_idx}.pt')
        _, single_history = train_loop(model, train_loader, val_loader, criterion,
                   optimizer, scheduler, device, n_epochs=n_epochs,
                   patience=patience, mixup_alpha=mixup_alpha,
                   stage_name='Train', save_path=model_path)

    # ── Final evaluation ──
    print('\n[5] Final evaluation...')
    val_results = evaluate(model, val_loader, criterion, device)
    test_results = evaluate(model, test_loader, criterion, device)

    for name, results in [('Validation', val_results), ('Test', test_results)]:
        print(f'\n  {name}:')
        print(f'    Accuracy:      {results["accuracy"]:.4f}')
        print(f'    F1 (macro):    {results["f1_macro"]:.4f}')
        print(f'    F1 (weighted): {results["f1_weighted"]:.4f}')
        print(f'    Kappa:         {results["kappa"]:.4f}')
        print(f'\n    Classification Report:')
        print(classification_report(results['true'], results['preds'],
                                    target_names=STAGE_NAMES, digits=3))
        print(f'    Confusion Matrix:')
        print(confusion_matrix(results['true'], results['preds']))

    # Save model
    if is_sequence_model:
        model_path = os.path.join(output_dir,
                                   f'{model_name}_fold{fold_idx}_L{seq_length}.pt')
    else:
        model_path = os.path.join(output_dir, f'{model_name}_fold{fold_idx}.pt')
    torch.save(model.state_dict(), model_path)
    print(f'\n  Saved model: {model_path}')

    # Save results
    results_dict = {
        'fold': fold_idx,
        'model': model_name,
        'seq_length': seq_length,
        'val_accuracy': float(val_results['accuracy']),
        'val_f1_macro': float(val_results['f1_macro']),
        'val_f1_weighted': float(val_results['f1_weighted']),
        'val_kappa': float(val_results['kappa']),
        'test_accuracy': float(test_results['accuracy']),
        'test_f1_macro': float(test_results['f1_macro']),
        'test_f1_weighted': float(test_results['f1_weighted']),
        'test_kappa': float(test_results['kappa']),
        'total_params': total_params,
    }

    results_path = os.path.join(output_dir, f'results_fold{fold_idx}.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # Save training history for plotting training curves
    all_history = {}
    if is_sequence_model and not args.skip_pretrain:
        all_history['cnn_pretrain'] = cnn_history
        all_history['temporal'] = temporal_history
        all_history['finetune'] = finetune_history
    else:
        all_history['train'] = single_history

    # Merge into flat lists for easy plotting
    merged = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
              'val_f1': [], 'val_kappa': [], 'stages': []}
    for stage_name, h in all_history.items():
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1', 'val_kappa']:
            merged[key].extend(h.get(key, []))
        merged['stages'].append({
            'name': stage_name,
            'start_epoch': len(merged['train_loss']) - len(h.get('train_loss', [])),
            'end_epoch': len(merged['train_loss']),
        })

    history_path = os.path.join(output_dir, f'history_fold{fold_idx}.json')
    with open(history_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f'  Saved training history: {history_path}')

    return results_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced SleepStageNet Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model selection
    parser.add_argument('--model', type=str, default='conformer',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--list_models', action='store_true',
                        help='List available models and exit')

    # Data
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to NPZ data files')
    parser.add_argument('--split_path', type=str, default=None,
                        help='Path to data_split_v1.npz')

    # Fold selection
    parser.add_argument('--fold', type=int, default=0, help='Fold index (0-19)')
    parser.add_argument('--all_folds', action='store_true',
                        help='Train all 20 folds')
    parser.add_argument('--n_folds', type=int, default=20,
                        help='Number of folds (for --all_folds)')

    # Training hyperparams (None = use model config defaults)
    parser.add_argument('--seq_len', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Total epochs per stage')
    parser.add_argument('--cnn_epochs', type=int, default=50,
                        help='CNN pre-training epochs')
    parser.add_argument('--temporal_epochs', type=int, default=50,
                        help='Temporal layer training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=30,
                        help='End-to-end fine-tuning epochs')

    # Options
    parser.add_argument('--skip_pretrain', action='store_true',
                        help='Skip CNN pre-training')
    parser.add_argument('--balanced_sampling', action='store_true',
                        help='Use balanced class sampling')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--drive_ckpt_dir', type=str, default=None,
                        help='Google Drive checkpoint dir — if set, copies fold '
                             'output here after each fold (for crash resilience)')

    args = parser.parse_args()

    if args.list_models:
        print('\nAvailable models:')
        print(f'{"Model":<20s} {"Default seq_len":<16s} {"Default batch":<14s}')
        print('-' * 50)
        for name, cfg in MODEL_CONFIGS.items():
            print(f'{name:<20s} {cfg["seq_length"]:<16d} {cfg["batch_size"]:<14d}')
        return

    config = MODEL_CONFIGS[args.model]

    if args.all_folds:
        # ── Restore previous results from Drive (if available) ──
        output_dir = os.path.join(args.output_dir, args.model)
        if args.drive_ckpt_dir:
            import shutil
            drive_model_dir = os.path.join(args.drive_ckpt_dir, args.model)
            if os.path.isdir(drive_model_dir):
                os.makedirs(output_dir, exist_ok=True)
                restored = 0
                for fname in os.listdir(drive_model_dir):
                    src = os.path.join(drive_model_dir, fname)
                    dst = os.path.join(output_dir, fname)
                    if not os.path.exists(dst):
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                            restored += 1
                if restored:
                    print(f'Restored {restored} file(s) from Drive: {drive_model_dir}')

        all_results = []
        for fold_idx in range(args.n_folds):
            # ── Resume support: skip folds that already have results ──
            output_dir = os.path.join(args.output_dir, args.model)
            results_file = os.path.join(output_dir, f'results_fold{fold_idx}.json')
            if os.path.exists(results_file):
                with open(results_file) as f:
                    cached = json.load(f)
                all_results.append(cached)
                print(f'\n{"=" * 70}')
                print(f'Fold {fold_idx} already complete — loaded from {results_file}')
                print(f'  test_acc={cached["test_accuracy"]:.4f}  '
                      f'test_f1={cached["test_f1_macro"]:.4f}  '
                      f'test_κ={cached["test_kappa"]:.4f}')
                continue

            result = run_fold(fold_idx, args, config)
            all_results.append(result)

            # Incremental upload to Drive after each fold
            if args.drive_ckpt_dir:
                import shutil
                src_dir = os.path.join(args.output_dir, args.model)
                dst_dir = os.path.join(args.drive_ckpt_dir, args.model)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print(f'  ✓ Fold {fold_idx} synced to Drive: {dst_dir}')

        # Summary
        print(f'\n{"=" * 70}')
        print(f'CROSS-VALIDATION SUMMARY — {args.model}')
        print(f'{"=" * 70}')

        metrics = ['test_accuracy', 'test_f1_macro', 'test_f1_weighted', 'test_kappa']
        for m in metrics:
            vals = [r[m] for r in all_results]
            print(f'  {m:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')

        # Save summary
        summary_path = os.path.join(args.output_dir, args.model, 'cv_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'model': args.model,
                'n_folds': args.n_folds,
                'per_fold': all_results,
                'mean': {m: float(np.mean([r[m] for r in all_results])) for m in metrics},
                'std': {m: float(np.std([r[m] for r in all_results])) for m in metrics},
            }, f, indent=2)
        print(f'\nSaved summary: {summary_path}')

    else:
        result = run_fold(args.fold, args, config)

        # Upload single fold to Drive if requested
        if args.drive_ckpt_dir:
            import shutil
            src_dir = os.path.join(args.output_dir, args.model)
            dst_dir = os.path.join(args.drive_ckpt_dir, args.model)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            print(f'  ✓ Fold {args.fold} synced to Drive: {dst_dir}')

        print(f'\n{"=" * 70}')
        print('SUMMARY')
        print(f'{"=" * 70}')
        for k, v in result.items():
            if isinstance(v, float):
                print(f'  {k}: {v:.4f}')
            else:
                print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
