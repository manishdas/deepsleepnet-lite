# Enhanced SleepStageNet: Attention & Transformer Models

Improved sleep stage classification models building on the CNN+BiLSTM baseline from `temporal/` and the DeepSleepNet-Lite TensorFlow baseline.

## Results are stored here: [SleepStageNet/enhanced](https://drive.google.com/drive/folders/1BA-HIQLNm0TSO1vYdLUDJxHB9au6AhDg)

## Models

| Model | Architecture | Key Innovation | Params |
|-------|-------------|----------------|--------|
| `cnn_only` | Multi-scale dual-path CNN | Fine + coarse filter paths (like DeepSleepNet-Lite) | ~191K |
| `cnn_bilstm` | Multi-scale CNN + 2-layer BiLSTM | Enhanced CNN backbone, deeper LSTM, longer sequences | ~851K |
| `conformer` | Multi-scale CNN + Conformer blocks | Combines attention (global) + depthwise conv (local) | ~1.34M |

## Improvements Over Baseline

### Architecture
- **Multi-scale CNN backbone**: Dual-path with small filters (50-sample, ~0.5s) capturing high-frequency spindles and large filters (400-sample, ~4s) capturing slow waves — inspired by DeepSleepNet-Lite's TensorFlow architecture
- **Conformer blocks** (conformer model): Macaron-style FFN → MHSA → Depthwise Conv → FFN, effective for both local transitions and global patterns
- **2-layer BiLSTM** (cnn_bilstm model): Deeper recurrence captures longer temporal dependencies
- **Learnable positional encoding**: Better than fixed sinusoidal for short sequences

### Training
- **Focal Loss** (γ=1.5, conformer only): Down-weights easy examples, focuses on hard-to-classify N1 epochs
- **Label smoothing** (0.05): Reduces overconfident predictions
- **Mixup augmentation** (α=0.1, cnn_bilstm & conformer): Interpolates training pairs for regularization
- **EEG data augmentation**: Time shift (±50 samples), amplitude scaling (±10%), Gaussian noise
- **Cosine annealing with warmup**: 3-epoch linear warmup, then cosine decay
- **3-stage training** (sequence models): Pretrain CNN → Train temporal (CNN frozen) → Fine-tune end-to-end
- **Balanced class sampling** option via `WeightedRandomSampler` (`--balanced_sampling` flag)
- **Gradient accumulation** supported in training loop

### Sequence Length
- Baseline used L=5 (±2 epochs of context)
- Enhanced models use L=11 (±5 epochs) by default for broader temporal context
- Longer sequences particularly help with N1/REM disambiguation

## Files

### Python Modules

| File | Description |
|------|-------------|
| `models.py` | **All model architectures & training utilities (766 lines).** Defines 5 architectures sharing a CNN backbone: `SleepCNNOnly`, `SleepCNNBiLSTM`, `SleepAttnBiLSTM` (experimental), `SleepTransformerNet` (experimental), `SleepConformer`. Includes `CNNFeatureExtractor` (single-scale) and `MultiScaleCNNExtractor` (dual-path k=50 + k=400). Also provides `LearnablePositionalEncoding`, `SinusoidalPositionalEncoding`, `FocalLoss`, `mixup_data()`, `mixup_criterion()`, and `MODEL_REGISTRY` / `MODEL_CONFIGS` / `build_model()` for easy model lookup and default hyperparameters. 3 models registered (`cnn_only`, `cnn_bilstm`, `conformer`); 2 unregistered experimental (`SleepAttnBiLSTM`, `SleepTransformerNet`). |
| `data_loader.py` | **Data loading pipeline (357 lines).** Loads per-subject NPZ recordings and the 20-fold LOSO-CV split. Builds sliding-window sequences of epochs via `create_sequences()` and `build_sequences_from_recordings()`. Provides two dataset classes: `SleepSequenceDataset` (returns L-epoch sequences for temporal models) and `SleepEpochDataset` (single epochs for CNN-only). Includes `EEGAugmentation` (time shift ±50 samples, amplitude scaling ±10%, Gaussian noise), `get_balanced_sampler()` using sklearn's `compute_class_weight`, and `get_fold_data()` — a single-call function that returns train/val/test datasets + class weights for any fold. |
| `train.py` | **Full training script (644 lines).** Supports all model architectures via `MODEL_REGISTRY`. Implements 3-stage training: (1) pretrain CNN, (2) train temporal layers with CNN frozen, (3) fine-tune end-to-end. Key components: `CosineAnnealingWithWarmup` scheduler, `train_one_epoch()` with optional Mixup, `evaluate()`, `train_loop()` with early stopping and best-model checkpointing, `run_fold()` for full fold orchestration, and `main()` for single-fold or all-20-fold CV with aggregated metrics. Supports `--drive_ckpt_dir` for Colab crash resilience, `--balanced_sampling`, gradient accumulation, and `--list_models` to enumerate the registry. |

### Notebooks

| File | Description |
|------|-------------|
| `notebooks/[Conformer] Enhanced_SleepNet_Colab.ipynb` | Google Colab notebook for training the **Conformer** model. Includes Drive mounting, data setup, 20-fold LOSO-CV execution, checkpoint sync to Drive, and results dashboard. |
| `notebooks/[LSTM] of Enhanced_SleepNet_Colab.ipynb` | Google Colab notebook for training the **CNN+BiLSTM** model. Same pipeline as the Conformer notebook but configured for the `cnn_bilstm` architecture. |

## Quick Start

```bash
# List available models
python train.py --list_models

# Train single fold (Conformer)
python train.py --model conformer --fold 0 \
    --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
    --split_path /path/to/data_split_v1.npz

# Train all 20 folds with Drive checkpoint sync (for crash resilience)
python train.py --model conformer --all_folds \
    --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
    --split_path /path/to/data_split_v1.npz \
    --drive_ckpt_dir /content/drive/MyDrive/deepsleepnet-lite

# Custom training stages
python train.py --model cnn_bilstm --fold 0 \
    --cnn_epochs 50 --temporal_epochs 50 --finetune_epochs 30 \
    --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
    --split_path /path/to/data_split_v1.npz

# Compare all models on fold 0
for model in cnn_only cnn_bilstm conformer; do
    python train.py --model $model --fold 0 \
        --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
        --split_path /path/to/data_split_v1.npz
done
```

### Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | `cnn_only`, `cnn_bilstm`, or `conformer` |
| `--fold` | 0 | Fold index (0–19) |
| `--all_folds` | — | Train all 20 folds |
| `--cnn_epochs` | 50 | CNN pre-training epochs (stage 1) |
| `--temporal_epochs` | 50 | Temporal training epochs with CNN frozen (stage 2) |
| `--finetune_epochs` | 30 | End-to-end fine-tuning epochs (stage 3) |
| `--skip_pretrain` | — | Skip CNN pre-training (stage 1) |
| `--balanced_sampling` | — | Use `WeightedRandomSampler` for class balance |
| `--drive_ckpt_dir` | — | Copy fold outputs to Drive after each fold |

## Default Hyperparameters

| Model | Seq Len | Batch | LR | Focal | Mixup | Label Smooth | Patience |
|-------|---------|-------|-----|-------|-------|-------------|----------|
| `cnn_only` | 1 | 64 | 1e-3 | No | 0.0 | 0.05 | 15 |
| `cnn_bilstm` | 11 | 32 | 5e-4 | No | 0.1 | 0.05 | 15 |
| `conformer` | 11 | 32 | 3e-4 | Yes (γ=1.5) | 0.1 | 0.05 | 20 |

## Architecture Diagrams

### Conformer Block
```
Input (B, L, D=128)
    │
    ├──→ LayerNorm → FFN (128→512→128, GELU) → ×0.5 → (+) ← residual
    │
    ├──→ LayerNorm → Multi-Head Self-Attention (4 heads) → (+) ← residual
    │
    ├──→ LayerNorm → Pointwise Conv (128→256) → GLU (→128) → Depthwise Conv (k=7) → BN → GELU → Pointwise Conv (128→128) → (+) ← residual
    │
    ├──→ LayerNorm → FFN (128→512→128, GELU) → ×0.5 → (+) ← residual
    │
    └──→ LayerNorm → Output (B, L, D=128)
```

### Multi-Scale CNN Feature Extractor
```
Input (B, 2, 3000)
    │
    ├─ Path A (fine-grained, ~0.5s):
    │  Conv1d(2→32, k=50, s=6) + BN + ReLU + MaxPool(8)
    │  Conv1d(32→64, k=8) + BN + ReLU
    │  Conv1d(64→64, k=8) + BN + ReLU
    │  Conv1d(64→64, k=8) + BN + ReLU + MaxPool(4)
    │  AdaptiveAvgPool(1) → (B, 64)
    │
    ├─ Path B (coarse temporal, ~4s):
    │  Conv1d(2→32, k=400, s=50) + BN + ReLU + MaxPool(4)
    │  Conv1d(32→64, k=6) + BN + ReLU
    │  Conv1d(64→64, k=6) + BN + ReLU
    │  Conv1d(64→64, k=6) + BN + ReLU + MaxPool(2)
    │  AdaptiveAvgPool(1) → (B, 64)
    │
    └─ Concat → Dropout(0.5) → Linear(128→D) → ReLU → (B, D)
```

## Resume & Crash Resilience

When training with `--all_folds --drive_ckpt_dir <path>`:
1. Before starting, existing results are restored from Drive → local output dir
2. Folds with existing `results_fold{N}.json` are automatically skipped
3. After each fold completes, outputs (checkpoint + results JSON + training history) are copied to Drive

This allows restarting interrupted Colab sessions without losing completed folds.

## HuggingFace Models Survey

We surveyed HuggingFace for pretrained EEG/sleep staging models:

| Model | Type | Relevance |
|-------|------|-----------|
| `karnamgyal/sleep-stage-classifier` | CNN-LSTM (PyTorch) | Same task but ~73% accuracy — lower than our baseline |
| `haseebnawazz/sleep-stage-classifier-model` | sklearn/Joblib | Not deep learning, feature-engineered |
| `Soromis/BP-transformer-EEG` | Transformer | EEG but for blood pressure, not sleep staging |

The EEG sleep staging field primarily uses custom architectures from papers rather than pretrained foundation models. Key references:
- **SleepTransformer** (Phan et al., 2022) — Epoch-level + sequence-level transformers
- **L-SeqSleepNet** (Phan et al., 2023) — Long sequence modeling with attention
- **U-Sleep** (Perslev et al., 2021) — U-Net style architecture for sleep staging

Our Conformer model draws inspiration from these papers, adapted to work with the Sleep-EDF v1 dataset and the existing DeepSleepNet-Lite data pipeline.


Checkpoints are stored here: https://drive.google.com/drive/folders/1-1RIm62i5ydsJal3fFTOOnRFSCnBByTM?usp=drive_link