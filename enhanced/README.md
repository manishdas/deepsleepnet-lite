# Enhanced SleepStageNet: Attention & Transformer Models

Improved sleep stage classification models building on the CNN+BiLSTM baseline from `temporal/` and the DeepSleepNet-Lite TensorFlow baseline.

## Results are stored here: [SleepStageNet/enhanced](https://drive.google.com/drive/folders/1BA-HIQLNm0TSO1vYdLUDJxHB9au6AhDg)

## Models

| Model | Architecture | Key Innovation | ~Params |
|-------|-------------|----------------|---------|
| `cnn_only` | Multi-scale dual-path CNN | Fine + coarse filter paths (like DeepSleepNet-Lite) | ~230K |
| `cnn_bilstm` | Multi-scale CNN + 2-layer BiLSTM | Enhanced CNN backbone, deeper LSTM, longer sequences | ~560K |

| `conformer` | Multi-scale CNN + Conformer blocks | Combines attention (global) + depthwise conv (local) | ~700K |

## Improvements Over Baseline

### Architecture
- **Multi-scale CNN backbone**: Dual-path with small filters (50-sample, ~0.5s) capturing high-frequency spindles and large filters (400-sample, ~4s) capturing slow waves — inspired by DeepSleepNet-Lite's TensorFlow architecture
- **Multi-Head Self-Attention**: Captures pairwise relationships between non-adjacent epochs (e.g., recognizing sleep cycle patterns)
- **Conformer blocks**: Macaron structure combining FFN → MHSA → Depthwise Conv → FFN, effective for both local transitions and global patterns
- **Pre-LayerNorm Transformer**: More stable training than Post-LN
- **Learnable positional encoding**: Better than fixed sinusoidal for short sequences

### Training
- **Focal Loss** (γ=2.0): Down-weights easy examples, focuses on hard-to-classify N1 epochs
- **Label smoothing** (0.05): Reduces overconfident predictions
- **Mixup augmentation**: Interpolates training pairs for regularization
- **EEG data augmentation**: Time shift (±50 samples), amplitude scaling (±10%), Gaussian noise
- **Cosine annealing with warmup**: Smoother LR schedule than ReduceLROnPlateau
- **3-stage training**: Pretrain CNN → Train temporal (CNN frozen) → Fine-tune end-to-end
- **Balanced class sampling** option via WeightedRandomSampler

### Sequence Length
- Baseline used L=5 (±2 epochs of context)
- Enhanced models support L=11 (±5) to L=21 (±10) for broader temporal context
- Longer sequences particularly help with N1/REM disambiguation

## Files

| File | Purpose |
|------|---------|
| `models.py` | All 3 model architectures + Focal Loss + Mixup utilities |
| `data_loader.py` | Data loading with augmentation, balanced sampling |
| `train.py` | Training script with 3-stage training, config presets |
| `notebooks/Enhanced_SleepNet_Colab.ipynb` | Google Colab notebook for full pipeline |

## Quick Start

```bash
# List available models
python train.py --list_models

# Train single fold (Conformer, ~15 min on GPU)
python train.py --model conformer --fold 0 \
    --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
    --split_path /path/to/data_split_v1.npz

# Train all 20 folds
python train.py --model conformer --all_folds \
    --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
    --split_path /path/to/data_split_v1.npz

# Compare all models on fold 0
for model in cnn_only cnn_bilstm conformer; do
    python train.py --model $model --fold 0 \
        --data_dir /path/to/eeg_FpzCz_PzOz_v1 \
        --split_path /path/to/data_split_v1.npz
done
```

## Default Hyperparameters

| Model | Seq Len | Batch | LR | Focal | Mixup | Label Smooth |
|-------|---------|-------|-----|-------|-------|-------------|
| `cnn_only` | 1 | 64 | 1e-3 | No | 0.0 | 0.05 |
| `cnn_bilstm` | 11 | 32 | 5e-4 | No | 0.1 | 0.05 |
| `conformer` | 11 | 32 | 3e-4 | Yes (γ=1.5) | 0.1 | 0.05 |

## Architecture Diagrams

### Conformer Block (recommended model)
```
Input (B, L, D)
    │
    ├──→ LayerNorm → FFN → ×0.5 → (+) ← residual
    │
    ├──→ LayerNorm → Multi-Head Self-Attention → (+) ← residual
    │
    ├──→ LayerNorm → Pointwise Conv → GLU → Depthwise Conv → BN → GELU → Pointwise Conv → (+) ← residual
    │
    ├──→ LayerNorm → FFN → ×0.5 → (+) ← residual
    │
    └──→ LayerNorm → Output (B, L, D)
```

### Multi-Scale CNN Feature Extractor
```
Input (B, 2, 3000)
    │
    ├─ Path A (fine-grained):
    │  Conv1d(2→32, k=50, s=6) + BN + ReLU + MaxPool(8)
    │  Conv1d(32→64, k=8) × 3 + MaxPool(4)
    │  AdaptiveAvgPool(1) → (B, 64)
    │
    ├─ Path B (coarse temporal):
    │  Conv1d(2→32, k=400, s=50) + BN + ReLU + MaxPool(4)
    │  Conv1d(32→64, k=6) × 3 + MaxPool(2)
    │  AdaptiveAvgPool(1) → (B, 64)
    │
    └─ Concat → Dropout(0.5) → Linear(128→D) → ReLU → (B, D)
```

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

Our Transformer and Conformer models draw inspiration from these papers, adapted to work with the Sleep-EDF v1 dataset and the existing DeepSleepNet-Lite data pipeline.
