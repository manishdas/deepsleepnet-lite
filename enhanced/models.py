#!/usr/bin/env python
"""
Enhanced model architectures for sleep stage classification.

Models (all use the same CNN backbone, differing in temporal modeling):
  1. SleepCNNOnly         — Single-epoch CNN classifier (baseline)
  2. SleepCNNBiLSTM       — CNN + BiLSTM (from temporal/)
  3. SleepAttnBiLSTM      — CNN + Multi-Head Attention + BiLSTM
  4. SleepTransformerNet  — CNN + Transformer Encoder (no LSTM)
  5. SleepConformer       — CNN + Conformer blocks (attention + conv)

Key improvements over the temporal/ models:
  - Multi-scale CNN feature extractor (small + large filters, like DeepSleepNet-Lite)
  - Multi-Head Self-Attention for capturing inter-epoch dependencies
  - Learnable positional encoding for sequence position
  - Pre-LayerNorm Transformer blocks with residual connections
  - Conformer: combines local convolution with global attention
  - Focal loss support for class imbalance
  - Label smoothing built into training
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CNN Feature Extractors
# ---------------------------------------------------------------------------

class CNNFeatureExtractor(nn.Module):
    """1D CNN (same as temporal/ baseline) for single-scale feature extraction.

    Conv1d(C→32, k=50) + BN + ReLU + MaxPool(4)
    Conv1d(32→64, k=25) + BN + ReLU + MaxPool(4)
    Conv1d(64→128, k=10) + BN + ReLU + AdaptiveAvgPool(1)
    Linear(128→feature_dim) + ReLU

    Input:  (batch, n_channels, 3000)
    Output: (batch, feature_dim)
    """

    def __init__(self, n_channels=2, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=50, padding=25)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=10, padding=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)
        x = self.relu(self.fc(x))
        return x


class MultiScaleCNNExtractor(nn.Module):
    """Dual-path CNN inspired by DeepSleepNet-Lite: small + large filters.

    Path A (fine-grained, ~50 Hz resolution):
      Conv1d(C→32, k=50, s=6) + BN + ReLU + MaxPool(8)
      Conv1d(32→64, k=8) + BN + ReLU
      Conv1d(64→64, k=8) + BN + ReLU
      Conv1d(64→64, k=8) + BN + ReLU + MaxPool(4)
      AdaptiveAvgPool(1) → (64,)

    Path B (coarse temporal, ~4s resolution):
      Conv1d(C→32, k=400, s=50) + BN + ReLU + MaxPool(4)
      Conv1d(32→64, k=6) + BN + ReLU
      Conv1d(64→64, k=6) + BN + ReLU
      Conv1d(64→64, k=6) + BN + ReLU + MaxPool(2)
      AdaptiveAvgPool(1) → (64,)

    Concat → (128,) → Linear → (feature_dim,)

    Input:  (batch, n_channels, 3000)
    Output: (batch, feature_dim)
    """

    def __init__(self, n_channels=2, feature_dim=128):
        super().__init__()
        self.feature_dim = feature_dim

        # Path A: small filters (high frequency features)
        self.a_conv1 = nn.Conv1d(n_channels, 32, kernel_size=50, stride=6, padding=25)
        self.a_bn1 = nn.BatchNorm1d(32)
        self.a_pool1 = nn.MaxPool1d(8)
        self.a_conv2 = nn.Conv1d(32, 64, kernel_size=8, padding=4)
        self.a_bn2 = nn.BatchNorm1d(64)
        self.a_conv3 = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.a_bn3 = nn.BatchNorm1d(64)
        self.a_conv4 = nn.Conv1d(64, 64, kernel_size=8, padding=4)
        self.a_bn4 = nn.BatchNorm1d(64)
        self.a_pool2 = nn.MaxPool1d(4)
        self.a_gap = nn.AdaptiveAvgPool1d(1)

        # Path B: large filters (low frequency / temporal patterns)
        self.b_conv1 = nn.Conv1d(n_channels, 32, kernel_size=400, stride=50, padding=200)
        self.b_bn1 = nn.BatchNorm1d(32)
        self.b_pool1 = nn.MaxPool1d(4)
        self.b_conv2 = nn.Conv1d(32, 64, kernel_size=6, padding=3)
        self.b_bn2 = nn.BatchNorm1d(64)
        self.b_conv3 = nn.Conv1d(64, 64, kernel_size=6, padding=3)
        self.b_bn3 = nn.BatchNorm1d(64)
        self.b_conv4 = nn.Conv1d(64, 64, kernel_size=6, padding=3)
        self.b_bn4 = nn.BatchNorm1d(64)
        self.b_pool2 = nn.MaxPool1d(2)
        self.b_gap = nn.AdaptiveAvgPool1d(1)

        # Merge
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Path A
        a = self.a_pool1(self.relu(self.a_bn1(self.a_conv1(x))))
        a = self.relu(self.a_bn2(self.a_conv2(a)))
        a = self.relu(self.a_bn3(self.a_conv3(a)))
        a = self.a_pool2(self.relu(self.a_bn4(self.a_conv4(a))))
        a = self.a_gap(a).squeeze(-1)  # (B, 64)

        # Path B
        b = self.b_pool1(self.relu(self.b_bn1(self.b_conv1(x))))
        b = self.relu(self.b_bn2(self.b_conv2(b)))
        b = self.relu(self.b_bn3(self.b_conv3(b)))
        b = self.b_pool2(self.relu(self.b_bn4(self.b_conv4(b))))
        b = self.b_gap(b).squeeze(-1)  # (B, 64)

        # Merge
        out = torch.cat([a, b], dim=1)  # (B, 128)
        out = self.dropout(out)
        out = self.relu(self.fc(out))   # (B, feature_dim)
        return out


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequences up to max_len."""

    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Model 1: CNN-Only (single epoch, no temporal context)
# ---------------------------------------------------------------------------

class SleepCNNOnly(nn.Module):
    """CNN-only classifier for comparison / pre-training."""

    def __init__(self, n_channels=2, n_classes=5, feature_dim=64,
                 cnn_type='single'):
        super().__init__()
        if cnn_type == 'multiscale':
            self.cnn = MultiScaleCNNExtractor(n_channels, feature_dim)
        else:
            self.cnn = CNNFeatureExtractor(n_channels, feature_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        # x may be (B, C, T) or (B, 1, C, T) from SleepSequenceDataset
        if x.ndim == 4:
            x = x.squeeze(1)  # (B, 1, C, T) -> (B, C, T)
        features = self.cnn(x)
        features = self.dropout(features)
        return self.classifier(features)


# ---------------------------------------------------------------------------
# Model 2: CNN + BiLSTM (baseline temporal model from temporal/)
# ---------------------------------------------------------------------------

class SleepCNNBiLSTM(nn.Module):
    """CNN + BiLSTM (same architecture as temporal/ for comparison)."""

    def __init__(self, n_channels=2, feature_dim=64, lstm_hidden=128,
                 lstm_layers=1, n_classes=5, dropout=0.3, cnn_type='single'):
        super().__init__()
        if cnn_type == 'multiscale':
            self.cnn = MultiScaleCNNExtractor(n_channels, feature_dim)
        else:
            self.cnn = CNNFeatureExtractor(n_channels, feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x):
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        features = self.cnn(x)
        features = features.reshape(B, L, -1)
        lstm_out, _ = self.lstm(features)
        center = L // 2
        center_feat = lstm_out[:, center, :]
        center_feat = self.dropout(center_feat)
        return self.classifier(center_feat)

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Model 3: CNN + Multi-Head Attention + BiLSTM
# ---------------------------------------------------------------------------

class SleepAttnBiLSTM(nn.Module):
    """CNN features → Multi-Head Self-Attention → BiLSTM → Classifier.

    The attention layer captures pairwise epoch relationships before
    the BiLSTM models sequential transitions. This combination allows
    the model to attend to relevant non-adjacent epochs while still
    respecting temporal ordering.

    Architecture:
      1. CNN: (B, L, C, T) → (B, L, feature_dim)
      2. Positional Encoding
      3. Multi-Head Self-Attention (n_heads, n_attn_layers)
      4. BiLSTM
      5. Center epoch → Classifier
    """

    def __init__(self, n_channels=2, feature_dim=128, n_heads=4,
                 n_attn_layers=2, lstm_hidden=128, lstm_layers=1,
                 n_classes=5, dropout=0.3, cnn_type='multiscale'):
        super().__init__()
        if cnn_type == 'multiscale':
            self.cnn = MultiScaleCNNExtractor(n_channels, feature_dim)
        else:
            self.cnn = CNNFeatureExtractor(n_channels, feature_dim)

        self.pos_enc = LearnablePositionalEncoding(feature_dim, dropout=dropout)

        # Transformer encoder layers (attention + FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm (more stable training)
        )
        self.attn_layers = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x):
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        features = self.cnn(x)                # (B*L, D)
        features = features.reshape(B, L, -1) # (B, L, D)

        # Attention
        features = self.pos_enc(features)
        features = self.attn_layers(features)
        features = self.attn_norm(features)

        # BiLSTM
        lstm_out, _ = self.lstm(features)
        center = L // 2
        center_feat = lstm_out[:, center, :]
        center_feat = self.dropout(center_feat)
        return self.classifier(center_feat)

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Model 4: CNN + Transformer Encoder (no LSTM)
# ---------------------------------------------------------------------------

class SleepTransformerNet(nn.Module):
    """Pure Transformer approach: CNN features → Transformer Encoder → Classifier.

    Inspired by SleepTransformer (Phan et al., 2022) and ViT-style classification.
    Uses a [CLS]-like approach: extracts the center epoch from the Transformer
    output for classification.

    Architecture:
      1. Multi-scale CNN: (B, L, C, T) → (B, L, D)
      2. Positional Encoding
      3. N x Transformer Encoder Layers (Pre-LN, GELU, Multi-Head Attention)
      4. Center epoch extraction → LayerNorm → Classifier
    """

    def __init__(self, n_channels=2, feature_dim=128, n_heads=8,
                 n_layers=4, ff_dim=512, n_classes=5, dropout=0.3,
                 cnn_type='multiscale', max_seq_len=25):
        super().__init__()
        if cnn_type == 'multiscale':
            self.cnn = MultiScaleCNNExtractor(n_channels, feature_dim)
        else:
            self.cnn = CNNFeatureExtractor(n_channels, feature_dim)

        self.pos_enc = LearnablePositionalEncoding(feature_dim, max_len=max_seq_len,
                                                    dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        features = self.cnn(x)                # (B*L, D)
        features = features.reshape(B, L, -1) # (B, L, D)

        features = self.pos_enc(features)
        features = self.transformer(features) # (B, L, D)
        features = self.norm(features)

        # Center epoch
        center = L // 2
        center_feat = features[:, center, :]
        center_feat = self.dropout(center_feat)
        return self.classifier(center_feat)

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Model 5: CNN + Conformer (Attention + Depthwise Conv)
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    """Conformer block: FFN → MHSA → Conv → FFN (Macaron structure).

    Combines the global context from self-attention with local pattern
    extraction from depth-wise convolution. Originally from speech
    recognition (Gulati et al., 2020), adapted for EEG epoch sequences.
    """

    def __init__(self, d_model, n_heads, conv_kernel=7, ff_expansion=4,
                 dropout=0.1):
        super().__init__()

        # First half-step FFN
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

        # Multi-Head Self-Attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),  # pointwise expand
            nn.GLU(dim=1),                                     # gated linear unit
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel,
                      padding=conv_kernel // 2, groups=d_model),  # depthwise
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),  # pointwise project
            nn.Dropout(dropout),
        )

        # Second half-step FFN
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, D)

        # First half-step FFN (with 0.5 residual)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        # MHSA
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.attn_dropout(attn_out)

        # Convolution module
        residual = x
        x_conv = self.conv_norm(x)
        x_conv = x_conv.transpose(1, 2)  # (B, D, L) for Conv1d
        x_conv = self.conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # (B, L, D)
        x = residual + x_conv

        # Second half-step FFN (with 0.5 residual)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))

        x = self.final_norm(x)
        return x


class SleepConformer(nn.Module):
    """CNN + Conformer for sleep stage classification.

    Conformer blocks combine self-attention (global inter-epoch relationships)
    with depthwise convolution (local transition patterns), making it
    particularly well-suited for sequential sleep data where both local
    transitions (N2→N3) and global patterns (sleep cycle structure) matter.

    Architecture:
      1. Multi-scale CNN: (B, L, C, T) → (B, L, D)
      2. Positional Encoding
      3. N x Conformer Blocks
      4. Center epoch → Classifier
    """

    def __init__(self, n_channels=2, feature_dim=128, n_heads=4,
                 n_layers=3, conv_kernel=7, n_classes=5, dropout=0.2,
                 cnn_type='multiscale', max_seq_len=25):
        super().__init__()
        if cnn_type == 'multiscale':
            self.cnn = MultiScaleCNNExtractor(n_channels, feature_dim)
        else:
            self.cnn = CNNFeatureExtractor(n_channels, feature_dim)

        self.pos_enc = LearnablePositionalEncoding(feature_dim, max_len=max_seq_len,
                                                    dropout=dropout)

        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(feature_dim, n_heads, conv_kernel,
                           ff_expansion=4, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        B, L, C, T = x.shape
        x = x.reshape(B * L, C, T)
        features = self.cnn(x)
        features = features.reshape(B, L, -1)

        features = self.pos_enc(features)
        for block in self.conformer_blocks:
            features = block(features)
        features = self.norm(features)

        center = L // 2
        center_feat = features[:, center, :]
        center_feat = self.dropout(center_feat)
        return self.classifier(center_feat)

    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) for handling class imbalance.

    Adds a modulating factor (1-p_t)^gamma to cross-entropy loss, focusing
    training on hard-to-classify examples (e.g., N1 stage).
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0,
                 reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                  reduction='none',
                                  label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Mixup Augmentation
# ---------------------------------------------------------------------------

def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch.

    Args:
        x: input tensor (B, L, C, T) or (B, C, T)
        y: label tensor (B,) — integer class indices
        alpha: Beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    'cnn_only': SleepCNNOnly,
    'cnn_bilstm': SleepCNNBiLSTM,
    'conformer': SleepConformer,
}


def build_model(model_name, **kwargs):
    """Build a model by name.

    Args:
        model_name: one of 'cnn_only', 'cnn_bilstm', 'conformer'
        **kwargs: model-specific arguments

    Returns:
        nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)


# ---------------------------------------------------------------------------
# Hyperparameter Configurations
# ---------------------------------------------------------------------------

# Default configurations for each model, tuned for Sleep-EDF dataset
MODEL_CONFIGS = {
    'cnn_only': {
        'model_kwargs': {
            'n_channels': 2, 'n_classes': 5, 'feature_dim': 128,
            'cnn_type': 'multiscale',
        },
        'seq_length': 1,  # single epoch
        'batch_size': 64,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 60,
        'patience': 15,
        'use_focal_loss': False,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.0,
    },
    'cnn_bilstm': {
        'model_kwargs': {
            'n_channels': 2, 'feature_dim': 128, 'lstm_hidden': 128,
            'lstm_layers': 2, 'n_classes': 5, 'dropout': 0.3,
            'cnn_type': 'multiscale',
        },
        'seq_length': 11,
        'batch_size': 32,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'epochs': 60,
        'patience': 15,
        'use_focal_loss': False,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.1,
    },
    'conformer': {
        'model_kwargs': {
            'n_channels': 2, 'feature_dim': 128, 'n_heads': 4,
            'n_layers': 3, 'conv_kernel': 7, 'n_classes': 5, 'dropout': 0.2,
            'cnn_type': 'multiscale', 'max_seq_len': 25,
        },
        'seq_length': 11,
        'batch_size': 32,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'epochs': 80,
        'patience': 20,
        'use_focal_loss': True,
        'focal_gamma': 1.5,
        'label_smoothing': 0.05,
        'mixup_alpha': 0.1,
    },
}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 70)
    print('Enhanced SleepStageNet Models Verification')
    print('=' * 70)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'Device: {device}\n')

    # Test each model
    B, L, C, T = 4, 11, 2, 3000
    x_seq = torch.randn(B, L, C, T, device=device)
    x_single = torch.randn(B, C, T, device=device)

    for name, cls in MODEL_REGISTRY.items():
        config = MODEL_CONFIGS[name]
        model = cls(**config['model_kwargs']).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        if name == 'cnn_only':
            out = model(x_single)
            in_shape = x_single.shape
        else:
            out = model(x_seq)
            in_shape = x_seq.shape

        print(f'{name:20s}  input={str(list(in_shape)):25s}  '
              f'output={str(list(out.shape)):15s}  params={n_params:>10,}')

        # Test freeze/unfreeze
        if hasattr(model, 'freeze_cnn'):
            model.freeze_cnn()
            t = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.unfreeze_cnn()
            u = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{"":20s}  frozen_trainable={t:,}  unfrozen_trainable={u:,}')

    # Test Focal Loss
    print('\nFocal Loss test:')
    logits = torch.randn(8, 5, device=device)
    targets = torch.randint(0, 5, (8,), device=device)
    fl = FocalLoss(gamma=2.0)
    loss = fl(logits, targets)
    print(f'  loss = {loss.item():.4f}')

    # Test Mixup
    print('\nMixup test:')
    mx, ya, yb, lam = mixup_data(x_seq, targets[:4], alpha=0.2)
    print(f'  lambda = {lam:.3f}  shapes: {mx.shape}, {ya.shape}, {yb.shape}')

    print('\nAll checks passed.')
