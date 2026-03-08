#!/usr/bin/env python
"""
Model definitions for SleepStageNet LSTM extension.

Models:
  - CNNFeatureExtractor: 3-block 1D CNN → 64-dim feature per epoch
  - SleepCNNBiLSTM: CNN features → BiLSTM → center epoch classifier
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """1D CNN that extracts a feature vector from a single EEG epoch.

    Architecture (same conv blocks as cnn_model.py SleepCNN):
      Conv1d(C→32, k=50, p=25) + BN + ReLU + MaxPool(4)   → (32, 750)
      Conv1d(32→64, k=25, p=12) + BN + ReLU + MaxPool(4)  → (64, 187)
      Conv1d(64→128, k=10, p=5) + BN + ReLU + AvgPool(1)  → (128,)
      Linear(128→64) + ReLU                                → (64,)

    Input:  (batch, n_channels, 3000)
    Output: (batch, 64)
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
        # x: (batch, C, 3000)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu(self.bn3(self.conv3(x))))
        x = x.squeeze(-1)             # (batch, 128)
        x = self.relu(self.fc(x))     # (batch, 64)
        return x


class SleepCNNOnly(nn.Module):
    """CNN-only classifier (no temporal context). For comparison / pre-training.

    Input:  (batch, n_channels, 3000)
    Output: (batch, n_classes)
    """

    def __init__(self, n_channels=2, n_classes=5, feature_dim=64):
        super().__init__()
        self.cnn = CNNFeatureExtractor(n_channels, feature_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = self.dropout(features)
        return self.classifier(features)


class SleepCNNBiLSTM(nn.Module):
    """CNN + BiLSTM for sequence-based sleep stage classification.

    Forward pass:
      1. Reshape (B, L, C, T) → (B*L, C, T)
      2. CNN feature extraction → (B*L, 64)
      3. Reshape → (B, L, 64)
      4. BiLSTM → (B, L, hidden*2)
      5. Extract center epoch → (B, hidden*2)
      6. Classify → (B, n_classes)

    Input:  (batch, seq_length, n_channels, 3000)
    Output: (batch, n_classes)
    """

    def __init__(self, n_channels=2, feature_dim=64, lstm_hidden=128,
                 lstm_layers=1, n_classes=5, dropout=0.3):
        super().__init__()
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

        # Extract per-epoch features
        x = x.reshape(B * L, C, T)
        features = self.cnn(x)           # (B*L, 64)
        features = features.reshape(B, L, -1)  # (B, L, 64)

        # Temporal modeling
        lstm_out, _ = self.lstm(features)  # (B, L, hidden*2)

        # Center epoch prediction
        center = L // 2
        center_feat = lstm_out[:, center, :]  # (B, hidden*2)
        center_feat = self.dropout(center_feat)

        return self.classifier(center_feat)  # (B, n_classes)

    def freeze_cnn(self):
        """Freeze CNN parameters for LSTM-only training."""
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze CNN for end-to-end fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    print('=' * 60)
    print('SleepStageNet Models Verification')
    print('=' * 60)

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # Test CNN feature extractor
    cnn = CNNFeatureExtractor(n_channels=2).to(device)
    x_single = torch.randn(4, 2, 3000, device=device)
    feat = cnn(x_single)
    print( '\nCNNFeatureExtractor:')
    print(f'  Input:  {x_single.shape}')
    print(f'  Output: {feat.shape}')
    print(f'  Params: {sum(p.numel() for p in cnn.parameters()):,}')

    # Test CNN-only classifier
    cnn_only = SleepCNNOnly(n_channels=2, n_classes=5).to(device)
    logits = cnn_only(x_single)
    print( '\nSleepCNNOnly:')
    print(f'  Input:  {x_single.shape}')
    print(f'  Output: {logits.shape}')
    print(f'  Params: {sum(p.numel() for p in cnn_only.parameters()):,}')

    # Test CNN+BiLSTM
    model = SleepCNNBiLSTM(n_channels=2, lstm_hidden=128, lstm_layers=1).to(device)
    x_seq = torch.randn(4, 5, 2, 3000, device=device)
    logits = model(x_seq)
    print( '\nSleepCNNBiLSTM (L=5):')
    print(f'  Input:  {x_seq.shape}')
    print(f'  Output: {logits.shape}')

    total = sum(p.numel() for p in model.parameters())
    cnn_params = sum(p.numel() for p in model.cnn.parameters())
    lstm_params = sum(p.numel() for p in model.lstm.parameters())
    cls_params = sum(p.numel() for p in model.classifier.parameters())
    print(f'  Params: {total:,} (CNN: {cnn_params:,}, LSTM: {lstm_params:,}, cls: {cls_params:,})')

    # Test freeze/unfreeze
    model.freeze_cnn()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  After freeze_cnn: {trainable:,} trainable')
    model.unfreeze_cnn()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  After unfreeze_cnn: {trainable:,} trainable')

    # Test backward pass
    loss = logits.sum()
    loss.backward()
    print('\n  Backward pass: OK')
    print('\nAll checks passed.')
