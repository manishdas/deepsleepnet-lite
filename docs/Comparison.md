# CNN+BiLSTM Architecture Comparison: Temporal vs Enhanced

Side-by-side comparison of the two CNN+BiLSTM models for sleep stage classification.

- **Temporal**: `temporal/models.py` → `SleepCNNBiLSTM`
- **Enhanced**: `enhanced/models.py` → `SleepCNNBiLSTM` (with `cnn_type='multiscale'`)

---

## Architectural Differences

| Aspect | Temporal | Enhanced |
|---|---|---|
| **CNN Backbone** | Single-scale: 3 conv blocks (k=50→k=25→k=10) | **Multi-scale dual-path**: Path A (fine, k=50/8/8/8) + Path B (coarse, k=400/6/6/6) |
| **CNN Depth** | 3 conv layers | **4 conv layers per path** (8 total) |
| **CNN Output Dim** | 64 | **128** (2× wider) |
| **CNN Dropout** | None inside CNN | **Dropout(0.5)** between concat and FC |
| **LSTM Layers** | 1 | **2** (deeper recurrence) |
| **LSTM Input Size** | 64 | **128** |
| **LSTM Hidden Size** | 128 (output: 256) | 128 (output: 256) — same |
| **Classifier Input** | 256 (128×2 directions) | 256 — same |
| **Sequence Length** | L=5 (±2 epochs) | **L=11** (±5 epochs, more context) |
| **Batch Size** | 64 | **32** |
| **Learning Rate** | 1e-3 | **5e-4** |
| **Label Smoothing** | None | **0.05** |
| **Mixup Augmentation** | None | **α=0.1** |
| **Total Params** | ~345K | **~851K** (2.5× larger) |

---

## Key Differences Explained

### 1. Multi-Scale CNN Backbone
The enhanced model uses two parallel filter paths inspired by the original DeepSleepNet-Lite TensorFlow architecture:
- **Path A** (fine-grained, k=50 first layer): Captures high-frequency features like sleep spindles (~0.5s resolution)
- **Path B** (coarse, k=400 first layer): Captures slow temporal patterns like delta waves (~4s resolution)

The temporal model uses a single filter path that can only capture one scale of features.

### 2. Deeper LSTM (2 layers vs 1)
Two stacked LSTM layers let the enhanced model learn hierarchical temporal abstractions — the first layer captures local transitions (e.g., N2→N3), while the second layer can model longer-range patterns (e.g., sleep cycle structure).

### 3. Wider Feature Dimension (128 vs 64)
Doubling the CNN output from 64 to 128 dimensions gives the LSTM richer per-epoch representations to work with, at the cost of more parameters.

### 4. Longer Sequences (L=11 vs L=5)
L=11 provides ±5 epochs of context (±2.5 minutes) compared to ±2 epochs (±1 minute) in the temporal model. This is particularly helpful for disambiguating N1 vs REM, which often requires seeing the broader sleep stage trajectory.

### 5. Training Regularization
The enhanced model adds three regularization techniques absent from the temporal version:
- **Label smoothing (0.05)**: Softens one-hot targets to reduce overconfidence
- **Mixup (α=0.1)**: Interpolates random training pairs for smoother decision boundaries
- **CNN dropout (0.5)**: Applied after concatenating the two CNN paths

---

## Architecture Diagrams

### Temporal CNN+BiLSTM (from `temporal/`)

```mermaid
flowchart TD
    subgraph Input
        A["Sequence of 5 EEG epochs<br/>(batch, 5, 2, 3000)"]
    end

    subgraph CNN["CNN Feature Extractor (shared weights, applied per-epoch)"]
        B["Conv1d(2→32, k=50) + BN + ReLU + MaxPool(4)"]
        C["Conv1d(32→64, k=25) + BN + ReLU + MaxPool(4)"]
        D["Conv1d(64→128, k=10) + BN + ReLU + AdaptiveAvgPool(1)"]
        E["Linear(128→64) + ReLU"]
        B --> C --> D --> E
    end

    subgraph Temporal["Temporal Modeling"]
        F["BiLSTM(input=64, hidden=128, bidirectional)"]
        G["Extract center epoch output<br/>index = L // 2 = 2"]
    end

    subgraph Classifier
        H["Dropout(0.3)"]
        I["Linear(256→5)"]
        J["Output: 5 sleep stage logits"]
    end

    A -->|"reshape to (B*5, 2, 3000)"| B
    E -->|"reshape to (B, 5, 64)"| F
    F -->|"(B, 5, 256)"| G
    G -->|"(B, 256)"| H
    H --> I --> J
```

#### Tensor shapes (Temporal)

```plaintext
Input:         (B, 5, 2, 3000)    -- 5 epochs, 2 EEG channels, 3000 samples each
  reshape  --> (B*5, 2, 3000)     -- flatten batch and sequence for CNN

  conv1    --> (B*5, 32, 750)     -- k=50, stride=1, pad=25, then MaxPool(4)
  conv2    --> (B*5, 64, 187)     -- k=25, stride=1, pad=12, then MaxPool(4)
  conv3    --> (B*5, 128, 1)      -- k=10, stride=1, pad=5, then AdaptiveAvgPool(1)
  fc       --> (B*5, 64)          -- linear projection

  reshape  --> (B, 5, 64)         -- restore sequence dimension

  BiLSTM   --> (B, 5, 256)        -- 128 hidden × 2 directions
  center   --> (B, 256)           -- take position [2] (center of 5)

  dropout  --> (B, 256)
  linear   --> (B, 5)             -- 5 sleep stage classes
```

---

### Enhanced CNN+BiLSTM (from `enhanced/`)

```mermaid
flowchart LR
    A["11 EEG Epochs<br/>(B, 11, 2, 3000)"] --> CNN

    subgraph CNN["Multi-Scale CNN (shared)"]
        direction TB
        PA["Path A: Fine filters<br/>Conv(k=50,s=6)→Conv(k=8)×3<br/>→ AvgPool → (64,)"]
        PB["Path B: Coarse filters<br/>Conv(k=400,s=50)→Conv(k=6)×3<br/>→ AvgPool → (64,)"]
        M["Concat → Dropout(0.5)<br/>Linear(128→128) + ReLU"]
        PA --> M
        PB --> M
    end

    CNN -->|"(B, 11, 128)"| LSTM["2-Layer BiLSTM<br/>hidden=128, bidir<br/>→ (B, 11, 256)"]
    LSTM -->|"center epoch"| CLS["Dropout(0.3) → Linear(256→5)<br/>5 sleep stage logits"]
```

#### Tensor shapes (Enhanced)

```plaintext
Input:         (B, 11, 2, 3000)   -- 11 epochs, 2 EEG channels, 3000 samples each
  reshape  --> (B*11, 2, 3000)    -- flatten batch and sequence for CNN

  Path A: conv1 → conv2 → conv3 → conv4 → AdaptiveAvgPool(1) → (B*11, 64)
  Path B: conv1 → conv2 → conv3 → conv4 → AdaptiveAvgPool(1) → (B*11, 64)
  concat   --> (B*11, 128)
  dropout  --> (B*11, 128)
  fc       --> (B*11, 128)        -- linear projection

  reshape  --> (B, 11, 128)       -- restore sequence dimension

  BiLSTM   --> (B, 11, 256)       -- 128 hidden × 2 directions (2 layers)
  center   --> (B, 256)           -- take position [5] (center of 11)

  dropout  --> (B, 256)
  linear   --> (B, 5)             -- 5 sleep stage classes
```

---

## Parameter Breakdown

### Temporal (~345K)

```plaintext
CNN Feature Extractor:  145,248  (conv1: 3,232  conv2: 51,264  conv3: 81,024  fc: 8,256  BN: 1,472)
BiLSTM:                 198,656  (64 input × 128 hidden × 4 gates × 2 directions + biases)
Classifier:               1,285  (256 × 5 + 5)
────────────────────────────────
Total:                  345,189
```

### Enhanced (~851K)

```plaintext
Multi-Scale CNN:        ~191,000  (Path A: ~56K  Path B: ~118K  merge FC: ~16K  BN: ~1K)
2-layer BiLSTM:         ~658,000  (layer 1: 128 in × 128 hidden × 4 × 2 + layer 2: 256 in × 128 hidden × 4 × 2)
Classifier:               1,285  (256 × 5 + 5)
────────────────────────────────
Total:                  ~851,000
```

---

## Training Pipeline Comparison

Both models use the same 3-stage training strategy, but with different hyperparameters:

| Training Aspect | Temporal | Enhanced |
|---|---|---|
| **Stage 1** (CNN pretrain) | Adam lr=1e-3, patience=15 | Adam lr=5e-4, patience=15 |
| **Stage 2** (LSTM, CNN frozen) | Adam lr=1e-3, patience=15, grad clip=1.0 | Adam lr=5e-4, patience=15, grad clip=1.0 |
| **Stage 3** (fine-tune all) | Adam lr=1e-4, patience=15, grad clip=1.0 | Adam lr=5e-5, patience=15, grad clip=1.0 |
| **LR Schedule** | ReduceLROnPlateau(patience=7, factor=0.5) | Cosine annealing with 3-epoch warmup |
| **Loss** | CrossEntropy with class weights | CrossEntropy with class weights + label smoothing (0.05) |
| **Data Augmentation** | None | Mixup (α=0.1), time shift, amplitude scaling, Gaussian noise |
| **Class Balancing** | `compute_class_weight('balanced')` | Same + optional `WeightedRandomSampler` |

---

## Enhanced CNN+BiLSTM — Slide-Ready Summary

### Slide 1: Why Add Temporal Modeling?

**Sleep stages are not independent — they follow a biological sequence.**

```mermaid
flowchart LR
    W(["Wake"]) --> N1(["N1"]) --> N2(["N2"]) --> N3(["N3"]) --> N2b(["N2"]) --> REM(["REM"]) --> W2(["Wake"])
    style W fill:#4CAF50,color:#fff
    style N1 fill:#FFC107,color:#000
    style N2 fill:#2196F3,color:#fff
    style N3 fill:#1a237e,color:#fff
    style N2b fill:#2196F3,color:#fff
    style REM fill:#E91E63,color:#fff
    style W2 fill:#4CAF50,color:#fff
```

```mermaid
block-beta
    columns 3
    block:problem:1
        columns 1
        A["❌ CNN Alone"]
        B["Each epoch scored independently"]
        C["No notion of before or after"]
    end
    block:arrow:1
        columns 1
        D["→ Add BiLSTM →"]
    end
    block:solution:1
        columns 1
        E["✅ CNN + BiLSTM"]
        F["Looks at surrounding context"]
        G["Learns transition patterns"]
    end
    style problem fill:#ffebee,stroke:#c62828
    style solution fill:#e8f5e9,stroke:#2e7d32
    style arrow fill:none,stroke:none
```

```mermaid
flowchart LR
    I["💡 Key Insight"] --- T["Adding temporal context lets the model<br/>mimic how human experts score sleep stages"]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

### Slide 2: Architecture Overview

```mermaid
flowchart LR
    A["11 EEG Epochs<br/>(B, 11, 2, 3000)"] --> CNN

    subgraph CNN["🔬 Multi-Scale CNN - extracts WHAT"]
        direction TB
        subgraph PA["Path A: Fine Filters k=50"]
            direction TB
            A1["Sleep Spindles · 12-14 Hz · ~0.5-1s"]
            A2["K-complexes · ~0.5 Hz · ~0.5s"]
            A3["Alpha Rhythm · 8-13 Hz · sustained"]
        end
        subgraph PB["Path B: Coarse Filters k=400"]
            direction TB
            B1["Delta / Slow Waves · 0.5-2 Hz · ~2-4s"]
        end
        M["Concat → 128-dim feature"]
        PA --> M
        PB --> M
    end

    CNN -->|"128-dim per epoch"| LSTM["🕐 2-Layer BiLSTM<br/>learns WHEN - stage<br/>transitions are likely"]
    LSTM -->|"center epoch"| CLS["→ 5 sleep stages"]

    style PA fill:#e3f2fd,stroke:#1565C0
    style PB fill:#fff3e0,stroke:#E65100
    style M fill:#e8eaf6,stroke:#283593
    style LSTM fill:#fff3e0,stroke:#E65100
```

---

### Slide 3: Why Multi-Scale CNN?

EEG signals contain features at different time scales:



```mermaid
flowchart LR
    I2["💡"] --- T2["Single-scale CNN captures one frequency range.<br/>Dual-path captures BOTH high-freq and low-freq, then concatenates."]
    style I2 fill:#FFF9C4,stroke:#F9A825,color:#000
    style T2 fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

### Slide 4: Why BiLSTM Helps

The bidirectional LSTM sees context in **both directions** — past and future epochs:

```mermaid
flowchart LR
    subgraph Forward["Forward LSTM →"]
        direction LR
        t5["t-5"] --> t4["t-4"] --> t3["t-3"] --> t2["t-2"] --> t1["t-1"] --> t0["⬤ t"]
    end
    subgraph Backward["← Backward LSTM"]
        direction RL
        t0b["⬤ t"] --> tp1["t+1"] --> tp2["t+2"] --> tp3["t+3"] --> tp4["t+4"] --> tp5["t+5"]
    end
    t0 -.->|"predict<br/>this epoch"| t0b
    style t0 fill:#E91E63,color:#fff,stroke-width:3px
    style t0b fill:#E91E63,color:#fff,stroke-width:3px
```

```mermaid
flowchart LR
    subgraph problem["N1 is hard (~7% of data, looks like Wake & REM)"]
        direction LR
        P1["Without context:<br/>CNN misclassifies N1 as Wake"] -->|"Add BiLSTM"| P2["With context:<br/>prev=N2, next=REM<br/>→ this is N1 transition ✅"]
    end
    style P1 fill:#ffcdd2,stroke:#c62828
    style P2 fill:#c8e6c9,stroke:#2e7d32
```

---

### Slide 5: Training Strategy

```mermaid
flowchart LR
    S1["🔵 Stage 1<br/>Train CNN<br/>lr=5e-4"]:::s1 -->|"CNN ready"| S2["🟠 Stage 2<br/>Freeze CNN, Train LSTM<br/>lr=5e-4"]:::s2 -->|"LSTM ready"| S3["🟣 Stage 3<br/>Fine-tune All<br/>lr=5e-5"]:::s3

    S1 -.- R1["<b>Training Improvements</b><br/>Label Smoothing · Mixup · EEG Augmentation · Class Weighting"]:::reg

    classDef s1 fill:#e3f2fd,stroke:#1565C0,color:#000
    classDef s2 fill:#fff3e0,stroke:#E65100,color:#000
    classDef s3 fill:#f3e5f5,stroke:#6A1B9A,color:#000
    classDef reg fill:#e8eaf6,stroke:#283593,color:#000
```

---

### Slide 6: Results (20-Fold LOSO-CV)

| Metric | CNN-Only Baseline | Enhanced CNN+BiLSTM |
|--------|-------------------|---------------------|
| Accuracy | 0.809 | **0.831** |
| Macro-F1 | 0.753 | **0.778** |
| Cohen's κ | 0.740 | **0.768** |
| Parameters | ~648K | ~851K |

```mermaid
flowchart LR
    G1["🎯 Biggest Gains"] --- G2["N1 recall +12%"] --- G3["REM precision +5%"]
    G3 --- G4["Exactly the stages that benefit<br/>most from temporal context"]
    style G1 fill:#FFF9C4,stroke:#F9A825,color:#000
    style G2 fill:#c8e6c9,stroke:#2e7d32
    style G3 fill:#c8e6c9,stroke:#2e7d32
    style G4 fill:#e8f5e9,stroke:#2e7d32
```
