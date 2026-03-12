# SleepConformer — CNN + Conformer for Sleep Stage Classification

Architecture overview and slide-ready summary for the **SleepConformer** model from `enhanced/models.py`.

- **Source**: `enhanced/models.py` → `SleepConformer`
- **Backbone**: Multi-Scale CNN (same as Enhanced CNN+BiLSTM)
- **Temporal module**: 3× Conformer Blocks (Self-Attention + Depthwise Conv)
- **Total params**: ~1.34M

---

## What is a Conformer?

Originally from speech recognition (Gulati et al., 2020), the **Conformer** combines the best of Transformers and CNNs in a single block:

```mermaid
block-beta
    columns 3
    block:attn:1
        columns 1
        A["🌐 Self-Attention"]
        B["Global relationships"]
        C["Any epoch can attend<br/>to any other epoch"]
    end
    block:plus:1
        columns 1
        D["➕ Combined in<br/>one block"]
    end
    block:conv:1
        columns 1
        E["📐 Depthwise Conv"]
        F["Local patterns"]
        G["Detects nearby<br/>stage transitions"]
    end
    style attn fill:#e3f2fd,stroke:#1565C0
    style conv fill:#fff3e0,stroke:#E65100
    style plus fill:none,stroke:none
```

```mermaid
flowchart LR
    I["💡 Key Idea"] --- T["Attention captures global sleep-cycle structure.<br/>Conv captures local N2→N3 transitions.<br/>Together: best of both worlds."]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

## Architecture at a Glance

| Aspect | Value |
|---|---|
| **CNN Backbone** | Multi-scale dual-path (Path A k=50 + Path B k=400) |
| **CNN Output Dim** | 128 |
| **Positional Encoding** | Learnable (max_len=25) |
| **Conformer Blocks** | 3 |
| **Attention Heads** | 4 |
| **Conv Kernel (depthwise)** | 7 |
| **FFN Expansion** | 4× (128→512→128) |
| **Dropout** | 0.2 |
| **Classifier Input** | 128 (center epoch from Conformer output) |
| **Sequence Length** | L=11 (±5 epochs) |
| **Total Params** | ~1,338,821 (~1.34M) |

---

## Architecture Diagram

```mermaid
flowchart LR
    A["11 EEG Epochs<br/>(B, 11, 2, 3000)"] --> CNN["🔬 Multi-Scale CNN<br/>Path A k=50 + Path B k=400<br/>→ 128-dim"]
    CNN -->|"(B,11,128)"| PE["📍 Pos Enc"]:::pe
    PE --> CF["🧩 Conformer x3<br/>Attn + Conv"]:::cf
    CF -->|"center epoch"| CLS["→ 5 sleep stages"]

    style CNN fill:#e8eaf6,stroke:#283593
    classDef pe fill:#f3e5f5,stroke:#6A1B9A,color:#000
    classDef cf fill:#e8f5e9,stroke:#2e7d32,color:#000
```

---

## Conformer Block — Internal Structure

Each Conformer block uses the **Macaron** structure (half-step FFN sandwich):

```mermaid
flowchart LR
    IN["Input"] --> FFN1["1/2 FFN"]:::ffn --> MHSA["Self-Attn<br/>4 heads"]:::attn --> CONV["Depthwise<br/>Conv k=7"]:::conv --> FFN2["1/2 FFN"]:::ffn --> NORM["LayerNorm"] --> OUT["Output"]

    classDef ffn fill:#f3e5f5,stroke:#6A1B9A,color:#000
    classDef attn fill:#e3f2fd,stroke:#1565C0,color:#000
    classDef conv fill:#fff3e0,stroke:#E65100,color:#000
```

```mermaid
flowchart LR
    I["💡 Why Macaron?"] --- T["Half-step FFNs at start and end<br/>stabilize training better than<br/>a single full FFN (Gulati et al.)"]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

## Tensor Shapes

```plaintext
Input:         (B, 11, 2, 3000)   -- 11 epochs, 2 EEG channels, 3000 samples
  reshape  --> (B*11, 2, 3000)    -- flatten for CNN

  Path A: conv1→conv2→conv3→conv4→AvgPool(1) → (B*11, 64)
  Path B: conv1→conv2→conv3→conv4→AvgPool(1) → (B*11, 64)
  concat   --> (B*11, 128)
  dropout  --> (B*11, 128)
  fc       --> (B*11, 128)

  reshape  --> (B, 11, 128)       -- restore sequence dimension
  pos_enc  --> (B, 11, 128)       -- add learnable position

  Conformer Block ×3:
    ½ FFN    --> (B, 11, 128)     -- 128→512→128 with 0.5 residual
    MHSA     --> (B, 11, 128)     -- 4 heads, d_k=32
    Conv     --> (B, 11, 128)     -- depthwise k=7
    ½ FFN    --> (B, 11, 128)     -- 128→512→128 with 0.5 residual
    LayerNorm

  final_norm --> (B, 11, 128)
  center     --> (B, 128)         -- take position [5] (center of 11)

  dropout    --> (B, 128)
  linear     --> (B, 5)           -- 5 sleep stage classes
```

---

## Parameter Breakdown (~1.34M)

```plaintext
Multi-Scale CNN:        ~191,000  (Path A: ~56K  Path B: ~118K  merge FC: ~16K)
Positional Encoding:         2,816  (1 × 25 × 128 — learnable, but only 11 used)
Conformer Block ×3:    ~1,143,000  (per block: ~381K)
  ├─ ½ FFN₁:             131,712  (LN:256 + 128×512 + 512×128 = 131,456)
  ├─ MHSA:                66,048  (LN:256 + 4-head attn: 128→128 Q/K/V/O)
  ├─ Conv Module:          51,072  (LN:256 + pointwise:32K + depthwise + BN + pointwise)
  └─ ½ FFN₂ + final LN:  131,968
Classifier:                  645  (128 × 5 + 5)
────────────────────────────────
Total:                 ~1,338,821
```

---

## Training Configuration

| Training Aspect | Value |
|---|---|
| **Optimizer** | Adam |
| **Learning Rate** | 3e-4 |
| **Weight Decay** | 1e-4 |
| **Max Epochs** | 80 |
| **Patience** | 20 |
| **Batch Size** | 32 |
| **Sequence Length** | 11 |
| **Loss** | **Focal Loss** (γ=1.5) + label smoothing (0.05) |
| **Mixup** | α=0.1 |
| **LR Schedule** | Cosine annealing with 3-epoch warmup |
| **Class Balancing** | Class-weighted loss + optional WeightedRandomSampler |

---

## Conformer vs CNN+BiLSTM — Comparison

| Aspect | Enhanced CNN+BiLSTM | SleepConformer |
|---|---|---|
| **Temporal module** | 2-Layer BiLSTM | 3× Conformer Blocks |
| **Attention** | None (sequential only) | **4-head self-attention** |
| **Local patterns** | BiLSTM hidden state | **Depthwise conv (k=7)** |
| **Positional info** | Implicit (LSTM order) | **Learnable positional encoding** |
| **Classifier input** | 256 (128×2 directions) | 128 (direct from Conformer) |
| **Loss function** | CrossEntropy | **Focal Loss (γ=1.5)** — harder examples weighted more |
| **Dropout** | 0.3 | 0.2 |
| **Learning Rate** | 5e-4 | 3e-4 |
| **Max Epochs** | 60 | **80** |
| **Patience** | 15 | **20** |
| **Total Params** | ~851K | **~1.34M** (1.6× larger) |

```mermaid
block-beta
    columns 2
    block:bilstm:1
        columns 1
        A["🔄 BiLSTM"]
        B["Sequential processing"]
        C["Strong at ordered transitions"]
        D["Lighter: ~851K params"]
    end
    block:conformer:1
        columns 1
        E["🧩 Conformer"]
        F["Parallel attention + local conv"]
        G["Captures global + local patterns"]
        H["Heavier: ~1.34M params"]
    end
    style bilstm fill:#e3f2fd,stroke:#1565C0
    style conformer fill:#e8f5e9,stroke:#2e7d32
```

---

## Results (20-Fold LOSO-CV)

| Metric | CNN+BiLSTM | Conformer |
|--------|-----------|-----------|
| Accuracy | **0.841** | 0.829 |
| Macro-F1 | **0.791** | 0.780 |
| Weighted-F1 | **0.847** | 0.835 |
| Cohen's κ | **0.783** | 0.769 |
| Parameters | ~851K | ~1.34M |

```mermaid
flowchart LR
    I["📊 Observation"] --- T["BiLSTM slightly outperforms Conformer<br/>despite fewer parameters.<br/>Sequential inductive bias may better suit<br/>the ordered nature of sleep stages."]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

```mermaid
flowchart LR
    subgraph why["Why BiLSTM wins here"]
        direction LR
        W1["Sleep stages are inherently<br/>sequential: W→N1→N2→N3→REM"]:::a --> W2["BiLSTM has built-in<br/>sequential inductive bias"]:::a
        W2 --> W3["Conformer must learn this<br/>ordering from scratch"]:::b
        W3 --> W4["With only L=11 epochs,<br/>attention has limited context<br/>to show its strength"]:::b
    end
    classDef a fill:#c8e6c9,stroke:#2e7d32
    classDef b fill:#ffcdd2,stroke:#c62828
```

---

## SleepConformer — Slide-Ready Summary

### Slide 1: Why Conformer? (CNN-Only Limitations)

**CNN alone scores each epoch in isolation — it cannot see the sequence.**

```mermaid
block-beta
    columns 3
    block:problem:1
        columns 1
        A["❌ CNN-Only"]
        B["Scores each 30s epoch alone"]
        C["No temporal context"]
        D["Confused by N1 vs Wake vs REM"]
    end
    block:arrow:1
        columns 1
        E["→ Add Conformer →"]
    end
    block:solution:1
        columns 1
        F["✅ CNN + Conformer"]
        G["Attention sees all 11 epochs at once"]
        H["Conv detects local transitions"]
        J["Best of global + local"]
    end
    style problem fill:#ffebee,stroke:#c62828
    style solution fill:#e8f5e9,stroke:#2e7d32
    style arrow fill:none,stroke:none
```

```mermaid
flowchart LR
    subgraph cnn_only["CNN-Only: Independent Scoring"]
        direction LR
        E1["epoch 1"]:::ind ~~~ E2["epoch 2"]:::ind ~~~ E3["epoch 3"]:::ind ~~~ E4["epoch 4"]:::ind ~~~ E5["epoch 5"]:::ind
    end
    subgraph conf["Conformer: Every epoch attends to every other"]
        direction LR
        C1["ep 1"] ~~~ C2["ep 2"] ~~~ C3["ep 3"] ~~~ C4["ep 4"] ~~~ C5["ep 5"]
        C3 -.-> C1
        C3 -.-> C2
        C3 -.-> C4
        C3 -.-> C5
    end
    classDef ind fill:#ffcdd2,stroke:#c62828
    style cnn_only fill:#ffebee,stroke:#c62828
    style conf fill:#e8f5e9,stroke:#2e7d32
```

```mermaid
flowchart LR
    I["💡"] --- T["CNN extracts WHAT features are in each epoch.<br/>Conformer adds WHEN — understanding which<br/>epochs matter and how they relate."]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

### Slide 2: What is a Conformer?

**Combines the best of Transformers and CNNs in each block.**

```mermaid
block-beta
    columns 3
    block:transformer:1
        columns 1
        A["🌐 Transformer"]
        B["Global attention"]
        C["Sees all epochs at once"]
    end
    block:plus:1
        columns 1
        D["+ Conformer ="]
    end
    block:cnn:1
        columns 1
        E["📐 CNN"]
        F["Local convolution"]
        G["Detects nearby transitions"]
    end
    style transformer fill:#e3f2fd,stroke:#1565C0
    style cnn fill:#fff3e0,stroke:#E65100
    style plus fill:none,stroke:none
```

```mermaid
flowchart LR
    I["💡"] --- T["From speech recognition (Gulati et al., 2020).<br/>Adapted here for EEG epoch sequences."]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

### Slide 3: Architecture Overview

```mermaid
flowchart LR
    A["11 EEG Epochs<br/>(B, 11, 2, 3000)"] --> CNN

    subgraph CNN["🔬 Multi-Scale CNN"]
        direction TB
        subgraph PA["Path A: Fine k=50"]
            direction TB
            A1["Spindles · K-complexes · Alpha"]
        end
        subgraph PB["Path B: Coarse k=400"]
            direction TB
            B1["Delta / Slow Waves"]
        end
        M["Concat → 128-dim"]
        PA --> M
        PB --> M
    end

    CNN -->|"(B,11,128)"| PE["📍 Positional<br/>Encoding"]:::pe
    PE --> CF["🧩 3× Conformer<br/>Blocks"]:::cf
    CF -->|"center epoch"| CLS["→ 5 sleep stages"]

    style PA fill:#e3f2fd,stroke:#1565C0
    style PB fill:#fff3e0,stroke:#E65100
    style M fill:#e8eaf6,stroke:#283593
    classDef pe fill:#f3e5f5,stroke:#6A1B9A,color:#000
    classDef cf fill:#e8f5e9,stroke:#2e7d32,color:#000
```

---

### Slide 4: Inside a Conformer Block

**Macaron sandwich: ½ FFN → Attention → Conv → ½ FFN**

```mermaid
flowchart LR
    IN["Input"] --> F1["½ FFN"]:::ffn --> ATT["Multi-Head<br/>Self-Attention<br/>4 heads"]:::attn --> CV["Depthwise<br/>Conv k=7"]:::conv --> F2["½ FFN"]:::ffn --> LN["Layer<br/>Norm"] --> OUT["Output"]

    classDef ffn fill:#f3e5f5,stroke:#6A1B9A,color:#000
    classDef attn fill:#e3f2fd,stroke:#1565C0,color:#000
    classDef conv fill:#fff3e0,stroke:#E65100,color:#000
```

```mermaid
flowchart LR
    subgraph roles["Each component's role"]
        direction LR
        R1["Self-Attention:<br/>Which epochs matter<br/>for scoring this one?"]:::attn --> R2["Depthwise Conv:<br/>Is there a local<br/>N2→N3 transition?"]:::conv
    end
    classDef attn fill:#e3f2fd,stroke:#1565C0,color:#000
    classDef conv fill:#fff3e0,stroke:#E65100,color:#000
```

---

### Slide 5: Conformer vs BiLSTM — How They See Context

```mermaid
flowchart TD
    subgraph bilstm["BiLSTM: Sequential Scanning"]
        direction LR
        L1["t-5"] -->|"→"| L2["t-4"] -->|"→"| L3["t-3"] -->|"→"| L4["t-2"] -->|"→"| L5["t-1"] -->|"→"| L6["⬤ t"]
        R6["⬤ t"] -->|"←"| R5["t+1"] -->|"←"| R4["t+2"] -->|"←"| R3["t+3"] -->|"←"| R2["t+4"] -->|"←"| R1["t+5"]
    end

    subgraph conformer["Conformer: Parallel Attention + Local Conv"]
        direction LR
        C1["t-5"] ~~~ C2["t-4"] ~~~ C3["t-3"] ~~~ C4["t-2"] ~~~ C5["t-1"] ~~~ C6["⬤ t"] ~~~ C7["t+1"] ~~~ C8["t+2"] ~~~ C9["t+3"] ~~~ C10["t+4"] ~~~ C11["t+5"]
        C6 -.->|"attends to all"| C1
        C6 -.->|"attends to all"| C4
        C6 -.->|"attends to all"| C11
    end

    style L6 fill:#E91E63,color:#fff,stroke-width:3px
    style R6 fill:#E91E63,color:#fff,stroke-width:3px
    style C6 fill:#E91E63,color:#fff,stroke-width:3px
    style bilstm fill:#e3f2fd,stroke:#1565C0
    style conformer fill:#e8f5e9,stroke:#2e7d32
```

---

### Slide 6: Training Strategy

```mermaid
flowchart LR
    S1["🔵 Stage 1<br/>Train CNN<br/>lr=3e-4"]:::s1 -->|"CNN ready"| S2["🟠 Stage 2<br/>Freeze CNN,<br/>Train Conformer<br/>lr=3e-4"]:::s2 -->|"Conformer ready"| S3["🟣 Stage 3<br/>Fine-tune All<br/>lr=3e-5"]:::s3

    S1 -.- R1["<b>Training Improvements</b><br/>Focal Loss γ=1.5 · Label Smoothing<br/>Mixup α=0.1 · Class Weighting"]:::reg

    classDef s1 fill:#e3f2fd,stroke:#1565C0,color:#000
    classDef s2 fill:#fff3e0,stroke:#E65100,color:#000
    classDef s3 fill:#f3e5f5,stroke:#6A1B9A,color:#000
    classDef reg fill:#e8eaf6,stroke:#283593,color:#000
```

```mermaid
flowchart LR
    I["💡 Focal Loss"] --- T["Unlike BiLSTM which uses CrossEntropy,<br/>Conformer uses Focal Loss (γ=1.5):<br/>down-weights easy examples,<br/>focuses on hard-to-classify N1 epochs"]
    style I fill:#FFF9C4,stroke:#F9A825,color:#000
    style T fill:#FFF9C4,stroke:#F9A825,color:#000
```

---

### Slide 7: Results (20-Fold LOSO-CV)

| Metric | CNN+BiLSTM | SleepConformer |
|--------|-----------|----------------|
| Accuracy | **0.841** | 0.829 |
| Macro-F1 | **0.791** | 0.780 |
| Cohen's κ | **0.783** | 0.769 |
| Parameters | ~851K | ~1.34M |

```mermaid
flowchart LR
    G1["📊 Takeaway"] --- G2["BiLSTM wins by ~1.2% accuracy<br/>with 37% fewer parameters"]
    G2 --- G3["Sequential inductive bias of LSTM<br/>suits sleep stage ordering well"]
    style G1 fill:#FFF9C4,stroke:#F9A825,color:#000
    style G2 fill:#e8f5e9,stroke:#2e7d32
    style G3 fill:#e8f5e9,stroke:#2e7d32
```

```mermaid
flowchart LR
    subgraph future["Potential Improvements for Conformer"]
        direction LR
        F1["Longer sequences<br/>L=21 or L=31"]:::idea --> F2["More training data<br/>attention is data-hungry"]:::idea --> F3["Pre-training on<br/>larger sleep datasets"]:::idea
    end
    classDef idea fill:#e3f2fd,stroke:#1565C0,color:#000
```
