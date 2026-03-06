# SleepStageNet: Team Handoff Document

## Project Overview

**Course:** CSEP 590A Deep Learning
**Task:** 5-class sleep stage classification (Wake, N1, N2, N3, REM) from raw EEG
**Dataset:** Sleep-EDF v1 from PhysioNet — 39 recordings, 20 subjects, 42,230 epochs (30s each, Fpz-Cz channel, 100Hz)
**Evaluation:** 20-fold leave-one-subject-out cross-validation (LOSO-CV)

---

## What's Already Done

### 1. Data Pipeline (Complete)
- Sleep-EDF data preprocessed into `.npz` files (3000 samples per epoch)
- Hosted on Google Drive for fast download via `gdown` (~28s)
- 20-fold LOSO-CV splits generated (`data_split_v1.npz`)

### 2. Baseline Model: DeepSleepNet-Lite (Complete)
- Architecture: Dual parallel 1D-CNN paths (fine-grained + coarse temporal filters) on 90-second windows (3 consecutive epochs)
- All TF 2.x compatibility issues resolved (runs on Colab with TF 2.19, Python 3.12, T4 GPU)
- Full 20-fold training complete, checkpoints saved to Google Drive
- Prediction and evaluation pipeline working end-to-end

### 3. Colab Notebook (Complete)
- **Repo:** `https://github.com/manishdas/deepsleepnet-lite`
- **Baseline notebook:** `notebooks/DeepSleepNet_Lite_BaseLineColab.ipynb` (with saved outputs)
- **Template notebook:** `notebooks/DeepSleepNet_Lite_Colab.ipynb` (for new experiments)
- Teammates can restore all 20 pre-trained fold models in ~13s via `gdown` (no retraining needed)
- Auto-saves new models to Drive after each fold (survives Colab disconnects)

### 4. Baseline Results (20-fold LOSO-CV)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 0.8092 |
| Macro F1 | 0.7527 |
| Weighted F1 | 0.8113 |
| Cohen's Kappa | 0.7400 |
| ECE (Expected Calibration Error) | 0.1099 |

**Per-class F1 scores:**

| Stage | F1 | Precision | Recall | Support |
|-------|-----|-----------|--------|---------|
| Wake (W) | 0.802 | 0.852 | 0.759 | 8,207 |
| N1 | 0.441 | 0.408 | 0.481 | 2,804 |
| N2 | 0.865 | 0.879 | 0.852 | 17,799 |
| N3 | 0.859 | 0.817 | 0.905 | 5,703 |
| REM | 0.795 | 0.780 | 0.811 | 7,717 |

**Key observation:** N1 (light sleep) is the clear bottleneck at 0.441 F1. It's frequently confused with Wake (379 misclassified) and REM (428 misclassified). This is expected — N1 is inherently ambiguous and severely underrepresented (6.6% of data vs. N2 at 42.1%).

---

## What Needs to Happen Next

All improvements build on top of the baseline. The goal is to show measurable gains over DeepSleepNet-Lite using the same 20-fold LOSO-CV protocol.

### Improvement 1: Temporal Context Modeling
**Goal:** Show that modeling transitions between sleep stages improves performance.
**Options (pick one or two):**
- **BiLSTM on top of CNN features:** Extract CNN features from DeepSleepNet-Lite, feed sequence into BiLSTM
- **Self-attention / Transformer block:** Add attention over the 3-epoch sequence
- **Longer sequence context:** Extend input from 3 epochs (90s) to 5-7 epochs
- **Effort:** ~3-5 days, most complex stage
- **Expected result:** +1-3% accuracy, +2-5% on N1 F1

### Improvement 2: Class Imbalance Mitigation
**Goal:** Improve N1 detection (our weakest class).
**Options:**
- **Focal loss:** Replace cross-entropy with focal loss (down-weights easy examples, focuses on hard ones like N1). Simplest change — modify loss function in `train.py`
- **Class-weighted cross-entropy:** Weight the loss inversely proportional to class frequency
- **Data augmentation:** Time shifts, Gaussian noise, amplitude scaling on minority classes
- **Effort:** Focal loss is ~half a day; augmentation is ~1-2 days
- **Expected result:** N1 F1 improvement from 0.44 to 0.48-0.52

### Improvement 3: Interpretability
**Goal:** Visualize what the model learns, relate to known sleep physiology.
**What to do:**
- Grad-CAM or saliency maps on the CNN filters
- Show that the model attends to known EEG patterns (delta waves in N3, sleep spindles in N2, etc.)
- Attention weight visualizations if using attention-based temporal model
- **Effort:** ~1-2 days
- **Expected result:** Qualitative figures for the report/presentation

---

## Suggested Task Assignment (5 people)

| Person | Task | Effort |
|--------|------|--------|
| Person A | Focal loss / class-weighted loss experiment | 1 day |
| Person B | Temporal context — BiLSTM or attention layer | 3-5 days |
| Person C | Temporal context — help Person B, or try data augmentation | 2-3 days |
| Person D | Interpretability (Grad-CAM / saliency maps) + report figures | 2-3 days |
| Person E | Report writing + ablation analysis | 2-3 days |

Everyone should coordinate on using the **same 20-fold LOSO-CV** and saving results in the same format for the comparison table.

---

## Metrics Explained (in context of this project)

### Overall Accuracy
- **What:** Fraction of correctly classified epochs out of all epochs.
- **Our value:** 0.8092 (80.9%)
- **Why it matters:** Simple to understand, but misleading with imbalanced classes. N2 dominates (42.1% of data), so a model predicting mostly N2 could get ~42% accuracy for free.
- **Limitation:** Doesn't reflect poor N1 performance (only 6.6% of data).

### Macro F1 Score
- **What:** Unweighted average of per-class F1 scores. Treats all 5 classes equally regardless of size.
- **Our value:** 0.7527
- **Why it matters:** This is our **primary metric**. It penalizes the model for doing poorly on rare classes (N1). A model that ignores N1 entirely would have Macro-F1 around 0.60 even with high accuracy.
- **How to improve:** Improving N1 has the biggest impact here.

### Weighted F1 Score
- **What:** Average of per-class F1 scores, weighted by class support (number of samples).
- **Our value:** 0.8113
- **Why it matters:** Reflects performance proportional to how often each stage appears. Closer to accuracy but accounts for precision/recall tradeoffs. Higher than Macro-F1 because the model does well on frequent classes (N2, W, REM).

### Cohen's Kappa
- **What:** Agreement between predictions and ground truth, corrected for chance agreement. Ranges from -1 to 1 (0 = random, 1 = perfect).
- **Our value:** 0.7400
- **Why it matters:** Standard metric in clinical sleep staging. A kappa of 0.74 indicates "substantial agreement" (0.61-0.80 range). For reference, inter-scorer agreement between human experts is typically kappa ~0.75-0.85. Our model is approaching the lower end of human-level agreement.
- **Clinical threshold:** Kappa > 0.60 is generally considered clinically usable.

### Per-class F1 Score
- **What:** Harmonic mean of precision and recall for each individual class.
- **Why it matters:** Reveals which stages the model struggles with. F1 balances two failure modes:
  - **Low precision** = too many false positives (predicting N1 when it's actually W/REM)
  - **Low recall** = too many false negatives (missing actual N1 epochs)
- **Our weak spot:** N1 at 0.441 — the model misses over half of N1 epochs.

### Precision and Recall (per class)
- **Precision:** Of all epochs the model *predicted* as class X, what fraction were actually X?
- **Recall (Sensitivity):** Of all epochs that *are* class X, what fraction did the model correctly identify?
- **Example — N1:** Precision=0.408 (many false N1 predictions), Recall=0.481 (misses ~52% of real N1)
- **Example — N3:** Precision=0.817, Recall=0.905 (model is good at finding deep sleep, occasional false positives)

### Expected Calibration Error (ECE)
- **What:** Measures how well the model's confidence scores match its actual accuracy. ECE=0 means perfect calibration.
- **Our value:** 0.1099 (~11%)
- **Why it matters:** When the model says it's 90% confident, is it actually correct 90% of the time? An ECE of 0.11 means there's about an 11% gap between confidence and accuracy on average. Important for clinical trust — doctors need to know when the model is uncertain.

### Confusion Matrix (how to read it)
- Rows = true labels, Columns = predicted labels
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- **Key patterns in our baseline:**
  - N1 is confused with W (379), N2 (627), and REM (428) — N1 is inherently ambiguous
  - N2 has some confusion with N3 (946) — adjacent stages are harder to distinguish
  - N3 is well-isolated (very few off-diagonal errors)
  - W is sometimes confused with N1 (913) and REM (618)

---

## Metrics Comparison Table (to fill in as experiments complete)

| Model | Accuracy | Macro F1 | Weighted F1 | Kappa | W F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 | ECE |
|-------|----------|----------|-------------|-------|------|-------|-------|-------|--------|-----|
| **DeepSleepNet-Lite (baseline)** | **0.8092** | **0.7527** | **0.8113** | **0.7400** | **0.802** | **0.441** | **0.865** | **0.859** | **0.795** | **0.110** |
| + Focal Loss | — | — | — | — | — | — | — | — | — | — |
| + Temporal Context (BiLSTM) | — | — | — | — | — | — | — | — | — | — |
| + Both | — | — | — | — | — | — | — | — | — | — |

**Primary comparison metric:** Macro F1 (treats all classes equally)
**Secondary:** Cohen's Kappa (clinical relevance), N1 F1 (our weakest class)

---

## How to Run Experiments

1. Clone the repo and open the Colab notebook
2. Run Cells 1-2 (setup + data download)
3. Run the model restore cell (downloads pre-trained baseline in ~13s)
4. For new experiments: copy the training cell, modify as needed (loss function, architecture, etc.)
5. Use the **same 20-fold LOSO-CV** with the same data splits
6. Save results using the same prediction/evaluation pipeline
7. Add your row to the comparison table above

**Important:** All experiments must use the same evaluation protocol for fair comparison. The 20-fold splits are fixed in `data/data_split_v1.npz`.

---

## Repository Structure

```
deepsleepnet-lite/
├── notebooks/
│   ├── DeepSleepNet_Lite_BaseLineColab.ipynb  # Baseline with outputs
│   └── DeepSleepNet_Lite_Colab.ipynb          # Template for experiments
├── deepsleeplite/                              # Model code
│   ├── model.py                                # DeepSleepNet-Lite architecture
│   ├── nn.py                                   # Neural network layers
│   ├── data_loader.py                          # Data loading + CV splits
│   └── utils.py                                # Training utilities
├── train.py                                    # Training entry point
├── predict.py                                  # Prediction entry point
├── summary_muquery.py                          # Evaluation metrics
├── prepare_physionet.py                        # Data preprocessing
├── results/                                    # CSV results
└── figures/                                    # Generated plots
```

## Google Drive Structure

```
MyDrive/SleepStageNet/
├── baseline_models.zip          # All 20 folds (276MB, shared via gdown)
├── models/v1/base/fold0-19/     # Individual fold checkpoints
├── results/                     # Prediction outputs
└── figures/                     # Plots
```
