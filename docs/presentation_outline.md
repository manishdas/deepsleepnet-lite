# SleepStageNet — Presentation Outline

**CSEP 590A Deep Learning | Group Project**
Shrinivas Acharya, Manish Das, Nithin Balachandran, Regith Lingesh, Thangakumar Dhanasekaran

---

## Slide 1: Title
**SleepStageNet: Deep Learning for Automated Sleep Stage Classification**
- Team names
- CSEP 590A Deep Learning

---

## Slide 2: Problem & Motivation
- Sleep disorders affect millions worldwide
- Diagnosis requires overnight polysomnography (PSG) — expensive, manual, expert-dependent
- A trained technician takes 2-4 hours to score one night of sleep (800-1000 epochs)
- **Goal:** Automate 5-class sleep stage classification from raw EEG signals
- **Clinical impact:** Reduce cost, enable scalable sleep monitoring

---

## Slide 3: Sleep Stages
- **5 classes** per AASM guidelines:
  - **Wake (W)** — alert/drowsy
  - **N1** — light sleep, transition stage
  - **N2** — intermediate sleep, sleep spindles
  - **N3** — deep/slow-wave sleep, delta waves
  - **REM** — rapid eye movement, dreaming
- Visual: Example EEG waveforms for each stage (from our notebook figures)
- Each 30-second epoch is classified independently

---

## Slide 4: Dataset
- **Sleep-EDF v1** from PhysioNet (public benchmark)
- 39 whole-night PSG recordings from 20 healthy subjects
- EEG channel: Fpz-Cz at 100 Hz (3000 samples per 30s epoch)
- **42,230 total epochs** across all recordings
- **Class distribution** (include bar chart from notebook):
  - W: 8,207 (19.4%)
  - N1: 2,804 (6.6%) — severely underrepresented
  - N2: 17,799 (42.1%) — dominant class
  - N3: 5,703 (13.5%)
  - REM: 7,717 (18.3%)
- **Challenge:** Severe class imbalance, subject variability, ambiguous transitions

---

## Slide 5: Evaluation Protocol
- **20-fold leave-one-subject-out cross-validation (LOSO-CV)**
  - Each fold: train on 19 subjects, test on 1 held-out subject
  - Ensures model generalizes to unseen subjects (no data leakage)
- **Metrics:**
  - Macro F1 (primary — treats all classes equally)
  - Cohen's Kappa (clinical standard — corrects for chance agreement)
  - Overall Accuracy, Per-class F1, Confusion Matrix
- **Why Macro F1?** Accuracy is misleading with imbalanced classes — a model predicting all N2 gets 42% accuracy

---

## Slide 6: Approach Overview
- Start from a strong CNN baseline, then improve systematically:

```
Baseline: CNN on Raw EEG       →  DeepSleepNet-Lite
                                   (end-to-end deep learning baseline)
            ↓
Improvement 1: Class Imbalance →  Focal Loss / Weighted Loss
                                   (improve N1 detection)
            ↓
Improvement 2: Temporal Context →  BiLSTM / Attention over epochs
                                   (model sleep stage transitions)
            ↓
Analysis: Interpretability      →  Grad-CAM / Saliency maps
                                   (what does the model learn?)
```

- All experiments compared using the same 20-fold LOSO-CV protocol

---

## Slide 7: Baseline Architecture — DeepSleepNet-Lite
- Input: 90-second window (3 consecutive 30s epochs) = 9000 samples
- **Dual parallel CNN paths:**
  - Path 1: Small filters (captures fine-grained, high-frequency features)
  - Path 2: Large filters (captures coarse, low-frequency features)
- Concatenated features → Fully connected → Softmax over 5 classes
- Training: Adam optimizer, lr=1e-4, 100 epochs, batch size 100
- Visual: Architecture diagram

---

## Slide 8: Baseline Results
- **Overall performance:**

| Metric | Value |
|--------|-------|
| Accuracy | 80.9% |
| Macro F1 | 0.753 |
| Cohen's Kappa | 0.740 |

- **Per-class F1:** (include bar chart)
  - W: 0.802, N1: **0.441**, N2: 0.865, N3: 0.859, REM: 0.795
- **Key finding:** N1 is the bottleneck (0.441 F1)
  - Confused with Wake (similar low-amplitude EEG) and REM (similar frequency content)
  - Only 6.6% of data — model biased toward majority classes
- Include confusion matrix (normalized) showing misclassification patterns
- Cohen's Kappa 0.74 = "substantial agreement" (human expert range: 0.75-0.85)

---

## Slide 9: Improvement — Focal Loss for Class Imbalance
- **Problem:** Standard cross-entropy treats all samples equally — model focuses on easy/frequent classes (N2)
- **Focal Loss** (Lin et al., 2017): Down-weights easy examples, focuses learning on hard ones
  - FL(p) = -α(1-p)^γ · log(p)
  - γ controls how much to focus on hard examples (γ=0 → standard CE)
- **Results:** [fill in after experiment]
- **Expected:** N1 F1 improvement, possibly at slight cost to N2/N3
- Addresses Research Q4: mitigating class imbalance

---

## Slide 10: Improvement — Temporal Context Modeling
- **Problem:** CNN classifies each epoch mostly independently — doesn't model sleep stage transitions
- **Solution:** Add BiLSTM / self-attention over sequence of epoch features
- Sleep naturally follows patterns: W → N1 → N2 → N3 → N2 → REM → ...
- Temporal model can learn that N1 is unlikely after N3 (should transition through N2)
- **Results:** [fill in after experiment]
- Addresses Research Q2: temporal context improves performance

---

## Slide 11: Interpretability
- **Grad-CAM / Saliency maps** on CNN filters
- Show that the model learns physiologically meaningful patterns:
  - N3 → model attends to delta waves (0.5-4 Hz, high amplitude)
  - N2 → model detects sleep spindles (12-14 Hz bursts)
  - REM → model picks up theta activity and low muscle tone
- Visual: Saliency map overlaid on EEG waveforms for different stages
- Builds clinical trust in the model's decisions

---

## Slide 12: Results Comparison

| Model | Accuracy | Macro F1 | Kappa | N1 F1 |
|-------|----------|----------|-------|-------|
| **DeepSleepNet-Lite (baseline)** | **0.809** | **0.753** | **0.740** | **0.441** |
| + Focal Loss | — | — | — | — |
| + Temporal Context | — | — | — | — |
| + Both | — | — | — | — |

- Highlight the progression: baseline → + focal loss → + temporal context
- Show that each improvement adds measurable value
- Focus on N1 F1 improvement as the key story

---

## Slide 13: Key Takeaways
1. DeepSleepNet-Lite achieves 80.9% accuracy and Kappa 0.74 — approaching human inter-scorer agreement (0.75-0.85)
2. N1 (light sleep) is the hardest stage at 0.441 F1 — even human experts disagree on it
3. [Focal loss / temporal context] improved Macro F1 from 0.753 to [Y]
4. Interpretability analysis confirms the model learns physiologically meaningful patterns
5. Class imbalance mitigation is critical for clinical deployment

---

## Slide 14: Limitations & Future Work
- **Limitations:**
  - Single EEG channel (Fpz-Cz only) — multi-channel could help
  - Sleep-EDF v1 is relatively small (20 subjects) — larger datasets like SHHS could improve generalization
  - N1 detection remains challenging even with improvements
- **Future work:**
  - Multi-modal input (EEG + EOG + EMG)
  - Transformer-based architectures (e.g., AttnSleep, U-Sleep)
  - Transfer learning across datasets
  - Real-time inference for wearable sleep monitors

---

## Slide 15: Demo
- Live walkthrough of the Colab notebook (or screenshots):
  1. Data loading and class distribution
  2. Model restore from pre-trained checkpoints (~13s)
  3. Run prediction on all 20 folds
  4. Confusion matrix and per-class F1 visualization
  5. Training curves (mean +/- std across 20 folds)

---

## Appendix Slides (if asked)

### A: Detailed Confusion Matrix
- Full 5x5 normalized confusion matrix with annotations
- Highlight the N1 ↔ W and N1 ↔ REM confusion patterns

### B: Training Curves
- Mean +/- std across 20 folds
- Show convergence behavior

### C: Calibration
- ECE = 0.110
- Reliability diagram showing confidence vs. accuracy

### D: Per-fold Accuracy Variation
- Box plot or table showing how accuracy varies across the 20 subjects
- Demonstrates subject-to-subject variability
