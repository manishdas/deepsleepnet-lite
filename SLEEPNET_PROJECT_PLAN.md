# SleepStageNet: Project Plan
> CSEP 590A Deep Learning — Group Project
> Presentation: Wednesday, March 12, 2025
> Timeline: 10 days (Mar 2 – Mar 12)

---

## Project Overview

Automated 5-class sleep stage classification (Wake, N1, N2, N3, REM) using the Sleep-EDF dataset.
Starting from the DeepSleepNet-Lite baseline, we progressively add temporal context modeling,
class imbalance strategies, and interpretability analysis.

**Baseline repo:** https://github.com/biomedical-signal-processing/deepsleepnet-lite
**Team repo (fork):** https://github.com/manishdas/deepsleepnet-lite

---

## Team Task Board

| Owner | Primary Responsibility |
|-------|----------------------|
| Person 1 | Repo setup, environment, data pipeline |
| Person 2 | Baseline training (Colab, full k-fold) |
| Person 3 | Classical ML baseline + comparison table |
| Person 4 | Temporal context ablation experiments |
| Person 5 | Class imbalance experiments + interpretability |
| All | Slides (Day 8–10) |

---

## 10-Day Sprint

### Day 1 — Sun Mar 2 | Setup
**Goal:** Everyone can clone the repo and run a single training fold on Colab.

```bash
# 1. Open notebooks/DeepSleepNet_Lite_Colab.ipynb in Google Colab
# 2. Set runtime to GPU (T4)
# 3. Run all cells through Section 2 (data download)
# 4. Run Cell 3a (single fold training) to verify everything works
```

**Deliverable:** All 5 teammates have run fold 0 successfully on Colab.

---

### Day 2 — Mon Mar 3 | Data Download & Preprocessing
**Goal:** Sleep-EDF dataset downloaded and preprocessed into fold structure.

```bash
# The Colab notebook handles this automatically via prepare_physionet.py
# After running, save processed data to Google Drive (Section 6 of notebook)
# Share the Drive folder with teammates so they don't each re-download
```

**Deliverable:** Processed `.npz` files in shared Drive, data loading verified.

---

### Day 3 — Tue Mar 4 | Reproduce Baseline (Sanity Check)
**Goal:** Run baseline on fold 0 only — confirm training works before committing Colab GPU time.

```bash
# In the Colab notebook, run Cell 3a (single fold)
# Once confirmed, kick off full k-fold (Cell 3b) — runs overnight
```

**Deliverable:** Fold 0 trains without errors. Full run started on Colab.

---

### Day 4 — Wed Mar 5 | Classical ML Baseline + Full Baseline Results
**Goal:** Classical ML comparison table + confirmed full baseline numbers.

#### Task A — Classical ML (Person 3)
```python
# notebooks/classical_baseline.ipynb
# Feature extraction: band-power per 30s epoch
# Bands: delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), sigma (12-15Hz), beta (13-30Hz)
# Classifiers: RandomForest, SVM
# Metrics: accuracy, macro F1, Cohen's kappa
```

#### Task B — Collect Full Baseline Numbers (Person 2)
```bash
# After Colab run completes, run Section 4 of the notebook (prediction + summary)
```

Record into `results/baseline_results.csv`:
- Overall accuracy, Macro F1, Cohen's kappa
- Per-class F1 (W, N1, N2, N3, REM)
- Confusion matrix

**Deliverable:** `results/baseline_results.csv` committed to repo.

---

### Day 5 — Thu Mar 6 | Temporal Context Ablation
**Goal:** Compare 1-epoch (no context) vs 3-epoch (baseline) vs extended context window.

```python
# notebooks/temporal_context_ablation.ipynb
EXPERIMENTS = {
    'no_context':    {'window': 1, 'description': 'Single epoch, no temporal context'},
    'base_3epoch':   {'window': 3, 'description': 'Baseline: current + prev + next'},
    'context_5epoch':{'window': 5, 'description': 'Extended: 2 prev + current + 2 next'},
    'context_7epoch':{'window': 7, 'description': 'Extended: 3 prev + current + 3 next'},
}
```

Run each on **5 folds only** to save time.

**Key question:** Does more context monotonically help, or does it plateau? Does N1 F1 improve?

**Deliverable:** Ablation table + plot (context window size vs. macro F1 / N1 F1).

---

### Day 6 — Fri Mar 7 | Class Imbalance Experiments
**Goal:** Improve N1 detection using loss design and sampling strategies.

```python
# notebooks/class_imbalance_experiments.ipynb
# Experiment A: Weighted Cross-Entropy
# Experiment B: Focal Loss (gamma=2.0)
# Experiment C: Label Smoothing variants (already supported by baseline)
```

```bash
# Label smoothing variants (already supported!)
# Uniform: smooth_value=0.1, smooth_stats=False
# Conditional: smooth_value=0.1, smooth_stats=True
```

**Deliverable:** Table: N1 F1 and macro F1 across all loss strategies vs. baseline.

---

### Day 7 — Sat Mar 8 | Interpretability + All Figures
**Goal:** Generate all visualizations needed for the presentation.

```python
# notebooks/interpretability.ipynb
# Figure 1: Confusion matrices (baseline vs. best experiment)
# Figure 2: Per-class F1 bar chart across all experiments
# Figure 3: Training curves (loss + accuracy over epochs)
# Figure 4: Saliency map / Grad-CAM on CNN filters
# Figure 5: Class distribution in dataset
```

**Deliverable:** All figures saved to `figures/` and committed to repo.

---

### Day 8 — Sun Mar 9 | Results Synthesis + Slide Draft
**Goal:** Complete draft slide deck.

#### Results Summary Table
```
| Model                        | Accuracy | Macro F1 | N1 F1 | kappa |
|------------------------------|----------|----------|-------|-------|
| Classical ML (SVM/RF)        |          |          |       |       |
| DeepSleepNet-Lite (baseline) | ~84.0%   | ~78.0%   |       |       |
| + Temporal context (5-epoch) |          |          |       |       |
| + Focal loss                 |          |          |       |       |
| + Label smoothing (uniform)  |          |          |       |       |
| + Label smoothing (stats)    |          |          |       |       |
| Best combined model          |          |          |       |       |
```

#### Slide Deck Outline (12 slides, ~12 min)
1. Title, team names
2. Motivation: the clinical bottleneck (sleep disorders, manual PSG)
3. Problem formulation (input/output, dataset, metrics)
4. Architecture: DeepSleepNet-Lite diagram
5. Deep Learning course connections (representation learning, generalization, robustness)
6. Baseline results + confusion matrix
7. Classical ML vs. deep learning comparison
8. Temporal context ablation results
9. Class imbalance experiments: N1 F1 improvement
10. Interpretability: saliency maps + what the model learned
11. Key takeaways + limitations
12. Future work + Q&A

**Deliverable:** Draft slide deck for feedback.

---

### Day 9 — Mon Mar 10 | Polish + Rehearsal
- Finalize all figures (consistent colors, fonts, axis labels)
- Fill any gaps in results table
- Full rehearsal — aim for **12 minutes** (leave 3 min for Q&A)

**Common Q&A questions to prep for:**
- "Why is N1 so hard to classify?"
- "How does Monte Carlo dropout work and why does it help?"
- "Did you try Transformers?"
- "How would this generalize to a wearable device?"

---

### Day 10 — Tue Mar 11 | Buffer + Final Dry Run
- Fix anything from rehearsal
- Re-run any experiments with bugs
- Final slide upload / formatting check

---

### Day 11 — Wed Mar 12 | Presentation

---

## Key Files & Structure

```
deepsleepnet-lite/
├── notebooks/
│   ├── DeepSleepNet_Lite_Colab.ipynb   ← Main Colab notebook
│   ├── classical_baseline.ipynb         ← Day 4
│   ├── temporal_context_ablation.ipynb  ← Day 5
│   ├── class_imbalance_experiments.ipynb ← Day 6
│   └── interpretability.ipynb           ← Day 7
├── figures/
│   ├── confusion_matrices.png
│   ├── class_distribution.png
│   ├── training_curves_and_f1.png
│   └── saliency_maps.png
├── results/
│   └── baseline_results.csv
├── prepare_physionet.py                 ← Data download + preprocessing
├── data/                                ← gitignored
├── output/                              ← gitignored
└── SLEEPNET_PROJECT_PLAN.md             ← this file
```

---

## Git Workflow

```bash
# Start a new task
git checkout main && git pull
git checkout -b feature/YOUR_NAME-task-description

# Save work daily
git add notebooks/ figures/ results/
git commit -m "Day 5: temporal context ablation — 5-fold results"
git push origin feature/day5-temporal-ablation

# Open PR -> teammate reviews -> merge to main
```

---

## Minimum Viable Presentation (if time runs short)
1. Reproduced baseline numbers (must have)
2. One ablation result — temporal context OR focal loss (must have)
3. Confusion matrix visualization (must have)
4. Classical ML comparison (nice to have)
5. Interpretability / saliency maps (nice to have)
6. All experiments completed (ideal)

---

## Useful Commands Quick Reference

```bash
# Train single fold
python train.py --data_dir data/eeg_FpzCz_PzOz_v1 \
  --output_dir output/model/v1/base --n_folds 20 --fold_idx 0 \
  --train_epochs 100 --smooth_value 0 --smooth_stats=False --resume=False

# Train all folds (in notebook Cell 3b, or manually)
for i in $(seq 0 19); do
  python train.py --data_dir data/eeg_FpzCz_PzOz_v1 \
    --output_dir output/model/v1/base --n_folds 20 --fold_idx $i \
    --train_epochs 100 --smooth_value 0 --smooth_stats=False --resume=False
done

# Predict
python predict.py --data_dir data/eeg_FpzCz_PzOz_v1 \
  --model_dir output/model/v1/base --output_dir output/results/v1/base

# Summary metrics
python summary_muquery.py --data_dir output/results/v1/base
```
