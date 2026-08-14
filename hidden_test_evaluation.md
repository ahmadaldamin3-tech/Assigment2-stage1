# Stage 2 — Hidden Test Evaluation

**Model:** frozen `distilbert-base-uncased-finetuned-sst-2-english` backbone →
5 document-level sentiment features `[mean, max, min, std, frac_positive]` →
Logistic Regression head. Predictions were produced by loading the **exact Stage 1
checkpoint** (`model_checkpoint/classifier.joblib`); the model was **not** retrained,
fine-tuned, or modified. See `stage2_notebook.ipynb` for the runnable inference.

## Hidden test accuracy

| Metric | Value |
|---|---|
| Accuracy | **0.8833** (530 / 600) |
| Balanced accuracy | **0.8833** |
| Set size / balance | 600 reviews (300 negative / 300 positive) |

## Hidden test confusion matrix

Rows = true label, columns = predicted label.

|            | pred negative | pred positive |
|------------|:-------------:|:-------------:|
| **true negative** | 256 | 44 |
| **true positive** | 26  | 274 |

Per-class: negative precision 0.91 / recall 0.85; positive precision 0.86 /
recall 0.91.

## Public vs. hidden comparison

| | Public test | Hidden test |
|---|---|---|
| Size / balance | 400 (200 / 200) | 600 (300 / 300) |
| Accuracy | 0.9025 | 0.8833 |
| Balanced accuracy | 0.9025 | 0.8833 |
| Confusion matrix | `[[174, 26], [13, 187]]` | `[[256, 44], [26, 274]]` |

The hidden-test accuracy is only **~1.9 points** below the public-test accuracy
(gap = +0.0192) — a small, expected drop on fresh, unseen reviews. Both sets are
balanced and both confusion matrices are close to symmetric, so the model is **not
biased toward the majority (positive) class** it saw in training: freezing a strong
pretrained sentiment backbone and training only a 6-weight `class_weight="balanced"`
head transferred as intended. Errors on the hidden set lean slightly toward false
positives (44 vs. 26), giving marginally higher recall on the positive class,
consistent with the mild positive lean of the training data. Cross-validation
(~0.93), public (~0.90), and hidden (~0.88) accuracy are all close, indicating the
pipeline generalized rather than memorized.

## If we had more time or compute

- **Parameter-efficient fine-tuning (LoRA / adapters)** of the backbone with strong
  regularization and early stopping, instead of freezing it.
- **Richer document aggregation** — attention pooling over chunk embeddings, or
  extra length / percentile / logit features — instead of 5 hand-picked statistics.
- **Minority-class augmentation** (back-translation, synonym replacement) to ease
  the 3:1 imbalance without hand-labeling data.
- **Decision-threshold calibration** on a validation fold to correct the slight
  false-positive skew.
- **Ensembling** with a second pretrained sentiment model for robustness on
  out-of-distribution reviews.
- **More systematic evaluation** — bootstrap confidence intervals and error
  analysis of misclassified reviews (sarcasm, mixed sentiment, very long reviews).
