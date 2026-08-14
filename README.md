# Assignment 2 — Stage 1: Movie Review Sentiment Classification

Predicts `0 = negative` / `1 = positive` for Pang & Lee movie reviews, trained on a
small, imbalanced set (240 reviews: 180 positive / 60 negative) and evaluated on a
balanced public test set (400 reviews).

## Model

- **Frozen pretrained backbone:** `distilbert-base-uncased-finetuned-sst-2-english`
  (a DistilBERT LM already fine-tuned for sentiment). Its weights are **not**
  trained. Because reviews average ~788 words (past the 256-token window), each
  review is split into ~200-word chunks and the backbone scores `P(positive)` per
  chunk.
- **Trained head:** a **Logistic Regression** (6 weights) over five aggregate
  features of the chunk probabilities `[mean, max, min, std, fraction_positive]`,
  with `class_weight="balanced"` and the L2 strength `C` chosen by 5-fold
  stratified cross-validation on the training set only.

Training only 6 weights on top of a strong pretrained signal is what makes this
robust on 240 imbalanced examples without overfitting. Full design rationale and
the answers to all Stage 1 questions are in `stage1_notebook.ipynb`.

**Public test result:** accuracy **0.9025**, balanced accuracy 0.9025.
Confusion matrix `[[TN, FP], [FN, TP]] = [[174, 26], [13, 187]]`.

## Repository contents

| File | Purpose |
|---|---|
| `stage1_notebook.ipynb` | Self-contained: trains, evaluates, saves checkpoint & predictions |
| `model_checkpoint/classifier.joblib` | The trained Logistic Regression head |
| `model_checkpoint/metadata.json` | Config + recorded metrics |
| `public_test_predictions.csv` | `id,predicted_label` for the public test set |
| `requirements.txt` | Pinned dependencies |
| `train.csv`, `public_test.csv` | Data used by the notebook |

> The frozen backbone is **not** committed (its weights are ~260 MB, above
> GitHub's file-size limit). It is a public model loaded **by name** and cached on
> first use — this is not retraining. To run fully offline, save it once into
> `model_checkpoint/backbone/` and the notebook will load it from there instead of
> the hub.

## How to run

```bash
# 1. Create an environment (CPU-only is fine) and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the notebook top-to-bottom (~1–2 min on CPU).
#    It downloads the backbone on first run, then trains the head,
#    saves model_checkpoint/, and writes public_test_predictions.csv.
jupyter nbconvert --to notebook --execute --inplace stage1_notebook.ipynb
#    (or open it in Jupyter and Run All)
```

## Predict from the saved checkpoint (no retraining)

The notebook's **section 6** defines `predict_from_checkpoint(texts)`, which reloads
`model_checkpoint/classifier.joblib` plus the frozen backbone and returns 0/1
predictions for any list of reviews — the exact inference path (reused unchanged in
Stage 2).

## Use of AI

Generative AI tools were used while developing this submission. This section
discloses that use, as required.

- **Tool used:** Claude (Anthropic).
- **What it was used for:**
  - Implementing the chunking of long reviews into ~200-word windows and the
    aggregation of per-chunk `P(positive)` into the five features
    `[mean, max, min, std, fraction_positive]`.
  - Debugging the 5-fold stratified cross-validation used to pick the L2 strength
    `C`, and wiring up the checkpoint save/reload path.
- **Representative prompts:**
  - "I have 240 movie reviews (180 positive / 60 negative) and a 400-review
    balanced test set. Reviews are long. What's a robust approach that won't
    overfit such a small, imbalanced set?"
  - "Reviews are longer than the 256-token window. Write code to split each review
    into ~200-word chunks, score each chunk with a frozen SST-2 DistilBERT, and
    aggregate the chunk probabilities into features for a Logistic Regression head."
  - "Set up 5-fold stratified cross-validation to choose the regularization
    strength C, using class_weight='balanced', and save the trained head plus its
    metadata to a checkpoint I can reload without retraining."
- **How the output was verified:** the notebook was run end-to-end and the reported
  metrics (public-test accuracy 0.9025, confusion matrix `[[174, 26], [13, 187]]`)
  were reproduced from the saved checkpoint; each AI-suggested step was reviewed
  against the course material and checked for correctness and reproducibility
  before being kept.
- **Stage 2:** Claude also helped run the inference-only hidden-test evaluation —
  loading the Stage 1 checkpoint (no retraining), generating
  `hidden_test_predictions.csv`, and drafting `stage2_notebook.ipynb` /
  `hidden_test_evaluation.md`. The hidden-test result (accuracy 0.8833) was
  reproduced from the same checkpoint and reviewed before submission.

All modeling decisions, the final approach, and every result were chosen, run, and
verified by the author; AI was used as an assistant, not a replacement.

## References

- **Dataset:** Pang & Lee movie-review polarity corpus, redistributed via NLTK.
- **Pretrained model:** `distilbert-base-uncased-finetuned-sst-2-english`
  (Hugging Face), used as a frozen backbone.
- **Libraries:** PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy.

## Credits

- **Author:** Ahmad Al Damin ([@ahmadaldamin3-tech](https://github.com/ahmadaldamin3-tech))
  — designed the approach, ran the experiments, and verified all results.
- **AI assistance:** Claude (Anthropic), as described in the *Use of AI* section above.
