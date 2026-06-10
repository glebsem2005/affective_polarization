# Target-Aspect Sentiment Analysis (TABSA) — Russian Political Comments

Two-model pipeline for target-based sentiment analysis of Russian Telegram
comments in a political discourse context.

## Models

### `sentiment/` — Sentiment classifier

Given a (comment, political_target) pair, predicts whether the comment
expresses **negative** or **positive** sentiment toward that target.

- **Architecture:** BERT-based sequence classifier
- **Input format:** `[CLS] comment_text [SEP] target_entity [SEP]`
- **Output:** 2 labels — `0 = positive`, `1 = negative`
- **Key output column:** `absa_neg_prob` (probability of negative sentiment)

### `relevance/` — Relevance filter

Filters out comment–target pairs where the comment does not actually
discuss the target entity. Applied before sentiment scoring; irrelevant
pairs receive `absa_neg_prob = NaN`.

## Performance

Evaluated on 698 gold-annotated (comment, target) pairs:

| Metric | Value |
|--------|-------|
| Macro F1 — pos/neg (primary) | **0.912 ± 0.028** |
| AUC-ROC | **0.977 ± 0.011** |
| Accuracy | 0.927 |
| Macro F1 — 3-class incl. neutral | 0.652 |
| Majority baseline macro F1 | 0.421 |

5-fold cross-validation, stratified by target entity.

## Target entities

- `pro-government` (Kremlin, Putin, state apparatus)
- `opposition` (liberal opposition, Navalny et al.)
- `radical-nationalist` (Wagner, Prigozhin, ultra-patriots)
- `ukraine` / `west`

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("models/tabsa/sentiment")
mdl = AutoModelForSequenceClassification.from_pretrained("models/tabsa/sentiment")
mdl.eval()

def absa_neg_prob(comment: str, target: str) -> float:
    enc = tok(comment, target, return_tensors="pt",
              truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=-1)[0]
    return probs[1].item()

score = absa_neg_prob("путин развязал эту войну", "pro-government")
# → ~0.77
```

## Pipeline integration

`absa_neg_prob` scores are aggregated per (src_cluster, tgt_cluster, date)
to produce the daily affective polarization series used in ITS analysis.

## Limitations

- Domain-specific: Russian political Telegram (2022–2024)
- Neutral handling uses a confidence threshold (≥ 0.6), not a trained class
- Implicit sentiment (sarcasm) performs surprisingly well (F1=0.949)
  vs. explicit (F1=0.881) — likely annotation artifact in explicit subset
