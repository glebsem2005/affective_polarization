# Political Cluster Classifier — Russian Telegram Comments

Silver-trained BERT classifier assigning Russian Telegram comments to one of
three politically coherent user communities. Trained and evaluated on
agent-labeled (silver) data. **This is the final inference model used in the pipeline.**

## Task

Given a sequence of comments from a single discussion thread (concatenated with
`[SEP]`), predict which political cluster the author belongs to:

| Label | Description |
|-------|-------------|
| `pro-government` | Pro-Kremlin / statist discourse; supportive of the war, hostile to Western framing |
| `opposition` | Liberal / anti-war / pro-Western discourse |
| `radical-nationalist` | Ethnic-nationalist / ultra-patriotic discourse; critical of the Kremlin from the right |

## Architecture

- **Base model:** `DeepPavlov/rubert-base-cased`
- **Classification head:** custom MLP (hidden=256) with per-class thresholds
- **Extra features:** NLI scores (gov/opp/ethnic/ukr/west), mean sentiment, toxicity
- **Input:** up to 5 concatenated comments per author, joined with `[SEP]`,
  max 256 tokens
- **Confidence threshold:** 0.6 (low-confidence predictions flagged separately)

## Performance

5-fold cross-validation on agent-labeled (silver) data:

| Metric | Value |
|--------|-------|
| Macro F1 | **0.696 ± 0.019** |
| Accuracy | **0.701 ± 0.018** |

Gold-label fine-tuning was evaluated separately (n=438 manual annotations,
macro F1=0.42) and was insufficient — silver-trained model is used for inference.

## Per-class thresholds

| Class | Threshold |
|-------|-----------|
| pro-government | 0.35 |
| opposition | 0.30 |
| radical-nationalist | 0.45 |

## Usage

```python
import torch
from pathlib import Path
# See scripts/pipeline.py for full inference pipeline
# model loads via UserClassifier from scripts/train_user_classifier.py
```

## Limitations

- Trained on agent-labeled data: label quality depends on LLM annotation
- Low-confidence predictions (~27%) are excluded from AP analysis
- Systemic-opposition not covered by the three-class scheme
- Domain-specific: Russian political Telegram 2022–2024
