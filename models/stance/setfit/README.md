# Political Cluster Classifier — Russian Telegram Comments

Few-shot SetFit classifier assigning Russian Telegram comments to one of three
politically coherent user communities identified in the AP pipeline study.

## Task

Given a sequence of comments from a single discussion thread (concatenated with
`[SEP]`), predict which political cluster the author belongs to:

| Label | Description |
|-------|-------------|
| `pro-government` | Pro-Kremlin / statist discourse; supportive of the war, hostile to Western framing |
| `opposition` | Liberal / anti-war / pro-Western discourse |
| `radical-nationalist` | Ethnic-nationalist / ultra-patriotic discourse; critical of the Kremlin from the right |

## Architecture

- **Base model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Method:** SetFit (Tunstall et al., 2022) — contrastive fine-tuning of the
  sentence encoder + logistic regression head
- **Input:** up to 5 concatenated comments per author, joined with `[SEP]`,
  truncated to 128 tokens
- **Output:** one of three cluster labels

## Performance

5-fold cross-validation on silver-labelled held-out data:

| Metric | Mean | Std |
|--------|------|-----|
| Macro F1 | **0.696** | 0.019 |
| Accuracy | **0.701** | 0.018 |

(Silver labels derived from the full unsupervised clustering pipeline;
gold-label validation on 400 manually annotated examples showed consistent
cluster separation.)

## Usage

```bash
pip install setfit
```

```python
from setfit import SetFitModel

model = SetFitModel.from_pretrained("path/to/models/stance/setfit")

# Single author: pass list of their recent comments joined by [SEP]
text = "Путин всё делает правильно [SEP] Запад хочет уничтожить Россию [SEP] Наша армия победит"
pred = model.predict([text])
print(pred)  # ['pro-government']
```

## Limitations

- Trained on Russian-language Telegram comments (2022–2024, war context)
- Three-class typology reflects discourse communities in this specific corpus;
  not a universal political typology
- Low-confidence predictions (~27% of corpus) are flagged and handled
  separately in the pipeline (`pct_low_conf` column in CV results)
- Systemic-opposition users (moderate pro-system critics) are excluded from
  the three-class scheme and classified separately

## Citation

```bibtex
@article{tunstall2022setfit,
  title={Efficient Few-Shot Learning Without Prompts},
  author={Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and others},
  journal={arXiv preprint arXiv:2209.11055},
  year={2022}
}
```
