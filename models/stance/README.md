# Stance models

## setfit/
Final SetFit stance classifier (paraphrase-multilingual-MiniLM-L12-v2 backbone).
Three axes: kremlin (pro/anti/neu), svo (pro/anti/neu), ethnic (yes/no).
Load: SetFitModel.from_pretrained('setfit/')
Requires: pip install setfit
