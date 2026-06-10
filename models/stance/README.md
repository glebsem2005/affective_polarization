# Stance models

## setfit/
Final SetFit stance classifier (paraphrase-multilingual-MiniLM-L12-v2 backbone).
Three axes: kremlin (pro/anti/neu), svo (pro/anti/neu), ethnic (yes/no).
Load: SetFitModel.from_pretrained('setfit/')
Requires: pip install setfit

## checkpoint/
Intermediate SentenceTransformer checkpoint (step 822, embedding_loss=0.091).
Stage 1 of SetFit contrastive training. Use setfit/ for inference.
Load: SentenceTransformer('checkpoint/')
