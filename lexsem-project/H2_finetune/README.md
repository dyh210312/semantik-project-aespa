# Fine-tuning (H2) Relation Classification

## Setup
pip install transformers pandas torch scikit-learn matplotlib seaborn

## Files
- train.tsv / val.tsv / test.tsv
- prepare_t5_data.py
- train_t5_relation.py
- relitu.py

## Steps

1. Prepare data:
python prepare_t5_data.py

2. Train model:
python train_t5_relation.py

3. Evaluate:
python relitu.py

## Output
- submission.tsv
- h2_final_heatmap.png
- Accuracy / Macro-F1 (printed in console)