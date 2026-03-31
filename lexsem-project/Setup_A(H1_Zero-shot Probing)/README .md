# Cloze Zero-shot Relation Classification

## Setup
pip install transformers pandas torch

## Files
- train.tsv / val.tsv / test.tsv
- run_h1_test_final.py
- evaluate_h1_all_relations.py

## Steps

1. Template selection:
python probe_templates_test.py

2. Run test:
python run_h1_test_final.py

3. Evaluate:
python evaluate_h1_all_relations.py

## Output
- cloze_test_predictions.jsonl
- h1_test_final_results.csv