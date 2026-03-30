import json
import torch
from transformers import pipeline
from tqdm import tqdm
from collections import defaultdict
import pandas as pd


print("Loading RoBERTa-large-mnli...")
classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1)

def run_full_evaluation(input_file):
    stats = defaultdict(lambda: {"correct": 0, "total": 0}) 
    detailed_results = []

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} samples...")
    for line in tqdm(lines):
        data = json.loads(line)
        label = data['original_label']
        

        prediction = classifier(f"{data['premise']} </s> </s> {data['hypothesis']}")
        pred_label = prediction[0]['label'].upper() # ENTAILMENT, NEUTRAL, CONTRADICTION


        is_correct = (pred_label == 'ENTAILMENT')
        if is_correct:
            stats[label]["correct"] += 1
        stats[label]["total"] += 1
        

        data['predicted_nli'] = pred_label
        data['conf_score'] = prediction[0]['score']
        detailed_results.append(data)


    report = []
    total_correct = 0
    total_samples = 0
    
    for rel, s in stats.items():
        acc = s["correct"] / s["total"]
        report.append({"Relation": rel, "Total": s["total"], "Correct": s["correct"], "Accuracy": f"{acc:.2%}"})
        total_correct += s["correct"]
        total_samples += s["total"]


    df_report = pd.DataFrame(report).sort_values(by="Accuracy", ascending=False)
    print("\n" + "="*50)
    print("NLI RECASTING DIAGNOSTIC REPORT")
    print("="*50)
    print(df_report.to_string(index=False))
    print("-"*50)
    print(f"OVERALL ACCURACY: {total_correct/total_samples:.2%}")
    print("="*50)


    with open("nli_results_full.jsonl", 'w', encoding='utf-8') as f:
        for item in detailed_results:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    
    run_full_evaluation("train_nli.jsonl")