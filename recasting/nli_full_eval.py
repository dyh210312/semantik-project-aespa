import json
import torch
from transformers import pipeline
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import os

# 1. 加载模型
print("Loading RoBERTa-large-mnli...")
classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1)

def run_full_evaluation(input_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(current_dir, input_file)
    
    if not os.path.exists(full_input_path):
        print(f"找不到输入文件: {full_input_path}")
        return
    stats = defaultdict(lambda: {"correct": 0, "total": 0}) 
    detailed_results = []

    with open(full_input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} samples...")
    for line in tqdm(lines):
        data = json.loads(line)
        label = data['original_label']
        
        # 运行推理
        prediction = classifier(f"{data['premise']} </s> </s> {data['hypothesis']}")
        pred_label = prediction[0]['label'].upper() # ENTAILMENT, NEUTRAL, CONTRADICTION

        # 统计
        is_correct = (pred_label == 'ENTAILMENT')
        if is_correct:
            stats[label]["correct"] += 1
        stats[label]["total"] += 1
        
        # 保存结果用于后续错误分析
        data['predicted_nli'] = pred_label
        data['conf_score'] = prediction[0]['score']
        detailed_results.append(data)

    # 2. 生成报告
    report = []
    total_correct = 0
    total_samples = 0
    
    for rel, s in stats.items():
        acc = s["correct"] / s["total"]
        report.append({"Relation": rel, "Total": s["total"], "Correct": s["correct"], "Accuracy": f"{acc:.2%}"})
        total_correct += s["correct"]
        total_samples += s["total"]

    # 打印表格
    df_report = pd.DataFrame(report).sort_values(by="Accuracy", ascending=False)
    print("\n" + "="*50)
    print("NLI RECASTING DIAGNOSTIC REPORT")
    print("="*50)
    print(df_report.to_string(index=False))
    print("-"*50)
    print(f"OVERALL ACCURACY: {total_correct/total_samples:.2%}")
    print("="*50)

    # 3. 保存详细结果（给 Wenshuang 做可解释性分析用）
    # 根据输入文件名生成对应的输出文件名
    output_file_name = input_file.replace("_nli.jsonl", "_results_full.jsonl")
    full_output_path = os.path.join(current_dir, output_file_name)
    
    with open(full_output_path, 'w', encoding='utf-8') as f:
        for item in detailed_results:
            f.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    splits = ["train_nli.jsonl", "val_nli.jsonl", "test_nli.jsonl"]
    for s in splits:
        run_full_evaluation(s)
