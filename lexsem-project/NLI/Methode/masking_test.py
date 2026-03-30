import json
import os
from transformers import pipeline
from tqdm import tqdm

# 1. 加载模型（确保 device=-1 使用 CPU，如果有 GPU 改为 0）
classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1)

def run_masking_test(input_file, mask_type=None):
    """
    mask_type: None (原始), 'n1' (遮住n1), 'n2' (遮住n2)
    """
    correct = 0
    total = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Running Masking Test [{mask_type if mask_type else 'Original'}]...")
    
    for line in tqdm(lines):
        data = json.loads(line)
        
        # 改进的拆分逻辑
        words = data['compound'].split()
        if len(words) < 2:
            continue # 防止有些脏数据只有一个词
            
        n1 = words[0]
        n2 = " ".join(words[1:])
        
        # 构建遮掩后的前提 (Premise)
        if mask_type == 'n1':
            premise = f"This is a [MASK] {n2}."
        elif mask_type == 'n2':
            premise = f"This is a {n1} [MASK]."
        else:
            premise = data['premise']

        # 运行推理
        prediction = classifier(f"{premise} </s> </s> {data['hypothesis']}")
        pred_label = prediction[0]['label'].upper()

        if pred_label == 'ENTAILMENT':
            correct += 1
        total += 1

    acc = correct / total
    return acc

if __name__ == "__main__":
    # 获取当前脚本所在的绝对路径 (即 .../recasting/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接出 test_nli.jsonl 的完整路径
    test_file = os.path.join(current_dir, "test_nli.jsonl")
    
    # 依次跑三种情况
    original_acc = run_masking_test(test_file, mask_type=None)
    mask_n1_acc = run_masking_test(test_file, mask_type='n1')
    mask_n2_acc = run_masking_test(test_file, mask_type='n2')

    print("\n" + "="*40)
    print("ROBUSTNESS (MASKING) REPORT")
    print("="*40)
    print(f"Original Accuracy:   {original_acc:.2%}")
    print(f"Mask n1 Accuracy:     {mask_n1_acc:.2%}")
    print(f"Mask n2 Accuracy:     {mask_n2_acc:.2%}")
    print("-" * 40)
    
    # 计算两个方向的下降率
    drop_n1 = original_acc - mask_n1_acc
    drop_n2 = original_acc - mask_n2_acc
    
    print(f"Drop after masking n1: {drop_n1:.2%}")
    print(f"Drop after masking n2: {drop_n2:.2%}")
    print("="*40)

    if drop_n1 > 0.5 and drop_n2 > 0.5:
        print("Conclusion: Model is highly compositional (Good!).")
    else:
        print("Conclusion: Potential Lexical Bias detected.")