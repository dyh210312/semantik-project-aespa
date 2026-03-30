import pandas as pd
import os
import json


# --- Setup B 映射任务 ---
# NLI 设计的核心逻辑
RELATION_TEMPLATES = {
    "owner_emp_use": "The [n2] is employed, owned, or used by the [n1].",
    "purpose": "The [n2] is used for the purpose of [n1].",
    "complement": "The [n2] acts as a complement or part of the [n1].",
    "attribute": "The [n1] is an attribute or characteristic of the [n2].",
    "cause": "The [n1] is the cause of the [n2].",
    "objective": "The [n1] is the object or target of the action of [n2].",
    "causal": "The [n2] is caused or motivated by the [n1].",
    "loc_part_whole": "The [n2] is located in or is a part of the [n1].",
    "topical": "The [n2] is about the topic of [n1].",
    "containment": "The [n2] contains or holds the [n1].",
    "time": "The [n1] indicates the time or duration of the [n2].",
    "other": "The [n2] has a different type of relationship with [n1]."
}

def process_split(split_name):
    # 1. 获取当前脚本所在的绝对路径 (即 .../recasting/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. 向上找一级，再进入 classification
    input_path = os.path.join(current_dir, "..", "classification", "data", "tratz_coarse_grained_lexical", f"{split_name}.tsv")
    
    # 3. 输出文件直接放在脚本同级目录下
    output_path = os.path.join(current_dir, f"{split_name}_nli.jsonl")
    
    # 打印一下路径，方便调试
    print(f"Checking path: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ 仍然找不到文件，请确认路径是否存在: {input_path}")
        return
    
    print(f"--- 正在读取: {input_path} ---")
    df = pd.read_csv(input_path, sep='\t', header=None, names=['n1', 'n2', 'label'])
    nli_data = []

    for _, row in df.iterrows():
        n1, n2, label = row['n1'], row['n2'], row['label']
        compound = f"{n1} {n2}"
        
        # 生成逻辑句子
        premise = f"This is a {compound}."
        hypothesis = RELATION_TEMPLATES[label].replace("[n1]", n1).replace("[n2]", n2)

        nli_data.append({
            "compound": compound,
            "premise": premise,
            "hypothesis": hypothesis,
            "original_label": label,
            "label": "entailment" # NLI 任务的目标标签
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in nli_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"已完成 {split_name} 的转换，保存至 {output_path}")

if __name__ == "__main__":
    # 一次性处理三个数据集
    for split in ["train", "val", "test"]:
        process_split(split)