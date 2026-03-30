import pandas as pd
import os
import json


DATA_PATH = r"C:\Users\XccccZ\OneDrive\Desktop\uni-Heidelberg\Semantik\panic-master\classification\data\tratz_coarse_grained_lexical\train.tsv"


if not os.path.exists(DATA_PATH):
    print(f"仍然无法找到文件，请在 VS Code 中右键点击 train.tsv 选择 'Copy Path'，然后替换上面的 DATA_PATH。")
else:
    print(f"成功定位数据：{DATA_PATH}")


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

def generate_nli():

    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['n1', 'n2', 'label'])
    nli_data = []
    
    for _, row in df.iterrows():
        n1, n2, label = row['n1'], row['n2'], row['label']

        premise = f"This is a {row['n1']} {row['n2']}."
        template = RELATION_TEMPLATES.get(row['label'], "The [n1] and [n2] are related.")
        hypothesis = template.replace("[n1]", row['n1']).replace("[n2]", row['n2'])
        current_compound = f"{n1} {n2}"

        nli_data.append({
            "compound": current_compound,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": "entailment",
            "original_label": row['label']            
        })
    

    output_path = "train_nli.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in nli_data:
            f.write(json.dumps(entry) + '\n')
    print(f"完成！已生成 NLI 格式文件：{os.path.abspath(output_path)}")

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        generate_nli()