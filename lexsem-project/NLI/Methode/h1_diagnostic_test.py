import json
from transformers import pipeline

# 1. 加载模型 (RoBERTa-large-mnli)
classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1)

# 2. 定义我们要测试的“重灾区”样本 (来自 Dong 的 Failed Examples )
test_samples = [
    {"compound": "combat aircraft", "label": "purpose", "n1": "combat", "n2": "aircraft"},
    {"compound": "college employee", "label": "owner_emp_use", "n1": "college", "n2": "employee"},
    {"compound": "abortion opponent", "label": "objective", "n1": "abortion", "n2": "opponent"}
]

# 3. 定义新旧模板对比
templates = {
    "purpose": {
        "old": "The [n2] is used for the purpose of [n1].",
        "new": "The [n2] is a type of [n2] specifically designed for [n1]."
    },
    "owner_emp_use": {
        "old": "The [n2] is employed, owned, or used by the [n1].",
        "new": "The [n1] is the employer of the [n2]."
    },
    "objective": {
        "old": "The [n1] is the object or target of the action of [n2].",
        "new": "The [n2] is defined by its specific stance against [n1]."
    }
}

print(f"{'Compound':<20} | {'Rel':<15} | {'Old Conf':<10} | {'New Conf':<10} | {'Result'}")
print("-" * 75)

for sample in test_samples:
    rel = sample['label']
    n1, n2 = sample['n1'], sample['n2']
    
    # 填充模板
    premise_old = templates[rel]['old'].replace("[n1]", n1).replace("[n2]", n2)
    premise_new = templates[rel]['new'].replace("[n1]", n1).replace("[n2]", n2)
    hypothesis = f"The relationship is {rel}." # 保持 Hypothesis 一致

    # 运行推理
    res_old = classifier(f"{premise_old} </s> </s> {hypothesis}")[0]
    res_new = classifier(f"{premise_new} </s> </s> {hypothesis}")[0]

    # 获取置信度 (Confidence Score)
    score_old = res_old['score'] if res_old['label'] == 'ENTAILMENT' else 0.0
    score_new = res_new['score'] if res_new['label'] == 'ENTAILMENT' else 0.0
    
    status = "UP " if score_new > score_old else "DOWN "
    print(f"{sample['compound']:<20} | {rel:<15} | {score_old:.4f} | {score_new:.4f} | {status}")