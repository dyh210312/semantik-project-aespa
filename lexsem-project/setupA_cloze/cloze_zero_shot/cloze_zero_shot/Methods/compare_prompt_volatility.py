import json
from collections import defaultdict
import pandas as pd
from transformers import pipeline

# =========================
# 1. 配置
# =========================
INPUT_JSONL = "test_nli.jsonl"
MODEL_NAME = "roberta-base"
TOP_K = 5

TEMPLATES = {
    "T1_original": "A {n1} {n2} refers to a {n2} that <mask> {n1}.",
    "T2_reverse": "A {n2} that <mask> {n1} is called a {n1} {n2}.",
    "T3_short": "A {n2} that <mask> {n1}."
}

# =========================
# 2. 关键词（必须完整）
# =========================
RELATION_KEYWORDS = {
    "containment": {"contain", "contains", "contained", "holding", "hold", "holds"},
    "purpose": {"use", "uses", "used", "using", "for"},
    "attribute": {"have", "has", "include", "includes"},
    "loc_part_whole": {"in", "inside", "within"},
    "topical": {"about", "on", "regarding"},
    "time": {"during", "after", "before"},
    "complement": {"with", "include", "includes"},
    "causal": {"cause", "causes", "produce"},
    "objective": {"achieve", "gain"},
    "owner_emp_use": {"use", "uses", "manage", "work"},
    "cause": {"from", "due"},
    "other": set()
}

def normalize(x):
    return x.strip().lower()

def load_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            comp = item["compound"].strip()
            label = item["original_label"]

            parts = comp.split()
            if len(parts) != 2:
                continue

            rows.append({
                "n1": parts[0],
                "n2": parts[1],
                "label": label
            })
    return pd.DataFrame(rows)

# =========================
# 3. 主逻辑
# =========================
df = load_data(INPUT_JSONL)
print(f"Loaded {len(df)} samples")

mlm = pipeline("fill-mask", model=MODEL_NAME)

for name, template in TEMPLATES.items():
    stats = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0})

    print("\n====================")
    print(f"Running {name}")
    print("====================")

    for _, row in df.iterrows():
        n1, n2, label = row["n1"], row["n2"], row["label"]

        if label not in RELATION_KEYWORDS or label == "other":
            continue

        prompt = template.format(n1=n1, n2=n2)

        try:
            preds = mlm(prompt, top_k=TOP_K)
        except:
            continue

        words = [normalize(p["token_str"]) for p in preds]

        gold = RELATION_KEYWORDS[label]

        top1_hit = words[0] in gold if words else False
        top5_hit = any(w in gold for w in words[:5])

        stats[label]["total"] += 1
        if top1_hit:
            stats[label]["top1"] += 1
        if top5_hit:
            stats[label]["top5"] += 1

    # 汇总
    top1_sum, top5_sum, count = 0, 0, 0

    for rel, s in stats.items():
        if s["total"] == 0:
            continue

        t1 = s["top1"] / s["total"]
        t5 = s["top5"] / s["total"]

        top1_sum += t1
        top5_sum += t5
        count += 1

    avg_t1 = top1_sum / count if count else 0
    avg_t5 = top5_sum / count if count else 0

    print(f"{name} → Avg Top-1: {avg_t1:.4f} | Avg Top-5: {avg_t5:.4f}")