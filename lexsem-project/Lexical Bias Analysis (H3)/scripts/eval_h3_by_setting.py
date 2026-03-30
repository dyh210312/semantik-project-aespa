import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

print("Loading RoBERTa-large-mnli...")
classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1)


def evaluate_file(input_file):
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    detailed_results = []

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} samples...")

    for line in tqdm(lines):
        data = json.loads(line)

        prediction = classifier(f"{data['premise']} </s> </s> {data['hypothesis']}")
        pred_label = prediction[0]["label"].upper()

        setting = data["setting"]
        is_correct = (pred_label == "ENTAILMENT")

        if is_correct:
            stats[setting]["correct"] += 1
        stats[setting]["total"] += 1

        data["predicted_nli"] = pred_label
        data["conf_score"] = prediction[0]["score"]
        detailed_results.append(data)

    rows = []
    for setting, s in stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        rows.append({
            "setting": setting,
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(acc, 4)
        })

    df = pd.DataFrame(rows).sort_values("setting")

    print("\n" + "=" * 50)
    print("H3 RESULTS BY SETTING")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50)

    df.to_csv("h3_setting_results.csv", index=False)
    print("saved: h3_setting_results.csv")

    with open("h3_detailed_results.jsonl", "w", encoding="utf-8") as f:
        for item in detailed_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("saved: h3_detailed_results.jsonl")


if __name__ == "__main__":
    evaluate_file("test_h3_nli.jsonl")