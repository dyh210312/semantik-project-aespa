import os
print("Working dir:", os.getcwd())
import pandas as pd
import json
from transformers import pipeline

def get_article(word: str) -> str:
    if not word:
        return "a"
    return "an" if word[0].lower() in "aeiou" else "a"


df = pd.read_csv("val.tsv", sep="\t", header=None, names=["n1", "n2", "label"])


print("Loading roberta-base fill-mask pipeline...")
mlm = pipeline("fill-mask", model="roberta-base")


templates = {
    "T2_better": "{article_cap_n1} {n1} {n2} is {article_n2} {n2} that <mask> {n1}.",
    "T3_best": "{article_cap_n1} {n1} {n2} refers to {article_n2} {n2} that <mask> {n1}.",
}

all_results = []

for template_name, template in templates.items():
    print(f"\nRunning template: {template_name}")

    for _, row in df.iterrows():
        n1 = str(row["n1"]).strip()
        n2 = str(row["n2"]).strip()
        label = str(row["label"]).strip()

        article_n1 = get_article(n1)
        article_n2 = get_article(n2)

        prompt = template.format(
            n1=n1,
            n2=n2,
            article_cap_n1=article_n1.capitalize(),
            article_n2=article_n2
        )

        try:
            preds = mlm(prompt, top_k=5)
        except Exception as e:
            print(f"Error: {prompt}")
            print(e)
            continue

        result = {
            "template": template_name,
            "compound": f"{n1} {n2}",
            "n1": n1,
            "n2": n2,
            "label": label,
            "prompt": prompt,
            "predictions": [
                {"word": p["token_str"].strip(), "score": float(p["score"])}
                for p in preds
            ]
        }
        all_results.append(result)

# 4. 保存结果
with open("cloze_val_predictions.jsonl", "w", encoding="utf-8") as f:
    for item in all_results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nDone. Saved {len(all_results)} predictions to cloze_val_predictions.jsonl")
