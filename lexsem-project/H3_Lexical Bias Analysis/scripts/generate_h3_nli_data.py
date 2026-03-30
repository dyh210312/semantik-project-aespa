import pandas as pd
import json

RELATION_TEMPLATES = {
    "owner_emp_use": "The head is employed, owned, or used by the modifier.",
    "purpose": "The head is used for the purpose of the modifier.",
    "complement": "The head complements or completes the modifier.",
    "attribute": "The modifier describes an attribute of the head.",
    "cause": "The modifier causes the head.",
    "objective": "The modifier is the object or target of the head.",
    "causal": "The head is caused or motivated by the modifier.",
    "loc_part_whole": "The head is located in or part of the modifier.",
    "topical": "The head is about the topic of the modifier.",
    "containment": "The head contains the modifier.",
    "time": "The modifier indicates the time of the head.",
    "other": "The head has some other relation to the modifier."
}


def convert_file(input_tsv, output_jsonl):
    df = pd.read_csv(input_tsv, sep="\t")
    results = []

    for _, row in df.iterrows():
        label = row["label"]
        hypothesis = RELATION_TEMPLATES[label]

        results.append({
            "compound": f"{row['n1']} {row['n2']}",
            "setting": row["setting"],
            "premise": row["compound_text"],
            "hypothesis": hypothesis,
            "original_label": label
        })

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"saved: {output_jsonl}")


if __name__ == "__main__":
    convert_file("train_h3_degenerate.tsv", "train_h3_nli.jsonl")
    convert_file("val_h3_degenerate.tsv", "val_h3_nli.jsonl")
    convert_file("test_h3_degenerate.tsv", "test_h3_nli.jsonl")