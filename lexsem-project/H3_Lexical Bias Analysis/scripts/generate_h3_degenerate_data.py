import pandas as pd
from pathlib import Path

INPUT_FILES = ["train.tsv", "val.tsv", "test.tsv"]
MASK_TOKEN = "[MASK]"


def make_full(n1, n2):
    return f"modifier: {n1} ; head: {n2}"


def make_mask_modifier(n1, n2):
    return f"modifier: {MASK_TOKEN} ; head: {n2}"


def make_mask_head(n1, n2):
    return f"modifier: {n1} ; head: {MASK_TOKEN}"


def generate_degenerate_versions(input_path):
    df = pd.read_csv(input_path, sep="\t", header=None, names=["n1", "n2", "label"])

    rows = []

    for _, row in df.iterrows():
        n1 = row["n1"]
        n2 = row["n2"]
        label = row["label"]

        rows.append({
            "n1": n1,
            "n2": n2,
            "label": label,
            "setting": "full",
            "compound_text": make_full(n1, n2)
        })

        rows.append({
            "n1": n1,
            "n2": n2,
            "label": label,
            "setting": "mask_modifier",
            "compound_text": make_mask_modifier(n1, n2)
        })

        rows.append({
            "n1": n1,
            "n2": n2,
            "label": label,
            "setting": "mask_head",
            "compound_text": make_mask_head(n1, n2)
        })

    out_df = pd.DataFrame(rows)

    input_path = Path(input_path)
    output_path = input_path.with_name(input_path.stem + "_h3_degenerate.tsv")
    out_df.to_csv(output_path, sep="\t", index=False)

    print(f"saved: {output_path}")
    print(out_df.head(9).to_string(index=False))


if __name__ == "__main__":
    for file_name in INPUT_FILES:
        generate_degenerate_versions(file_name)