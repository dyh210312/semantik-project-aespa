import pandas as pd

def convert_tsv_to_t5_csv(input_path: str, output_path: str):
    df = pd.read_csv(input_path, sep="\t", header=None, names=["n1", "n2", "label"])

    df["input_text"] = "compound: " + df["n1"].astype(str) + " " + df["n2"].astype(str)
    df["target_text"] = df["label"].astype(str)

    out_df = df[["input_text", "target_text"]]
    out_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {output_path} with {len(out_df)} rows")

convert_tsv_to_t5_csv("train.tsv", "train_t5.csv")
convert_tsv_to_t5_csv("val.tsv", "val_t5.csv")
convert_tsv_to_t5_csv("test.tsv", "test_t5.csv")