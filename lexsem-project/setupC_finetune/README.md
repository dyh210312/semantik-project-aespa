# T5 Relation Classification (H2)

## 📌 Overview

This folder contains the implementation of **H2: supervised fine-tuning** for relation classification using a transformer-based model.

---

## 📂 Files

* `prepare_t5_data.py` – convert TSV data to T5 format
* `train_t5_relation.py` – train model and generate predictions
* `relitu.py` – utility functions (if used)

---

## ⚙️ Requirements

Install dependencies:

pip install transformers torch pandas scikit-learn tqdm

---

## ▶️ Run Order

### 1. Prepare data

python prepare_t5_data.py

→ generates:

* `train_t5.csv`
* `val_t5.csv`
* `test_t5.csv`

---

### 2. Train model & predict

python train_t5_relation.py

→ generates:

* `submission.tsv`

---

### 3. (Optional) Evaluate

Use your evaluation script to compute:

* Accuracy
* Macro-F1
* Confusion Matrix

---

## 📊 Output

* `submission.tsv` – predicted relations
* (optional) `h2_final_heatmap.png`

---

## ⚠️ Notes

* Make sure `train.tsv`, `val.tsv`, `test.tsv` are in the same folder
* GPU is recommended but not required
* First run will download the model automatically

---
