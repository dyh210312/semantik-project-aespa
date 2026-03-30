- Evaluation:
- Top-1 Accuracy
- Top-5 Accuracy
- Keyword matching → relation mapping

👉 Goal: Test whether relational knowledge exists **without training**

---

### 🔹 Prompt Volatility (H1 Extension)

We evaluate robustness by comparing different prompt templates:

- T1 (original)
- T2 (rephrased)
- T3 (simplified)

👉 Finding:
Performance varies significantly → model is highly sensitive to prompt design

---

### 🔹 NLI Recasting (Zero-shot Reasoning)

- Model: `RoBERTa-large-MNLI`
- Reformulation:
- Premise: "This is a bread knife."
- Hypothesis: "The knife is used for bread."

- Task:
Predict **entailment / neutral / contradiction**

👉 Result:
- Accuracy: **89.22%**

👉 Insight:
NLI provides stronger semantic scaffolding than cloze probing

---

### 🔹 H2: Supervised Fine-tuning (Closed-world)

- Model: `RoBERTa-large`
- Task: 12-class classification
- Input: noun compounds
- Output: predefined relation label

- Training:
- Epochs: 10
- Learning rate: 1e-5
- Batch size: 16

👉 Results:
- Accuracy: **67.35%**
- Macro-F1: **58.96%**

👉 Insight:
Fine-tuning improves stability but limits generalization

---

### 🔹 H3: Lexical Bias Analysis

We test whether models rely on lexical shortcuts:

Input settings:
- Full: `plaintiff attorney`
- Mask modifier: `[MASK] attorney`
- Mask head: `plaintiff [MASK]`

👉 Results:
- Full: 77.41%
- Mask modifier: 73.04%
- Mask head: 63.03%

👉 Insight:
Models rely heavily on **head noun bias**

---

### 🔹 SHAP Interpretability Analysis

We apply SHAP to analyze model predictions:

- Identify feature attribution
- Detect lexical bias
- Reveal "Template Trap" in NLI

👉 Key finding:
Model decisions are influenced by:
- lexical co-occurrence
- prompt structure
- syntactic artifacts

---

## 📂 Repository Structure
