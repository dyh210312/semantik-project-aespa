# Compositional Reasoning or Lexical Prior?
## Semantic Relation Classification in Noun Compounds

This repository contains the implementation and experiments for our project:

**"Compositional Reasoning or Lexical Prior? A Study on Semantic Relation Classification in Noun Compounds"**

Authors: Yunhe Dong, Wenshuang Hu, Yuzhou Shi, Xiaoci Zhang  
Course: Formale und Computationelle Semantik  
University of Heidelberg  

---

## 📌 Overview

Nominal compounds (e.g., *water bottle*, *bread knife*) encode implicit semantic relations between two nouns. This project investigates whether pretrained language models (PLMs) truly capture these compositional relations or rely on shallow lexical patterns.

We study this problem through three complementary settings:

- **H1: Cloze-based Zero-shot Probing**
- **NLI Recasting (Zero-shot reasoning)**
- **H2: Supervised Fine-tuning (Closed-world classification)**
- **H3: Lexical Bias Analysis (Interpretability)**

---

## 🧪 Experimental Setup

### 🔹 H1: Cloze-based Zero-shot Probing

- Model: `RoBERTa-base`
- Task: Predict masked token in prompts
- Example:
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
