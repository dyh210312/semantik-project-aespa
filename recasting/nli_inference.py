from transformers import pipeline
import json

# 1. 加载预训练的 NLI 模型 (RoBERTa-large-mnli 是目前的 SOTA 之一)
# 这对应了你大纲中提到的 Methods [cite: 22]
classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1) # CPU运行设为-1

def run_inference(input_file):
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 拼接 Premise 和 Hypothesis
            # 模型会输出: 'entailment', 'neutral', 或 'contradiction'
            out = classifier(f"{data['premise']} </s> </s> {data['hypothesis']}")
            
            results.append({
                "compound": data['compound'],
                "original_label": data['original_label'],
                "predicted_nli": out[0]['label'],
                "score": out[0]['score']
            })
            
            # 先测试前5条看效果
            if len(results) >= 5: break 
            
    for res in results:
        print(f"Compound: {res['compound']} | Label: {res['original_label']}")
        print(f"Prediction: {res['predicted_nli']} ({res['score']:.4f})\n")

if __name__ == "__main__":
    run_inference("train_nli.jsonl")