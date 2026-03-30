from transformers import pipeline
import json


classifier = pipeline("text-classification", model="roberta-large-mnli", device=-1) # CPU运行设为-1

def run_inference(input_file):
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            out = classifier(f"{data['premise']} </s> </s> {data['hypothesis']}")
            
            results.append({
                "compound": data['compound'],
                "original_label": data['original_label'],
                "predicted_nli": out[0]['label'],
                "score": out[0]['score']
            })
            
           
            if len(results) >= 5: break 
            
    for res in results:
        print(f"Compound: {res['compound']} | Label: {res['original_label']}")
        print(f"Prediction: {res['predicted_nli']} ({res['score']:.4f})\n")

if __name__ == "__main__":
    run_inference("train_nli.jsonl")