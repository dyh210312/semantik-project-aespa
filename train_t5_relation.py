import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm  # 导入进度条库
import os


# 1. 数据加载
def load_and_preprocess(file_path, label2id=None):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['w1', 'w2', 'label'])
    df['text'] = df['w1'] + " " + df['w2']
    if label2id is not None:
        df['label_id'] = df['label'].map(label2id)
    return df


# 初始化标签映射
train_raw = pd.read_csv('train.tsv', sep='\t', header=None, names=['w1', 'w2', 'label'])
unique_labels = sorted(train_raw['label'].unique().tolist())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)


class RelationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_tensors='pt')
        data = {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}
        if self.labels is not None:
            data['labels'] = torch.tensor(self.labels[item], dtype=torch.long)
        return data


# 3. 训练主程序
def main():
    # --- 参数配置 ---
    MAX_LEN = 64
    BATCH_SIZE = 16  # 4060 推荐值
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    MODEL_NAME = 'roberta-large'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 30}")
    print(f"运行设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 未检测到 GPU，当前正在使用 CPU 运行，速度会很慢！")
    print(f"{'=' * 30}\n")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    train_df = load_and_preprocess('train.tsv', label2id)

    train_ds = RelationDataset(train_df['text'].values, train_df['label_id'].values, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * EPOCHS)

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # 添加进度条控制台输出
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            # 在进度条右侧实时显示 Loss
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 完成，平均 Loss: {avg_loss:.4f}")

        # 每轮清理一次显存缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. 预测阶段添加进度条
    print("\n开始对测试集进行预测...")
    test_df = load_and_preprocess('test.tsv')
    test_ds = RelationDataset(test_df['text'].values, None, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", unit="batch"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend([id2label[p] for p in preds])

    test_df['predicted_label'] = all_preds
    test_df[['w1', 'w2', 'predicted_label']].to_csv('submission.tsv', sep='\t', index=False, header=False)
    print(f"\n{'=' * 30}")
    print("任务完成！预测结果已存入 submission.tsv")
    print(f"{'=' * 30}")


if __name__ == "__main__":
    main()