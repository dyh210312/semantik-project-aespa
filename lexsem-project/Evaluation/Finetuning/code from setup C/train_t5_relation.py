import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm  # 导入进度条库
import os

# ==========================================
# 1. 数据清洗与加载函数
# ==========================================
def load_and_preprocess(file_path, label2id=None):
    # 1. 正常读取逗号分隔的 CSV
    df = pd.read_csv(file_path)
    
    # 2. 数据清洗：去掉 "compound: " 前缀，并按空格拆分出 w1 和 w2
    df['clean_text'] = df['input_text'].str.replace('compound: ', '')
    df[['w1', 'w2']] = df['clean_text'].str.split(' ', n=1, expand=True)
    df['label'] = df['target_text']
    
    # 3. 删掉有缺失值的无效行
    df = df[['w1', 'w2', 'label']].dropna()
    
    # 4. 拼接文本给模型输入
    texts = (df['w1'] + " " + df['w2']).tolist()
    labels = df['label'].tolist()
    
    if label2id is not None:
        # 将文字标签转换为数字 ID
        labels = [label2id[l] for l in labels]
        
    return texts, labels


# ==========================================
# 2. 全局标签字典提取 (依赖训练集)
# ==========================================
# 读取带有表头的真实 CSV 以获取所有可能的标签种类
train_raw = pd.read_csv('train_t5.csv')
# 标签在 target_text 列里，去重并排序
unique_labels = sorted(train_raw['target_text'].dropna().unique().tolist())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)


# ==========================================
# 3. PyTorch Dataset 定义
# ==========================================
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
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        data = {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            data['labels'] = torch.tensor(self.labels[item], dtype=torch.long)
            
        return data


# ==========================================
# 4. 主训练与预测流程
# ==========================================
def main():
    # --- 参数配置 ---
    MAX_LEN = 64
    BATCH_SIZE = 16  # 显存够的话可以使用 16，如果 OOM 可以改成 8
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

    # --- 1. 准备模型与数据 ---
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    print("正在加载训练集数据...")
    train_texts, train_labels = load_and_preprocess('train_t5.csv', label2id)
    train_ds = RelationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS
    )

    # --- 2. 训练循环 ---
    print("\n🚀 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # 进度条控制台输出
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

    # --- 3. 预测阶段 ---
    print("\n🎯 开始对测试集进行预测...")
    # 假设你的测试集文件名为 test_t5.csv，如果不是，请在这里修改
    test_texts, test_labels = load_and_preprocess('test_t5.csv', label2id)
    test_ds = RelationDataset(test_texts, test_labels, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting", unit="batch"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            # 直接在这里将数字转换回文字标签
            all_preds.extend([id2label[p] for p in preds])

    # --- 4. 保存模型与结果 ---
    print("\n💾 正在保存模型到本地...")
    save_dir = './my_roberta_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ 模型保存成功！路径: {save_dir}")

    print("📊 正在生成预测文件...")
    # 直接使用 test_texts 和 all_preds 创建 DataFrame
    submission_df = pd.DataFrame({
        'input_text': test_texts,
        'predicted_label': all_preds
    })
    
    submission_df.to_csv('submission.csv', index=False)
    
    print(f"\n{'=' * 30}")
    print("🎉 任务圆满完成！预测结果已存入 submission.csv")
    print(f"{'=' * 30}")

if __name__ == "__main__":
    main()
