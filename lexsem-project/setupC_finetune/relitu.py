import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# 1. 加载预测结果
# 检查：如果第一行是数据而不是表头，请把 header=0 改成 header=None
df_pred = pd.read_csv("submission.tsv", sep='\t', header=0)

# 2. 加载标准答案
df_test = pd.read_csv("test_t5.csv")

# 3. 核心对齐逻辑：打印长度检查
print(f"原始测试集行数: {len(df_test)}")
print(f"原始预测集行数: {len(df_pred)}")

# 为了计算，我们取两者的交集，确保长度一致
min_len = min(len(df_test), len(df_pred))
y_true = df_test['target_text'].values[:min_len] # 假设列名是 target_text
y_pred = df_pred.iloc[:min_len, 2].values       # 强制取第3列作为预测结果

# 4. 计算指标
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"\n--- T5 Fine-tuning (H2) 实验结果 ---")
print(f"准确率 (Accuracy): {acc:.4f}")
print(f"Macro-F1 分数: {f1:.4f}")

# 5. 生成热力图
labels = sorted(list(set(y_true) | set(y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("H2: T5 Fine-tuning Confusion Matrix")
plt.xlabel("Predicted Relation")
plt.ylabel("Gold Relation")
plt.xticks(rotation=45)
plt.tight_layout()

# 6. 保存
plt.savefig("h2_final_heatmap.png")
print("\n✅ 热力图已保存为: h2_final_heatmap.png")