import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# 1. Vorhersageergebnisse laden
# Hinweis: Falls die erste Zeile Daten und kein Header ist, ändern Sie header=0 zu header=None
df_pred = pd.read_csv("submission.tsv", sep='\t', header=0)

# 2. Goldstandard laden
df_test = pd.read_csv("test_t5.csv")

# 3. Zentrale Alignment-Logik: Längen überprüfen
print(f"Anzahl der Testdaten (Original): {len(df_test)}")
print(f"Anzahl der Vorhersagen (Original): {len(df_pred)}")

# Für die Berechnung verwenden wir die Schnittmenge, um gleiche Länge sicherzustellen
min_len = min(len(df_test), len(df_pred))
y_true = df_test['target_text'].values[:min_len] # Annahme: Spaltenname ist target_text
y_pred = df_pred.iloc[:min_len, 2].values       # Die 3. Spalte wird als Vorhersage verwendet

# 4. Metriken berechnen
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"\n--- T5 Fine-tuning (H2) Versuchsergebnisse ---")
print(f"Genauigkeit (Accuracy): {acc:.4f}")
print(f"Macro-F1-Wert: {f1:.4f}")

# 5. Heatmap erzeugen
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

# 6. Speichern
plt.savefig("h2_final_heatmap.png")
print("\n✅ Heatmap wurde gespeichert als: h2_final_heatmap.png")
