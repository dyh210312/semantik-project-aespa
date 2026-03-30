import matplotlib.pyplot as plt

settings = ["Full", "Mask_modifier", "Mask_head"]
accuracies = [0.7741, 0.7304, 0.6303]

plt.figure(figsize=(8, 6))
bars = plt.bar(settings, accuracies)

plt.ylim(0.0, 0.85)
plt.ylabel("Accuracy")
plt.xlabel("Condition")
plt.title("H3 Results: Accuracy under Degenerate Input Conditions")

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        acc + 0.01,
        f"{acc:.2%}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("h3_accuracy_barplot_final.png", dpi=300)
plt.show()

print("saved: h3_accuracy_barplot_final.png")