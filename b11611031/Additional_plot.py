import matplotlib.pyplot as plt


versions = ["v1", "v2", "v3", "v4", "v5"]
recall = [0.2244, 0.6290, 0.3755, 0.5993, 0.8486]
mrr = [0.0440, 0.1314, 0.0903, 0.1199, 0.2604]
cos = [0.2073, 0.2080, 0.2047, 0.2122, 0.2420]

plt.figure(figsize=(8,5))
plt.plot(versions, recall, marker='o', label="Recall@10", linewidth=2)
plt.plot(versions, mrr, marker='s', label="MRR@10", linewidth=2)
plt.plot(versions, cos, marker='^', label="Bi-Encoder CosSim", linewidth=2)
plt.xlabel("Retriever Version", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.title("Performance Change Across Retriever Fine-tuning Versions", fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()