import matplotlib.pyplot as plt

# Reranker training loss data
data = [
    {"epoch": 1, "val_mse": 11.858075141906738},
    {"epoch": 2, "val_mse": 9.690593719482422},
    {"epoch": 3, "val_mse": 9.732178688049316},
    {"epoch": 4, "val_mse": 10.769713401794434},
    {"epoch": 5, "val_mse": 12.423750877380371},
    {"epoch": 6, "val_mse": 14.84790325164795},
    {"epoch": 7, "val_mse": 18.259571075439453},
    {"epoch": 8, "val_mse": 23.420108795166016},
]

# Extract data
epochs = [d["epoch"] for d in data]
loss = [d["val_mse"] for d in data]

# Plot
plt.figure(figsize=(7,5))
plt.plot(epochs, loss, marker='o', color='royalblue', linewidth=2)
plt.title("Reranker Training Loss Curve", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()