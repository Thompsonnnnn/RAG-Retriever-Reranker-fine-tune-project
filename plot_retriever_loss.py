import matplotlib.pyplot as plt

# 紀錄
data = [
    {"epoch": 0, "train_triplet_loss": 0.14307903366088867},
    {"epoch": 1, "train_triplet_loss": 0.009067894965410233},
    {"epoch": 2, "train_triplet_loss": 0.006746531146764755},
    {"epoch": 3, "train_triplet_loss": 0.005871293014287949},
    {"epoch": 4, "train_triplet_loss": 0.005313361757993698},
    {"epoch": 5, "train_triplet_loss": 0.004985526600480079},
    {"epoch": 6, "train_triplet_loss": 0.004934150767326355}
]


epochs = [d["epoch"] for d in data]
train_loss = [d["train_triplet_loss"] for d in data]


plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o', color='blue', label='Train Triplet Loss')
plt.title("Retriever Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Train Triplet Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()