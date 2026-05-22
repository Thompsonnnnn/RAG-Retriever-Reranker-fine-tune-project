import argparse, json, os, random, time
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sentence_transformers import CrossEncoder, InputExample


class RerankerDataset(Dataset):
    def __init__(self, path):
        self.samples: List[InputExample] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                q, p, label = obj['query'], obj['passage'], obj['label']
                self.samples.append(InputExample(texts=[q, p], label=float(label)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='cross-encoder/ms-marco-MiniLM-L-12-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=7e-6)
    parser.add_argument('--max_seq_length', type=int, default=512)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--eval_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: CrossEncoder, dataset, batch_size: int = 32):
    """Use CrossEncoder.predict for MSE validation"""
    model.eval()
    if not isinstance(dataset, list):
        dataset = list(dataset)

    # Extract [query, passage] pairs
    text_pairs = [ex.texts for ex in dataset]

    # Predict scores
    preds = model.predict(text_pairs, batch_size=batch_size, show_progress_bar=True)

    # Compute MSE
    labels = [ex.label for ex in dataset]
    mse = float(torch.mean((torch.tensor(preds) - torch.tensor(labels)) ** 2))
    return mse


def save_checkpoint(model: CrossEncoder, output_dir: str, epoch: int):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-ep{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(ckpt_dir)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Info] Loading training data from {args.train_path}")
    dataset = RerankerDataset(args.train_path)
    val_size = int(len(dataset) * args.eval_ratio)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)

    print(f"[Info] Model: {args.model_name}")
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_seq_length)

    warmup_steps = int(len(train_loader) * args.epochs * args.warmup_ratio)
    print(f"[Info] Warmup steps: {warmup_steps}")

    best_mse = float('inf')
    history = []

    for ep in range(1, args.epochs + 1):
        print(f"\n[Train] ===== Epoch {ep}/{args.epochs} =====")
        model.fit(
            train_dataloader=train_loader,
            evaluator=None,
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': args.lr},
            weight_decay=args.weight_decay,
            output_path=None,
            use_amp=True
        )

        # 評估 Validation MSE
        val_examples = [val_ds[i] for i in range(len(val_ds))]
        val_mse = evaluate(model, val_examples, batch_size=args.batch_size)
        print(f"[Eval] Epoch {ep} val_MSE = {val_mse:.6f}")
        history.append({'epoch': ep, 'val_mse': val_mse, 'time': int(time.time())})

        # 儲存 checkpoint
        save_checkpoint(model, args.output_dir, ep)

        # 儲存最佳模型
        if val_mse < best_mse:
            best_mse = val_mse
            best_dir = os.path.join(args.output_dir, "best_model")
            model.save(best_dir)
            print(f"[Info]  New best model saved to {best_dir}")

    with open(os.path.join(args.output_dir, "val_mse_curve.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] All checkpoints in: {args.output_dir}")
    print(f"[Done] Best model saved to: {best_dir}")
    print(f"[Summary] Best val MSE = {best_mse:.6f}")


if __name__ == '__main__':
    main()

# Reference
# - https://medium.com/rahasak/optimizing-rag-supervised-embeddings-reranking-with-your-data-with-llamaindex-88344ff89da7
# - https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data#train-the-bi-encoder-for-retrieval
# - https://colab.research.google.com/github/argilla-io/argilla/blob/main/docs/_source/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.ipynb
# - ChatGPT