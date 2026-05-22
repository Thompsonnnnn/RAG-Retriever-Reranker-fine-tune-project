from __future__ import annotations
import os, json, argparse, random, time, shutil, re
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses

QUERY_PREFIX   = "query: "
PASSAGE_PREFIX = "passage: "


# 讀取資料
def load_corpus(path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            pid = str(obj.get("id"))
            txt = obj.get("text", "")
            if pid and isinstance(txt, str):
                mp[pid] = txt
    if not mp:
        raise ValueError(f"corpus 為空或格式不符: {path}")
    return mp

def load_qrels(path: str) -> Dict[str, List[str]]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    q2pos: Dict[str, List[str]] = {}
    # 先嘗試整檔JSON
    try:
        root = json.loads(txt)
        if isinstance(root, dict):
            for qid, d in root.items():
                if isinstance(d, dict):
                    pos = [str(pid) for pid, lbl in d.items() if int(lbl) == 1]
                    if not pos and d:
                        pos = [str(pid) for pid in d.keys()]
                    if pos:
                        q2pos[str(qid)] = pos
            if q2pos:
                return q2pos
    except json.JSONDecodeError:
        pass
    # 再讀 JSONL
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            for qid, d in obj.items():
                if isinstance(d, dict):
                    pos = [str(pid) for pid, lbl in d.items() if int(lbl) == 1]
                    if not pos and d:
                        pos = [str(pid) for pid in d.keys()]
                    if pos:
                        q2pos[str(qid)] = pos
    if not q2pos:
        raise ValueError(f"qrels 為空或格式不符: {path}")
    return q2pos

def iter_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

# 僅用於過濾重複/等同（不改變真實文本）
def _normalize_for_compare(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def resolve_evidence_text(evi_item: Any, corpus_map: Dict[str, str]) -> str | None:
    if isinstance(evi_item, dict):
        if isinstance(evi_item.get("text"), str):
            return evi_item["text"]
        pid = evi_item.get("id")
        if pid is not None:
            return corpus_map.get(str(pid))
        return None
    if isinstance(evi_item, str):
        return corpus_map.get(evi_item, evi_item)
    return None

# 資料集
class TripletDataset(Dataset):
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = triples
    def __len__(self): return len(self.triples)
    def __getitem__(self, idx: int) -> InputExample:
        a, p, n = self.triples[idx]
        return InputExample(texts=[a, p, n])

class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx: int) -> InputExample:
        a, p = self.pairs[idx]
        return InputExample(texts=[a, p])

def build_pairs_and_triplets(
    train_path: str,
    qrels_map: Dict[str, List[str]],
    corpus_map: Dict[str, str],
    hard_neg_k: int = 8,
    seed: int = 1234,
):
    """產生：
       - pairs:給 MNR,用 (anchor=query, positive=pos_passage)
       - triples: 給 TripletLoss,用 (anchor, pos, neg),neg 來自 evidences 標 0 與隨機補
    """
    rng = random.Random(seed)
    all_pids = list(corpus_map.keys())
    pairs: List[Tuple[str, str]] = []
    triples: List[Tuple[str, str, str]] = []

    for obj in tqdm(iter_items(train_path), desc="Building pairs & triplets"):
        qid = str(obj.get("qid") or obj.get("id") or "")
        if not qid or qid not in qrels_map: 
            continue

        query = (obj.get("rewrite") or "").strip()
        if not query:
            continue

        pos_pid  = str(qrels_map[qid][0])
        pos_text = corpus_map.get(pos_pid)
        if not pos_text:
            continue

        # pair for MNR
        q_fmt = f"{QUERY_PREFIX}{query}"
        p_fmt = f"{PASSAGE_PREFIX}{pos_text}"
        pairs.append((q_fmt, p_fmt))

        # negatives for Triplet
        pos_norm = _normalize_for_compare(pos_text)
        neg_texts: List[str] = []
        evidences = obj.get("evidences") or []
        labels    = obj.get("retrieval_labels") or []

        if isinstance(evidences, list):
            if labels and isinstance(labels, list) and len(labels) == len(evidences):
                pairs_e = zip(evidences, labels)
            else:
                pairs_e = [(e, 0) for e in evidences]

            seen = set()
            for evi, lab in pairs_e:
                try:
                    if int(lab) != 0:
                        continue
                except Exception:
                    pass
                txt = resolve_evidence_text(evi, corpus_map)
                if not txt: 
                    continue
                key = _normalize_for_compare(txt)
                if key == pos_norm or key in seen:
                    continue
                seen.add(key)
                neg_texts.append(txt)

        need = max(1, hard_neg_k) - len(neg_texts)
        for _ in range(max(0, need)):
            while True:
                pid = rng.choice(all_pids)
                if pid != pos_pid:
                    break
            neg_texts.append(corpus_map[pid])

        neg_texts = neg_texts[: max(1, hard_neg_k)]
        for neg in neg_texts:
            n_fmt = f"{PASSAGE_PREFIX}{neg}"
            triples.append((q_fmt, p_fmt, n_fmt))

    if not pairs or not triples:
        raise RuntimeError("資料不足:pairs 或 triples 為空，請檢查資料/qrels")
    return pairs, triples


# 無梯度 Triplet loss（記錄用）
def compute_triplet_loss(
    model: SentenceTransformer,
    triples: List[Tuple[str, str, str]],
    margin: float = 0.2,
    batch_size: int = 64,
    sample_limit: int | None = 5000,
) -> float:
    if sample_limit is not None and len(triples) > sample_limit:
        triples = random.sample(triples, sample_limit)

    model.eval()
    device = model.device
    total_loss, count = 0.0, 0

    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    with torch.no_grad():
        for batch in chunk(triples, batch_size):
            a = [x[0] for x in batch]; p = [x[1] for x in batch]; n = [x[2] for x in batch]
            a_emb = model.encode(a, convert_to_tensor=True, device=device, normalize_embeddings=True)
            p_emb = model.encode(p, convert_to_tensor=True, device=device, normalize_embeddings=True)
            n_emb = model.encode(n, convert_to_tensor=True, device=device, normalize_embeddings=True)
            cos_ap = torch.sum(a_emb * p_emb, dim=1)
            cos_an = torch.sum(a_emb * n_emb, dim=1)
            loss = torch.clamp(cos_an - cos_ap + margin, min=0.0)
            total_loss += float(loss.sum()); count += int(loss.numel())
    return total_loss / max(1, count)


# 參數與主程式
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADL2025-HW3 retriever fine-tuning (MNR + Triplet, loss logging)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-small")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--max_seq_length", type=int, default=384)

    # MNR / Triplet 批量可分開設定
    p.add_argument("--batch_size_mnr", type=int, default=128)
    p.add_argument("--batch_size_triplet", type=int, default=32)

    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--hard_neg_k", type=int, default=8)
    p.add_argument("--eval_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_sample_train", type=int, default=5000)
    p.add_argument("--log_sample_val", type=int, default=5000)
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model: SentenceTransformer, out_dir: str, epoch: int):
    model.save(out_dir)  # 最新
    ckpt_dir = os.path.join(out_dir, f"checkpoint-ep{epoch:02d}")
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    model.save(ckpt_dir)

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    corpus_path = os.path.join(args.data_dir, "corpus.txt")
    qrels_path  = os.path.join(args.data_dir, "qrels.txt")
    train_path  = os.path.join(args.data_dir, "train.txt")

    print(f"[Info] Loading corpus: {corpus_path}")
    corpus_map = load_corpus(corpus_path)
    print(f"[Info] Loading qrels : {qrels_path}")
    qrels_map  = load_qrels(qrels_path)

    print(f"[Info] Building pairs & triplets from {train_path}")
    pairs, triples = build_pairs_and_triplets(
        train_path, qrels_map, corpus_map,
        hard_neg_k=args.hard_neg_k, seed=args.seed
    )

    # 切 val（只用來記錄 triplet loss）
    eval_size  = max(1, int(len(triples) * args.eval_ratio))
    train_size = max(1, len(triples) - eval_size)
    train_subset, val_subset = random_split(
        triples, [train_size, eval_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_triples = [triples[i] for i in train_subset.indices]
    val_triples   = [triples[i] for i in val_subset.indices]

    # DataLoader
    mnr_ds = PairDataset(pairs)
    trip_ds = TripletDataset(train_triples)
    mnr_loader = DataLoader(
        mnr_ds, batch_size=args.batch_size_mnr, shuffle=True,
        collate_fn=SentenceTransformer.smart_batching_collate
    )
    trip_loader = DataLoader(
        trip_ds, batch_size=args.batch_size_triplet, shuffle=True,
        collate_fn=SentenceTransformer.smart_batching_collate
    )

    print(f"[Info] Loading base model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    # Losses
    mnr_loss = losses.MultipleNegativesRankingLoss(model)
    triplet_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin
    )

    # 記錄 epoch 0 的 triplet loss
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.json")
    init_train = compute_triplet_loss(model, train_triples, args.margin, sample_limit=args.log_sample_train)
    init_val   = compute_triplet_loss(model, val_triples,   args.margin, sample_limit=args.log_sample_val)
    with open(loss_curve_path, "w", encoding="utf-8") as f:
        json.dump([{
            "epoch": 0, "train_triplet_loss": float(init_train),
            "val_triplet_loss": float(init_val), "time": int(time.time())
        }], f, ensure_ascii=False, indent=2)
    print(f"[Eval] epoch=0 train_loss={init_train:.6f} val_loss={init_val:.6f}")

    # 聯合訓練（每個 epoch 都跑 MNR 與 Triplet）
    steps_mnr = max(1, len(mnr_loader))
    warmup_steps = int(steps_mnr * args.warmup_ratio)  # 以 MNR 步數估暖身

    for ep in range(1, args.epochs + 1):
        print(f"\n[Train] ===== Epoch {ep}/{args.epochs} =====")
        model.fit(
            train_objectives=[(mnr_loader, mnr_loss), (trip_loader, triplet_loss)],
            epochs=1,
            warmup_steps=warmup_steps,
            scheduler="WarmupLinear",
            optimizer_params={"lr": args.lr},
            weight_decay=args.weight_decay,
            show_progress_bar=True,
            output_path=None,
            save_best_model=False,
            use_amp=True,
        )

        tr_loss = compute_triplet_loss(model, train_triples, args.margin, sample_limit=args.log_sample_train)
        va_loss = compute_triplet_loss(model, val_triples,   args.margin, sample_limit=args.log_sample_val)
        print(f"[Eval] epoch={ep} train_triplet_loss={tr_loss:.6f} val_triplet_loss={va_loss:.6f}")

        try:
            arr = json.load(open(loss_curve_path, "r", encoding="utf-8"))
        except Exception:
            arr = []
        arr.append({
            "epoch": ep,
            "train_triplet_loss": float(tr_loss),
            "val_triplet_loss": float(va_loss),
            "time": int(time.time())
        })
        with open(loss_curve_path, "w", encoding="utf-8") as f:
            json.dump(arr, f, ensure_ascii=False, indent=2)

        save_checkpoint(model, args.output_dir, ep)

    print(f"\n[Done] Saved retriever to: {args.output_dir}")
    

if __name__ == "__main__":
    main()


# Reference
# - https://medium.com/rahasak/optimizing-rag-supervised-embeddings-reranking-with-your-data-with-llamaindex-88344ff89da7
# - https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data#train-the-bi-encoder-for-retrieval
# - https://colab.research.google.com/github/argilla-io/argilla/blob/main/docs/_source/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.ipynb
# - ChatGPT