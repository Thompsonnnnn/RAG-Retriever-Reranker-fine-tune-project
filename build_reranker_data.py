# build_reranker_data.py
import os, json, argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--qrels', type=str, required=True)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--output', type=str, default='reranker_train.jsonl')
parser.add_argument('--retriever_path', type=str, required=True)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--max_neg_per_query', type=int, default=5)
parser.add_argument('--max_total_samples', type=int, default=None)
args = parser.parse_args()

# Load corpus
corpus = {}
with open(args.corpus, 'r') as f:
    for line in f:
        obj = json.loads(line)
        corpus[str(obj['id'])] = obj['text']

# Load qrels
with open(args.qrels, 'r') as f:
    qrels = json.load(f)

# Load retriever
print("[Info] Loading retriever model...")
retriever = SentenceTransformer(args.retriever_path)
retriever.max_seq_length = 384
passage_ids = list(corpus.keys())
passage_texts = [f"passage: {corpus[pid]}" for pid in passage_ids]
passage_embeddings = retriever.encode(
    passage_texts, batch_size=64, convert_to_tensor=True, show_progress_bar=True
)

# Build training pairs
pairs = []
seen_pairs = set()
total_count = 0

with open(args.train, 'r') as f:
    for line in tqdm(f, desc="Building training pairs"):
        if args.max_total_samples and total_count >= args.max_total_samples:
            break

        obj = json.loads(line)
        qid = str(obj.get('qid') or obj.get('id'))
        query = obj.get('rewrite', '').strip()
        if not query or qid not in qrels:
            continue

        qrels_pos = set(str(pid) for pid, lbl in qrels[qid].items() if int(lbl) == 1)
        query_embed = retriever.encode(f"query: {query}", convert_to_tensor=True)
        scores = util.cos_sim(query_embed, passage_embeddings)[0]
        top_k_ids = scores.topk(k=args.top_k).indices.tolist()

        # Positive samples
        for pid in qrels_pos:
            if pid in corpus:
                key = (qid, pid)
                if key not in seen_pairs:
                    pairs.append({'query': query, 'passage': corpus[pid], 'label': 1})
                    seen_pairs.add(key)
                    total_count += 1
                    if args.max_total_samples and total_count >= args.max_total_samples:
                        break

        # Hard negatives from top-k
        neg_count = 0
        for idx in top_k_ids:
            if args.max_total_samples and total_count >= args.max_total_samples:
                break
            pid = passage_ids[idx]
            if pid not in qrels_pos:
                key = (qid, pid)
                if key not in seen_pairs:
                    pairs.append({'query': query, 'passage': corpus[pid], 'label': 0})
                    seen_pairs.add(key)
                    neg_count += 1
                    total_count += 1
                    if neg_count >= args.max_neg_per_query:
                        break

# Save
with open(args.output, 'w') as f:
    for item in pairs:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"[Done] Saved {len(pairs)} pairs to {args.output}")