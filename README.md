# RAG Retriever and Reranker Fine-tuning Project

## Project Overview

This project fine-tunes a dual-encoder retriever and cross-encoder reranker for improving Retrieval-Augmented Generation (RAG) performance. The model is trained using contrastive learning techniques (MNR + Triplet Loss) for dense passage retrieval and ranking.

## Tech Stack

- **Retriever**: Multilingual E5 (intfloat/multilingual-e5-small)
- **Reranker**: Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-12-v2)
- **Framework**: PyTorch, Hugging Face Transformers, Sentence Transformers
- **Training Methods**: MNR Loss, Triplet Loss, MSE Loss

## Key Features

- **Retriever Training**: Dense passage retrieval with MNR and Triplet loss
- **Reranker Training**: Cross-encoder for relevance scoring
- **Batch Processing**: Efficient inference pipeline with batch support
- **Embeddings**: FAISS vector database for similarity search

## Files

- `train_retriever.py` - Retriever model training script
- `train_reranker.py` - Reranker model training script
- `build_reranker_data.py` - Data preprocessing for reranker training
- `save_embeddings.py` - Generate and save embeddings to FAISS
- `inference_batch.py` - Batch inference pipeline
- `utils.py` - Helper functions

## Installation

```bash
pip install -r requirements.txt
```

## References

- https://medium.com/rahasak/optimizing-rag-supervised-embeddings-reranking-with-your-data-with-llamaindex-88344ff89da7
- https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data
- Sentence Transformers: Siamese BERT networks for semantic text similarity