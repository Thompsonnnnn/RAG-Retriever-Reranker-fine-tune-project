使用 train_retriever.py 和 train_reranker.py 來訓練
使用 build_reranker_data.py 來處理要訓練的資料

以下為訓練的指令
CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/Thompson/ADL_HW3/train_retriever.py   --data_dir /home/ubuntu/Thompson/ADL_HW3/HW3_For_student/data   --output_dir /home/ubuntu/Thompson/ADL_HW3/retriever_v5_mnr_triplet   --model_name intfloat/multilingual-e5-small   --epochs 6   --max_seq_length 384   --batch_size_mnr 128   --batch_size_triplet 32   --lr 3e-5   --warmup_ratio 0.1   --weight_decay 0.01   --margin 0.2   --hard_neg_k 8   --eval_ratio 0.1   --log_sample_train 5000   --log_sample_val 5000   --seed 42

CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/Thompson/ADL_HW3/train_reranker.py \
--train_path /home/ubuntu/Thompson/ADL_HW3/reranker_train_lite.jsonl \
--output_dir /home/ubuntu/Thompson/ADL_HW3/reranker_best_final \
--model_name cross-encoder/ms-marco-MiniLM-L-12-v2 \
--epochs 8 \
--batch_size 48 \
--lr 5e-6 \
--eval_ratio 0.05 \
--warmup_ratio 0.1 \
--max_seq_length 512 \
--seed 42


Reference
- https://medium.com/rahasak/optimizing-rag-supervised-embeddings-reranking-with-your-data-with-llamaindex-88344ff89da7
- https://huggingface.co/blog/sdiazlor/fine-tune-modernbert-for-rag-with-synthetic-data#train-the-bi-encoder-for-retrieval
- https://colab.research.google.com/github/argilla-io/argilla/blob/main/docs/_source/tutorials_and_integrations/tutorials/feedback/fine-tuning-sentencesimilarity-rag.ipynb
- ChatGPT