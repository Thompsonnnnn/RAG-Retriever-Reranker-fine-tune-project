mkdir -p models


gdown --fuzzy https://drive.google.com/file/d/1HVmfxST-R1yqelbhd8aY5XATv-mUyDOC/view?usp=sharing -O retriever.zip
unzip -q retriever.zip -d models/retriever


gdown --fuzzy https://drive.google.com/file/d/1xdySL2qs7CTBoDiqcCWC5RZ_nI7DcWD8/view?usp=sharing -O reranker.zip
unzip -q reranker.zip -d models/reranker
