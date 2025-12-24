nohup python src/index/build_pyserini.py \
  --input data/chunks/chunks.jsonl \
  --corpus-dir data/pyserini/corpus_full \
  --index-dir data/pyserini/index_full \
  --threads 16 \
  > logs/pyserini_index.log 2>&1 &