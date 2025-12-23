#!/usr/bin/env bash

python indexing.py \
  --chunks ../../data/chunks/chunks.jsonl \
  --bm25-output ../../data/index/bm25.pkl \
  --faiss-index-output ../../data/index/faiss.index \
  --faiss-meta-output ../../data/index/faiss_meta.jsonl \
  --max-chunks 1000000
echo "== index files =="
ls -lh ../../data/index
