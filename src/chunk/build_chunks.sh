#!/usr/bin/env bash

python chunking.py \
  --input-root ../../data/corpus/oa_texts \
  --output ../../data/chunks/chunks.jsonl \

echo "== line count =="
wc -l ../../data/chunks/chunks.jsonl

echo "== head =="
head -n 5 ../../data/chunks/chunks.jsonl
