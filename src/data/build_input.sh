#!/usr/bin/env bash

python build_input.py \
  --prompts ../../data/input/prompts.csv \
  --annotations ../../data/input/annotations.csv \
  --output ../../data/input/prompts_merged.csv

echo "== merged file =="
ls -lh ../../data/input/prompts_merged.csv
