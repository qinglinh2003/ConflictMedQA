#!/usr/bin/env bash

python build_placebo.py \
  --input ../../data/input/prompts_merged.csv \
  --output-all ../../data/input/prompts_placebo_all.csv \
  --output-dir ../../data/input/placebo_subsets

echo "== placebo outputs =="
ls -lh ../../data/input/prompts_placebo_all.csv
ls -lh ../../data/input/placebo_subsets
