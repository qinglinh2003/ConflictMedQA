#!/usr/bin/env python
import argparse
import os
import pandas as pd
import numpy as np


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(str(candidates))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="../../data/input/prompts_merged.csv",
    )
    parser.add_argument(
        "--output-all",
        default="../../data/input/prompts_placebo_all.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="../../data/input/placebo_subsets",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    comp_col = find_col(df, ["Comparator", "Comparator_prompt", "Comparator_x"])
    comp_norm = df[comp_col].astype(str).str.strip().str.lower()
    mask_placebo = comp_norm == "placebo"
    df_placebo = df[mask_placebo].copy()

    os.makedirs(os.path.dirname(args.output_all), exist_ok=True)
    df_placebo.to_csv(args.output_all, index=False)
    print(f"Total placebo rows: {len(df_placebo)} -> {args.output_all}")

    os.makedirs(args.output_dir, exist_ok=True)
    parts = np.array_split(df_placebo, 4)
    for i, part in enumerate(parts, start=1):
        out_path = os.path.join(args.output_dir, f"placebo_part{i}.csv")
        part.to_csv(out_path, index=False)
        print(f"part{i}: {len(part)} rows -> {out_path}")


if __name__ == "__main__":
    main()
