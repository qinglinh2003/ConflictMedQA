#!/usr/bin/env python
import argparse
import pandas as pd
import os


def to_bool(series):
    return series.astype(str).str.lower().map({"true": True, "false": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        default="../data/prompts.csv",
    )
    parser.add_argument(
        "--annotations",
        default="../data/annotations.csv",
    )
    parser.add_argument(
        "--output",
        default="../data/prompts_annotations_merged_clean.csv",
    )
    args = parser.parse_args()

    print(f"Loading prompts from {args.prompts}")
    df_prompts = pd.read_csv(args.prompts)

    print(f"Loading annotations from {args.annotations}")
    df_ann = pd.read_csv(args.annotations)

    if "Valid Label" in df_ann.columns:
        df_ann["Valid Label"] = to_bool(df_ann["Valid Label"])
        mask_valid_label = df_ann["Valid Label"]
    else:
        mask_valid_label = True

    if "Valid Reasoning" in df_ann.columns:
        df_ann["Valid Reasoning"] = to_bool(df_ann["Valid Reasoning"])
        mask_valid_reason = df_ann["Valid Reasoning"]
    else:
        mask_valid_reason = True

    df_ann_valid = df_ann[mask_valid_label & mask_valid_reason].copy()
    print(f"Annotations: {len(df_ann)} rows -> {len(df_ann_valid)} with valid label+reasoning")

    def all_labels_agree(group):
        return group["Label"].nunique() == 1

    df_ann_agree = (
        df_ann_valid
        .groupby("PromptID")
        .filter(all_labels_agree)
        .copy()
    )
    print(f"After enforcing agreement: {df_ann_agree['PromptID'].nunique()} prompts remain")

    df_ann_one = (
        df_ann_agree
        .sort_values(["PromptID", "UserID"])
        .groupby("PromptID", as_index=False)
        .first()
    )

    print(f"Collapsed to one annotation per PromptID: {len(df_ann_one)} rows")

    df_merged = df_prompts.merge(df_ann_one, on="PromptID", how="inner", suffixes=("_prompt", "_annot"))

    print(f"Merged prompts: {len(df_prompts)} -> {len(df_merged)} with agreed annotations")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_merged.to_csv(args.output, index=False)
    print(f"Saved merged CSV to {args.output}")


if __name__ == "__main__":
    main()
