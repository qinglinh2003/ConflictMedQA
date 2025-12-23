#!/usr/bin/env python
import os
import re
import json
import argparse
from collections import defaultdict
import re
from tqdm import tqdm 


SECTION_HEADING_MAP = {
    "introduction": "INTRODUCTION",
    "background": "INTRODUCTION",
    "results": "RESULTS",
    "results and discussion": "RESULTS",
    "discussion": "DISCUSSION",
    "conclusion": "CONCLUSION",
    "conclusions": "CONCLUSION",
    "materials and methods": "METHODS",
    "methods": "METHODS",
}

# section_name -> (window_size, stride)
SECTION_WINDOW = {
    "ABSTRACT": (2, 1),
    "INTRODUCTION": (4, 2),
    "RESULTS": (3, 1),
    "DISCUSSION": (3, 2),
    "CONCLUSION": (3, 2),
    # FULL_TEXT is kept as a fallback
    "FULL_TEXT": (3, 2),
}


def split_into_sentences(text: str):
    """Very simple sentence splitter."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def parse_sections(path: str):
    """
    Parse sections from a PMC plain-text file with blocks:
    ==== Front / ==== Body / ==== Refs.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    sections = defaultdict(list)
    mode = None      # None | "front" | "body" | "refs"
    current_sec = None

    for raw in lines:
        line = raw.strip()
        low = line.lower()

        # Block markers
        if low.startswith("===="):
            if "front" in low:
                mode = "front"
                current_sec = "FRONT"
                continue
            if "body" in low:
                mode = "body"
                current_sec = None
                continue
            if "refs" in low:
                # Stop at references
                mode = "refs"
                break
            # Other ==== lines can be ignored
            continue

        if not line:
            continue

        if mode == "front":
            sections["FRONT"].append(line)
            continue

        if mode == "body":
            # Detect section headings like "Introduction", "Results", etc.
            if low in SECTION_HEADING_MAP:
                mapped = SECTION_HEADING_MAP[low]
                current_sec = mapped
                continue

            # If we are in the body but have not seen any heading yet,
            # treat early content as INTRODUCTION.
            if current_sec is None:
                current_sec = "INTRODUCTION"

            sections[current_sec].append(line)
            continue

        # If mode is None or "refs", we ignore content.

    out = {}

    # Treat the whole Front block as an ABSTRACT-like summary
    if "FRONT" in sections:
        out["ABSTRACT"] = " ".join(sections["FRONT"])

    # Copy the main sections if they exist
    for sec_name in ["INTRODUCTION", "RESULTS", "DISCUSSION", "CONCLUSION"]:
        if sec_name in sections:
            out[sec_name] = " ".join(sections[sec_name])

    # Fallback: if we somehow did not detect anything, keep the whole file
    if not out:
        full_text = " ".join(ln.strip() for ln in lines)
        out["FULL_TEXT"] = full_text

    return out


def build_chunks_for_section(pmcid: str, section_name: str, text: str):
    """Create sliding-window chunks for one section."""
    if section_name not in SECTION_WINDOW:
        return []

    window_size, stride = SECTION_WINDOW[section_name]
    sentences = split_into_sentences(text)
    n = len(sentences)
    chunks = []

    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + window_size, n)
        sent_slice = sentences[start:end]
        if not sent_slice:
            break

        chunk_text = " ".join(sent_slice)
        chunk = {
            "chunk_id": f"{pmcid}_{section_name}_{start:04d}_{end:04d}",
            "pmcid": pmcid,
            "section": section_name,
            "sent_start": start,
            "sent_end": end,  # half-open [start, end)
            "text": chunk_text,
        }
        chunks.append(chunk)

        if end == n:
            break
        start += stride

    return chunks


def iter_txt_paths(root: str):
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith(".txt"):
                yield os.path.join(dirpath, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        help="Root directory containing PMC*.txt files",
    )
    parser.add_argument(
        "--output",
        help="Output JSONL file for chunks",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Optional limit on number of txt files (for quick testing)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    txt_paths = list(iter_txt_paths(args.input_root))
    if args.max_files is not None:
        txt_paths = txt_paths[: args.max_files]

    print(f"Found {len(txt_paths)} txt files under {args.input_root}")

    num_chunks = 0
    with open(args.output, "w", encoding="utf-8") as out_f:
        for path in tqdm(txt_paths, desc="Building chunks", unit="file"):
            dir_name = os.path.basename(os.path.dirname(path))   # e.g. "PMC000xxxxxx"
            file_name = os.path.splitext(os.path.basename(path))[0]  # e.g. "PMC176545"
            dir_digits = re.findall(r"\d+", dir_name)[0]   # e.g. "000"
            file_digits = re.findall(r"\d+", file_name)[0] # e.g. "176545"
            prefix = dir_digits[-3:]
            suffix = file_digits[-6:]
            pmcid = f"PMC{prefix}{suffix}"   # e.g. "PMC000176545"

            sections = parse_sections(path)

            for sec_name, sec_text in sections.items():
                chunks = build_chunks_for_section(pmcid, sec_name, sec_text)
                for ch in chunks:
                    out_f.write(json.dumps(ch) + "\n")
                num_chunks += len(chunks)


    print(f"Done. Wrote {num_chunks} chunks to {args.output}")


if __name__ == "__main__":
    main()
