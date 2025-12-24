#!/usr/bin/env python
"""
Build BM25 index using Pyserini (Lucene backend).

Advantages over rank_bm25:
- Disk-based index (not memory-bound)
- Handles billions of documents
- Fast indexing with multi-threading
- Battle-tested (used in MS MARCO, BEIR, etc.)

Install: pip install pyserini faiss-cpu

For 250M chunks:
- Time: ~3-6 hours
- Disk: ~20-50GB index
- RAM: ~8-16GB (manageable)
"""
import os
import json
import argparse
import subprocess
from tqdm import tqdm


def prepare_jsonl_for_pyserini(input_path: str, output_dir: str, max_chunks: int = None):
    """
    Convert chunks.jsonl to Pyserini's expected format.
    
    Pyserini expects: {"id": "...", "contents": "..."}
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "corpus.jsonl")
    
    print(f"Converting {input_path} to Pyserini format...")
    
    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in tqdm(fin, desc="Converting", unit=" docs"):
            if max_chunks and count >= max_chunks:
                break
            
            obj = json.loads(line)
            
            # Pyserini format
            doc = {
                "id": obj["chunk_id"],
                "contents": obj.get("text", ""),
                # Store metadata as additional fields
                "pmcid": obj.get("pmcid", ""),
                "section": obj.get("section", ""),
            }
            fout.write(json.dumps(doc) + "\n")
            count += 1
    
    print(f"Wrote {count:,} documents to {output_path}")
    return output_path


def build_index_with_pyserini(corpus_dir: str, index_dir: str, threads: int = 8):
    """
    Build Lucene index using Pyserini's indexer.
    """
    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(threads),
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Index built at {index_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../../data/chunks/chunks.jsonl")
    parser.add_argument("--corpus-dir", default="../../data/pyserini/corpus")
    parser.add_argument("--index-dir", default="../../data/pyserini/index")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-convert", action="store_true", help="Skip conversion if already done")
    args = parser.parse_args()
    
    if not args.skip_convert:
        prepare_jsonl_for_pyserini(args.input, args.corpus_dir, args.max_chunks)
    
    build_index_with_pyserini(args.corpus_dir, args.index_dir, args.threads)