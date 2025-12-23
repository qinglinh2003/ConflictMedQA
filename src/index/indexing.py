#!/usr/bin/env python
import os
import re
import json
import argparse
from typing import List, Dict

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import pickle


QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase and keep only a-z0-9."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks(chunks_path: str, max_chunks: int = None) -> List[Dict]:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="Loading chunks", unit="line")):
            if max_chunks is not None and i >= max_chunks:
                break
            obj = json.loads(line)
            chunks.append(obj)
    return chunks



def build_bm25(chunks: List[Dict]):
    tokenized_corpus = []
    bm25_meta = []

    for ch in tqdm(chunks, desc="Preparing BM25 corpus", unit="chunk"):
        text = ch["text"]
        tokenized_corpus.append(simple_tokenize(text))
        bm25_meta.append({
            "chunk_id": ch["chunk_id"],
            "pmcid": ch["pmcid"],
            "section": ch.get("section", ""),
            "sent_start": ch.get("sent_start", None),
            "sent_end": ch.get("sent_end", None),
        })

    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, bm25_meta


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling with attention mask, followed by L2 normalization."""
    mask = attention_mask.unsqueeze(-1).bool()
    hidden = last_hidden_state.masked_fill(~mask, 0.0)
    summed = hidden.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    emb = summed / counts
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb


def encode_qwen(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int = 128,
    max_length: int = 512,
) -> np.ndarray:
    all_vecs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with Qwen3", unit="batch"):
        batch = texts[i:i + batch_size]
        with torch.no_grad():
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(model.device)
            out = model(**enc)
            last_hidden_state = out.last_hidden_state
            emb = mean_pooling(last_hidden_state, enc["attention_mask"])
            all_vecs.append(emb.cpu().numpy().astype("float32"))

    return np.vstack(all_vecs)


def build_faiss(
    chunks: List[Dict],
    index_path: str,
    meta_path: str,
    max_chunks: int = None,
    batch_size: int = 16,
):
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
    model = AutoModel.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    texts = []
    metas = []
    for i, ch in enumerate(chunks):
        if max_chunks is not None and i >= max_chunks:
            break
        texts.append(ch["text"])
        metas.append({
            "chunk_id": ch["chunk_id"],
            "pmcid": ch["pmcid"],
            "section": ch.get("section", ""),
            "sent_start": ch.get("sent_start", None),
            "sent_end": ch.get("sent_end", None),
        })

    emb = encode_qwen(texts, tokenizer, model, batch_size=batch_size)
    dim = emb.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        default="../../data/chunks/chunks.jsonl",
        help="Path to chunks.jsonl",
    )
    parser.add_argument(
        "--bm25-output",
        default="../../data/index/bm25.pkl",
        help="Output path for BM25 index pickle",
    )
    parser.add_argument(
        "--faiss-index-output",
        default="../../data/index/faiss.index",
        help="Output path for FAISS index",
    )
    parser.add_argument(
        "--faiss-meta-output",
        default="../../data/index/faiss_meta.jsonl",
        help="Output path for FAISS meta JSONL",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional limit on number of chunks for building indexes (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for Qwen3 encoding",
    )
    args = parser.parse_args()

    index_dir = os.path.dirname(os.path.abspath(args.bm25_output))
    os.makedirs(index_dir, exist_ok=True)

    print(f"Loading chunks from {args.chunks} ...")
    chunks = load_chunks(args.chunks, max_chunks=args.max_chunks)
    print(f"Loaded {len(chunks)} chunks")

    # Build BM25
    print("Building BM25 index ...")
    bm25, bm25_meta = build_bm25(chunks)
    with open(args.bm25_output, "wb") as f:
        pickle.dump(
            {
                "bm25": bm25,
                "chunks": bm25_meta,
                "tokenizer": "simple_tokenize",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved BM25 index to {args.bm25_output}")

    # Build FAISS (Qwen3 embedding)
    print(f"Building FAISS index with {QWEN_MODEL_NAME} embeddings ...")
    build_faiss(
        chunks,
        index_path=args.faiss_index_output,
        meta_path=args.faiss_meta_output,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
    )
    print(f"Saved FAISS index to {args.faiss_index_output}")
    print(f"Saved FAISS meta to {args.faiss_meta_output}")


if __name__ == "__main__":
    main()
