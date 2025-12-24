#!/usr/bin/env python
"""
Two-stage retriever using Pyserini (BM25) + Dense rerank.

Usage:
    # BM25 only (no GPU)
    python pyserini_retriever.py --query "aspirin heart disease" --no-rerank
    
    # BM25 + Dense rerank
    python pyserini_retriever.py --query "aspirin heart disease"
"""
import json
import argparse
from typing import List
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Pyserini
from pyserini.search.lucene import LuceneSearcher


QWEN_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"


@dataclass
class RetrievalResult:
    chunk_id: str
    pmcid: str
    section: str
    score: float
    text: str


class PyseriniRetriever:
    """
    Production retriever using Pyserini BM25 + optional dense rerank.
    
    Scales to billions of documents.
    """
    
    def __init__(self, index_dir: str, device: str = "auto"):
        print(f"Loading Pyserini index from {index_dir} ...")
        self.searcher = LuceneSearcher(index_dir)
        print(f"  Index loaded: {self.searcher.num_docs:,} documents")
        
        # Lazy load dense model
        self.tokenizer = None
        self.model = None
        self.device = device
    
    def _ensure_model_loaded(self):
        if self.model is not None:
            return
        
        print(f"Loading {QWEN_MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=self.device if self.device != "auto" else "auto",
        )
        self.model.eval()
    
    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        self._ensure_model_loaded()
        
        with torch.no_grad():
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            out = self.model(**enc)
            
            mask = enc["attention_mask"].unsqueeze(-1).bool()
            hidden = out.last_hidden_state.masked_fill(~mask, 0.0)
            emb = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            
            return emb.cpu().numpy().astype("float32")
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """BM25 search only (fast, no GPU)."""
        hits = self.searcher.search(query, k=top_k)
        
        results = []
        for hit in hits:
            # Get stored document
            doc = self.searcher.doc(hit.docid)
            raw = json.loads(doc.raw())
            
            results.append(RetrievalResult(
                chunk_id=hit.docid,
                pmcid=raw.get("pmcid", ""),
                section=raw.get("section", ""),
                score=hit.score,
                text=raw.get("contents", ""),
            ))
        
        return results
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        recall_k: int = 100,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        """Two-stage: BM25 recall + dense rerank."""
        # Stage 1: BM25
        candidates = self.search_bm25(query, top_k=recall_k)
        
        if not rerank or not candidates:
            return candidates[:top_k]
        
        # Stage 2: Dense rerank
        query_emb = self._encode_batch([query])[0]
        candidate_texts = [c.text for c in candidates]
        candidate_embs = self._encode_batch(candidate_texts)
        
        similarities = candidate_embs @ query_emb
        reranked_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            RetrievalResult(
                chunk_id=candidates[i].chunk_id,
                pmcid=candidates[i].pmcid,
                section=candidates[i].section,
                score=float(similarities[i]),
                text=candidates[i].text,
            )
            for i in reranked_indices
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="../../data/pyserini/index")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--recall-k", type=int, default=100)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    
    retriever = PyseriniRetriever(args.index)
    
    import time
    
    # BM25 only
    start = time.time()
    bm25_results = retriever.search_bm25(args.query, top_k=args.top_k)
    bm25_time = time.time() - start
    
    print(f"\nQuery: {args.query}")
    print(f"{'='*60}")
    print(f"\n[BM25] {bm25_time*1000:.1f}ms\n")
    
    for i, r in enumerate(bm25_results, 1):
        print(f"  [{i}] {r.score:.4f} | {r.chunk_id}")
        print(f"      {r.text[:100]}...")
    
    if not args.no_rerank:
        start = time.time()
        reranked = retriever.search(args.query, top_k=args.top_k, recall_k=args.recall_k)
        rerank_time = time.time() - start
        
        print(f"\n[BM25 + Rerank] {rerank_time*1000:.1f}ms\n")
        for i, r in enumerate(reranked, 1):
            print(f"  [{i}] {r.score:.4f} | {r.chunk_id}")
            print(f"      {r.text[:100]}...")


if __name__ == "__main__":
    main()