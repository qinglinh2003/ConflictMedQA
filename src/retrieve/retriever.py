#!/usr/bin/env python
"""
Three-stage Medical Retriever:

    BM25 (500) → Qwen3-Embedding-8B (100) → MedCPT-Cross-Encoder (10)
    
Stage 1 - BM25: Fast lexical recall
Stage 2 - Qwen 8B: Large model semantic filtering  
Stage 3 - MedCPT CE: Medical domain expert ranking
"""
import json
import os
import argparse
from typing import List, Literal
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from pyserini.search.lucene import LuceneSearcher


@dataclass
class RetrieverConfig:
    bm25_k: int = 500
    dense_k: int = 100
    top_k: int = 10
    max_length: int = 512


@dataclass
class RetrievalResult:
    chunk_id: str
    pmcid: str
    section: str
    score: float
    text: str
    bm25_rank: int = 0
    dense_rank: int = 0


class ThreeStageRetriever:
    """
    BM25 → Dense → Cross-encoder
    """
    
    # Models
    DENSE_MODEL = "Qwen/Qwen3-Embedding-8B"  # 8B, strong semantic
    CE_MODEL = "ncbi/MedCPT-Cross-Encoder"   # 110M, medical expert
    
    def __init__(
        self,
        index_dir: str,
        config: RetrieverConfig = None,
        device: str = "auto",
    ):
        self.config = config or RetrieverConfig()
        self.device = device
        
        # Load BM25 index
        print(f"Loading index from {index_dir} ...")
        self.searcher = LuceneSearcher(index_dir)
        print(f"  {self.searcher.num_docs:,} documents")
        
        # Lazy load models
        self.dense_encoder = None
        self.dense_tokenizer = None
        self.ce_model = None
        self.ce_tokenizer = None
    
    def _get_device(self):
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
    
    # ==================== Dense Bi-encoder (Qwen 8B) ====================
    
    def _load_dense(self):
        if self.dense_encoder is not None:
            return
        
        print(f"Loading Dense encoder: {self.DENSE_MODEL} ...")
        self.dense_tokenizer = AutoTokenizer.from_pretrained(self.DENSE_MODEL)
        self.dense_encoder = AutoModel.from_pretrained(
            self.DENSE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.dense_encoder.eval()
        print(f"  Dense encoder loaded (8B params)")
    
    def _encode_dense(self, texts: List[str]) -> np.ndarray:
        """Encode texts with Qwen embedding."""
        self._load_dense()
        
        all_embs = []
        batch_size = 8  # Smaller batch for 8B model
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.dense_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.dense_encoder.device)
                
                outputs = self.dense_encoder(**inputs)
                
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).bool()
                hidden = hidden.masked_fill(~mask, 0.0)
                emb = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                
                all_embs.append(emb.cpu().float().numpy())
        
        return np.vstack(all_embs)
    
    def _score_dense(self, query: str, documents: List[str]) -> np.ndarray:
        """Score documents with Qwen embedding similarity."""
        query_emb = self._encode_dense([query])[0]
        doc_embs = self._encode_dense(documents)
        return doc_embs @ query_emb
    
    # ==================== Cross-encoder (MedCPT) ====================
    
    def _load_ce(self):
        if self.ce_model is not None:
            return
        
        print(f"Loading Cross-encoder: {self.CE_MODEL} ...")
        
        self.ce_tokenizer = AutoTokenizer.from_pretrained(self.CE_MODEL)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(
            self.CE_MODEL,
            use_safetensors=True,  # Force safetensors to avoid torch.load security issue
        ).to(self._get_device())
        
        self.ce_model.eval()
        print(f"  Cross-encoder loaded (110M params, medical)")
    
    def _score_ce(self, query: str, documents: List[str]) -> np.ndarray:
        """Score with MedCPT cross-encoder."""
        self._load_ce()
        
        pairs = [[query, doc] for doc in documents]
        scores = []
        
        batch_size = 32  # MedCPT is small, can use larger batch
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.ce_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.ce_model.device)
                
                outputs = self.ce_model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().float().numpy()
                
                if batch_scores.ndim == 0:
                    scores.append(batch_scores.item())
                else:
                    scores.extend(batch_scores.tolist())
        
        return np.array(scores)
    
    # ==================== Search Pipeline ====================
    
    def search_bm25(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """Stage 1: BM25."""
        top_k = top_k or self.config.bm25_k
        hits = self.searcher.search(query, k=top_k)
        
        results = []
        for rank, hit in enumerate(hits):
            doc = self.searcher.doc(hit.docid)
            raw = json.loads(doc.raw())
            
            results.append(RetrievalResult(
                chunk_id=hit.docid,
                pmcid=raw.get("pmcid", ""),
                section=raw.get("section", ""),
                score=hit.score,
                text=raw.get("contents", ""),
                bm25_rank=rank + 1,
            ))
        
        return results
    
    def search(
        self,
        query: str,
        stages: Literal["bm25", "bm25+dense", "full"] = "full",
    ) -> List[RetrievalResult]:
        """
        Full retrieval pipeline.
        """
        # Stage 1: BM25
        candidates = self.search_bm25(query, top_k=self.config.bm25_k)
        
        if stages == "bm25" or not candidates:
            return candidates[:self.config.top_k]
        
        # Stage 2: Dense bi-encoder
        texts = [c.text for c in candidates]
        dense_scores = self._score_dense(query, texts)
        
        # Keep top dense_k
        dense_top_indices = np.argsort(dense_scores)[::-1][:self.config.dense_k]
        candidates_stage2 = []
        for new_rank, idx in enumerate(dense_top_indices):
            c = candidates[idx]
            candidates_stage2.append(RetrievalResult(
                chunk_id=c.chunk_id,
                pmcid=c.pmcid,
                section=c.section,
                score=float(dense_scores[idx]),
                text=c.text,
                bm25_rank=c.bm25_rank,
                dense_rank=new_rank + 1,
            ))
        
        if stages == "bm25+dense":
            return candidates_stage2[:self.config.top_k]
        
        # Stage 3: Cross-encoder
        texts_stage2 = [c.text for c in candidates_stage2]
        ce_scores = self._score_ce(query, texts_stage2)
        
        # Final ranking
        ce_top_indices = np.argsort(ce_scores)[::-1][:self.config.top_k]
        
        return [
            RetrievalResult(
                chunk_id=candidates_stage2[i].chunk_id,
                pmcid=candidates_stage2[i].pmcid,
                section=candidates_stage2[i].section,
                score=float(ce_scores[i]),
                text=candidates_stage2[i].text,
                bm25_rank=candidates_stage2[i].bm25_rank,
                dense_rank=candidates_stage2[i].dense_rank,
            )
            for i in ce_top_indices
        ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="../../data/pyserini/index_full")
    parser.add_argument("--query", type=str, help="Single query (interactive mode)")
    parser.add_argument("--input", type=str, help="Input CSV file with 'Prompt' column (batch mode)")
    parser.add_argument("--output", type=str, help="Output JSONL file (batch mode)")
    parser.add_argument("--bm25-k", type=int, default=500)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--stages", choices=["bm25", "bm25+dense", "full"], default="full")
    # Sharding support
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=None, help="Total number of shards")
    args = parser.parse_args()
    
    config = RetrieverConfig(
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
        top_k=args.top_k,
    )
    
    retriever = ThreeStageRetriever(args.index, config=config)
    
    import time
    
    # Batch mode: read from CSV, write to JSONL
    if args.input:
        import os
        import pandas as pd
        from tqdm import tqdm
        import logging
        
        if not args.output:
            args.output = args.input.replace(".csv", "_retrieved.jsonl")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        
        # Setup logging
        log_file = args.output.replace(".jsonl", ".log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )
        logger = logging.getLogger(__name__)
        
        shard_info = f" (shard {args.shard}/{args.num_shards})" if args.shard is not None else ""
        logger.info(f"Batch mode{shard_info}:")
        logger.info(f"  Input:  {args.input}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Log:    {log_file}")
        logger.info(f"  Pipeline: BM25({config.bm25_k}) → Qwen8B({config.dense_k}) → MedCPT-CE({config.top_k})")
        
        df = pd.read_csv(args.input)
        
        if "Prompt" not in df.columns:
            raise ValueError(f"CSV must have 'Prompt' column. Found: {list(df.columns)}")
        
        # Apply sharding if specified
        if args.shard is not None and args.num_shards is not None:
            total = len(df)
            shard_size = (total + args.num_shards - 1) // args.num_shards  # Ceiling division
            start_idx = args.shard * shard_size
            end_idx = min(start_idx + shard_size, total)
            df = df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Update output filename with shard number
            if args.output:
                base, ext = os.path.splitext(args.output)
                args.output = f"{base}_shard{args.shard}{ext}"
            
            logger.info(f"  Shard {args.shard}/{args.num_shards}: rows {start_idx}-{end_idx} ({len(df)} prompts)")
        
        logger.info(f"  Loaded {len(df)} prompts")
        
        start_time = time.time()
        
        # Write to local temp file first, then move (safer for HPC filesystems)
        import shutil
        
        temp_output = args.output + ".tmp"
        
        with open(temp_output, "w", encoding="utf-8") as f_out:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Retrieving"):
                query = row["Prompt"]
                
                # Retrieve
                results = retriever.search(query, stages=args.stages)
                
                # Build output record (handle NaN values)
                record = {}
                for k, v in row.items():
                    if pd.isna(v):
                        record[k] = None
                    else:
                        record[k] = v
                
                record["retrieved"] = [
                    {
                        "rank": i + 1,
                        "chunk_id": r.chunk_id,
                        "pmcid": r.pmcid,
                        "section": r.section,
                        "score": r.score,
                        "bm25_rank": r.bm25_rank,
                        "dense_rank": r.dense_rank,
                        "text": r.text,
                    }
                    for i, r in enumerate(results)
                ]
                
                # Write line
                line = json.dumps(record, ensure_ascii=False) + "\n"
                f_out.write(line)
                f_out.flush()
                os.fsync(f_out.fileno())  # Force write to disk
                
                # Log progress every 100 queries
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (idx + 1)
                    remaining = avg_time * (len(df) - idx - 1)
                    logger.info(f"  Progress: {idx+1}/{len(df)} | Avg: {avg_time:.2f}s/query | ETA: {remaining/60:.1f}min")
        
        # Move temp file to final location
        shutil.move(temp_output, args.output)
        
        total_time = time.time() - start_time
        logger.info(f"Done! Processed {len(df)} queries in {total_time/60:.1f} minutes")
        logger.info(f"Saved to {args.output}")
    
    # Interactive mode: single query
    elif args.query:
        print(f"\nQuery: {args.query}")
        print(f"Pipeline: BM25({config.bm25_k}) → Qwen8B({config.dense_k}) → MedCPT-CE({config.top_k})")
        print(f"{'='*70}")
        
        start = time.time()
        results = retriever.search(args.query, stages=args.stages)
        total_time = time.time() - start
        
        print(f"\n[{args.stages.upper()}] {total_time*1000:.1f}ms\n")
        
        for i, r in enumerate(results, 1):
            rank_info = f"BM25#{r.bm25_rank}"
            if r.dense_rank:
                rank_info += f" → Dense#{r.dense_rank}"
            print(f"  [{i}] score={r.score:.4f} ({rank_info})")
            print(f"      {r.chunk_id}")
            print(f"      {r.text[:70]}...")
            print()
    
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  # Single query:")
        print("  python retriever.py --query 'Does aspirin prevent cardiovascular events?'")
        print("  # Batch mode:")
        print("  python retriever.py --input prompts.csv --output results.jsonl")


if __name__ == "__main__":
    main()