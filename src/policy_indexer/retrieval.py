from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np


class HybridRetriever:
    """Hybrid retrieval layer for policy evidence.

    Combines dense semantic retrieval and sparse BM25 lexical retrieval,
    then applies a lightweight reranking step before returning policy chunks.
    """

    def __init__(self, index_data: Dict[str, Any]):
        self.chunks = index_data["chunks"]
        self.embeddings = index_data["dense_embeddings"]
        self.index = index_data["dense_index"]
        self.bm25 = index_data["bm25_index"]

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a user query into a vector embedding for semantic search."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vector, dtype="float32")

    def _dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve semantically closest policy chunks using dense embeddings."""
        if len(self.embeddings) == 0:
            return []

        query_vector = self._encode_query(query)
        scores, indices = self.index.search(query_vector, min(top_k, len(self.embeddings)))

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "heading": chunk["heading"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "dense_score": float(score),
                    "sparse_score": 0.0,
                    "fused_score": float(score),
                    "rerank_score": float(score),
                }
            )
        return results

    def _sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve exact keyword matches using BM25 lexical scoring."""
        if not self.bm25:
            return []

        query_tokens = re.findall(r"\w+", query.lower())
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[: min(top_k, len(scores))]

        results: List[Dict[str, Any]] = []
        for idx in ranked:
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "heading": chunk["heading"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "dense_score": 0.0,
                    "sparse_score": float(scores[idx]),
                    "fused_score": float(scores[idx]),
                    "rerank_score": float(scores[idx]),
                }
            )
        return results

    def _fuse_scores(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge dense and sparse candidates into a single ranked result set."""
        combined: Dict[str, Dict[str, Any]] = {}

        for result in dense_results:
            combined[result["chunk_id"]] = result.copy()

        for result in sparse_results:
            if result["chunk_id"] in combined:
                combined[result["chunk_id"]]["sparse_score"] = result["sparse_score"]
                combined[result["chunk_id"]]["dense_score"] = combined[result["chunk_id"]].get("dense_score", 0.0)
                combined[result["chunk_id"]]["fused_score"] = (
                    combined[result["chunk_id"]].get("dense_score", 0.0) + result["sparse_score"]
                )
            else:
                combined[result["chunk_id"]] = result.copy()
                combined[result["chunk_id"]]["fused_score"] = result["sparse_score"]

        ordered = sorted(
            combined.values(),
            key=lambda item: float(item["fused_score"]),
            reverse=True,
        )
        return ordered

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Apply a lightweight rerank pass using query-token overlap as a relevance boost.

        This keeps the implementation lightweight and deterministic while making the
        final evidence ordering more useful for policy-grounded reasoning.
        """
        query_tokens = set(re.findall(r"\w+", query.lower()))
        reranked: List[Dict[str, Any]] = []

        for candidate in candidates:
            text_tokens = set(re.findall(r"\w+", candidate["text"].lower()))
            overlap = len(query_tokens.intersection(text_tokens))
            candidate["rerank_score"] = float(candidate.get("fused_score", 0.0)) + (overlap * 0.25)
            reranked.append(candidate)

        reranked.sort(key=lambda item: float(item["rerank_score"]), reverse=True)
        return reranked[:top_k]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return hybrid retrieval results with dense, sparse, fused, and reranked scores."""
        dense_results = self._dense_search(query, top_k=top_k)
        sparse_results = self._sparse_search(query, top_k=top_k)
        fused = self._fuse_scores(dense_results, sparse_results)
        reranked = self.rerank(query, fused, top_k=top_k)
        return reranked
