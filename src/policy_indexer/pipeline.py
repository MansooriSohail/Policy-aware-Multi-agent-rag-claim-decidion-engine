from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import faiss
import fitz
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class PolicyChunk:
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section: str
    heading: str | None = None
    source: str = "policy.pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_headings_from_page(page_text: str) -> List[Tuple[str, str]]:
    lines = [line.strip() for line in page_text.splitlines()]
    normalized: List[Tuple[str, str]] = []
    for line in lines:
        if not line:
            continue
        if len(line) <= 80 and line.isupper() or re.fullmatch(r"[A-Z0-9\- /&()]+", line) and len(line) <= 90:
            normalized.append((line, "heading"))
    return normalized


def _build_policy_chunks(pdf_path: str | Path) -> List[PolicyChunk]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    chunks: List[PolicyChunk] = []

    curr_section = "Untitled Section"
    curr_heading = "Untitled Heading"
    current_parts: List[str] = []
    page_start = 1
    page_end = 1

    for page_number in range(doc.page_count):
        page = doc[page_number]
        page_text = page.get_text("text")
        lines = [line.rstrip() for line in page_text.splitlines()]

        for line in lines:
            text = _normalize_space(line)
            if not text:
                continue

            is_heading = (
                len(text) <= 120 and text.isupper()
                or re.fullmatch(r"[A-Z0-9\- /&()]+", text) and len(text) <= 120
            )

            if is_heading:
                if current_parts:
                    chunk_text = "\n".join(current_parts).strip()
                    if chunk_text:
                        chunks.append(
                            PolicyChunk(
                                chunk_id=f"chunk_{len(chunks)+1:04d}",
                                text=chunk_text,
                                page_start=page_start,
                                page_end=page_end,
                                section=curr_section,
                                heading=curr_heading,
                                metadata={"page_count": page_end - page_start + 1},
                            )
                        )
                curr_heading = text
                curr_section = text if not current_parts else curr_section
                current_parts = []
                page_start = page_number + 1
                page_end = page_number + 1
                continue

            if text.endswith(':') and len(text) <= 120:
                if current_parts:
                    chunk_text = "\n".join(current_parts).strip()
                    if chunk_text:
                        chunks.append(
                            PolicyChunk(
                                chunk_id=f"chunk_{len(chunks)+1:04d}",
                                text=chunk_text,
                                page_start=page_start,
                                page_end=page_end,
                                section=curr_section,
                                heading=curr_heading,
                                metadata={"page_count": page_end - page_start + 1},
                            )
                        )
                    current_parts = []
                curr_heading = text[:-1]
                curr_section = curr_section
                page_start = page_number + 1
                page_end = page_number + 1
                continue

            current_parts.append(text)
            page_end = page_number + 1

        if current_parts:
            chunk_text = "\n".join(current_parts).strip()
            if chunk_text:
                chunks.append(
                    PolicyChunk(
                        chunk_id=f"chunk_{len(chunks)+1:04d}",
                        text=chunk_text,
                        page_start=page_start,
                        page_end=page_end,
                        section=curr_section,
                        heading=curr_heading,
                        metadata={"page_count": page_end - page_start + 1},
                    )
                )
            current_parts = []

    if not chunks:
        page = doc[0]
        text = page.get_text("text")
        chunks = [
            PolicyChunk(
                chunk_id="chunk_0001",
                text=_normalize_space(text),
                page_start=1,
                page_end=doc.page_count,
                section="Document",
                heading="Document",
            )
        ]

    def _finalize_chunks() -> List[PolicyChunk]:
        final: List[PolicyChunk] = []
        for chunk in chunks:
            text = _normalize_space(chunk.text)
            if len(text) < 30:
                continue
            final.append(chunk)
        return final

    return _finalize_chunks()


def _embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype="float32")


def build_policy_index(pdf_path: str | Path, output_dir: str | Path | None = None) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    chunks = _build_policy_chunks(pdf_path)
    texts = [chunk.text for chunk in chunks]

    dense_embeddings = _embed_texts(texts)
    dim = dense_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense_embeddings)

    tokenized = [re.findall(r"\w+", text.lower()) for text in texts]
    bm25 = BM25Okapi(tokenized)

    result = {
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "section": chunk.section,
                "heading": chunk.heading,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "source": chunk.source,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
        "dense_index": index,
        "bm25_index": bm25,
        "dense_embeddings": dense_embeddings,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump(result["chunks"], f, ensure_ascii=False, indent=2)

    return result
