"""
Document chunker for Knowledge Base (KB) documents.

Splits long policy documents (PDF/web/excel) into overlapping text chunks
suitable for embedding and FAISS/Qdrant indexing.

Product records (short, <100 tokens) → NOT chunked, handled by rag.py.
Policy docs (long, >200 tokens) → chunked here before indexing.

Strategy: sentence-aware sliding window
  - chunk_size: ~300 tokens (~1500 chars) — fits in BERT context window
  - overlap: 50 tokens — ensures continuity across chunk boundaries
"""

import re
from dataclasses import dataclass, field

_DEFAULT_CHUNK_CHARS = int(600)  # ~200-300 tokens Vietnamese; model max_seq=256 tokens
_DEFAULT_OVERLAP_CHARS = int(100)

# Vietnamese sentence boundary: end with . ? ! followed by space/newline
_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")


@dataclass
class Chunk:
    text: str
    source: str  # file path or URL
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return f"{self.source}::{self.chunk_index}"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using Vietnamese-aware regex."""
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = _DEFAULT_CHUNK_CHARS,
    overlap: int = _DEFAULT_OVERLAP_CHARS,
    metadata: dict | None = None,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full document text.
        source: Identifier for the source (file path or URL).
        chunk_size: Max characters per chunk.
        overlap: Overlap characters between consecutive chunks.
        metadata: Extra metadata attached to every chunk (e.g. title, category).

    Returns:
        List of Chunk objects.
    """
    if not text or not text.strip():
        return []

    sentences = _split_sentences(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    for sent in sentences:
        sent_len = len(sent)

        # If single sentence exceeds chunk_size, split by chars
        if sent_len > chunk_size:
            if current:
                chunks.append(
                    Chunk(
                        text=" ".join(current),
                        source=source,
                        chunk_index=chunk_idx,
                        metadata=metadata or {},
                    )
                )
                chunk_idx += 1
                current = []
                current_len = 0
            # Hard split the oversized sentence
            for start in range(0, sent_len, chunk_size - overlap):
                sub = sent[start : start + chunk_size]
                if sub.strip():
                    chunks.append(
                        Chunk(
                            text=sub,
                            source=source,
                            chunk_index=chunk_idx,
                            metadata=metadata or {},
                        )
                    )
                    chunk_idx += 1
            continue

        if current_len + sent_len > chunk_size and current:
            chunks.append(
                Chunk(
                    text=" ".join(current),
                    source=source,
                    chunk_index=chunk_idx,
                    metadata=metadata or {},
                )
            )
            chunk_idx += 1

            # Keep overlap: rewind until overlap chars reached
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(
            Chunk(
                text=" ".join(current),
                source=source,
                chunk_index=chunk_idx,
                metadata=metadata or {},
            )
        )

    return chunks


def chunk_documents(docs: list[dict]) -> list[Chunk]:
    """
    Chunk a list of document dicts.

    Each dict must have: {"text": str, "source": str, "metadata": dict (optional)}
    """
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_text(
            text=doc.get("text", ""),
            source=doc.get("source", "unknown"),
            metadata=doc.get("metadata", {}),
        )
        all_chunks.extend(chunks)
    return all_chunks
