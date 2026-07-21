"""
KB ingestion pipeline: S3/MinIO → load → chunk → embed → FAISS index.

Called by:
  - Airflow DAG  dags/kb_reindex_pipeline.py  (scheduled / webhook-triggered)
  - POST /admin/kb/reindex  (direct hot-reload via agent-api)

Supports PDF, Excel (.xlsx), HTML, plain text files stored in S3 under kb-docs/.
"""

import logging
import os
import tempfile
from pathlib import Path

from src.serving.agent_api.chunker import chunk_documents
from src.serving.agent_api.indexer import KBIndexer
from src.serving.agent_api.ingestion.s3_loader import S3KBLoader

logger = logging.getLogger(__name__)

_KB_INDEX_PATH = os.getenv("KB_INDEX_PATH", "/app/artifacts/kb_index")


_KB_LOCAL_PATH = os.getenv(
    "KB_LOCAL_PATH", ""
)  # non-empty = read from local folder instead of S3


def run_ingestion() -> KBIndexer:
    """
    Full pipeline: load files (local folder or S3/MinIO) → parse → chunk → embed → save FAISS.

    Source priority:
      KB_LOCAL_PATH set  → read .txt/.html/.pdf files from that folder (no S3 needed)
      KB_LOCAL_PATH unset → read from S3/MinIO (S3_ENDPOINT_URL controls MinIO vs AWS)
    """
    if _KB_LOCAL_PATH:
        docs = _load_from_local(_KB_LOCAL_PATH)
    else:
        docs = _load_from_s3()

    if not docs:
        logger.warning("kb_ingestion_no_docs — returning empty indexer")
        return KBIndexer()

    chunks = chunk_documents(docs)
    indexer = KBIndexer()
    indexer.add_chunks(chunks)
    indexer.save(_KB_INDEX_PATH)
    logger.info("kb_ingestion_done docs=%d chunks=%d", len(docs), len(chunks))
    return indexer


def _load_from_local(folder: str) -> list[dict]:
    """Read all supported files from a local directory."""
    base = Path(folder)
    if not base.exists():
        logger.warning("kb_local_path not found: %s", folder)
        return []
    docs: list[dict] = []
    for path in sorted(base.iterdir()):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        filename = path.name
        try:
            if suffix == ".txt":
                docs.append(
                    {
                        "text": path.read_text(encoding="utf-8", errors="replace"),
                        "source": filename,
                        "metadata": {"title": path.stem, "loader": "txt"},
                    }
                )
            elif suffix in (".html", ".htm"):
                docs.extend(_load_html(path.read_bytes(), filename))
            elif suffix == ".pdf":
                docs.extend(_load_pdf(path.read_bytes(), filename))
            else:
                logger.debug("kb_local_skip %s (unsupported)", filename)
        except Exception as exc:
            logger.error("kb_local_file_error %s: %s", filename, exc)
    logger.info("kb_ingestion_local folder=%s docs=%d", folder, len(docs))
    return docs


def _load_from_s3() -> list[dict]:
    """Read all supported files from S3/MinIO."""
    s3 = S3KBLoader()
    files = s3.list_files()
    logger.info(
        "kb_ingestion_start files=%d bucket=%s prefix=%s",
        len(files),
        s3._bucket,
        s3._prefix,
    )
    docs: list[dict] = []
    for f in files:
        key, filename = f["key"], f["filename"]
        suffix = Path(filename).suffix.lower()
        try:
            data = s3.download_bytes(key)
            if suffix == ".pdf":
                docs.extend(_load_pdf(data, filename))
            elif suffix in (".xlsx", ".xls"):
                docs.extend(_load_excel(data, filename))
            elif suffix in (".html", ".htm"):
                docs.extend(_load_html(data, filename))
            elif suffix == ".txt":
                docs.append(
                    {
                        "text": data.decode("utf-8", errors="replace"),
                        "source": filename,
                        "metadata": {"title": Path(filename).stem, "loader": "txt"},
                    }
                )
            else:
                logger.warning("kb_ingestion_skip key=%s (unsupported type)", key)
        except Exception as exc:
            logger.error("kb_ingestion_file_error key=%s: %s", key, exc)
    return docs


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_pdf(data: bytes, filename: str) -> list[dict]:
    from src.serving.agent_api.ingestion.pdf_loader import PDFLoader

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data)
            tmp_path = f.name
        docs = PDFLoader().load_file(tmp_path)
        # Replace temp path in source with the original filename
        for doc in docs:
            page = doc["metadata"].get("page", "")
            doc["source"] = f"{filename}::page_{page}" if page else filename
            doc["metadata"]["title"] = Path(filename).stem
        return docs
    finally:
        if tmp_path:
            os.unlink(tmp_path)


def _load_excel(data: bytes, filename: str) -> list[dict]:
    import io

    import pandas as pd

    df = pd.read_excel(io.BytesIO(data))
    cols = df.columns.tolist()
    docs: list[dict] = []
    for _, row in df.iterrows():
        text = ". ".join(
            f"{c}: {row[c]}"
            for c in cols
            if pd.notna(row.get(c)) and str(row.get(c, "")).strip()
        )
        if text.strip():
            docs.append(
                {
                    "text": text,
                    "source": filename,
                    "metadata": {"title": Path(filename).stem, "loader": "excel"},
                }
            )
    return docs


def _load_html(data: bytes, filename: str) -> list[dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = "\n".join(l.strip() for l in soup.get_text("\n").splitlines() if l.strip())
    if not text:
        return []
    return [
        {
            "text": text,
            "source": filename,
            "metadata": {"title": Path(filename).stem, "loader": "html"},
        }
    ]
