"""
PDF loader for KB documents.

Reads local PDF files (policy docs, product catalogues) and extracts text
page-by-page. Each page becomes a separate document dict for chunking.

Requires: pypdf>=4.0.0
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_KB_DIR = os.getenv("KB_PDF_DIR", "/app/artifacts/kb_docs/pdf")


class PDFLoader:
    """Load PDF files from a directory or explicit file list."""

    def load_file(self, path: str) -> list[dict]:
        """Extract text from each page of a PDF. Returns one doc dict per page."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.error("pypdf not installed — run: pip install pypdf>=4.0.0")
            return []

        docs: list[dict] = []
        try:
            reader = PdfReader(path)
            stem = Path(path).stem
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        {
                            "text": text.strip(),
                            "source": f"{path}::page_{page_num + 1}",
                            "metadata": {
                                "title": stem,
                                "page": page_num + 1,
                                "total_pages": len(reader.pages),
                                "loader": "pdf",
                            },
                        }
                    )
        except Exception as e:
            logger.warning("pdf_loader failed %s: %s", path, e)
        return docs

    def load_dir(self, dir_path: str = _DEFAULT_KB_DIR) -> list[dict]:
        """Load all PDFs from a directory."""
        all_docs: list[dict] = []
        pdf_dir = Path(dir_path)
        if not pdf_dir.exists():
            logger.warning("pdf_loader: directory not found: %s", dir_path)
            return []
        for pdf_file in sorted(pdf_dir.glob("*.pdf")):
            docs = self.load_file(str(pdf_file))
            logger.info("pdf_loader loaded %d pages from %s", len(docs), pdf_file.name)
            all_docs.extend(docs)
        return all_docs
