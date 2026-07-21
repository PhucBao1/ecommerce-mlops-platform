"""
Excel loader for tabular KB data (e.g. product spec sheets, FAQ tables).

Reads .xlsx/.csv files and converts each row into a natural-language sentence
suitable for embedding. Handles multi-column rows by joining as key: value pairs.

Requires: openpyxl>=3.1.0 (already a pandas dep)
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_KB_DIR = os.getenv("KB_EXCEL_DIR", "/app/artifacts/kb_docs/excel")


def _row_to_text(row: pd.Series, columns: list[str]) -> str:
    """Convert a DataFrame row to a natural language string."""
    parts = []
    for col in columns:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    return ". ".join(parts)


class ExcelLoader:
    """Load Excel/CSV files and convert rows to document dicts."""

    def load_file(self, path: str, text_column: str | None = None) -> list[dict]:
        """
        Load a single file.

        Args:
            path: Path to .xlsx or .csv file.
            text_column: If set, use this single column as text. Otherwise
                         join all non-null columns as key-value pairs.
        """
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            logger.warning("excel_loader failed %s: %s", path, e)
            return []

        docs: list[dict] = []
        stem = Path(path).stem
        columns = df.columns.tolist()

        for i, row in df.iterrows():
            if text_column and text_column in df.columns:
                text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            else:
                text = _row_to_text(row, columns)

            if text.strip():
                docs.append(
                    {
                        "text": text.strip(),
                        "source": f"{path}::row_{i}",
                        "metadata": {
                            "title": stem,
                            "row": int(i),
                            "loader": "excel",
                        },
                    }
                )

        logger.info("excel_loader loaded %d rows from %s", len(docs), Path(path).name)
        return docs

    def load_dir(self, dir_path: str = _DEFAULT_KB_DIR) -> list[dict]:
        """Load all Excel and CSV files from a directory."""
        all_docs: list[dict] = []
        kb_dir = Path(dir_path)
        if not kb_dir.exists():
            logger.warning("excel_loader: directory not found: %s", dir_path)
            return []
        for f in sorted(kb_dir.glob("*")):
            if f.suffix.lower() in (".xlsx", ".xls", ".csv"):
                all_docs.extend(self.load_file(str(f)))
        return all_docs
