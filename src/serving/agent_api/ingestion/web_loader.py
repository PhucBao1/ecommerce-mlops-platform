"""
Web loader for Tiki policy pages.

Scrapes Vietnamese e-commerce policy URLs and returns clean text documents.
Target pages (from ROADMAP Task 95):
  - tiki.vn/chinh-sach-doi-tra (return policy)
  - tiki.vn/chinh-sach-bao-hanh (warranty)
  - tiki.vn/chinh-sach-thanh-toan (payment)
  - tiki.vn/van-chuyen (shipping)
  - tiki.vn/faq
  - TikiNOW, TikiPRO info pages
"""

import logging
import os
import time
from typing import Iterator

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = int(os.getenv("WEB_LOADER_TIMEOUT", "15"))
_DEFAULT_DELAY = float(os.getenv("WEB_LOADER_DELAY", "1.0"))

TIKI_POLICY_URLS = [
    ("https://tiki.vn/chinh-sach-doi-tra.html", "Chính sách đổi trả"),
    ("https://tiki.vn/chinh-sach-bao-hanh.html", "Chính sách bảo hành"),
    ("https://tiki.vn/chinh-sach-thanh-toan.html", "Chính sách thanh toán"),
    ("https://tiki.vn/van-chuyen.html", "Chính sách vận chuyển"),
    ("https://tiki.vn/faq.html", "FAQ"),
    ("https://tiki.vn/tikinow.html", "TikiNOW"),
    ("https://tiki.vn/tikipro.html", "TikiPRO"),
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; RecSys-KB-Bot/1.0; "
        "+https://github.com/pbao2910/ecommerce-recsys)"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}


def _extract_text(html: str, url: str) -> str:
    """Extract readable text from HTML, removing nav/footer/script."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=lambda c: c and "content" in c.lower())
        or soup.body
    )
    if main is None:
        return ""
    lines = [line.strip() for line in main.get_text("\n").splitlines() if line.strip()]
    return "\n".join(lines)


class WebLoader:
    """Scrapes a list of URLs and returns document dicts for chunking."""

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT, delay: float = _DEFAULT_DELAY):
        self._timeout = timeout
        self._delay = delay

    def load_url(self, url: str, title: str = "") -> dict | None:
        """Load a single URL. Returns None on failure."""
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=self._timeout)
            resp.raise_for_status()
            text = _extract_text(resp.text, url)
            if not text.strip():
                logger.warning("web_loader empty content: %s", url)
                return None
            return {
                "text": text,
                "source": url,
                "metadata": {"title": title or url, "loader": "web"},
            }
        except Exception as e:
            logger.warning("web_loader failed %s: %s", url, e)
            return None

    def load_urls(
        self, urls: list[tuple[str, str]] = TIKI_POLICY_URLS
    ) -> Iterator[dict]:
        """Load multiple URLs with rate limiting. Yields document dicts."""
        for url, title in urls:
            doc = self.load_url(url, title)
            if doc:
                yield doc
            time.sleep(self._delay)

    def load_all(self, urls: list[tuple[str, str]] = TIKI_POLICY_URLS) -> list[dict]:
        return list(self.load_urls(urls))
