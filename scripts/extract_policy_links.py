"""
Bước 1: Chạy script này để extract tất cả links từ 1 trang.
Sau đó copy links cần thiết sang crawl_tiki_policies.py.

Usage:
    pip install playwright
    playwright install chromium
    python scripts/extract_policy_links.py <url>

Example:
    python scripts/extract_policy_links.py https://tiki.vn/chinh-sach
"""

import asyncio
import sys
from urllib.parse import urlparse

from playwright.async_api import async_playwright


async def extract_links(start_url: str):
    base_domain = urlparse(start_url).netloc

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(
            locale="vi-VN",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
        )

        print(f"Đang mở: {start_url}")
        await page.goto(start_url, wait_until="networkidle", timeout=30_000)
        await page.wait_for_timeout(2000)

        # Lấy tất cả <a href>
        links = await page.eval_on_selector_all(
            "a[href]", "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))"
        )

        await browser.close()

    # Filter: chỉ giữ link cùng domain
    same_domain = [
        l
        for l in links
        if urlparse(l["href"]).netloc == base_domain and l["href"] != start_url
    ]

    # Dedup
    seen = set()
    unique = []
    for l in same_domain:
        if l["href"] not in seen:
            seen.add(l["href"])
            unique.append(l)

    print(f"\nTìm thấy {len(unique)} links:\n")
    for l in unique:
        print(f"  {l['href']}")
        if l["text"]:
            print(f"    └─ {l['text'][:80]}")

    # Export dạng python list để copy vào crawler
    print("\n--- Copy vào crawl_tiki_policies.py ---")
    print("POLICIES = [")
    for l in unique:
        slug = l["href"].rstrip("/").split("/")[-1] or "index"
        text = l["text"][:40].strip().replace('"', "") if l["text"] else slug
        print(f'    ("{slug}", "{l["href"]}"),  # {text}')
    print("]")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://tiki.vn/chinh-sach"
    asyncio.run(extract_links(url))
