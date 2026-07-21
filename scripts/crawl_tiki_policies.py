"""
Crawl Tiki KB (hotro.tiki.vn) theo 2 tầng:
  Level 1: /knowledge-base/<id>      — category page → extract article links
  Level 2: /knowledge-base/post/<id> — article page  → extract text content

Lưu ra local (kb-docs/) hoặc upload MinIO/S3.
Sau đó trigger agent-api reindex để FAISS được cập nhật.

Usage:
    pip install playwright boto3 beautifulsoup4
    playwright install chromium

    # Lưu local (mặc định):
    python scripts/crawl_tiki_policies.py

    # Upload lên MinIO:
    python scripts/crawl_tiki_policies.py --minio

Env vars:
    S3_ENDPOINT_URL       http://localhost:9000   (empty = AWS S3)
    AWS_ACCESS_KEY_ID     admin
    AWS_SECRET_ACCESS_KEY password
    KB_BUCKET             warehouse
    KB_PREFIX             kb-docs/
    KB_LOCAL_PATH         kb-docs
"""

import asyncio
import os
import re
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

# ── Chỉ crawl các category liên quan đến chính sách / hỗ trợ ──────────────────
CATEGORY_URLS = [
    "https://hotro.tiki.vn/knowledge-base/226",  # Tài khoản của tôi
    "https://hotro.tiki.vn/knowledge-base/229",  # Đặt hàng và Thanh toán
    "https://hotro.tiki.vn/knowledge-base/232",  # Giao và Nhận hàng
    "https://hotro.tiki.vn/knowledge-base/235",  # Đổi trả - Bảo hành và Bồi Thường
    "https://hotro.tiki.vn/knowledge-base/241",  # Thông tin và Chính sách
    "https://hotro.tiki.vn/knowledge-base/250",  # Dịch vụ và Chương trình
]

ARTICLE_PATTERN = re.compile(r"https://hotro\.tiki\.vn/knowledge-base/post/\d+")

# ── Storage ────────────────────────────────────────────────────────────────────
LOCAL_DIR = Path(os.getenv("KB_LOCAL_PATH", "kb-docs"))
BUCKET = os.getenv("KB_BUCKET", "warehouse")
PREFIX = os.getenv("KB_PREFIX", "kb-docs/")
ENDPOINT = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")


def _s3():
    import boto3
    from botocore.client import Config

    kw = dict(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "admin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "password"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    if ENDPOINT:
        kw["endpoint_url"] = ENDPOINT
        kw["config"] = Config(signature_version="s3v4")
    return boto3.client("s3", **kw)


def _ensure_bucket(s3):
    try:
        s3.head_bucket(Bucket=BUCKET)
    except Exception:
        s3.create_bucket(Bucket=BUCKET)
        print(f"  Created bucket: {BUCKET}")


def save(slug: str, title: str, text: str, use_minio: bool):
    content = f"# {title}\n\nNguồn: https://hotro.tiki.vn/knowledge-base/post/{slug}\n\n{text}"
    if use_minio:
        s3 = _s3()
        _ensure_bucket(s3)
        key = f"{PREFIX}post-{slug}.txt"
        s3.put_object(
            Bucket=BUCKET,
            Key=key,
            Body=content.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        print(f"    ✓ → s3://{BUCKET}/{key}")
    else:
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)
        path = LOCAL_DIR / f"post-{slug}.txt"
        path.write_text(content, encoding="utf-8")
        print(f"    ✓ → {path}")


# ── Playwright helpers ─────────────────────────────────────────────────────────


async def make_page(browser) -> Page:
    return await browser.new_page(
        locale="vi-VN",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
    )


async def get_article_links(page: Page, category_url: str) -> list[str]:
    """Vào trang category → lấy tất cả link /knowledge-base/post/<id>."""
    await page.goto(category_url, wait_until="networkidle", timeout=30_000)
    await page.wait_for_timeout(1500)
    hrefs = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
    links = list(
        {h for h in hrefs if ARTICLE_PATTERN.match(h.split("?")[0].split("#")[0])}
    )
    return links


async def get_article_content(page: Page, url: str) -> tuple[str, str]:
    """Vào trang bài viết → lấy title + text."""
    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    # Đợi body render xong
    await page.wait_for_selector("body", timeout=10_000)
    await page.wait_for_timeout(1000)
    title = await page.title()
    # Dùng inner_text() của Playwright — an toàn hơn evaluate trên document.body
    text = await page.inner_text("body")
    return title, text.strip()


# ── Main ───────────────────────────────────────────────────────────────────────


async def run(use_minio: bool):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        # ── Bước 1: collect article links từ tất cả categories ──
        all_articles: dict[str, str] = {}  # url → category

        print("=" * 60)
        print("[1] Thu thập article links từ các category...")
        print("=" * 60)

        for cat_url in CATEGORY_URLS:
            cat_name = cat_url.split("/")[-1]
            print(f"\n  Category {cat_name}: {cat_url}")
            try:
                page = await make_page(browser)
                links = await get_article_links(page, cat_url)
                await page.close()
                print(f"  → {len(links)} articles")
                for l in links:
                    all_articles[l] = cat_name
            except Exception as e:
                print(f"  ✗ {e}")

        print(f"\nTổng cộng: {len(all_articles)} articles (đã dedup)\n")

        # ── Bước 2: crawl từng article ──
        print("=" * 60)
        print("[2] Crawling articles...")
        print("=" * 60)

        ok, fail = 0, 0
        for url, cat in all_articles.items():
            slug = url.rstrip("/").split("/")[-1]
            print(f"\n  [{cat}] post/{slug}")
            try:
                page = await make_page(browser)
                title, text = await get_article_content(page, url)
                await page.close()

                if len(text) < 50:
                    print(f"    ✗ quá ngắn ({len(text)} chars), bỏ qua")
                    continue

                save(slug, title, text, use_minio)
                ok += 1
            except Exception as e:
                print(f"    ✗ {e}")
                fail += 1

        await browser.close()

    print(f"\n{'='*60}")
    print(f"Xong: {ok} OK, {fail} lỗi")

    if not use_minio:
        print(f"Files: {LOCAL_DIR.resolve()}/")

    print("\nTrigger reindex để agent-api cập nhật KB:")
    print("  curl -X POST http://localhost:8003/admin/kb/reindex")


if __name__ == "__main__":
    use_minio = "--minio" in sys.argv
    mode = "MinIO" if use_minio else f"local ({LOCAL_DIR}/)"
    print(f"Mode: {mode}\n")
    asyncio.run(run(use_minio))
