import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────
RECSYS_URL = os.getenv("RECSYS_URL", "http://localhost:8001")
SENTIMENT_URL = os.getenv("SENTIMENT_URL", "http://localhost:8000")
PRODUCER_URL = os.getenv("PRODUCER_URL", "http://localhost:8002")
AGENT_URL = os.getenv("AGENT_URL", "http://localhost:8003")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

SAMPLE = json.loads(
    (Path(__file__).parent / "sample_data.json").read_text(encoding="utf-8")
)

_ROOT = Path(__file__).resolve().parent
for _candidate in [
    _ROOT.parents[3] / "artifacts/recsys_models/data_menu/item_lookup.parquet",
    Path("artifacts/recsys_models/data_menu/item_lookup.parquet"),
]:
    if _candidate.exists():
        ITEM_LOOKUP_PATH = _candidate
        break
else:
    ITEM_LOOKUP_PATH = None


# ─── Data helpers ─────────────────────────────────────────────────────
@st.cache_data
def _load_item_lookup() -> pd.DataFrame | None:
    if ITEM_LOOKUP_PATH is None:
        return None
    df = pd.read_parquet(ITEM_LOOKUP_PATH)
    df["product_id"] = df["product_id"].astype(str)
    return df


@st.cache_data
def _get_top_categories(n: int = 10) -> list[dict]:
    """Top N categories theo số sản phẩm."""
    df = _load_item_lookup()
    if df is None:
        return []
    return (
        df.groupby(["category_id", "category_name"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(n)
        .to_dict(orient="records")
    )


@st.cache_data
def _get_products_by_category(category_id: str, n: int = 9) -> list[dict]:
    """Top N sản phẩm của category, sort by avg_item_sentiment."""
    df = _load_item_lookup()
    if df is None:
        return []
    subset = df[df["category_id"].astype(str) == str(category_id)]
    return (
        subset.sort_values("avg_item_sentiment", ascending=False)
        .head(n)
        .to_dict(orient="records")
    )


@st.cache_data
def _search_products(query: str, n: int = 12) -> list[dict]:
    """Tìm sản phẩm theo tên/brand/category — client-side (item_lookup đã
    load sẵn local cho demo, không cần round-trip qua API)."""
    df = _load_item_lookup()
    if df is None:
        return []
    q = query.lower().strip()
    mask = (
        df["product_name"].str.lower().str.contains(q, na=False, regex=False)
        | df["brand_name"].str.lower().str.contains(q, na=False, regex=False)
        | df["category_name"].str.lower().str.contains(q, na=False, regex=False)
    )
    subset = df[mask]
    return (
        subset.sort_values("avg_item_sentiment", ascending=False)
        .head(n)
        .to_dict(orient="records")
    )


@st.cache_data(ttl=3600)
def _fetch_image(url: str) -> bytes | None:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None


# ─── API helpers ──────────────────────────────────────────────────────
def _demo_recommend(
    customer_id: str, top_k: int, exclude_ids: set | None = None
) -> dict:
    df = _load_item_lookup()
    if df is None:
        data = SAMPLE["recsys"]["existing_user"].copy()
        data["customer_id"] = customer_id
        data["recommendations"] = list(data["recommendations"][:top_k])
        return data

    seed = int(hashlib.md5(customer_id.encode()).hexdigest(), 16) % (2**31)
    pool = df if not exclude_ids else df[~df["product_id"].isin(exclude_ids)]
    sampled = pool.sample(n=min(top_k * 3, len(pool)), random_state=seed)
    sampled = sampled.sort_values("avg_item_sentiment", ascending=False).head(top_k)

    recs = []
    for rank, (_, row) in enumerate(sampled.iterrows()):
        recs.append(
            {
                "product_id": str(row["product_id"]),
                "product_name": str(row.get("product_name", "")),
                "thumbnail_url": str(row.get("thumbnail_url", "")),
                "price": float(row.get("price", 0)),
                "category_id": str(row.get("category_id", "")),
                "category_name": str(row.get("category_name", "")),
                "brand_name": str(row.get("brand_name", "")),
                "avg_item_sentiment": float(row.get("avg_item_sentiment", 0)),
                "predict_score": round(0.95 - rank * 0.05, 3),
            }
        )
    return {
        "status": "success",
        "customer_id": customer_id,
        "source": "model",
        "recommendations": recs,
    }


def _call_recsys(customer_id: str, top_k: int, exclude_ids: set | None = None) -> dict:
    if DEMO_MODE:
        return _demo_recommend(customer_id, top_k, exclude_ids)
    resp = requests.post(
        f"{RECSYS_URL}/recommend",
        json={"customer_id": customer_id, "top_k": top_k},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _submit_review(
    customer_id: str, product_id: str, comment: str, rating: int
) -> bool:
    if DEMO_MODE:
        return True
    try:
        resp = requests.post(
            f"{PRODUCER_URL}/api/v1/reviews",
            json={
                "customer_id": customer_id,
                "product_id": product_id,
                "comment": comment,
                "rating": rating,
                "purchased_at": "2026-06-25T10:00:00Z",
            },
            timeout=5,
        )
        return resp.status_code == 202
    except Exception:
        return False


def _simulate_personalization(
    recs: list, reviewed_product_id: str, rating: int, customer_id: str, top_k: int
) -> list:
    df = _load_item_lookup()
    reviewed = next((r for r in recs if r["product_id"] == reviewed_product_id), None)
    target_cat = reviewed.get("category_id") if reviewed else None
    shown_ids = {r["product_id"] for r in recs}

    if df is not None and target_cat:
        pool = (
            df[
                (df["category_id"].astype(str) == target_cat)
                & (~df["product_id"].isin(shown_ids))
            ]
            if rating >= 4
            else df[
                (df["category_id"].astype(str) != target_cat)
                & (~df["product_id"].isin(shown_ids))
            ]
        )
        if len(pool) >= 1:
            seed = int(
                hashlib.md5((customer_id + reviewed_product_id).encode()).hexdigest(),
                16,
            ) % (2**31)
            n_replace = min(3, len(pool), len(recs))
            new_rows = pool.sample(n=n_replace, random_state=seed)
            new_recs = [
                {
                    "product_id": str(row["product_id"]),
                    "product_name": str(row.get("product_name", "")),
                    "thumbnail_url": str(row.get("thumbnail_url", "")),
                    "price": float(row.get("price", 0)),
                    "category_id": str(row.get("category_id", "")),
                    "category_name": str(row.get("category_name", "")),
                    "brand_name": str(row.get("brand_name", "")),
                    "avg_item_sentiment": float(row.get("avg_item_sentiment", 0)),
                    "predict_score": round(0.90 - i * 0.05, 3),
                }
                for i, (_, row) in enumerate(new_rows.iterrows())
            ]
            return recs[:-n_replace] + new_recs

    same = [
        r
        for r in recs
        if r.get("category_id") == target_cat and r["product_id"] != reviewed_product_id
    ]
    diff = [r for r in recs if r.get("category_id") != target_cat]
    return (same + diff) if rating >= 4 else (diff + same)


# ─── UI helpers ───────────────────────────────────────────────────────
def _product_card(
    item: dict, key_prefix: str, show_buy: bool = False, bought: bool = False
):
    """Render 1 product card. Trả về True nếu user bấm 'Mua'."""
    with st.container(border=True):
        if img := _fetch_image(item.get("thumbnail_url")):
            st.image(img, width=180)
        name = item.get("product_name") or f"Product {item['product_id']}"
        st.markdown(f"**{name[:70]}{'…' if len(name) > 70 else ''}**")
        brand = item.get("brand_name", "")
        st.caption(f"{brand + ' · ' if brand else ''}{item.get('category_name', '')}")
        st.markdown(f"💰 **{item.get('price', 0):,.0f}đ**")
        sentiment = float(item.get("avg_item_sentiment", 0))
        st.progress(sentiment, text=f"Sentiment {sentiment:.2f}")

        if show_buy:
            if bought:
                st.success("✓ Đã mua", icon="🛒")
                return False
            return st.button(
                "🛒 Mua ngay",
                key=f"{key_prefix}_{item['product_id']}",
                use_container_width=True,
            )
    return False


# ─── Page setup ───────────────────────────────────────────────────────
st.set_page_config(page_title="E-commerce AI Demo", page_icon="🛍️", layout="wide")
st.title("🛍️ E-commerce AI Demo")
st.caption("Two-Tower Recommendation · PhoBERT Sentiment · Real-time Personalization")

if DEMO_MODE:
    st.info("🎬 **Demo Mode** — dữ liệu mẫu từ sản phẩm thật.", icon="ℹ️")

# ─── Session state ────────────────────────────────────────────────────
if "customer_id" not in st.session_state:
    st.session_state.customer_id = "2083331"
if "selected_cat" not in st.session_state:
    st.session_state.selected_cat = None
if "cart" not in st.session_state:
    st.session_state.cart = {}  # pid → item dict
if "reviewed" not in st.session_state:
    st.session_state.reviewed = set()
if "recs" not in st.session_state:
    st.session_state.recs = None
if "recs_source" not in st.session_state:
    st.session_state.recs_source = None
if "review_count" not in st.session_state:
    st.session_state.review_count = 0

tab_shop, tab_sentiment, tab_agent, tab_health = st.tabs(
    [
        "🛍️ Mua sắm & Gợi ý",
        "💬 Phân tích cảm xúc",
        "🤖 Shopping Agent",
        "⚙️ System Health",
    ]
)

# ═════════════════════════════════════════════════════════════════════
# TAB 1: SHOPPING FLOW
# ═════════════════════════════════════════════════════════════════════
with tab_shop:

    # ── Customer ID bar ──────────────────────────────────────────────
    c1, c2 = st.columns([4, 1])
    with c1:
        cid_input = st.text_input(
            "Customer ID",
            value=st.session_state.customer_id,
            label_visibility="collapsed",
            placeholder="Nhập customer_id...",
        )
        if cid_input != st.session_state.customer_id:
            st.session_state.customer_id = cid_input
            st.session_state.cart.clear()
            st.session_state.reviewed.clear()
            st.session_state.recs = None
    with c2:
        if st.session_state.cart:
            st.metric("Giỏ hàng", f"{len(st.session_state.cart)} sản phẩm")

    st.divider()

    # ── Tìm kiếm sản phẩm ─────────────────────────────────────────────
    search_query = st.text_input(
        "🔍 Tìm sản phẩm",
        placeholder="Nhập tên sản phẩm, thương hiệu, hoặc danh mục...",
    )
    if search_query.strip():
        search_results = _search_products(search_query, n=12)
        if search_results:
            st.markdown(f'**{len(search_results)} kết quả cho "{search_query}"**')
            scols = st.columns(3)
            for i, prod in enumerate(search_results):
                with scols[i % 3]:
                    bought = prod["product_id"] in st.session_state.cart
                    if _product_card(
                        prod, key_prefix="search", show_buy=True, bought=bought
                    ):
                        st.session_state.cart[prod["product_id"]] = prod
                        st.rerun()
        else:
            st.warning(f'Không tìm thấy sản phẩm nào khớp "{search_query}".')

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 1: KHÁM PHÁ THEO DANH MỤC
    # ══════════════════════════════════════════════════════════════════
    st.subheader("① Khám phá theo danh mục")
    st.caption("Chọn danh mục → xem sản phẩm → bấm **Mua ngay**")

    categories = _get_top_categories(10)

    if categories:
        # Hiển thị categories như button chips
        cat_cols = st.columns(min(len(categories), 5))
        for i, cat in enumerate(categories):
            with cat_cols[i % 5]:
                is_selected = st.session_state.selected_cat == cat["category_id"]
                label = f"{'▶ ' if is_selected else ''}{cat['category_name']} ({cat['count']})"
                if st.button(
                    label,
                    key=f"cat_{cat['category_id']}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_cat = cat["category_id"]
                    st.rerun()

        # Product grid cho category đang chọn
        if st.session_state.selected_cat:
            products = _get_products_by_category(st.session_state.selected_cat, n=9)
            if products:
                cat_name = next(
                    (
                        c["category_name"]
                        for c in categories
                        if c["category_id"] == st.session_state.selected_cat
                    ),
                    st.session_state.selected_cat,
                )
                st.markdown(f"**{cat_name}** — {len(products)} sản phẩm nổi bật")
                cols = st.columns(3)
                for i, prod in enumerate(products):
                    with cols[i % 3]:
                        bought = prod["product_id"] in st.session_state.cart
                        if _product_card(
                            prod, key_prefix="browse", show_buy=True, bought=bought
                        ):
                            st.session_state.cart[prod["product_id"]] = prod
                            st.rerun()
        else:
            st.info("← Chọn một danh mục để xem sản phẩm")
    else:
        st.warning("Không tải được danh mục. Kiểm tra item_lookup.parquet.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 2: GIỎ HÀNG & ĐÁNH GIÁ
    # ══════════════════════════════════════════════════════════════════
    st.subheader("② Đánh giá sản phẩm đã mua")

    if not st.session_state.cart:
        st.info("Chưa có sản phẩm trong giỏ. Bấm **Mua ngay** ở bước 1.")
    else:
        st.caption(
            "Review → Kafka → PhoBERT sentiment → Feast `recent_sentiment_score` "
            "→ Redis cache invalidation → gợi ý cập nhật"
        )
        for pid, item in list(st.session_state.cart.items()):
            name = item.get("product_name", pid)[:60]
            already_reviewed = pid in st.session_state.reviewed
            icon = "✅" if already_reviewed else "⭐"

            with st.expander(f"{icon} {name}", expanded=not already_reviewed):
                if already_reviewed:
                    st.success("Đã gửi đánh giá — gợi ý bên dưới đã được cập nhật.")
                    continue

                ec1, ec2 = st.columns([3, 1])
                with ec1:
                    comment = st.text_area(
                        "Nhận xét của bạn",
                        placeholder="Sản phẩm này...",
                        key=f"comment_{pid}",
                        height=90,
                    )
                with ec2:
                    rating = st.select_slider(
                        "Rating", options=[1, 2, 3, 4, 5], value=5, key=f"rating_{pid}"
                    )
                    st.markdown("⭐" * rating + "☆" * (5 - rating))

                if st.button("📤 Gửi đánh giá", key=f"submit_{pid}", type="primary"):
                    if not comment.strip():
                        st.warning("Nhập nhận xét trước khi gửi.")
                    else:
                        with st.spinner("Gửi vào Kafka..."):
                            ok = _submit_review(
                                customer_id=st.session_state.customer_id,
                                product_id=pid,
                                comment=comment,
                                rating=rating,
                            )
                        if not ok:
                            st.error("Không kết nối được Kafka producer (:8002).")
                        else:
                            st.session_state.reviewed.add(pid)
                            st.session_state.review_count += 1

                            # Cập nhật gợi ý
                            bought_ids = set(st.session_state.cart.keys())
                            if DEMO_MODE:
                                if st.session_state.recs:
                                    st.session_state.recs = _simulate_personalization(
                                        st.session_state.recs,
                                        pid,
                                        rating,
                                        st.session_state.customer_id,
                                        12,
                                    )
                                else:
                                    data = _call_recsys(
                                        st.session_state.customer_id,
                                        12,
                                        exclude_ids=bought_ids,
                                    )
                                    st.session_state.recs = data.get(
                                        "recommendations", []
                                    )
                                    st.session_state.recs_source = data.get(
                                        "source", "model"
                                    )
                            else:
                                # Bug thật 17/7/2026: sleep(3) cố định không đủ —
                                # đo trực tiếp pipeline thật (Kafka → PhoBERT →
                                # Feast → invalidate cache Redis) trên EC2 mất tới
                                # ~4.2s trong lúc box chịu tải (2vCPU chia sẻ toàn
                                # bộ stack), khiến lần gọi lại /recommend đôi khi
                                # tới TRƯỚC khi cache kịp invalidate, vẫn trả kết
                                # quả cũ y hệt → tưởng gợi ý "không load lại".
                                # Poll thay vì sleep mù: hỏi lại tới khi list sản
                                # phẩm thực sự đổi (hoặc hết timeout), có margin an
                                # toàn thay vì đoán 1 con số cố định.
                                old_ids = {
                                    r["product_id"]
                                    for r in (st.session_state.recs or [])
                                }
                                with st.spinner(
                                    "Chờ pipeline xử lý (Kafka → PhoBERT → Feast → Redis)..."
                                ):
                                    data = None
                                    for _ in range(8):  # tối đa ~8s, poll mỗi 1s
                                        time.sleep(1)
                                        try:
                                            data = _call_recsys(
                                                st.session_state.customer_id,
                                                12,
                                                exclude_ids=bought_ids,
                                            )
                                        except Exception:
                                            continue
                                        new_ids = {
                                            r["product_id"]
                                            for r in data.get("recommendations", [])
                                        }
                                        if new_ids != old_ids:
                                            break
                                try:
                                    if data is not None:
                                        st.session_state.recs = data.get(
                                            "recommendations", []
                                        )
                                        st.session_state.recs_source = data.get(
                                            "source", "model"
                                        )
                                    else:
                                        st.error(
                                            "Không gọi được recsys-api để refresh gợi ý."
                                        )
                                except Exception as e:
                                    st.error(f"Lỗi refresh gợi ý: {e}")

                            st.rerun()

    st.divider()

    # ══════════════════════════════════════════════════════════════════
    # BƯỚC 3: GỢI Ý CÁ NHÂN HÓA
    # ══════════════════════════════════════════════════════════════════
    st.subheader("③ Gợi ý cá nhân hóa")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 Lấy gợi ý", type="primary", use_container_width=True):
            bought_ids = set(st.session_state.cart.keys())
            with st.spinner("Two-Tower → FAISS → Reranker..."):
                try:
                    data = _call_recsys(
                        st.session_state.customer_id, 12, exclude_ids=bought_ids
                    )
                    st.session_state.recs = data.get("recommendations", [])
                    st.session_state.recs_source = data.get("source", "model")
                except Exception as e:
                    st.error(f"Lỗi recsys-api: {e}")
    with col_info:
        if st.session_state.recs:
            source_label = {
                "model": "🤖 Two-Tower Model",
                "trending": "🔥 Trending (Cold Start)",
                "redis_cache": "⚡ Redis Cache",
            }.get(st.session_state.recs_source, st.session_state.recs_source or "—")
            badge = (
                f"✍️ {st.session_state.review_count} review"
                if st.session_state.review_count
                else ""
            )
            st.info(
                f"**Nguồn:** {source_label} &nbsp;|&nbsp; `{st.session_state.customer_id}` &nbsp; {badge}"
            )

    if st.session_state.recs:
        cols = st.columns(3)
        for i, item in enumerate(st.session_state.recs):
            with cols[i % 3]:
                _product_card(item, key_prefix="rec", show_buy=False)
    elif not st.session_state.recs:
        st.caption("← Đánh giá sản phẩm hoặc bấm **Lấy gợi ý** để xem kết quả.")

    with st.expander("💡 Cold start demo"):
        st.write("Nhập customer_id mới → API trả trending items thay vì chạy model")
        st.code('customer_id: "new_user_99999"')

# ═════════════════════════════════════════════════════════════════════
# TAB 2: SENTIMENT
# ═════════════════════════════════════════════════════════════════════
with tab_sentiment:
    st.subheader("Phân tích cảm xúc review (PhoBERT)")
    st.caption("Vietnamese BERT fine-tuned on e-commerce reviews → NEG / POS / NEU")

    sample_texts = [
        "Sản phẩm rất tốt, giao hàng nhanh, đóng gói cẩn thận",
        "Hàng kém chất lượng, không như mô tả, rất thất vọng",
        "Tạm được, không có gì đặc biệt lắm",
    ]
    texts_input = st.text_area(
        "Nhập các review (mỗi dòng 1 review):",
        value="\n".join(sample_texts),
        height=150,
    )

    if st.button("📊 Phân tích", type="primary"):
        texts = [t.strip() for t in texts_input.strip().split("\n") if t.strip()]
        if not texts:
            st.warning("Nhập ít nhất 1 dòng.")
        else:
            with st.spinner("Đang chạy PhoBERT..."):
                try:
                    if DEMO_MODE:
                        results = []
                        for i, text in enumerate(texts):
                            s = SAMPLE["sentiment"]["results"][i % 3].copy()
                            s["text"] = text
                            results.append(s)
                        data = {
                            "results": results,
                            "latency": SAMPLE["sentiment"]["latency"],
                        }
                    else:
                        resp = requests.post(
                            f"{SENTIMENT_URL}/predict",
                            json={"texts": texts},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                except Exception as e:
                    st.error(f"Lỗi sentiment-api: {e}")
                    st.stop()

            if lat := data.get("latency"):
                st.caption(f"⏱️ Latency: {lat:.3f}s")

            label_map = {
                "POS": ("😊 Tích cực", "green"),
                "NEG": ("😞 Tiêu cực", "red"),
                "NEU": ("😐 Trung tính", "orange"),
            }
            for r in data.get("results", []):
                label = r.get("sentiment", "?")
                display, color = label_map.get(label, (label, "gray"))
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(r.get("text", ""))
                    with c2:
                        st.markdown(f":{color}[**{display}**]")
                        st.caption(f"conf: {r.get('confidence', 0):.2f}")

# ═════════════════════════════════════════════════════════════════════
# TAB 3: SHOPPING AGENT
# ═════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with tab_agent:
    st.subheader("🤖 Shopping Agent")
    st.caption(
        "Trợ lý mua sắm AI — tìm kiếm sản phẩm, gợi ý cá nhân hóa, trả lời câu hỏi bằng tiếng Việt"
    )

    if DEMO_MODE:
        st.info("Demo Mode — Agent API không chạy. Hiển thị kết quả mẫu.", icon="🎬")

    agent_cid = st.text_input(
        "Customer ID (cho gợi ý cá nhân hóa)",
        value=st.session_state.get("customer_id", "2083331"),
        key="agent_cid",
    )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander("🔧 Tool calls", expanded=False):
                    for tc in msg["tool_calls"]:
                        st.code(
                            f"{tc.get('tool')}({json.dumps(tc.get('input', {}), ensure_ascii=False)})\n→ {tc.get('output', '')[:200]}"
                        )

    # Chat input
    user_input = st.chat_input("Hỏi về sản phẩm, tìm kiếm, hoặc yêu cầu gợi ý...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if DEMO_MODE:
                with st.spinner("Đang xử lý..."):
                    df_lookup = _load_item_lookup()
                    if df_lookup is not None:
                        hits = df_lookup[
                            df_lookup["product_name"].str.contains(
                                user_input.split()[0] if user_input else "",
                                case=False,
                                na=False,
                            )
                        ].head(3)
                        if not hits.empty:
                            lines = [
                                f"- **{r['product_name']}** — {int(r['price']):,} VND"
                                for _, r in hits.iterrows()
                            ]
                            response_text = (
                                "Tìm thấy một số sản phẩm liên quan:\n"
                                + "\n".join(lines)
                            )
                        else:
                            response_text = (
                                "Không tìm thấy sản phẩm phù hợp trong Demo Mode."
                            )
                    else:
                        response_text = "Demo Mode — không có dữ liệu sản phẩm."
                    tool_calls = []
                st.markdown(response_text)
            else:
                # Streaming response via SSE
                response_text, tool_calls = "", []
                placeholder = st.empty()
                placeholder.markdown("_Đang suy nghĩ..._")
                try:
                    with requests.post(
                        f"{AGENT_URL}/chat/stream",
                        json={"customer_id": agent_cid, "message": user_input},
                        stream=True,
                        timeout=(
                            10,
                            300,
                        ),  # connect=10s, read=300s (LLM on CPU is slow)
                    ) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line or not line.startswith(b"data: "):
                                continue
                            event = json.loads(line[6:])
                            etype = event.get("type", "")
                            if etype == "tool_start":
                                names = ", ".join(event.get("tools", []))
                                placeholder.markdown(f"_Đang dùng: {names}..._")
                            elif etype == "token":
                                response_text += event.get("content", "")
                                placeholder.markdown(response_text + "▌")
                            elif etype == "done":
                                tool_calls = event.get("tool_calls", [])
                                placeholder.markdown(response_text)
                            elif etype == "blocked":
                                response_text = "⚠️ " + event.get(
                                    "message", "Yêu cầu không hợp lệ."
                                )
                                placeholder.markdown(response_text)
                            elif etype == "error":
                                response_text = "❌ " + event.get(
                                    "message", "Lỗi không xác định."
                                )
                                placeholder.markdown(response_text)
                except requests.exceptions.ConnectionError:
                    response_text = "❌ Agent API chưa khởi động. Chạy: `docker compose --profile agent up agent-api`"
                    placeholder.markdown(response_text)
                except Exception as e:
                    response_text = f"❌ Lỗi: {e}"
                    placeholder.markdown(response_text)

            if tool_calls:
                with st.expander("🔧 Tool calls", expanded=False):
                    for tc in tool_calls:
                        st.code(
                            f"{tc.get('tool')}({json.dumps(tc.get('input', {}), ensure_ascii=False)})\n→ {tc.get('output', '')[:200]}"
                        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls if not DEMO_MODE else [],
            }
        )

    if st.session_state.chat_history and st.button(
        "🗑️ Xóa lịch sử chat", key="clear_chat"
    ):
        st.session_state.chat_history = []
        st.rerun()

# ═════════════════════════════════════════════════════════════════════
# TAB 4: HEALTH
# ═════════════════════════════════════════════════════════════════════
with tab_health:
    st.subheader("System Health")

    if DEMO_MODE:
        st.warning("Demo Mode — APIs không chạy.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**recsys-api** (`:8001`)")
            for k, v in {
                "model": "ok",
                "device": "cpu",
                "num_items": 1744,
                "redis": "unavailable",
                "feast": "ok",
            }.items():
                st.write(
                    f"{'✅' if v in ('ok','cpu') or isinstance(v, int) else '⚠️'} `{k}`: {v}"
                )
        with c2:
            st.markdown("**sentiment-api** (`:8000`)")
            for k, v in {
                "status": "ok",
                "model": "phobert-base",
                "device": "cpu",
            }.items():
                st.write(f"✅ `{k}`: {v}")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**recsys-api** (`:8001`)")
            try:
                h = requests.get(f"{RECSYS_URL}/health", timeout=3).json()
                for k, v in h.items():
                    icon = (
                        "✅"
                        if v in ("ok", "cpu", "cuda") or isinstance(v, int)
                        else "⚠️"
                    )
                    st.write(f"{icon} `{k}`: {v}")
            except Exception as e:
                st.error(f"Offline: {e}")
        with c2:
            st.markdown("**sentiment-api** (`:8000`)")
            try:
                h = requests.get(f"{SENTIMENT_URL}/health", timeout=3).json()
                for k, v in h.items():
                    icon = (
                        "✅"
                        if v in ("ok", "healthy") or isinstance(v, (int, float))
                        else "⚠️"
                    )
                    st.write(f"{icon} `{k}`: {v}")
            except Exception as e:
                st.error(f"Offline: {e}")
