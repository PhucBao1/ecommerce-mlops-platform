"""
MCP Server for E-commerce RecSys API.

Exposes recommendation and search endpoints as AI-native tools via fastmcp.
Compatible with Claude Desktop, Cursor, and any MCP-compliant AI client.

Usage (stdio transport — Claude Desktop):
    python -m src.serving.recsys_api.mcp_server

Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "recsys": {
          "command": "python",
          "args": ["-m", "src.serving.recsys_api.mcp_server"],
          "env": {"RECSYS_API_URL": "http://localhost:8001"}
        }
      }
    }
"""

import os

import requests
from fastmcp import FastMCP

mcp = FastMCP("E-commerce RecSys — Tiki-style recommendation engine")

RECSYS_URL = os.getenv("RECSYS_API_URL", "http://localhost:8001")
AGENT_URL = os.getenv("AGENT_API_URL", "http://localhost:8003")
_TIMEOUT = int(os.getenv("MCP_REQUEST_TIMEOUT", "10"))


@mcp.tool()
def get_recommendations(customer_id: str, top_k: int = 5) -> list[dict]:
    """Get personalized product recommendations for a customer based on purchase history.

    Uses Two-Tower neural network with collaborative filtering.
    Returns cold-start trending products for new customers.

    Args:
        customer_id: Unique customer identifier (e.g. "12345")
        top_k: Number of recommendations to return (1-20, default 5)
    """
    top_k = max(1, min(20, top_k))
    resp = requests.post(
        f"{RECSYS_URL}/recommend",
        json={"customer_id": customer_id, "top_k": top_k},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in data.get("recommendations", []):
        item = {
            "product_id": str(r.get("product_id", "")),
            "name": r.get("product_name", ""),
            "price_vnd": int(r.get("price", 0)),
            "category": r.get("category_name", ""),
            "brand": r.get("brand_name", ""),
        }
        explanation = r.get("explanation", {})
        if explanation.get("top_reason"):
            item["reason"] = explanation["top_reason"]
        results.append(item)

    return results


@mcp.tool()
def search_products(
    query: str,
    max_price: float | None = None,
    min_price: float | None = None,
    top_k: int = 5,
) -> list[dict]:
    """Search products by Vietnamese natural language query with optional price filter.

    Uses hybrid search: dense (PhoBERT semantic) + sparse (TF-IDF keyword) + RRF fusion.

    Args:
        query: Search query in Vietnamese (e.g. "tai nghe bluetooth chong on")
        max_price: Maximum price in VND (e.g. 500000 for 500k)
        min_price: Minimum price in VND (default 0)
        top_k: Number of results to return (1-20, default 5)
    """
    top_k = max(1, min(20, top_k))
    params: dict = {"q": query, "top_k": top_k}
    if max_price is not None:
        params["max_price"] = max_price
    if min_price is not None:
        params["min_price"] = min_price

    resp = requests.get(f"{AGENT_URL}/search", params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    results = resp.json()

    return [
        {
            "product_id": r.get("product_id", ""),
            "name": r.get("product_name", ""),
            "price_vnd": int(r.get("price", 0)),
            "category": r.get("category_name", ""),
            "brand": r.get("brand_name", ""),
            "sentiment_score": round(float(r.get("avg_sentiment", 0)), 2),
        }
        for r in results
    ]


@mcp.tool()
def get_product_detail(product_id: str) -> dict:
    """Get detailed information about a specific product.

    Args:
        product_id: Product identifier string
    """
    resp = requests.get(
        f"{AGENT_URL}/search",
        params={"q": product_id, "top_k": 1},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return {"error": f"Product {product_id} not found"}
    r = results[0]
    return {
        "product_id": r.get("product_id", ""),
        "name": r.get("product_name", ""),
        "price_vnd": int(r.get("price", 0)),
        "category": r.get("category_name", ""),
        "brand": r.get("brand_name", ""),
        "sentiment_score": round(float(r.get("avg_sentiment", 0)), 2),
    }


if __name__ == "__main__":
    mcp.run()
