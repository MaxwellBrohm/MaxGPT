"""Web-search tool (the model's "internet access").

Returns short snippets for a query. Runs on the PC / with internet, via the free
DuckDuckGo search library (no API key). Degrades gracefully to [] if the library or
network is unavailable, so nothing breaks offline.
"""
from __future__ import annotations


def web_search(query: str, k: int = 3) -> list[str]:
    try:
        from duckduckgo_search import DDGS          # pip install duckduckgo_search
    except ImportError:
        try:
            from ddgs import DDGS                    # newer package name
        except ImportError:
            return []
    try:
        with DDGS() as ddgs:
            return [r.get("body", "") for r in ddgs.text(query, max_results=k) if r.get("body")]
    except Exception:
        return []
