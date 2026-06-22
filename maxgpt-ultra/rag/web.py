"""Web-search tool (the model's "internet access").

Returns real-source text for a query via the free DuckDuckGo search library (no API
key). Two quality knobs, both aimed at making the grounding fair + primary-source:

  1. Authoritative domains are preferred. We over-fetch results and re-rank so
     Wikipedia / .gov / .edu / official docs bubble to the top ahead of random
     blogs. DuckDuckGo's relevance order is preserved within each tier, so we only
     ever promote a source that was already relevant to the query.
  2. Optional full-text fetch. With fetch_full, we download the actual page and
     extract its main text instead of using the short search snippet. Worth it for
     the big-context model (ultra, 2048); snippets stay the default for the small
     one (mini, 1024). Fetches run concurrently so latency is ~one page, not k.

Everything degrades gracefully: no library / no network / a dead page -> we fall
back to the snippet, or to [] if search itself is unavailable. Nothing breaks
offline. Note: this only ever returns text excerpted from real web pages. It never
touches an "AI overview" / answer box (DuckDuckGo's .text() returns organic
results only), so the model grounds on sources, it is not handed another AI's
pre-written answer.
"""
from __future__ import annotations

import html as _html
import re
from urllib.parse import urlparse

# Domains we trust as primary / authoritative. A result whose host equals one of
# these (or is a subdomain of it), or sits on a .gov/.edu/.mil/.int TLD, is ranked
# ahead of general web results.
_AUTH_DOMAINS = (
    "wikipedia.org", "britannica.com", "nature.com", "science.org",
    "ncbi.nlm.nih.gov", "nih.gov", "who.int", "nasa.gov", "noaa.gov", "usgs.gov",
    "arxiv.org", "stackoverflow.com", "python.org", "docs.python.org",
    "developer.mozilla.org", "pytorch.org", "numpy.org", "wolframalpha.com",
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
)
_AUTH_TLDS = (".gov", ".edu", ".mil", ".int")


def _href(r: dict) -> str:
    return r.get("href") or r.get("link") or r.get("url") or ""


def _domain(url: str) -> str:
    try:
        net = urlparse(url).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return ""


def _auth_rank(url: str) -> int:
    """0 = authoritative, 1 = normal, 2 = unknown host. Lower sorts first."""
    d = _domain(url)
    if not d:
        return 2
    if any(d == a or d.endswith("." + a) for a in _AUTH_DOMAINS):
        return 0
    if any(d.endswith(t) for t in _AUTH_TLDS):
        return 0
    return 1


def _html_to_text(raw: str, max_chars: int) -> str:
    """Strip HTML to readable main text (best-effort). bs4 if present, regex fallback."""
    raw = raw[:3_000_000]   # bound parse cost on huge pages
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "nav", "aside", "form", "svg"]):
            tag.decompose()
        text = soup.get_text(" ")
    except ImportError:
        text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = _html.unescape(text)
    return " ".join(text.split())[:max_chars]


def _fetch_text(url: str, max_chars: int, timeout: float = 6.0) -> str:
    """Download a page and return its main text. Empty string on any failure."""
    if not url:
        return ""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MaxGPT-RAG/1.0)"}
    try:
        try:
            import requests
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            raw = resp.text
        except ImportError:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (RAG fetch)
                raw = r.read().decode("utf-8", "ignore")
    except Exception:
        return ""
    return _html_to_text(raw, max_chars)


def web_search(query: str, k: int = 3, fetch_full: bool = False,
               per_source_chars: int = 1500) -> list[tuple[str, str]]:
    """Search the web and return up to k (domain, text) pairs, most authoritative first.

    text is the full page (truncated to per_source_chars) when fetch_full, else the
    DuckDuckGo snippet. domain is the source host (for labeling / provenance).
    """
    try:
        from duckduckgo_search import DDGS          # pip install duckduckgo_search
    except ImportError:
        try:
            from ddgs import DDGS                    # newer package name
        except ImportError:
            return []
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max(k * 3, 10)))
    except Exception:
        return []

    raw.sort(key=lambda r: _auth_rank(_href(r)))    # stable: keeps DDG order within a tier
    chosen = raw[:k]

    texts = [""] * len(chosen)
    if fetch_full:
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(max(len(chosen), 1), 5)) as ex:
                texts = list(ex.map(lambda r: _fetch_text(_href(r), per_source_chars), chosen))
        except Exception:
            texts = [""] * len(chosen)               # fall back to snippets below

    out: list[tuple[str, str]] = []
    for r, full in zip(chosen, texts):
        text = full or (r.get("body") or "").strip()
        if text:
            out.append((_domain(_href(r)), text[:per_source_chars]))
    return out
