"""Duplicate removal. Layer 1 (exact URL / title), Layer 2 (near-duplicate text via shingle Jaccard).
Layer 3 (semantic near-duplicates across runs) lives in pipeline.py and uses Qdrant."""
import re
import zlib
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .types import Document

_TRACKING = ("utm_", "fbclid", "gclid", "ref", "source")


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(_TRACKING)]
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", urlencode(q), ""))


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    if len(words) <= n:
        return {zlib.crc32(" ".join(words).encode())}
    return {zlib.crc32(" ".join(words[i : i + n]).encode()) for i in range(len(words) - n + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedupe_documents(docs: list[Document], threshold: float = 0.8) -> list[Document]:
    kept: list[Document] = []
    urls: set[str] = set()
    titles: set[str] = set()
    sigs: list[set[int]] = []
    for d in docs:
        cu, nt = canonical_url(d.url), _norm_title(d.title)
        if cu in urls or (nt and nt in titles):
            continue
        sig = shingles(d.text)
        if any(jaccard(sig, s) >= threshold for s in sigs):
            continue
        urls.add(cu)
        titles.add(nt)
        sigs.append(sig)
        kept.append(d)
    return kept
