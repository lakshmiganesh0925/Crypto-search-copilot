"""Collectors for the three source types: news (RSS + article text), governance (Snapshot), whitepapers (PDF)."""
import html
import io
import logging
import re
import time
from datetime import datetime, timezone

import feedparser
import httpx
import trafilatura
from pypdf import PdfReader

from .types import Document

log = logging.getLogger("collectors")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
BROWSER = {"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
           "Accept-Language": "en-US,en;q=0.9"}
logging.getLogger("trafilatura").setLevel(logging.CRITICAL)

NEWS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Bitcoin Magazine": "https://bitcoinmagazine.com/.rss/full/",
}

# Snapshot space ids (DAO governance forums). Add your own.
SNAPSHOT_SPACES = {
    "aave.eth": "Aave",
    "uniswapgovernance.eth": "Uniswap",
    "ens.eth": "ENS",
    "lido-snap.eth": "Lido",
    "arbitrumfoundation.eth": "Arbitrum",
    "gitcoindao.eth": "Gitcoin",
}

# name -> (pdf url, publication date). Edit freely; failures are logged and skipped.
WHITEPAPERS = {
    "Bitcoin": ("https://bitcoin.org/bitcoin.pdf", "2008-10-31"),
    "Ethereum": (
        "https://ethereum.org/content/whitepaper/whitepaper-pdf/Ethereum_Whitepaper_-_Buterin_2014.pdf",
        "2014-01-01",
    ),
    "Solana": ("https://solana.com/solana-whitepaper.pdf", "2017-11-01"),
    "Uniswap v3": ("https://uniswap.org/whitepaper-v3.pdf", "2021-03-23"),
    "Chainlink": ("https://research.chain.link/whitepaper-v1.pdf", "2017-09-04"),
    "Aave": (
        "https://raw.githubusercontent.com/aave/aave-protocol/master/docs/Aave_Protocol_Whitepaper_v1_0.pdf",
        "2020-01-08",
    ),
}


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def collect_news(limit_per_feed: int = 15) -> list[Document]:
    docs: list[Document] = []
    with httpx.Client(timeout=30, follow_redirects=True, headers=BROWSER) as client:
        for name, feed_url in NEWS_FEEDS.items():
            try:
                r = client.get(feed_url)
                r.raise_for_status()
                feed = feedparser.parse(r.content)
            except Exception as e:  # noqa: BLE001
                log.warning("feed failed %s: %s", name, e)
                continue
            for e in feed.entries[:limit_per_feed]:
                link, title = e.get("link"), (e.get("title") or "").strip()
                if not link or not title:
                    continue
                text = ""
                try:
                    page = client.get(link)
                    page.raise_for_status()
                    text = trafilatura.extract(page.text, include_comments=False, include_tables=False) or ""
                except Exception:  # noqa: BLE001  (403 etc.: fall back to the RSS body)
                    pass
                if len(text) < 400:
                    body = e["content"][0].get("value", "") if e.get("content") else ""
                    text = _strip_html(body or e.get("summary", "")) or text
                if len(text) < 80:
                    continue
                ts = e.get("published_parsed") or e.get("updated_parsed")
                published = (
                    datetime(*ts[:6], tzinfo=timezone.utc) if ts else datetime.now(timezone.utc)
                ).isoformat()
                docs.append(Document(link, title, "news", name, published, text))
            log.info("news: %s done (%d docs so far)", name, len(docs))
    return docs


_SNAPSHOT_Q = """
query($spaces:[String!], $first:Int!){
  proposals(first:$first, where:{space_in:$spaces}, orderBy:"created", orderDirection:desc){
    id title body created state choices scores space{ id name }
  }
}"""


def collect_governance(per_space: int = 20) -> list[Document]:
    docs: list[Document] = []
    with httpx.Client(timeout=45, headers=BROWSER) as client:
        for space, label in SNAPSHOT_SPACES.items():
            try:
                r = client.post(
                    "https://hub.snapshot.org/graphql",
                    json={"query": _SNAPSHOT_Q, "variables": {"spaces": [space], "first": per_space}},
                )
                r.raise_for_status()
                proposals = r.json()["data"]["proposals"]
            except Exception as e:  # noqa: BLE001
                log.warning("snapshot failed %s: %s", space, e)
                continue
            for p in proposals:
                body = (p.get("body") or "").strip()
                if len(body) < 100:
                    continue
                results = ", ".join(
                    f"{c}: {round(s, 2)}" for c, s in zip(p.get("choices") or [], p.get("scores") or [])
                )
                header = (
                    f"Governance proposal ({label} DAO via Snapshot)\nTitle: {p['title']}\n"
                    f"Status: {p['state']}\nVote tally: {results or 'n/a'}\n\n"
                )
                url = f"https://snapshot.org/#/{space}/proposal/{p['id']}"
                published = datetime.fromtimestamp(p["created"], tz=timezone.utc).isoformat()
                docs.append(Document(url, p["title"], "governance", f"{label} DAO", published, header + body))
    return docs


def collect_whitepapers() -> list[Document]:
    docs: list[Document] = []
    with httpx.Client(timeout=90, follow_redirects=True, headers=BROWSER) as client:
        for name, (url, date) in WHITEPAPERS.items():
            try:
                for attempt in range(1, 4):  # polite retry on 429 / transient errors
                    r = client.get(url)
                    if r.status_code != 429:
                        break
                    time.sleep(8 * attempt)
                r.raise_for_status()
                reader = PdfReader(io.BytesIO(r.content))
                text = "\n\n".join((pg.extract_text() or "") for pg in reader.pages)
                text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # re-flow hard line breaks
                text = re.sub(r"[ \t]+", " ", text).strip()
                if len(text) < 1000:
                    raise ValueError("too little text extracted")
                docs.append(
                    Document(url, f"{name} whitepaper", "whitepaper", name, f"{date}T00:00:00+00:00", text)
                )
            except Exception as e:  # noqa: BLE001
                log.warning("whitepaper failed %s: %s", name, e)
    return docs
