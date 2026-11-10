"""Ingestion: collect -> dedupe -> split (LangChain) -> embed + upsert (QdrantVectorStore hybrid)."""
import hashlib
import logging
import uuid
from datetime import datetime

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings
from ..lc import build_filter, ensure_collection, get_client, get_vectorstore
from .collectors import collect_governance, collect_news, collect_whitepapers
from .dedupe import canonical_url, dedupe_documents

log = logging.getLogger("ingest")


def make_splitter() -> RecursiveCharacterTextSplitter:
    s = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )


def _existing_hashes(ids: list[str]) -> dict[str, str]:
    s = get_settings()
    recs = get_client().retrieve(s.qdrant_collection, ids=ids, with_payload=True, with_vectors=False)
    return {str(r.id): ((r.payload or {}).get("metadata") or {}).get("content_hash", "") for r in recs}


def run_ingest(source_types: list[str]) -> dict:
    s = get_settings()
    ensure_collection()
    hybrid, dense = get_vectorstore("hybrid"), get_vectorstore("dense")
    splitter = make_splitter()

    docs = []
    if "news" in source_types:
        docs += collect_news(s.max_articles_per_feed)
    if "governance" in source_types:
        docs += collect_governance(s.max_proposals_per_space)
    if "whitepaper" in source_types:
        docs += collect_whitepapers()
    collected = len(docs)
    docs = dedupe_documents(docs)
    stats = {"collected": collected, "after_dedupe": len(docs), "docs_skipped_unchanged": 0,
             "docs_skipped_near_dup": 0, "docs_indexed": 0, "chunks_indexed": 0}

    for doc in docs:
        cu = canonical_url(doc.url)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, cu))
        chunks = [c for c in splitter.split_text(doc.text) if len(c) >= 80] or [doc.text[: s.chunk_size]]
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{cu}#{i}")) for i in range(len(chunks))]
        hashes = [hashlib.sha256(c.encode()).hexdigest()[:16] for c in chunks]

        existing = _existing_hashes(ids)  # idempotent re-runs: only embed new/changed chunks
        todo = [i for i in range(len(chunks)) if existing.get(ids[i]) != hashes[i]]
        if not todo:
            stats["docs_skipped_unchanged"] += 1
            continue

        ts = int(datetime.fromisoformat(doc.published_at).timestamp())
        lc_docs = [
            LCDocument(
                page_content=f"{doc.title}\n\n{chunks[i]}",  # title prefix helps both BM25 and dense recall
                metadata={
                    "doc_id": doc_id, "chunk_index": i, "title": doc.title, "url": doc.url,
                    "source_type": doc.source_type, "source_name": doc.source_name,
                    "published_at": doc.published_at, "published_ts": ts, "content_hash": hashes[i],
                },
            )
            for i in todo
        ]

        if doc.source_type == "news" and not existing:  # semantic near-dup (syndicated / rewritten stories)
            near = dense.similarity_search_with_score(
                lc_docs[0].page_content, k=1, filter=build_filter(exclude_doc_id=doc_id)
            )
            if near and near[0][1] >= s.near_dup_threshold:
                stats["docs_skipped_near_dup"] += 1
                log.info("near-duplicate skipped: %s ~ %s", doc.title, near[0][0].metadata.get("title"))
                continue

        hybrid.add_documents(lc_docs, ids=[ids[i] for i in todo])
        stats["docs_indexed"] += 1
        stats["chunks_indexed"] += len(lc_docs)
        log.info("indexed %s (%d chunks)", doc.title[:70], len(lc_docs))

    stats["collection_points"] = get_client().count(s.qdrant_collection, exact=True).count
    return stats
