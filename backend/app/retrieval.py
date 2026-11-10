"""Retrieval: Qdrant hybrid search (dense + BM25 + RRF) -> cross-encoder rerank -> thresholded context."""
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from .config import get_settings
from .lc import build_filter, get_vectorstore
from .reranker import FastEmbedReranker


def get_retriever(source_types=None, days=None, k: int | None = None) -> Runnable:
    """LCEL pipeline: query -> Qdrant hybrid retriever -> cross-encoder rerank. invoke(query) -> list[Document]."""
    s = get_settings()
    k = k or s.retrieve_k
    base = get_vectorstore("hybrid").as_retriever(
        search_kwargs={"k": k, "filter": build_filter(source_types, days)}
    )
    reranker = FastEmbedReranker(model_name=s.reranker_model, top_n=k)
    return {"docs": base, "q": RunnablePassthrough()} | RunnableLambda(
        lambda x: list(reranker.compress_documents(x["docs"], x["q"]))
    )


def search_mode(query: str, mode: str, k: int, source_types=None, days=None) -> list[Document]:
    """Single-strategy search (dense | sparse | hybrid), no rerank. Used by the eval ablation."""
    return get_vectorstore(mode).similarity_search(query, k=k, filter=build_filter(source_types, days))


def select_context(docs: list[Document]) -> list[Document]:
    """Keep reranked docs above the relevance threshold, capped per source document and overall."""
    s = get_settings()
    out: list[Document] = []
    per_doc: dict[str, int] = {}
    for d in sorted(docs, key=lambda d: d.metadata.get("rerank_score", -99), reverse=True):
        if d.metadata.get("rerank_score", -99) < s.min_rerank_score:
            break
        did = d.metadata.get("doc_id", d.metadata.get("url", ""))
        if per_doc.get(did, 0) >= s.max_chunks_per_doc:
            continue
        per_doc[did] = per_doc.get(did, 0) + 1
        out.append(d)
        if len(out) >= s.rerank_top_n:
            break
    return out
