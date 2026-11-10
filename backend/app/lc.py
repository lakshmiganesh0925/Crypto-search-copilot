"""LangChain factories: Gemini chat model, Gemini embeddings, BM25 sparse embeddings, Qdrant Cloud vector store."""
import re
import threading
import time
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from .config import get_settings

DENSE = "dense"
SPARSE = "sparse"
META = "metadata"  # LangChain stores Document.metadata under this payload key


@lru_cache
def get_llm(temperature: float = 0.2, model: str | None = None) -> ChatGoogleGenerativeAI:
    s = get_settings()
    return ChatGoogleGenerativeAI(
        model=model or s.gemini_chat_model, google_api_key=s.gemini_api_key,
        temperature=temperature, max_retries=3,
    )


def llm_names() -> list[str]:
    s = get_settings()
    names = [s.gemini_chat_model] + [m.strip() for m in s.gemini_fallback_models.split(",") if m.strip()]
    return list(dict.fromkeys(names))


def with_fallbacks(make_chain, temperature: float = 0.2):
    chains = [make_chain(get_llm(temperature, m)) for m in llm_names()]
    return chains[0].with_fallbacks(chains[1:]) if len(chains) > 1 else chains[0]

class ThrottledEmbeddings(Embeddings):
    """Keeps under the Gemini free-tier limit (every text counts as one request per minute)
    and retries on 429 using the server's suggested delay."""

    def __init__(self, inner: Embeddings, per_minute: int = 80, batch: int = 20, max_attempts: int = 8) -> None:
        self.inner, self.per_minute, self.batch, self.max_attempts = inner, per_minute, batch, max_attempts
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def _reserve(self, n: int) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._stamps = [x for x in self._stamps if now - x < 60]
                if not self._stamps or len(self._stamps) + n <= self.per_minute:
                    self._stamps.extend([now] * n)
                    return
                time.sleep(max(60 - (now - self._stamps[0]) + 0.5, 0.5))

    def _call(self, fn, n: int):
        for attempt in range(1, self.max_attempts + 1):
            self._reserve(n)
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                    raise
                if "PerDay" in msg or "per day" in msg.lower():
                    raise RuntimeError(
                        "Gemini DAILY embedding quota reached. Wait until it resets, enable billing, "
                        "or ingest less (see MAX_ARTICLES_PER_FEED / MAX_PROPOSALS_PER_SPACE)."
                    ) from e
                m = re.search(r"retry in ([\d.]+)s", msg)
                time.sleep(float(m.group(1)) + 2 if m else 15 * attempt)
        raise RuntimeError("Gemini embedding rate limit: retries exhausted")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            part = texts[i : i + self.batch]
            out.extend(self._call(lambda part=part: self.inner.embed_documents(part), len(part)))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._call(lambda: self.inner.embed_query(text), 1)


@lru_cache
def get_embeddings() -> ThrottledEmbeddings:
    s = get_settings()
    # embed_documents -> RETRIEVAL_DOCUMENT, embed_query -> RETRIEVAL_QUERY (wrapper defaults)
    inner = GoogleGenerativeAIEmbeddings(model=f"models/{s.gemini_embed_model}", google_api_key=s.gemini_api_key)
    return ThrottledEmbeddings(inner, per_minute=s.embed_rpm)


@lru_cache
def get_sparse() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=get_settings().sparse_model)  # local BM25, no API calls


@lru_cache
def get_client() -> QdrantClient:
    s = get_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key, timeout=60)


def ensure_collection() -> None:
    s, client = get_settings(), get_client()
    if client.collection_exists(s.qdrant_collection):
        return
    client.create_collection(
        s.qdrant_collection,
        vectors_config={DENSE: models.VectorParams(size=s.embed_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field, schema in [
        (f"{META}.source_type", models.PayloadSchemaType.KEYWORD),
        (f"{META}.source_name", models.PayloadSchemaType.KEYWORD),
        (f"{META}.doc_id", models.PayloadSchemaType.KEYWORD),
        (f"{META}.published_ts", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(s.qdrant_collection, field, schema)


_MODES = {"hybrid": RetrievalMode.HYBRID, "dense": RetrievalMode.DENSE, "sparse": RetrievalMode.SPARSE}


@lru_cache
def get_vectorstore(mode: str = "hybrid") -> QdrantVectorStore:
    """mode: hybrid (dense + BM25, RRF fusion) | dense | sparse. Same collection, different query strategy."""
    kw: dict = dict(
        client=get_client(),
        collection_name=get_settings().qdrant_collection,
        retrieval_mode=_MODES[mode],
        vector_name=DENSE,
        sparse_vector_name=SPARSE,
    )
    if mode in ("dense", "hybrid"):
        kw["embedding"] = get_embeddings()
    if mode in ("sparse", "hybrid"):
        kw["sparse_embedding"] = get_sparse()
    return QdrantVectorStore(**kw)


def build_filter(
    source_types: list[str] | None = None, days: int | None = None, exclude_doc_id: str | None = None
) -> models.Filter | None:
    must, must_not = [], []
    if source_types:
        must.append(models.FieldCondition(key=f"{META}.source_type", match=models.MatchAny(any=source_types)))
    if days:
        must.append(
            models.FieldCondition(key=f"{META}.published_ts", range=models.Range(gte=int(time.time()) - days * 86400))
        )
    if exclude_doc_id:
        must_not.append(models.FieldCondition(key=f"{META}.doc_id", match=models.MatchValue(value=exclude_doc_id)))
    return models.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None
