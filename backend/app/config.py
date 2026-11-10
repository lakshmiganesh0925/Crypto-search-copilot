import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Gemini
    gemini_api_key: str
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_fallback_models: str = "gemini-3.6-flash,gemini-3.5-flash-lite"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_rpm: int = 80  # free tier = 100 texts/min; lower if you still see 429s
    embed_dim: int = 3072  # gemini-embedding-001 default size via LangChain

    # Qdrant Cloud
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "crypto_research"

    # Ingestion
    chunk_size: int = 1200
    chunk_overlap: int = 150
    max_articles_per_feed: int = 15
    max_proposals_per_space: int = 8
    near_dup_threshold: float = 0.96  # cosine; semantic near-duplicate articles

    # Retrieval
    retrieve_k: int = 30
    rerank_top_n: int = 6
    max_chunks_per_doc: int = 2
    # Cross-encoder logits: relevant ~ > 0, unrelated ~ < -5. Tune with eval/run_eval.py
    min_rerank_score: float = -3.0
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    sparse_model: str = "Qdrant/bm25"

    # Guardrails
    use_llm_guard: bool = True

    cors_origins: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
