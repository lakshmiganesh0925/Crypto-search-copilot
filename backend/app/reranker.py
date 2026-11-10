"""Cross-encoder reranker as a LangChain document compressor (runs locally via fastembed/ONNX)."""
from functools import lru_cache
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


@lru_cache
def _encoder(model_name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name)


class FastEmbedReranker(BaseDocumentCompressor):
    model_name: str
    top_n: int = 30

    def compress_documents(
        self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = list(_encoder(self.model_name).rerank(query, [d.page_content for d in documents]))
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)[: self.top_n]
        return [
            Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(sc)})
            for d, sc in ranked
        ]
