# Crypto Research Copilot

A Retrieval-Augmented Generation (RAG) chatbot that answers crypto questions from three source types
(**news articles, DAO governance proposals, whitepapers**), always with source links and publication dates.
Built with **LangChain**, **Gemini**, **Qdrant Cloud**, **FastAPI** and **Next.js**.

## Architecture

```
Next.js UI ──► FastAPI /chat
                 │
   input guard ──┤  regex rules + LLM intent check (ChatPromptTemplate | Gemini.with_structured_output)
   history rewrite (LCEL)
   Retriever (LCEL):  QdrantVectorStore(RetrievalMode.HYBRID)   ← Gemini dense + FastEmbed BM25, RRF fusion
                        └─► FastEmbedReranker (BaseDocumentCompressor, cross-encoder)
   relevance threshold ──► "insufficient information" when nothing reliable
   generation (LCEL):  ChatPromptTemplate | ChatGoogleGenerativeAI | StrOutputParser
   output guard ─── citation validity, advice-language check, INSUFFICIENT_INFORMATION sentinel

Ingestion: collectors (RSS+trafilatura, Snapshot GraphQL, PDF) → dedupe (URL/title, shingle Jaccard,
           semantic near-dup via Qdrant) → RecursiveCharacterTextSplitter → QdrantVectorStore.add_documents
```

LangChain pieces used: `ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`, `QdrantVectorStore`
(hybrid/dense/sparse modes), `FastEmbedSparse`, `RecursiveCharacterTextSplitter`, `ChatPromptTemplate`,
LCEL (`|`, `RunnableLambda`, `RunnablePassthrough`), `with_structured_output`, a custom `BaseDocumentCompressor`.

## Setup

Prereqs: Python 3.10+, Node 18+, a [Gemini API key](https://aistudio.google.com/apikey),
and a free [Qdrant Cloud](https://cloud.qdrant.io) cluster (URL + API key).

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # fill in GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY

python -m scripts.ingest                               # or: --types news governance whitepaper
uvicorn app.main:app --reload --port 8000
```
First run downloads two small local models (BM25 + reranker, ~100 MB). Re-running ingest is idempotent:
unchanged chunks are not re-embedded. You can also trigger ingestion with `POST /ingest` and poll `GET /ingest/status`.

### Frontend
```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev                                            # http://localhost:3000
```

## Free-tier notes (Gemini)
The free tier allows ~100 embedding requests/minute and ~1,000/day (each chunk = 1 request). Ingestion is throttled
(`EMBED_RPM=80`) and retries on 429, so a full run takes several minutes. If you hit the daily cap, re-run later:
already-indexed chunks are skipped, so it resumes where it stopped. Ingest the most valuable sources first:
`python -m scripts.ingest --types whitepaper`, then `governance`, then `news`. Lower `MAX_ARTICLES_PER_FEED` /
`MAX_PROPOSALS_PER_SPACE` or raise `CHUNK_SIZE` to use fewer requests.

## Guardrails
- **No financial advice**: rule-based input filter (buy/sell/hold, price predictions, "will X hit $…"), an LLM intent
  classifier for paraphrases, a system prompt that forbids recommendations, and an output filter for advice language.
- **Insufficient information**: returned when the best reranked chunk is below `MIN_RERANK_SCORE`, when the model
  replies `INSUFFICIENT_INFORMATION`, or when the answer has missing/invalid citations.
- **Prompt injection**: input patterns blocked; retrieved text is treated as untrusted data in the prompt.

## Evaluation
```bash
cd backend
python -m pytest                       # offline unit tests (guardrails, dedupe, splitter)
python -m eval.build_dataset           # Gemini writes questions for sampled chunks -> eval/datasets/generated.jsonl
python -m eval.run_eval --e2e 20       # retrieval ablation + guardrail/abstention + LLM-judge faithfulness
```
- **Retrieval ablation**: dense vs BM25 vs hybrid vs hybrid+rerank (hit@1/3/5/10, MRR@10). Note: questions are
  generated from the chunk itself, so lexical methods are slightly favoured. Treat it as a relative comparison.
- **Guardrails** (`eval/datasets/static.jsonl`): advice + injection must be refused, off-topic/fabricated must abstain,
  and legitimate questions must *not* be refused (checks over-blocking). Failures are listed in the report.
- Tune `MIN_RERANK_SCORE` using the unanswerable vs should-answer pass rates.

## Notes
- Whitepaper/RSS/Snapshot URLs live in `backend/app/ingestion/collectors.py`; edit freely. Failing sources are logged and skipped.
- Changing `GEMINI_EMBED_MODEL`/`EMBED_DIM` requires a new Qdrant collection (set `QDRANT_COLLECTION`).
- Informational research tool only, not financial advice.
