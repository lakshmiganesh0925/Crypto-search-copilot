import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .ingestion.pipeline import run_ingest
from .lc import ensure_collection, get_client
from .models import ChatRequest, ChatResponse, IngestRequest
from .rag import answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for _n in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)
INGEST_STATE: dict = {"running": False, "last": None, "error": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_collection()
    yield


app = FastAPI(title="Crypto Research Copilot", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    s = get_settings()
    return {"ok": True, "points": get_client().count(s.qdrant_collection, exact=True).count}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):  # sync def -> FastAPI runs it in a threadpool
    try:
        return answer(req)
    except Exception as e:  # noqa: BLE001
        logging.exception("chat failed")
        msg = str(e)
        if any(k in msg for k in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")):
            raise HTTPException(status_code=503, detail="Gemini is busy or rate limited. Try again in a minute.") from e
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {msg[:500]}") from e


def _ingest_job(types: list[str]) -> None:
    INGEST_STATE.update(running=True, error=None)
    try:
        INGEST_STATE["last"] = run_ingest(types)
    except Exception as e:  # noqa: BLE001
        logging.exception("ingest failed")
        INGEST_STATE["error"] = str(e)
    finally:
        INGEST_STATE["running"] = False


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, bg: BackgroundTasks):
    if INGEST_STATE["running"]:
        raise HTTPException(status_code=409, detail="Ingestion already running")
    bg.add_task(_ingest_job, req.source_types)
    return {"started": True, "source_types": req.source_types}


@app.get("/ingest/status")
def ingest_status():
    return INGEST_STATE
