"""Build a synthetic retrieval dataset from your own index: Gemini writes a question for sampled chunks;
the source document is the gold label.   python -m eval.build_dataset --per-type 15"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import get_settings
from app.lc import get_client, get_llm

OUT = Path(__file__).parent / "datasets" / "generated.jsonl"


class Q(BaseModel):
    question: str


PROMPT = ChatPromptTemplate.from_template(
    "Write ONE specific question a crypto researcher might ask that is answered by the passage below. "
    "Do not mention 'the passage' or 'the article'; do not copy long phrases verbatim; "
    "the question must be understandable on its own.\n\nTitle: {title}\n\nPassage:\n{text}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    s, client = get_settings(), get_client()

    by_type: dict[str, list] = defaultdict(list)
    offset = None
    while True:
        pts, offset = client.scroll(s.qdrant_collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
        for p in pts:
            md = (p.payload or {}).get("metadata", {})
            if len((p.payload or {}).get("page_content", "")) > 400:
                by_type[md.get("source_type", "?")].append(p)
        if offset is None:
            break

    chain = PROMPT | get_llm(0.3).with_structured_output(Q)
    rows = []
    for stype, pts in by_type.items():
        for p in random.sample(pts, min(args.per_type, len(pts))):
            md = p.payload["metadata"]
            q = chain.invoke({"title": md["title"], "text": p.payload["page_content"][:1500]}).question
            rows.append({"id": str(p.id), "kind": "retrieval", "question": q, "gold_doc_id": md["doc_id"],
                         "gold_url": md["url"], "source_type": stype})
            time.sleep(0.5)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} questions -> {OUT}")


if __name__ == "__main__":
    main()
