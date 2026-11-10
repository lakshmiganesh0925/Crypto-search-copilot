"""CLI:  python -m scripts.ingest --types news governance whitepaper"""
import argparse
import json
import logging

from app.ingestion.pipeline import run_ingest

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "trafilatura", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", nargs="+", default=["news", "governance", "whitepaper"],
                    choices=["news", "governance", "whitepaper"])
    args = ap.parse_args()
    print(json.dumps(run_ingest(args.types), indent=2))
