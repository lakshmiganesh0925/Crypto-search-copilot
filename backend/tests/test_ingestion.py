from app.ingestion.dedupe import canonical_url, dedupe_documents
from app.ingestion.types import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def doc(url, title, text):
    return Document(url, title, "news", "X", "2026-01-01T00:00:00+00:00", text)


def test_canonical_url_strips_tracking():
    assert canonical_url("https://A.com/x/?utm_source=t&id=2#frag") == "https://a.com/x?id=2"


def test_dedupe_exact_and_near():
    base = "Ethereum developers agreed on the next upgrade timeline after months of debate. " * 8
    docs = [
        doc("https://a.com/1", "Upgrade agreed", base),
        doc("https://a.com/1?utm_source=x", "Other title", "completely different words " * 40),  # same URL
        doc("https://b.com/2", "Syndicated copy", base + " Extra line."),  # near-dup text
        doc("https://c.com/3", "Different story", "Solana validators discuss fee markets and congestion. " * 10),
    ]
    kept = dedupe_documents(docs)
    assert [d.url for d in kept] == ["https://a.com/1", "https://c.com/3"]


def test_splitter_sizes():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(6))
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_text(text)
    assert len(chunks) > 1 and all(len(c) <= 1000 for c in chunks)
