from dataclasses import dataclass


@dataclass
class Document:
    url: str
    title: str
    source_type: str  # news | governance | whitepaper
    source_name: str
    published_at: str  # ISO-8601, UTC
    text: str
