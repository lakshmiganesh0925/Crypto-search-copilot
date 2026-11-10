"""Guardrails. Pure-Python rules (fast, testable offline); an optional LLM intent check lives in rag.py."""
import re
from dataclasses import dataclass
from typing import Callable, Optional

DISCLAIMER = "Informational research only — not financial advice."
SENTINEL = "INSUFFICIENT_INFORMATION"

REFUSAL_MESSAGE = (
    "I can't give buy/sell/hold recommendations, price predictions, or personal financial advice. "
    "I can explain what recent news, governance proposals, and whitepapers say about a project — "
    "its design, risks, recent decisions, or tradeoffs. Try rephrasing along those lines."
)
INJECTION_MESSAGE = "I can only answer crypto research questions from my indexed sources. Please rephrase your question."
INSUFFICIENT_MESSAGE = (
    "I don't have enough reliable information in my indexed sources to answer that. "
    "Try naming a specific protocol, proposal, or time frame, or widen the source filters."
)


@dataclass
class GuardResult:
    allowed: bool
    category: Optional[str] = None  # financial_advice | prompt_injection | empty | no_citations | bad_citation | ...


def _c(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.I) for p in patterns]


_ADVICE_IN = _c([
    r"\b(should|shall|can|could|would)\s+(i|we)\s+(buy|sell|hold|invest|short|long|ape|stake|swap|dca|trade)\b",
    r"\b(buy|sell|hold)\s+(or|vs\.?)\s+(buy|sell|hold)\b",
    r"\bwhat\s+(coins?|tokens?|cryptos?|altcoins?)\s+(should|to)\s+(i\s+)?(buy|invest|hold)",
    r"\b(best|top|safest)\s+(coins?|tokens?|cryptos?|altcoins?)\s+to\s+(buy|invest|hold)",
    r"\bis\s+.{1,40}\s+(a\s+)?(good|great|bad|smart|safe)\s+(buy|investment)\b",
    r"\bis\s+.{1,40}\s+worth\s+(buying|investing|holding)\b",
    r"\bprice\s+(prediction|forecast|target)s?\b",
    r"\bwhen\s+(should|to)\s+(i\s+)?(buy|sell|enter|exit)\b",
    r"\bwill\b.{0,50}\b(moon|pump|dump|crash|10x|100x|1000x|go\s+up|go\s+down)\b",
    r"\bwill\b.{0,50}\b(reach|hit|exceed|surpass)\s+\$\s?\d",
    r"\b(entry|exit)\s+(point|price)s?\b",
    r"\bgive\s+me\s+(financial|investment|trading)\s+advice\b",
])

_INJECTION = _c([
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
    r"disregard\s+(the\s+)?(system|previous|above)",
    r"(reveal|show|print|repeat)\s+(your\s+|the\s+)?(system\s+prompt|instructions|hidden\s+prompt)",
    r"\byou\s+are\s+now\b",
    r"\bjailbreak\b|\bDAN\s+mode\b",
])

_FINANCE_HINT = re.compile(
    r"\b(buy|sell|hold|invest\w*|trad(e|ing)|worth|profit|moon|pump|dump|portfolio|predict\w*|forecast|"
    r"target|100x|10x|bullish|bearish|stake|long|short|savings|money|funds|wise|safe|risky|returns?|gains?|bet|allocate)\b",
    re.I,
)

_ADVICE_OUT = _c([
    r"\byou\s+should\s+(buy|sell|hold|invest|accumulate|dca)\b",
    r"\bi\s+(would\s+)?(recommend|suggest|advise)\s+(buying|selling|holding|investing)",
    r"\b(is|are)\s+(a\s+)?(great|good|strong|solid|smart)\s+(buy|investment)\b",
    r"\bprice\s+target\b",
    r"\b(will|is\s+going\s+to)\s+(moon|pump|dump|skyrocket|crash)\b",
    r"\bnot\s+a\s+bad\s+time\s+to\s+(buy|sell)\b",
])

_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def check_input(query: str, intent_fn: Optional[Callable[[str], str]] = None) -> GuardResult:
    q = (query or "").strip()
    if not q:
        return GuardResult(False, "empty")
    if any(p.search(q) for p in _INJECTION):
        return GuardResult(False, "prompt_injection")
    if any(p.search(q) for p in _ADVICE_IN):
        return GuardResult(False, "financial_advice")
    # Paraphrased advice requests slip past regexes: ask the LLM, but only when finance-y words are present.
    if intent_fn and _FINANCE_HINT.search(q):
        try:
            if intent_fn(q) == "advice":
                return GuardResult(False, "financial_advice")
        except Exception:  # noqa: BLE001  (fail open: regexes + output guard + system prompt still apply)
            pass
    return GuardResult(True)


def cited_ids(answer: str) -> list[int]:
    ids: set[int] = set()
    for m in _CITE.finditer(answer):
        ids.update(int(x) for x in re.split(r"\s*,\s*", m.group(1)))
    return sorted(ids)


def check_output(answer: str, n_sources: int) -> GuardResult:
    if not answer or SENTINEL in answer:
        return GuardResult(False, "insufficient")
    ids = cited_ids(answer)
    if not ids:
        return GuardResult(False, "no_citations")
    if any(i < 1 or i > n_sources for i in ids):
        return GuardResult(False, "bad_citation")
    if any(p.search(answer) for p in _ADVICE_OUT):
        return GuardResult(False, "financial_advice")
    return GuardResult(True)
