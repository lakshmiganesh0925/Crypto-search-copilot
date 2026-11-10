"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Answer from "@/components/Answer";
import Sources from "@/components/Sources";
import { ask, ChatResponse, HistoryItem, SourceType } from "@/lib/api";

interface Turn {
  id: number;
  question: string;
  response?: ChatResponse;
  error?: string;
}

const EXAMPLES = [
  "What did the latest Aave governance proposals decide?",
  "How does concentrated liquidity in Uniswap v3 work?",
  "What does the Bitcoin whitepaper say about preventing double spending?",
  "What are recent news stories about Ethereum upgrades?",
];

const TYPES: { id: SourceType; label: string }[] = [
  { id: "news", label: "News" },
  { id: "governance", label: "Governance" },
  { id: "whitepaper", label: "Whitepapers" },
];

export default function Page() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [types, setTypes] = useState<SourceType[]>([]);
  const [days, setDays] = useState<number | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [turns, loading]);

  const toggle = (t: SourceType) =>
    setTypes((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  async function submit(q: string) {
    const query = q.trim();
    if (!query || loading) return;
    const id = Date.now();
    const history: HistoryItem[] = turns.flatMap((t) =>
      t.response
        ? [
            { role: "user" as const, content: t.question },
            { role: "assistant" as const, content: t.response.answer },
          ]
        : []
    );
    setTurns((cur) => [...cur, { id, question: query }]);
    setInput("");
    setLoading(true);
    try {
      const response = await ask(query, history.slice(-6), types, days);
      setTurns((cur) => cur.map((t) => (t.id === id ? { ...t, response } : t)));
    } catch (e) {
      const error = e instanceof Error ? e.message : "Something went wrong.";
      setTurns((cur) => cur.map((t) => (t.id === id ? { ...t, error } : t)));
    } finally {
      setLoading(false);
    }
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    submit(input);
  };

  return (
    <>
      <main className="shell">
        <header className="head">
          <h1>Crypto Research Copilot</h1>
          <p>
            Ask about crypto news, DAO governance proposals and whitepapers. Every answer is cited, and it says so
            when the sources don&apos;t cover your question.
          </p>
        </header>

        <div className="filters" role="group" aria-label="Source filters">
          {TYPES.map((t) => (
            <button
              key={t.id}
              type="button"
              className="chip"
              aria-pressed={types.includes(t.id)}
              onClick={() => toggle(t.id)}
            >
              {t.label}
            </button>
          ))}
          <select
            aria-label="Time range"
            value={days ?? ""}
            onChange={(e) => setDays(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">Any date</option>
            <option value="7">Past 7 days</option>
            <option value="30">Past 30 days</option>
            <option value="365">Past year</option>
          </select>
        </div>

        {turns.length === 0 && (
          <section className="empty">
            <h2>Try a question</h2>
            <ul className="examples">
              {EXAMPLES.map((ex) => (
                <li key={ex}>
                  <button type="button" onClick={() => submit(ex)}>
                    {ex}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <div className="thread">
          {turns.map((t, idx) => (
            <article key={t.id}>
              <div className="q">{t.question}</div>
              <div className="a">
                {t.error && (
                  <div className="note err">
                    <b>Request failed</b>
                    {t.error}
                  </div>
                )}
                {!t.response && !t.error && idx === turns.length - 1 && loading && (
                  <div className="loading">Searching sources…</div>
                )}
                {t.response?.status === "answered" && (
                  <>
                    <Answer text={t.response.answer} messageId={t.id} />
                    <Sources sources={t.response.sources} messageId={t.id} />
                  </>
                )}
                {t.response?.status === "insufficient_information" && (
                  <div className="note">
                    <b>Not enough information</b>
                    {t.response.answer}
                  </div>
                )}
                {t.response?.status === "refused" && (
                  <div className="note">
                    <b>Outside what I can answer</b>
                    {t.response.answer}
                  </div>
                )}
                {t.response && <div className="disclaimer">{t.response.disclaimer}</div>}
              </div>
            </article>
          ))}
        </div>
        <div ref={endRef} />
      </main>

      <div className="composer">
        <form onSubmit={onSubmit}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about a protocol, proposal or whitepaper"
            aria-label="Your question"
            maxLength={1000}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Ask
          </button>
        </form>
      </div>
    </>
  );
}
