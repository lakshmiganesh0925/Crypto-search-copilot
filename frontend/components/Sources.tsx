import type { Source } from "@/lib/api";

const LABEL = { news: "News", governance: "Governance", whitepaper: "Whitepaper" } as const;

function fmt(iso: string) {
  const d = new Date(iso);
  return isNaN(d.getTime())
    ? iso
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export default function Sources({ sources, messageId }: { sources: Source[]; messageId: number }) {
  if (!sources.length) return null;
  return (
    <ol className="sources" aria-label="Sources">
      {sources.map((s) => (
        <li key={s.id} id={`src-${messageId}-${s.id}`} className={`src t-${s.source_type}`}>
          <span className="src-n">{s.id}</span>
          <div>
            <a href={s.url} target="_blank" rel="noopener noreferrer" className="src-title">
              {s.title}
            </a>
            <div className="src-meta">
              <span className={`tag t-${s.source_type}`}>{LABEL[s.source_type]}</span>
              {s.source_name}, {fmt(s.published_at)}
            </div>
            <p className="src-snip">{s.snippet}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}
