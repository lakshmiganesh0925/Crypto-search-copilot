import { Fragment } from "react";

const CITE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/** Renders answer text; [1], [2][3] and [1, 3] become links to the matching source row. */
export default function Answer({ text, messageId }: { text: string; messageId: number }) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  const re = new RegExp(CITE);
  while ((m = re.exec(text)) !== null) {
    parts.push(text.slice(last, m.index));
    m[1].split(",").forEach((n, i) => {
      const id = n.trim();
      parts.push(
        <a key={`${m!.index}-${i}`} className="cite" href={`#src-${messageId}-${id}`} aria-label={`Source ${id}`}>
          {id}
        </a>
      );
    });
    last = m.index + m[0].length;
  }
  parts.push(text.slice(last));
  return (
    <div className="answer">
      {parts.map((p, i) => (
        <Fragment key={i}>{p}</Fragment>
      ))}
    </div>
  );
}
