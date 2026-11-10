export type SourceType = "news" | "governance" | "whitepaper";
export type Status = "answered" | "insufficient_information" | "refused";

export interface Source {
  id: number;
  title: string;
  url: string;
  source_type: SourceType;
  source_name: string;
  published_at: string;
  snippet: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  status: Status;
  sources: Source[];
  disclaimer: string;
  latency_ms: Record<string, number>;
}

export interface HistoryItem {
  role: "user" | "assistant";
  content: string;
}

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function ask(
  query: string,
  history: HistoryItem[],
  sourceTypes: SourceType[],
  days: number | null
): Promise<ChatResponse> {
  const res = await fetch(`${API}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      history,
      source_types: sourceTypes.length ? sourceTypes : null,
      days,
    }),
  });
  if (!res.ok) {
    throw new Error(`The backend returned ${res.status}. Check that the API is running at ${API}.`);
  }
  return res.json();
}
