/**
 * Remote neural rerank client.
 *
 * HTTP only — never loads jina-ai/jina-reranker-v3.5 (or any cross-encoder) in-process.
 * Speaks the same Cohere-compatible /rerank wire format Hindsight uses:
 *   client → OmniRoute POST /v1/rerank {model, query, documents} → Jina
 *
 * Memorix must never call api.jina.ai (or any jina.ai host). OmniRoute is
 * the only hop that talks to Jina.
 */

// OmniRoute rejects the bare id `jina-reranker-v3.5` (HTTP 400).
// Hindsight and Memorix send the catalog id; OmniRoute maps it to Jina.
export const JINA_RERANKER_MODEL = 'jina-ai/jina-reranker-v3.5';

export type NeuralRerankProviderName = 'http';

export interface NeuralRerankRequestConfig {
  provider: NeuralRerankProviderName;
  model?: string;
  baseUrl: string;
  apiKey?: string;
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export interface RerankedDocument {
  index: number;
  score: number;
}

/**
 * True when a URL would send rerank traffic to Jina instead of OmniRoute.
 */
export function isForbiddenRerankHost(value: string | undefined): boolean {
  if (!value) return false;
  try {
    const host = new URL(value).hostname.toLowerCase();
    return host === 'jina.ai' || host.endsWith('.jina.ai');
  } catch {
    return /(?:^|[/.])jina\.ai(?:[:/]|$)/i.test(value);
  }
}

/**
 * Join a provider base URL with `/rerank`, leaving an explicit path intact.
 */
export function buildRerankUrl(baseUrl: string): string {
  const trimmed = baseUrl.trim().replace(/\/+$/, '');
  if (trimmed.endsWith('/rerank')) return trimmed;
  return `${trimmed}/rerank`;
}

/**
 * Accept Cohere/Jina `{results:[{index,relevance_score}]}` and TEI `[{index,score}]`.
 */
export function parseRerankResponse(payload: unknown): RerankedDocument[] {
  const rows = extractRows(payload);
  if (rows.length === 0) {
    throw new Error('Neural rerank response contained no ranked results');
  }
  return rows.sort((a, b) => b.score - a.score);
}

function extractRows(payload: unknown): RerankedDocument[] {
  if (Array.isArray(payload)) {
    return payload.map(rowFromUnknown).filter((row): row is RerankedDocument => row !== null);
  }
  if (payload && typeof payload === 'object' && 'results' in payload) {
    const results = (payload as { results: unknown }).results;
    if (Array.isArray(results)) {
      return results.map(rowFromUnknown).filter((row): row is RerankedDocument => row !== null);
    }
  }
  throw new Error('Neural rerank response was not Cohere or TEI shaped');
}

function rowFromUnknown(value: unknown): RerankedDocument | null {
  if (!value || typeof value !== 'object') return null;
  const row = value as { index?: unknown; relevance_score?: unknown; score?: unknown };
  if (typeof row.index !== 'number' || !Number.isFinite(row.index)) return null;
  const score = typeof row.relevance_score === 'number'
    ? row.relevance_score
    : typeof row.score === 'number'
      ? row.score
      : null;
  if (score === null || !Number.isFinite(score)) return null;
  return { index: row.index, score };
}

/**
 * POST query + documents to OmniRoute (or another non-Jina gateway) and return ranked indexes.
 */
export async function rerankViaHttp(
  query: string,
  documents: string[],
  config: NeuralRerankRequestConfig,
): Promise<RerankedDocument[]> {
  if (documents.length === 0) return [];
  if (isForbiddenRerankHost(config.baseUrl)) {
    throw new Error('Neural rerank must go through OmniRoute, not a Jina URL');
  }

  const url = buildRerankUrl(config.baseUrl);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
  };
  if (config.apiKey) {
    headers.authorization = `Bearer ${config.apiKey}`;
  }

  const controller = new AbortController();
  const timeoutMs = config.timeoutMs && config.timeoutMs > 0 ? config.timeoutMs : 5000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const fetchImpl = config.fetchImpl ?? fetch;
    const response = await fetchImpl(url, {
      method: 'POST',
      headers,
      signal: controller.signal,
      body: JSON.stringify({
        model: config.model || JINA_RERANKER_MODEL,
        query,
        documents,
        top_n: documents.length,
        return_documents: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Neural rerank HTTP ${response.status} from ${url}`);
    }

    return parseRerankResponse(await response.json());
  } finally {
    clearTimeout(timer);
  }
}
