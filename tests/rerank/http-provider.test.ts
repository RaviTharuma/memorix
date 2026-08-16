/**
 * HTTP neural rerank provider tests.
 *
 * Remote-only: OmniRoute Cohere-compatible POST /v1/rerank.
 * Memorix must never call api.jina.ai or load a local model.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  JINA_RERANKER_MODEL,
  buildRerankUrl,
  isForbiddenRerankHost,
  parseRerankResponse,
  rerankViaHttp,
} from '../../src/rerank/http-provider.js';

const OMNIROUTE_BASE = 'https://omniroute.jaguar-fish.ts.net/v1';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

describe('buildRerankUrl', () => {
  it('appends /rerank to an OmniRoute v1 base URL', () => {
    expect(buildRerankUrl(OMNIROUTE_BASE)).toBe(`${OMNIROUTE_BASE}/rerank`);
    expect(buildRerankUrl('https://omniroute.example/v1/')).toBe('https://omniroute.example/v1/rerank');
  });

  it('keeps an explicit /rerank path', () => {
    expect(buildRerankUrl('http://omniroute.local:8080/rerank')).toBe('http://omniroute.local:8080/rerank');
  });
});

describe('isForbiddenRerankHost', () => {
  it('rejects Jina hosts so Memorix cannot bypass OmniRoute', () => {
    expect(isForbiddenRerankHost('https://api.jina.ai/v1')).toBe(true);
    expect(isForbiddenRerankHost('https://api.jina.ai/v1/rerank')).toBe(true);
    expect(isForbiddenRerankHost('https://jina.ai')).toBe(true);
  });

  it('allows OmniRoute and other non-Jina gateways', () => {
    expect(isForbiddenRerankHost(OMNIROUTE_BASE)).toBe(false);
    expect(isForbiddenRerankHost('http://omniroute.omniroute.svc.cluster.local/v1')).toBe(false);
  });
});

describe('parseRerankResponse', () => {
  it('reads Cohere results[].relevance_score', () => {
    const ranked = parseRerankResponse({
      results: [
        { index: 2, relevance_score: 0.91 },
        { index: 0, relevance_score: 0.4 },
      ],
    });
    expect(ranked).toEqual([
      { index: 2, score: 0.91 },
      { index: 0, score: 0.4 },
    ]);
  });

  it('reads TEI [{index, score}] arrays', () => {
    const ranked = parseRerankResponse([
      { index: 1, score: 0.8 },
      { index: 0, score: 0.2 },
    ]);
    expect(ranked).toEqual([
      { index: 1, score: 0.8 },
      { index: 0, score: 0.2 },
    ]);
  });

  it('rejects empty or malformed payloads', () => {
    expect(() => parseRerankResponse({})).toThrow(/rerank/i);
    expect(() => parseRerankResponse({ results: [] })).toThrow(/rerank/i);
  });
});

describe('rerankViaHttp', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('posts the Cohere wire format to OmniRoute /v1/rerank', async () => {
    const fetchImpl = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(String(input)).toBe(`${OMNIROUTE_BASE}/rerank`);
      expect(init?.method).toBe('POST');
      const headers = new Headers(init?.headers);
      expect(headers.get('authorization')).toBe('Bearer omniroute-key');
      const body = JSON.parse(String(init?.body));
      expect(body).toEqual({
        model: JINA_RERANKER_MODEL,
        query: 'JWT expiry',
        documents: ['gotcha about tokens', 'unrelated database note'],
        top_n: 2,
        return_documents: false,
      });
      return jsonResponse({
        model: JINA_RERANKER_MODEL,
        results: [
          { index: 0, relevance_score: 0.88 },
          { index: 1, relevance_score: 0.11 },
        ],
      });
    });

    const ranked = await rerankViaHttp(
      'JWT expiry',
      ['gotcha about tokens', 'unrelated database note'],
      {
        provider: 'http',
        model: JINA_RERANKER_MODEL,
        baseUrl: OMNIROUTE_BASE,
        apiKey: 'omniroute-key',
        fetchImpl,
      },
    );

    expect(ranked.map((row) => row.index)).toEqual([0, 1]);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it('refuses to call api.jina.ai even if a caller passes that URL', async () => {
    const fetchImpl = vi.fn();
    await expect(rerankViaHttp('q', ['a'], {
      provider: 'http',
      baseUrl: 'https://api.jina.ai/v1',
      apiKey: 'jina-key',
      fetchImpl,
    })).rejects.toThrow(/OmniRoute/i);
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it('omits Authorization when no API key is configured', async () => {
    const fetchImpl = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const headers = new Headers(init?.headers);
      expect(headers.get('authorization')).toBeNull();
      return jsonResponse({ results: [{ index: 0, relevance_score: 1 }] });
    });

    await rerankViaHttp('q', ['only'], {
      provider: 'http',
      baseUrl: 'http://omniroute.omniroute.svc.cluster.local/v1',
      fetchImpl,
    });
  });

  it('throws on HTTP errors so the caller can fall back to LLM rerank', async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ error: 'nope' }, 502));

    await expect(rerankViaHttp('q', ['a', 'b'], {
      provider: 'http',
      baseUrl: OMNIROUTE_BASE,
      apiKey: 'k',
      fetchImpl,
    })).rejects.toThrow(/502/);
  });
});
