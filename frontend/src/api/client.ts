import type { ApiEnvelope } from './types.ts';

export class ApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function parseJson<T>(r: Response, path: string): Promise<ApiEnvelope<T>> {
  const text = await r.text();
  let j: ApiEnvelope<T>;
  try {
    j = JSON.parse(text) as ApiEnvelope<T>;
  } catch {
    const snippet = String(text || r.statusText || '').slice(0, 120);
    throw new ApiError(r.ok ? `Invalid JSON from ${path}` : `${r.status} ${path}: ${snippet}`);
  }
  if (!j.ok) {
    throw new ApiError(
      (j as { error?: { message?: string }; detail?: string }).error?.message ||
        (j as { detail?: string }).detail ||
        r.statusText ||
        'API error',
    );
  }
  return j;
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number | boolean | undefined | null>,
): Promise<{ data: T; meta?: Record<string, unknown> }> {
  const url = new URL(path, window.location.origin);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v == null || v === '') continue;
      url.searchParams.set(k, String(v));
    }
  }
  const r = await fetch(url.toString());
  const j = await parseJson<T>(r, path);
  return { data: j.data, meta: j.meta };
}

export function apiQuery(params: Record<string, string | boolean | undefined | null>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue;
    q.set(k, String(v));
  }
  return q.toString();
}
