interface JsonRequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  sourceLabel?: string;
  method?: 'GET' | 'POST' | 'PATCH';
  body?: unknown;
}

const responseMessage = async (response: Response) => {
  const body = (await response.text()).trim();
  return body.length > 280 ? `${body.slice(0, 277)}...` : body;
};

export async function getJson<T>(url: string, options: JsonRequestOptions = {}): Promise<T> {
  return requestJson<T>(url, options);
}

export async function requestJson<T>(
  url: string,
  {
    signal,
    timeoutMs = 12_000,
    sourceLabel = 'Le service',
    method = 'GET',
    body,
  }: JsonRequestOptions = {},
): Promise<T> {
  if (signal?.aborted) throw new DOMException('Requête annulée', 'AbortError');

  const controller = new AbortController();
  const abortFromCaller = () => controller.abort();
  signal?.addEventListener('abort', abortFromCaller, { once: true });
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
      body: body == null ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await responseMessage(response);
      throw new Error(body || `${sourceLabel} a répondu avec le statut ${response.status}.`);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (signal?.aborted) throw new DOMException('Requête annulée', 'AbortError');
    if (controller.signal.aborted) {
      throw new Error(`${sourceLabel} n’a pas répondu dans le délai attendu.`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener('abort', abortFromCaller);
  }
}
