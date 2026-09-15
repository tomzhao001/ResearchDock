function resolveApiUrl(input: string): string {
  if (/^https?:\/\//.test(input) || input === "/api" || input.startsWith("/api/")) {
    return input;
  }
  const path = input.startsWith("/") ? input : `/${input}`;
  return `/api${path}`;
}

export async function apiFetch(input: string, init?: RequestInit): Promise<Response> {
  return fetch(resolveApiUrl(input), {
    ...init,
    credentials: "include",
  });
}

export async function readApiErrorMessage(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as
    | { detail?: string | { message?: string } }
    | null;
  if (typeof body?.detail === "string" && body.detail) {
    return body.detail;
  }
  if (typeof body?.detail === "object" && body.detail?.message) {
    return body.detail.message;
  }
  return `请求失败 (${response.status})`;
}

export async function apiJson<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await apiFetch(input, init);
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response));
  }
  return (await response.json()) as T;
}

export function websocketUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${protocol}//${window.location.host}${normalized}`;
}
