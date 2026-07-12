import type { HmiSnapshot } from './types';

const DEFAULT_API_BASE = 'http://127.0.0.1:8088';

function apiBase(): string {
  if (import.meta.env.VITE_HS_HMI_API_URL) {
    return import.meta.env.VITE_HS_HMI_API_URL;
  }
  return import.meta.env.DEV ? DEFAULT_API_BASE : '';
}

export async function fetchSnapshot(): Promise<HmiSnapshot> {
  const response = await fetch(`${apiBase()}/api/snapshot`);
  if (!response.ok) {
    throw new Error(await responseText(response, 'snapshot read failed'));
  }
  return (await response.json()) as HmiSnapshot;
}

export async function writeCommand(id: string, value: number): Promise<void> {
  await writeJson(`${apiBase()}/api/commands/${encodeURIComponent(id)}`, { value });
}

export async function writeCoil(id: string, enabled: boolean): Promise<void> {
  await writeJson(`${apiBase()}/api/coils/${encodeURIComponent(id)}`, { enabled });
}

async function writeJson(url: string, body: unknown): Promise<void> {
  const response = await fetch(url, {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await responseText(response, 'command write failed'));
  }
}

async function responseText(response: Response, fallback: string): Promise<string> {
  try {
    const payload = (await response.json()) as { error?: string };
    return payload.error ?? fallback;
  } catch {
    return fallback;
  }
}
