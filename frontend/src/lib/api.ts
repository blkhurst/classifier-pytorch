import type { Prediction, Session, TrainingConfig } from "@/lib/types";

export const api = {
  status: (signal?: AbortSignal) => request<Session>("/status", { signal }),
  train: (config: TrainingConfig) =>
    request<Session>("/train", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  stop: () => request<Session>("/train/stop", { method: "POST" }),
  predict: (image: string, signal: AbortSignal) =>
    request<Prediction>("/predict", {
      method: "POST",
      body: JSON.stringify({ image }),
      signal,
    }),
  sample: (index: number, signal: AbortSignal) =>
    request<Prediction>(`/samples/${index}`, { signal }),
};

// Helpers
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(10_000)])
    : AbortSignal.timeout(10_000);
  const response = await fetch(`/api${path}`, {
    ...options,
    signal,
    headers: { "Content-Type": "application/json" },
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(
      typeof error?.detail === "string"
        ? error.detail
        : `Request failed (${response.status}). Check your settings and API.`,
    );
  }
  return response.json() as Promise<T>;
}
