import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import type { Session, TrainingConfig } from "@/lib/types";

export function useSession() {
  const [session, setSession] = useState<Session | null>(null);
  const [connected, setConnected] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const revision = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      const current = revision.current;
      try {
        const result = await api.status(controller.signal);
        if (!controller.signal.aborted && current === revision.current) {
          setSession(result);
          setConnected(true);
        }
      } catch {
        if (!controller.signal.aborted) setConnected(false);
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(poll, 1000);
      }
    }

    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, []);

  async function update(action: () => Promise<Session>) {
    // Ignore Polls Started Before This Action Finishes.
    revision.current++;
    setPending(true);
    setError(null);
    try {
      setSession(await action());
      setConnected(true);
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "Could not update training.",
      );
    } finally {
      revision.current++;
      setPending(false);
    }
  }

  return {
    session,
    connected,
    pending,
    error,
    start: (config: TrainingConfig) => update(() => api.train(config)),
    stop: () => update(api.stop),
  };
}
