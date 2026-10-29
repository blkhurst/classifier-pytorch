import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { Prediction, PredictionInput } from "@/lib/types";

export function usePrediction(
  input: PredictionInput,
  version: number,
  connected: boolean,
) {
  const [response, setResponse] = useState<{
    input: PredictionInput;
    version: number;
    result: Prediction | null;
    error: string | null;
  } | null>(null);

  useEffect(() => {
    if (!input || !connected) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const result =
          input.kind === "drawing"
            ? await api.predict(input.image, controller.signal)
            : await api.sample(input.index, controller.signal);
        if (!controller.signal.aborted)
          setResponse({ input, version, result, error: null });
      } catch (error) {
        if (!controller.signal.aborted) {
          setResponse({
            input,
            version,
            result: null,
            error:
              error instanceof Error ? error.message : "Prediction failed.",
          });
        }
      }
    }, 100);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [input, version, connected]);

  const current = input !== null && response?.input === input;
  return {
    result: current ? response.result : null,
    error: current ? response.error : null,
    busy:
      connected && input !== null && (!current || response.version !== version),
  };
}
