import { useState } from "react";
import { SiteHeader } from "@/components/site-header";
import { DrawingPad } from "@/features/prediction/drawing-pad";
import { ProbabilityBars } from "@/features/prediction/probability-bars";
import { usePrediction } from "@/features/prediction/use-prediction";
import { LossChart } from "@/features/training/loss-chart";
import { TrainingForm } from "@/features/training/training-form";
import { useSession } from "@/hooks/use-session";
import type { PredictionInput } from "@/lib/types";

export function App() {
  const { session, connected, pending, error, start, stop } = useSession();
  const [input, setInput] = useState<PredictionInput>(null);
  const prediction = usePrediction(
    input,
    session?.model_version ?? 0,
    connected,
  );
  const trainingError = error ?? session?.error;

  return (
    <>
      <SiteHeader connected={connected} />
      <main className="mx-auto max-w-5xl space-y-8 px-6 py-8 sm:px-8 sm:py-10">
        <section
          aria-label="Prediction"
          className="section-grid px-1 pb-2 sm:px-6"
        >
          <DrawingPad
            input={input}
            result={prediction.result}
            connected={connected}
            onInput={setInput}
          />
          <div>
            <ProbabilityBars
              result={prediction.result}
              busy={prediction.busy}
              epoch={session?.prediction_epoch ?? 0}
            />
            {prediction.error && (
              <p role="alert" className="mt-4 text-sm text-danger">
                {prediction.error}
              </p>
            )}
          </div>
        </section>
        <section
          aria-label="Training"
          className="grid overflow-hidden rounded-2xl border border-border bg-surface shadow-sm md:grid-cols-[328px_minmax(0,1fr)]"
        >
          <div className="border-b border-border bg-surface-muted/40 p-6 md:border-r md:border-b-0">
            {session ? (
              <TrainingForm
                session={session}
                disabled={!connected || pending}
                onStart={start}
                onStop={stop}
              />
            ) : (
              <div className="min-h-64">
                <h2 className="section-title">Train model</h2>
                <p className="mt-6 text-sm text-copy-secondary">
                  Waiting for the API…
                </p>
              </div>
            )}
            {trainingError && (
              <p role="alert" className="mt-4 text-sm text-danger">
                {trainingError}
              </p>
            )}
          </div>
          <div className="min-w-0 p-6">
            <LossChart history={session?.history ?? []} />
          </div>
        </section>
      </main>
    </>
  );
}
