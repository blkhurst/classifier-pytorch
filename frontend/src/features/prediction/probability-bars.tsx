import type { Prediction } from "@/lib/types";

export function ProbabilityBars({
  result,
  busy,
  epoch,
}: {
  result: Prediction | null;
  busy: boolean;
  epoch: number;
}) {
  const winner = result?.digit;
  const trainedEpochs = result?.trained_epochs ?? epoch;

  return (
    <div aria-busy={busy}>
      <div className="flex min-h-7 items-center justify-between gap-4">
        <h2 className="section-title">Probabilities</h2>
        <span className="text-xs text-copy-secondary">
          {trainedEpochs ? `Epoch ${trainedEpochs}` : "Untrained"}
        </span>
      </div>
      <div className="mt-6 grid gap-3" aria-label="Digit probabilities">
        {Array.from({ length: 10 }, (_, digit) => {
          const probability = (result?.probabilities[digit] ?? 0) * 100;
          return (
            <div
              key={digit}
              className="grid grid-cols-[1rem_1fr_3.5rem] items-center gap-3 text-xs tabular-nums"
            >
              <span
                className={
                  winner === digit
                    ? "font-semibold text-accent"
                    : "text-copy-secondary"
                }
              >
                {digit}
              </span>
              <div
                className="h-2.5 overflow-hidden rounded-full bg-surface-muted"
                role="progressbar"
                aria-label={`Digit ${digit}`}
                aria-valuenow={probability}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className={`h-full rounded-full transition-[width,background-color] duration-300 ease-out motion-reduce:transition-none ${winner === digit ? "bg-accent" : "bg-bar-muted"}`}
                  style={{ width: `${probability}%` }}
                />
              </div>
              <span
                className={`text-right ${winner === digit ? "font-medium text-accent" : "text-copy-secondary"}`}
              >
                {probability.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
      <div
        className="mt-6 flex min-h-6 items-center justify-between border-t border-border-subtle pt-4 text-xs text-copy-secondary"
        aria-live="polite"
      >
        <span>
          {busy ? (
            "Predicting…"
          ) : winner != null ? (
            <>
              Prediction{" "}
              <strong className="ml-1 text-sm text-copy-primary">
                {winner}
              </strong>
            </>
          ) : (
            "Prediction —"
          )}
        </span>
        {result?.label !== undefined && (
          <span>Sample label {result.label}</span>
        )}
      </div>
    </div>
  );
}
