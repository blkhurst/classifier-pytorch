import { useState, type FormEvent } from "react";
import { Button } from "@/components/button";
import type { Session, TrainingConfig } from "@/lib/types";

type TrainingFormProps = {
  session: Session;
  disabled: boolean;
  onStart: (config: TrainingConfig) => Promise<void>;
  onStop: () => Promise<void>;
};

export function TrainingForm({
  session,
  disabled,
  onStart,
  onStop,
}: TrainingFormProps) {
  const [config, setConfig] = useState<TrainingConfig>(() => ({
    epochs: session.config.epochs,
    batch_size: session.config.batch_size,
    learning_rate: session.config.learning_rate,
    seed: session.config.seed,
  }));
  const running = session.phase === "training" || session.phase === "stopping";
  const epoch = session.history.at(-1)?.epoch ?? 0;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void onStart(config);
  }

  return (
    <form onSubmit={submit}>
      <h2 className="section-title">Train model</h2>
      <fieldset disabled={disabled || running} className="mt-5">
        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Epochs"
            value={config.epochs}
            min={1}
            max={100}
            onChange={(epochs) => setConfig({ ...config, epochs })}
          />
          <NumberField
            label="Batch size"
            value={config.batch_size}
            min={1}
            max={1024}
            onChange={(batch_size) => setConfig({ ...config, batch_size })}
          />
        </div>
        <details className="group mt-4">
          <summary className="flex cursor-pointer list-none items-center justify-between rounded py-1 text-xs text-copy-secondary transition-colors hover:text-copy-primary focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent [&::-webkit-details-marker]:hidden">
            More settings
            <svg
              aria-hidden="true"
              viewBox="0 0 16 16"
              fill="none"
              className="size-4 transition-transform group-open:rotate-180 motion-reduce:transition-none"
            >
              <path
                d="m4 6 4 4 4-4"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </summary>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <NumberField
              label="Learning rate"
              value={config.learning_rate}
              min={0.000001}
              max={1}
              step="any"
              onChange={(learning_rate) =>
                setConfig({ ...config, learning_rate })
              }
            />
            <NumberField
              label="Seed"
              value={config.seed}
              min={0}
              max={4294967295}
              onChange={(seed) => setConfig({ ...config, seed })}
            />
          </div>
        </details>
      </fieldset>
      <div className="mt-5 flex flex-wrap items-center gap-3">
        {running ? (
          <Button
            variant="secondary"
            className="w-full"
            disabled={disabled || session.phase === "stopping"}
            onClick={() => void onStop()}
          >
            {session.phase === "stopping" ? "Stopping…" : "Stop after epoch"}
          </Button>
        ) : (
          <Button
            type="submit"
            className="w-full"
            disabled={disabled}
            title="Start a new model from random weights"
          >
            {session.history.length ? "Train again" : "Train model"}
          </Button>
        )}
        <span
          className="text-xs tabular-nums text-copy-secondary"
          role="status"
        >
          {running
            ? `Epoch ${epoch} / ${session.config.epochs}`
            : session.phase === "completed"
              ? "Complete"
              : session.phase === "stopped"
                ? "Stopped"
                : ""}
        </span>
      </div>
    </form>
  );
}

// Helpers
function NumberField({
  label,
  value,
  onChange,
  ...limits
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: string;
}) {
  return (
    <label className="block text-xs text-copy-secondary">
      {label}
      <input
        className="mt-1.5 h-9 w-full min-w-0 rounded-lg border border-border bg-surface px-2.5 text-xs tabular-nums text-copy-primary outline-none transition-colors focus:border-accent focus:ring-2 focus:ring-accent-soft disabled:bg-background disabled:text-copy-muted"
        type="number"
        required
        value={Number.isNaN(value) ? "" : value}
        onChange={(event) => onChange(event.target.valueAsNumber)}
        {...limits}
      />
    </label>
  );
}
