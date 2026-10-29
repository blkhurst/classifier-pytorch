import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpochMetrics } from "@/lib/types";

export function LossChart({ history }: { history: EpochMetrics[] }) {
  const latest = history.at(-1);
  return (
    <div>
      <div className="flex min-h-7 items-center justify-between gap-4">
        <h2 className="section-title">Loss</h2>
        {latest && (
          <span className="text-xs tabular-nums text-copy-secondary">
            Validation accuracy {(latest.validation_accuracy * 100).toFixed(1)}%
          </span>
        )}
      </div>
      <div
        className="relative mt-5 h-52 w-full"
        role="img"
        aria-label="Training and validation loss by epoch"
      >
        <ResponsiveContainer width="100%" height="100%" minWidth={0}>
          <LineChart
            data={history}
            margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
          >
            <CartesianGrid
              stroke="var(--color-border)"
              vertical={false}
              strokeDasharray="3 3"
            />
            <XAxis
              dataKey="epoch"
              type="number"
              domain={[1, (maximum: number) => Math.max(2, maximum)]}
              allowDecimals={false}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: "var(--color-copy-secondary)" }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: "var(--color-copy-secondary)" }}
              width={50}
            />
            <Tooltip
              cursor={{ stroke: "var(--color-border)" }}
              labelFormatter={(value) => `Epoch ${value}`}
              formatter={(value) => Number(value).toFixed(4)}
              contentStyle={{
                border: "1px solid var(--color-border)",
                background: "var(--color-surface)",
                color: "var(--color-copy-primary)",
                borderRadius: 6,
                fontSize: 12,
              }}
            />
            <Line
              type="linear"
              dataKey="training_loss"
              name="Training"
              stroke="var(--color-chart-training)"
              strokeWidth={2.5}
              dot={history.length === 1}
              isAnimationActive={false}
            />
            <Line
              type="linear"
              dataKey="validation_loss"
              name="Validation"
              stroke="var(--color-chart-validation)"
              strokeWidth={2.5}
              dot={history.length === 1}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
        {!history.length && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center text-xs text-copy-muted">
            No epochs yet
          </div>
        )}
      </div>
      <div className="mt-3 flex justify-between text-xs text-copy-secondary">
        <div className="flex gap-4">
          <span className="flex items-center gap-2">
            <span className="h-0.5 w-3 bg-chart-training" />
            Training
          </span>
          <span className="flex items-center gap-2">
            <span className="h-0.5 w-3 bg-chart-validation" />
            Validation
          </span>
        </div>
        <span>Epoch</span>
      </div>
    </div>
  );
}
