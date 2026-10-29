export interface TrainingConfig {
  epochs: number;
  batch_size: number;
  learning_rate: number;
  seed: number;
}

export interface EpochMetrics {
  epoch: number;
  training_loss: number;
  training_accuracy: number;
  validation_loss: number;
  validation_accuracy: number;
}

export interface Session {
  phase: "idle" | "training" | "stopping" | "completed" | "stopped" | "error";
  error: string | null;
  device: string;
  config: TrainingConfig;
  history: EpochMetrics[];
  model_version: number;
  prediction_epoch: number;
}

export type PredictionInput =
  { kind: "drawing"; image: string } | { kind: "sample"; index: number } | null;

export interface Prediction {
  digit: number | null;
  probabilities: number[];
  pixels: number[];
  model_version: number;
  trained_epochs: number;
  label?: number;
  sample?: number;
}
