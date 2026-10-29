"""Keep one model in memory and train it."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock, Thread
import math

import torch

from classifier_pytorch.device import resolve_device
from classifier_pytorch.feedforward import FeedForward, FeedForwardConfig
from classifier_pytorch.inference import predict
from classifier_pytorch.mnist import DEFAULT_DATA_DIR, load_batches
from classifier_pytorch.training import EpochMetrics, TrainingConfig, train


class TrainingSession:
    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.device = resolve_device()
        self.model = FeedForward(FeedForwardConfig()).eval()
        self.config = TrainingConfig()
        self.history = []
        self.phase = "idle"
        self.error = None
        self.version = 0
        self.prediction_epoch = 0
        self.lock = Lock()
        self.stop = Event()
        self.thread = None
        self.closed = False

    def status(self):
        with self.lock:
            return {
                "phase": self.phase,
                "error": self.error,
                "device": str(self.device),
                "config": asdict(self.config),
                "history": list(self.history),
                "model_version": self.version,
                "prediction_epoch": self.prediction_epoch,
            }

    def prediction(self, image):
        # Read A Stable Snapshot While The Worker Trains Its Own Model
        with self.lock:
            model, version, epoch = self.model, self.version, self.prediction_epoch
        result = predict(model, image) if bool(image.any()) else None
        return {
            "digit": result.digit if result else None,
            "probabilities": result.probabilities.tolist() if result else [0.0] * 10,
            "pixels": image.flatten().tolist(),
            "model_version": version,
            "trained_epochs": epoch,
        }

    def start(self, config: TrainingConfig):
        with self.lock:
            if self.closed:
                raise RuntimeError("The training session is shutting down.")
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError("Training is already running.")
            self.config = config
            self.history = []
            self.phase, self.error = "training", None
            self.stop.clear()
            self.thread = Thread(target=self._run, args=(config,), daemon=True)
            self.thread.start()
        return self.status()

    def request_stop(self):
        with self.lock:
            if self.phase == "training":
                self.phase = "stopping"
                self.stop.set()
        return self.status()

    def close(self):
        with self.lock:
            self.closed = True
            self.stop.set()
            thread = self.thread
        if thread is not None:
            thread.join()

    # Training Runs On One Background Thread
    def _run(self, config):
        try:
            torch.manual_seed(config.seed)
            training, validation = load_batches(
                config.batch_size,
                self.data_dir,
                validation_size=config.validation_size,
            )
            model = FeedForward(FeedForwardConfig()).to(self.device)

            # Publish The Fresh Model Without Evaluating Epoch Zero
            snapshot = deepcopy(model).cpu().eval()
            with self.lock:
                self.model = snapshot
                self.prediction_epoch = 0
                self.version += 1

            def report_epoch(metrics: EpochMetrics):
                if not all(
                    math.isfinite(value)
                    for value in (metrics.training_loss, metrics.validation_loss)
                ):
                    raise ValueError("Loss is not finite. Try a lower learning rate.")

                # Publish Metrics & A CPU Prediction Copy After Each Epoch
                snapshot = deepcopy(model).cpu().eval()
                with self.lock:
                    self.model = snapshot
                    self.prediction_epoch = metrics.epoch
                    self.version += 1
                    self.history.append(asdict(metrics))
                if self.stop.is_set():
                    raise _TrainingStopped()

            train(
                model, training, validation, config,
                on_epoch=report_epoch, evaluate_initial=False,
            )
            with self.lock:
                self.phase = "stopped" if self.stop.is_set() else "completed"
        except _TrainingStopped:
            with self.lock:
                self.phase = "stopped"
        except Exception as error:
            # Worker Errors Must Be Returned To The Interface
            with self.lock:
                self.phase, self.error = "error", str(error)


class _TrainingStopped(Exception):
    """End training at an epoch boundary without changing the library loop."""
