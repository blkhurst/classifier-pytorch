from dataclasses import dataclass
from collections.abc import Callable
import math
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .evaluate import evaluate


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    validation_size: int = 10_000
    seed: int = 42

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Weight decay must be finite and nonnegative.")
        if self.epochs <= 0 or self.batch_size <= 0 or self.validation_size <= 0:
            raise ValueError(
                "Epochs, batch size, and validation size must be positive."
            )
        if not 0 <= self.seed < 2**63:
            raise ValueError("Seed must be between 0 and 2**63 - 1.")


@dataclass
class EpochMetrics:
    epoch: int
    training_loss: float
    training_accuracy: float
    validation_loss: float
    validation_accuracy: float


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_function: nn.Module,
) -> float:
    """Train one epoch in place and return the average loss per sample."""
    model.train()
    parameter = next(model.parameters())

    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        labels = labels.to(parameter.device)

        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # Compute predictions & calculate loss
        logits = model(images)
        loss = loss_function(logits, labels)

        # Backpropagate the loss to compute gradients for weights and biases.
        loss.backward()

        # Update weights and biases using those gradients.
        optimizer.step()

        # Weight by batch size; smaller final batch is counted correctly.
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The training loader contains no samples.")

    return total_loss / total_samples


def train(
    model: nn.Module,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> list[EpochMetrics]:
    """Train in place and report metrics."""

    # SGD Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Cross Entropy Loss
    loss_function = nn.CrossEntropyLoss()

    history = []
    for epoch in range(config.epochs + 1):
        if epoch > 0:
            train_epoch(model, training_loader, optimizer, loss_function)

        # Evaluate Training/Validation Loss & Accuracy
        training = evaluate(model, training_loader, loss_function)
        validation = evaluate(model, validation_loader, loss_function)
        metrics = EpochMetrics(
            epoch,
            training.loss,
            training.accuracy,
            validation.loss,
            validation.accuracy,
        )

        history.append(metrics)

        # Callback
        if on_epoch is not None:
            on_epoch(metrics)

    return history
