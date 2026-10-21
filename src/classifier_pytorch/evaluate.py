from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class Evaluation:
    loss: float
    accuracy: float


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_function: nn.Module
) -> Evaluation:
    """Measure loss and accuracy without updating weights."""
    model.eval()
    parameter = next(model.parameters())

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device=parameter.device, dtype=parameter.dtype)
        labels = labels.to(parameter.device)

        # Compute predictions & calculate loss
        logits = model(images)
        loss = loss_function(logits, labels)

        # Weight by batch size; smaller final batch is counted correctly.
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The evaluation loader contains no examples.")

    return Evaluation(total_loss / total_samples, total_correct / total_samples)
