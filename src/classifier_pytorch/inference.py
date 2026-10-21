from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Prediction:
    digit: int
    logits: torch.Tensor
    probabilities: torch.Tensor


@torch.inference_mode()
def predict(model: nn.Module, image: torch.Tensor) -> Prediction:
    """Predict one image shaped [channels, height, width]."""
    if image.ndim != 3:
        raise ValueError("Expected one image shaped [channels, height, width].")

    parameter = next(model.parameters())
    inputs = image.unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
    model.eval()
    logits = model(inputs)[0].float().cpu()
    probabilities = logits.softmax(dim=0)
    return Prediction(int(logits.argmax()), logits, probabilities)
