import torch


def resolve_device(name: str = "auto") -> torch.device:
    """Choose a device for inference or training."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return torch.device(name)
