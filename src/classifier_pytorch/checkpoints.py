from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors import SafetensorError, safe_open

from .feedforward import FeedForward, FeedForwardConfig


@dataclass
class CheckpointMetadata:
    model_config: dict
    training_config: dict
    completed_epochs: int = 0
    format_version: int = 1


@dataclass
class Checkpoint:
    model: FeedForward
    metadata: CheckpointMetadata

    @property
    def epochs(self) -> int:
        return self.metadata.completed_epochs


def save_checkpoint(model, metadata: CheckpointMetadata, path: str | Path):
    # Collect Weights & Settings
    checkpoint = {
        "metadata": asdict(metadata),
        "model": model.state_dict(),
    }

    # Write Completely Before Replacing The Previous Checkpoint
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        torch.save(checkpoint, temporary)
        temporary.replace(path)


def load_checkpoint(path: str | Path) -> Checkpoint:
    # Read Weights & Metadata
    values = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    metadata = CheckpointMetadata(**values["metadata"])
    if metadata.format_version != 1:
        raise ValueError("Unsupported checkpoint format.")

    # Rebuild Model & Restore Weights
    model = FeedForward(FeedForwardConfig(**metadata.model_config))
    model.load_state_dict(values["model"], strict=True, assign=True)
    model.eval()
    return Checkpoint(model, metadata)


def inspect_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser()

    # Inspect A PyTorch Checkpoint
    if path.suffix != ".safetensors":
        values = torch.load(path, map_location="cpu", weights_only=True)
        tensors = {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(values["model"].items())
        }
        return {"metadata": values["metadata"], "tensors": tensors}

    # Inspect A Safetensor
    try:
        with safe_open(path, framework="pt", device="cpu") as file:
            tensors = {}
            for name in sorted(file.keys()):
                tensor = file.get_slice(name)
                tensors[name] = {
                    "shape": tensor.get_shape(),
                    "dtype": tensor.get_dtype(),
                }
            return {"metadata": file.metadata() or {}, "tensors": tensors}
    except SafetensorError as error:
        raise ValueError(f"Could not read safetensors checkpoint: {error}") from error
