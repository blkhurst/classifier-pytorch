from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from .feedforward import FeedForward, FeedForwardConfig
from .training import TrainingConfig


@dataclass
class CheckpointMetadata:
    model_config: FeedForwardConfig
    training_config: TrainingConfig

    completed_epochs: int = 0
    model_type: str = "feedforward"
    activation: str = "relu"
    format_version: str = "1"

    def __post_init__(self):
        if not isinstance(self.training_config, TrainingConfig):
            raise ValueError("A TrainingConfig is required for every checkpoint.")
        if self.format_version != "1":
            raise ValueError("Unsupported checkpoint format version.")
        if self.model_type != "feedforward":
            raise ValueError(f"Unsupported checkpoint model type: {self.model_type!r}.")
        if self.activation != "relu":
            raise ValueError("Unsupported checkpoint activation; expected relu.")
        if type(self.completed_epochs) is not int or self.completed_epochs < 0:
            raise ValueError("Completed epochs must be a nonnegative integer.")
        sizes = asdict(self.model_config).values()
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValueError("Layer sizes must be positive integers.")

    def to_metadata(self) -> dict[str, str]:
        return {
            name: value if isinstance(value, str) else json.dumps(value)
            for name, value in asdict(self).items()
        }

    @staticmethod
    def from_metadata(values: dict[str, str]) -> "CheckpointMetadata":
        try:
            model_config = json.loads(values["model_config"])
            if set(model_config) != {field.name for field in fields(FeedForwardConfig)}:
                raise ValueError("Missing or unexpected model_config.")

            training_config = json.loads(values["training_config"])
            if not isinstance(training_config, dict):
                raise ValueError("training_config must contain training settings.")

            metadata = CheckpointMetadata(
                model_config=FeedForwardConfig(**model_config),
                training_config=TrainingConfig(**training_config),
                completed_epochs=int(
                    values["completed_epochs"]
                    if "completed_epochs" in values
                    else values["epochs"]
                ),
                model_type=values["model_type"],
                activation=values["activation"],
                format_version=values["format_version"],
            )

            return metadata
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid checkpoint metadata: {error}") from error


@dataclass
class Checkpoint:
    model: FeedForward
    metadata: CheckpointMetadata

    @property
    def epochs(self) -> int:
        return self.metadata.completed_epochs


def save_checkpoint(
    model: FeedForward,
    path: str | Path,
    *,
    training_config: TrainingConfig,
    epochs: int = 0,
) -> None:
    # Check Model & Config
    if not isinstance(model, FeedForward):
        raise ValueError("Only FeedForward checkpoints are supported.")

    # Prepare Checkpoint Metadata
    metadata = CheckpointMetadata(model.config, training_config, epochs)

    # Collect Weights On CPU
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }

    # Save First, Then Replace The Checkpoint
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        save_file(tensors, temporary, metadata=metadata.to_metadata())
        temporary.replace(path)


def load_checkpoint(path: str | Path) -> Checkpoint:
    # Read Weights & Metadata
    path = Path(path).expanduser()
    try:
        with safe_open(path, framework="pt", device="cpu") as file:
            values = file.metadata() or {}
            tensors = {name: file.get_tensor(name) for name in file.keys()}
    except SafetensorError as error:
        raise ValueError(f"Could not read safetensors checkpoint: {error}") from error

    # Restore Checkpoint Metadata
    metadata = CheckpointMetadata.from_metadata(values)

    # Rebuild Model
    model = FeedForward(metadata.model_config)

    # Restore Weights
    model.load_state_dict(tensors, strict=True, assign=True)
    model.eval()
    return Checkpoint(model, metadata)


def inspect_checkpoint(path: str | Path) -> dict:
    path = Path(path).expanduser()
    try:
        with safe_open(path, framework="pt", device="cpu") as file:
            # Metadata & Tensor Details
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
