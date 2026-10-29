from pydantic import BaseModel, ConfigDict, Field

from classifier_pytorch.training import TrainingConfig


# Expose Only The Settings Used By The Interface
_defaults = TrainingConfig()


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    epochs: int = Field(default=_defaults.epochs, ge=1, le=100)
    batch_size: int = Field(default=_defaults.batch_size, ge=1, le=1024)
    learning_rate: float = Field(default=_defaults.learning_rate, gt=0, le=1)
    seed: int = Field(default=_defaults.seed, ge=0, le=2**32 - 1)


class DrawingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(max_length=500_000)
