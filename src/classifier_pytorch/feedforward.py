from dataclasses import dataclass
from torch import nn


@dataclass(frozen=True)
class FeedForwardConfig:
    input_size: int = 784
    hidden_units: int = 128
    output_size: int = 10


class FeedForward(nn.Module):
    """A feed-forward network with one ReLU hidden layer."""

    def __init__(self, config: FeedForwardConfig):
        super().__init__()

        self.config = config
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.input_size, config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units, config.output_size),
        )

    def forward(self, x):
        return self.layers(x)
