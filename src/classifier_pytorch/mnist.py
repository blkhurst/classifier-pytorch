from pathlib import Path
from dataclasses import dataclass
from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = Path("data")


def download(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir).expanduser()
    MNIST(root=data_dir, train=True, download=True)
    MNIST(root=data_dir, train=False, download=True)
