from pathlib import Path
from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = Path("data")


def download(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir).expanduser()
    MNIST(root=data_dir, train=True, download=True)
    MNIST(root=data_dir, train=False, download=True)


def explore(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    """Display the first six training images."""
    from .plotting import show_samples

    data_dir = Path(data_dir).expanduser()
    try:
        dataset = MNIST(root=data_dir, train=True, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            f"MNIST training data was not found in {data_dir}. "
            "Run 'classifier data download' first."
        ) from error

    show_samples(dataset)


def load_sample(index: int, data_dir: str | Path = DEFAULT_DATA_DIR):
    """Return one training image as [1, 28, 28] floats and its correct label."""
    from torchvision.transforms.functional import to_tensor

    try:
        dataset = MNIST(root=Path(data_dir).expanduser(), train=True, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            "MNIST training data is missing. Run 'classifier data download' first."
        ) from error
    if not 0 <= index < len(dataset):
        raise ValueError(f"Sample index must be between 0 and {len(dataset) - 1}.")
    image, label = dataset[index]
    return to_tensor(image), label
