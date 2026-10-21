from pathlib import Path
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_tensor

DEFAULT_DATA_DIR = Path("data")


def _load_dataset(data_dir: str | Path, transform=None) -> MNIST:
    data_dir = Path(data_dir).expanduser()
    try:
        return MNIST(root=data_dir, train=True, transform=transform, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            f"MNIST training data was not found in {data_dir}. "
            "Run 'classifier data download' first."
        ) from error


def download(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir).expanduser()
    MNIST(root=data_dir, train=True, download=True)
    MNIST(root=data_dir, train=False, download=True)


def explore(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    """Display the first six training images."""
    from .plotting import show_samples

    dataset = _load_dataset(data_dir)
    show_samples(dataset)


def load_sample(index: int, data_dir: str | Path = DEFAULT_DATA_DIR):
    """Return one training image as [1, 28, 28] floats and its correct label."""
    dataset = _load_dataset(data_dir)

    if not 0 <= index < len(dataset):
        raise ValueError(f"Sample index must be between 0 and {len(dataset) - 1}.")

    image, label = dataset[index]
    return to_tensor(image), label


def load_batches(
    batch_size: int,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    validation_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Split MNIST training data into training and validation loaders."""
    dataset = _load_dataset(data_dir, transform=ToTensor())

    training_size = len(dataset) - validation_size
    training, validation = random_split(dataset, [training_size, validation_size])

    training_loader = DataLoader(training, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader
