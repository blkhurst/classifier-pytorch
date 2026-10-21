from pathlib import Path
from torchvision.datasets import MNIST

DEFAULT_DATA_DIR = Path("data")


def download(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir).expanduser()
    MNIST(root=data_dir, train=True, download=True)
    MNIST(root=data_dir, train=False, download=True)


def explore(data_dir: str | Path = DEFAULT_DATA_DIR) -> None:
    """Display the first six training images."""
    import matplotlib.pyplot as plt

    data_dir = Path(data_dir).expanduser()
    try:
        dataset = MNIST(root=data_dir, train=True, download=False)
    except RuntimeError as error:
        raise FileNotFoundError(
            f"MNIST training data was not found in {data_dir}. "
            "Run 'classifier data download' first."
        ) from error

    figure, axes = plt.subplots(2, 3, figsize=(9, 6))
    figure.suptitle("MNIST training examples")

    for index, axis in enumerate(axes.flat):
        image, label = dataset[index]
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"Label: {label}")
        axis.axis("off")

    figure.tight_layout()
    plt.show()
