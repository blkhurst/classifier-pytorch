import matplotlib.pyplot as plt

from .inference import Prediction


def show_samples(dataset) -> None:
    """Display the first six MNIST images and their labels."""
    figure, axes = plt.subplots(2, 3, figsize=(9, 6))
    figure.suptitle("MNIST training examples")
    for index, axis in enumerate(axes.flat):
        image, label = dataset[index]
        axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axis.set_title(f"Label: {label}")
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def show_prediction(
    image, result: Prediction, label: int, *, title: str = "Prediction"
) -> None:
    figure, (image_axis, score_axis) = plt.subplots(1, 2, figsize=(9, 4))
    figure.suptitle(f"{title} | Predicted digit: {result.digit}")
    image_axis.imshow(image.detach().cpu()[0], cmap="gray", vmin=0, vmax=1)
    image_axis.set_title(f"True label: {label}")
    image_axis.axis("off")
    score_axis.bar(range(10), result.probabilities.numpy())
    score_axis.set_xticks(range(10))
    score_axis.set_ylim(0, 1)
    score_axis.set_xlabel("Digit")
    score_axis.set_ylabel("Softmax probability")
    figure.tight_layout()
    plt.show()
