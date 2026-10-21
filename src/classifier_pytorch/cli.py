import argparse

from .device import resolve_device
from .feedforward import FeedForward, FeedForwardConfig
from .inference import predict
from .mnist import DEFAULT_DATA_DIR, download, explore, load_sample


def data_download(args: argparse.Namespace) -> None:
    print(f"Preparing MNIST in {DEFAULT_DATA_DIR.resolve()}...")
    try:
        download()
    except Exception as error:
        raise SystemExit(f"Could not prepare MNIST: {error}")
    print("MNIST training and test data are ready.")


def data_explore(args: argparse.Namespace) -> None:
    try:
        explore()
    except (OSError, RuntimeError) as error:
        raise SystemExit(f"Could not explore MNIST: {error}")


def predict_sample(args: argparse.Namespace) -> None:
    from .plotting import show_prediction

    image, label = load_sample(args.sample)
    device = resolve_device()
    model = FeedForward(FeedForwardConfig()).to(device)
    result = predict(model, image)

    print(f"Device: {device}")
    print(f"Predicted digit: {result.digit} | True label: {label}")
    show_prediction(image, result, label, title="Untrained model")


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST digit classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    # data
    data_parser = commands.add_parser("data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)

    # data download
    download_parser = data_commands.add_parser(
        "download", help="Download MNIST dataset."
    )
    download_parser.set_defaults(func=data_download)

    # data explore
    explore_parser = data_commands.add_parser(
        "explore", help="Display MNIST training images."
    )
    explore_parser.set_defaults(func=data_explore)

    # predict
    predict_parser = commands.add_parser("predict", help="Predict one MNIST sample.")
    predict_parser.add_argument(
        "--random",
        action="store_true",
        required=True,
        help="Explicitly use fresh, untrained weights.",
    )
    predict_parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="MNIST training example index (default: 0).",
    )
    predict_parser.set_defaults(func=predict_sample)

    # Parse arguments and execute
    args = parser.parse_args()
    args.func(args)
