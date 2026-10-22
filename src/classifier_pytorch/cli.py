import argparse
import torch

from .checkpoints import load_checkpoint, save_checkpoint
from .device import resolve_device
from .feedforward import FeedForward, FeedForwardConfig
from .inference import predict
from .mnist import DEFAULT_DATA_DIR, download, explore, load_sample, load_batches
from .training import EpochMetrics, TrainingConfig, train


def data_download(args: argparse.Namespace) -> None:
    print(f"Preparing MNIST in {DEFAULT_DATA_DIR.resolve()}...")
    download()
    print("MNIST training and test data are ready.")


def data_explore(args: argparse.Namespace) -> None:
    print("Displaying six MNIST training samples...")
    explore()


def predict_sample(args: argparse.Namespace) -> None:
    from .plotting import show_prediction

    # Load or Create Model
    if args.load:
        checkpoint = load_checkpoint(args.load)
        model, epochs = checkpoint.model, checkpoint.epochs
        print(f"Loaded checkpoint: {args.load} ({checkpoint.epochs} epochs)")
    else:
        model, epochs = FeedForward(FeedForwardConfig()), 0

    device = resolve_device()
    model = model.to(device)

    image, label = load_sample(args.sample)
    result = predict(model, image)

    print(f"Device: {device}")
    print(f"Predicted digit: {result.digit} | True label: {label}")
    show_prediction(
        image,
        result,
        label,
        title=(
            f"Checkpoint · {epochs} completed epochs"
            if args.load
            else "Untrained model"
        ),
    )


def train_model(args: argparse.Namespace) -> None:
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    # Resolve Device & Set Seed
    device = resolve_device()
    torch.manual_seed(config.seed)

    # Load Training & Validation Batches
    training_loader, validation_loader = load_batches(
        config.batch_size, validation_size=config.validation_size
    )

    # Load or Create Model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model, completed_epochs = checkpoint.model, checkpoint.epochs
        print(f"Loaded checkpoint: {args.resume} ({completed_epochs} completed epochs)")
    else:
        model, completed_epochs = FeedForward(FeedForwardConfig()), 0
    model = model.to(device)

    print(
        f"Device: {device} | "
        f"Optimizer: SGD | "
        f"Learning rate: {config.learning_rate}\n"
        f"Training examples: {len(training_loader.dataset)} | "
        f"Validation examples: {len(validation_loader.dataset)}"
    )

    def report_epoch(metrics: EpochMetrics) -> None:
        print(
            f"Epoch {completed_epochs + metrics.epoch}/{completed_epochs + config.epochs} | "
            f"Training loss: {metrics.training_loss:.4f} | "
            f"Validation loss: {metrics.validation_loss:.4f} | "
            f"Validation accuracy: {metrics.validation_accuracy:.2%}"
        )

    # Train & Save
    train(model, training_loader, validation_loader, config, on_epoch=report_epoch)

    if args.save:
        save_checkpoint(
            model,
            args.save,
            training_config=config,
            epochs=completed_epochs + config.epochs,
        )
        print(f"Saved checkpoint: {args.save}")
    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Digit classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    # data
    data_parser = commands.add_parser("data", help="MNIST dataset utilities")
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
    predict_parser = commands.add_parser(
        "predict",
        help="Predict one sample.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    weights = predict_parser.add_mutually_exclusive_group(required=True)
    weights.add_argument(
        "--random",
        action="store_true",
        help="Explicitly use fresh, untrained weights.",
    )
    weights.add_argument(
        "--load",
        metavar="PATH",
        help="Load model weights from a safetensors checkpoint.",
    )
    predict_parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="MNIST training sample index.",
    )
    predict_parser.set_defaults(func=predict_sample)

    # train
    defaults = TrainingConfig()
    train_parser = commands.add_parser(
        "train",
        help="Train a classifier model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=defaults.epochs,
        help="Number of epochs to run.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=defaults.batch_size,
        help="Number of samples per batch.",
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
        help="SGD learning rate.",
    )
    train_parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
        help="SGD weight decay.",
    )
    train_parser.add_argument(
        "--validation-size",
        type=int,
        default=defaults.validation_size,
        help="Number of training samples held out for validation.",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed.",
    )
    train_parser.add_argument(
        "--resume",
        metavar="PATH",
        help="Resume training saved weights using the current training settings.",
    )
    train_parser.add_argument(
        "--save", metavar="PATH", help="Save a checkpoint after training completes."
    )
    train_parser.set_defaults(func=train_model)

    # Parse arguments and execute
    args = parser.parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
