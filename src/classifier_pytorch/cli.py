import argparse
from pathlib import Path

from .mnist import DEFAULT_DATA_DIR, download


def data_download(args: argparse.Namespace) -> None:
    data_dir = args.data_dir.expanduser()
    print(f"Preparing MNIST in {data_dir.resolve()}...")
    try:
        download(data_dir)
    except Exception as error:
        raise SystemExit(f"Could not prepare MNIST: {error}")
    print("MNIST training and test data are ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST digit classifier")
    commands = parser.add_subparsers(dest="command", required=True)

    # data
    data_parser = commands.add_parser("data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    download_parser = data_commands.add_parser(
        "download", help="Download MNIST dataset."
    )
    download_parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Dataset directory (default: %(default)s).",
    )
    download_parser.set_defaults(func=data_download)

    # Parse arguments and execute
    args = parser.parse_args()
    args.func(args)
