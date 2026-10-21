import argparse

from .mnist import DEFAULT_DATA_DIR, download, explore


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

    # Parse arguments and execute
    args = parser.parse_args()
    args.func(args)
