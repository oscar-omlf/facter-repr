"""Download supported raw datasets into the repository data cache.

This is a small CLI wrapper around `facter.data.download.download_dataset`.
"""

import argparse

from facter.data.download import download_dataset


def main() -> None:
    """Parse CLI arguments and download the requested dataset.

    Returns:
        None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ml-1m", "amazon"], required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    path = download_dataset(dataset=args.dataset, force=args.force)
    print(f"Downloaded to: {path}")



if __name__ == "__main__":
    main()
