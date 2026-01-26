import argparse

from facter.data.download import download_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ml-1m", "amazon", "sushi3-2016"], required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    path = download_dataset(dataset=args.dataset, force=args.force)
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
