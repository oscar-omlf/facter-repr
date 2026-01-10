import argparse

from facter.data.download import download_amazon_movies_tv_5, download_movielens_1m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ml-1m", "amazon"], required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.dataset == "ml-1m":
        path = download_movielens_1m(force=args.force)
    else:
        path = download_amazon_movies_tv_5(force=args.force)

    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
