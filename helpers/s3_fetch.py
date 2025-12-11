from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Set

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from .coco import COCO

# Environment variables expected:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_DEFAULT_REGION (optional, falls back to us-east-1)
# - AWS_S3_ENDPOINT (optional, for custom endpoints)


def load_coco(path: Path) -> COCO:
    return COCO.from_file(path)


def parse_s3_uris(images: Iterable[str]) -> Set[str]:
    uris: Set[str] = set()
    for uri in images:
        if uri.startswith("s3://"):
            uris.add(uri)
    return uris


def download_images(coco_path: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    coco = load_coco(coco_path)
    s3_uris = parse_s3_uris([img.file_name for img in coco.images])
    session = _make_session()
    s3 = session.resource(
        "s3",
        endpoint_url=os.getenv("AWS_S3_ENDPOINT"),
        config=Config(signature_version="s3v4"),
    )

    downloaded = 0
    for uri in sorted(s3_uris):
        bucket, key = _split_s3_uri(uri)
        filename = Path(key).name
        dest_file = destination_dir / filename
        if dest_file.exists():
            print(f"exists, skipping: {dest_file}")
            continue
        print(f"downloading {uri} -> {dest_file}")
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        _download_object(s3, bucket, key, dest_file)
        downloaded += 1

    missing = [uri for uri in s3_uris if not (destination_dir / Path(_split_s3_uri(uri)[1]).name).exists()]
    if missing:
        raise RuntimeError(f"Incomplete dataset, missing {len(missing)} files: {missing[:5]}")

    print(
        "Download complete.",
        f"expected={len(s3_uris)}",
        f"skipped={len(s3_uris) - downloaded}",
        f"downloaded={downloaded}",
    )


def _download_object(s3_resource, bucket: str, key: str, dest: Path) -> None:
    try:
        s3_resource.Bucket(bucket).download_file(key, str(dest))
    except ClientError as exc:
        raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {exc}") from exc


def _split_s3_uri(uri: str) -> tuple[str, str]:
    _, _, rest = uri.partition("s3://")
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid S3 uri: {uri}")
    return bucket, key


def _make_session():
    return boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download images referenced in a COCO json from S3."
    )
    parser.add_argument("coco_json", type=str, help="Path to COCO json file.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination directory for images. Defaults to COCO parent / images",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to a .env file with AWS credentials (defaults to searching .env).",
    )
    args = parser.parse_args()

    # Load environment variables early so boto can see them.
    load_dotenv(dotenv_path=args.env_file)

    coco_path = Path(args.coco_json)
    if args.out:
        dest = Path(args.out)
    else:
        dest = coco_path.parent / "images"

    download_images(coco_path, dest)


if __name__ == "__main__":
    main()
