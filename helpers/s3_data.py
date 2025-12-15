from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Set, Optional

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    boto3 = None  # type: ignore[assignment]
    Config = ClientError = None  # type: ignore[assignment]
from dotenv import load_dotenv

from .coco import COCO


def load_coco(path: Path) -> COCO:
    return COCO.from_file(path)


def parse_s3_uris(images: Iterable[str]) -> Set[str]:
    uris: Set[str] = set()
    for uri in images:
        if isinstance(uri, str) and uri.startswith("s3://"):
            uris.add(uri)
    return uris


def _normalize_s3_prefix(prefix: Optional[str]) -> str:
    """
    Turn user prefix like:
      "", None, "folder", "folder/", "/folder/", "folder\\sub"
    into:
      "" or "folder/sub"
    """
    if not prefix:
        return ""
    return prefix.replace("\\", "/").strip("/")


def _basename_any_path(p: str) -> str:
    """
    Robust basename for inputs like:
      "../../a/b.jpg"
      "..\\..\\a\\b.jpg"
      "s3://bucket/folder/b.jpg"
    """
    if p.startswith("s3://"):
        _, key = _split_s3_uri(p)
        return Path(key).name
    return Path(p.replace("\\", "/")).name


def _join_s3_key(prefix: str, filename: str) -> str:
    prefix = _normalize_s3_prefix(prefix)
    return f"{prefix}/{filename}" if prefix else filename


def _format_file_name(filename_only: str, prefix: str, bucket: Optional[str]) -> str:
    """
    If bucket is provided -> s3://bucket/prefix/filename
    else -> prefix/filename (or just filename if no prefix)
    """
    key = _join_s3_key(prefix, filename_only)
    if bucket:
        return f"s3://{bucket}/{key}"
    return key


def rewrite_coco_image_paths_inplace(
    coco_json_path: Path,
    *,
    prefix: str = "",
    bucket: Optional[str] = None,
    output: Optional[Path] = None
) -> Path:
    """
    Rewrites COCO JSON images[*].file_name to:
      - s3://<bucket>/<prefix>/<filename>   (if bucket provided)
      - <prefix>/<filename>                (if no bucket but prefix provided)
      - <filename>                         (if neither provided)

    Also supports "style" output:
      - ensure_ascii=True => Chinese becomes \\uXXXX
    """
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Invalid COCO json: 'images' is not a list in {coco_json_path}")

    changed = 0
    for img in images:
        if not isinstance(img, dict):
            continue
        old = img.get("file_name")
        if not isinstance(old, str) or not old:
            continue

        filename_only = _basename_any_path(old)
        new = _format_file_name(filename_only, prefix=prefix, bucket=bucket)

        if new != old:
            img["file_name"] = new
            changed += 1

    out_path = output if output is not None else coco_json_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=True, indent=2)
    json_text = json_text.replace("/", "\\/")
    out_path.write_text(json_text, encoding="utf-8")
    print(f"[rename] wrote: {out_path} (updated {changed} image file_name entries)")
    return out_path


def download_images(coco_path: Path, destination_dir: Path) -> None:
    if boto3 is None:
        raise RuntimeError("boto3 is required to download images. Please install boto3 to use this helper.")
    destination_dir.mkdir(parents=True, exist_ok=True)
    coco = load_coco(coco_path)
    s3_uris = parse_s3_uris([img.file_name for img in coco.images])

    if not s3_uris:
        raise RuntimeError(
            "No s3:// URIs found in COCO images[].file_name.\n"
            "Either:\n"
            "  (1) run with --rename --bucket <...> --prefix <...> first, OR\n"
            "  (2) ensure your COCO already contains s3://... in file_name."
        )

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

    missing = [
        uri
        for uri in s3_uris
        if not (destination_dir / Path(_split_s3_uri(uri)[1]).name).exists()
    ]
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
    if boto3 is None:
        raise RuntimeError("boto3 is required to create an AWS session.")
    return boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Default: download images from s3:// URIs in COCO. Optional: rename/rewrite file_name entries."
    )
    parser.add_argument("coco_json", type=str, help="Path to COCO json file.")

    # Mode switch
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Rewrite images[].file_name (filename-only + optional prefix + optional s3://bucket).",
    )

    # Rename options (only used when --rename)
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="Bucket name for s3://<bucket>/<prefix>/<filename> when using --rename.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Folder/prefix to prepend before the filename when using --rename (supports Chinese).",
    )
    parser.add_argument(
        "--write-json",
        type=str,
        default=None,
        help="Output path for rewritten COCO json (defaults to overwrite input) when using --rename.",
    )

    # Download options (default behavior)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination directory for downloaded images. Defaults to COCO parent / images",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Also download after --rename. (Without --rename, download happens by default.)",
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to a .env file with AWS credentials (defaults to searching .env).",
    )

    args = parser.parse_args()
    load_dotenv(dotenv_path=args.env_file)

    coco_path = Path(args.coco_json)

    # If rename mode: rewrite JSON (maybe also download)
    if args.rename:
        out_json = Path(args.write_json) if args.write_json else None
        rewritten_path = rewrite_coco_image_paths_inplace(
            coco_path,
            prefix=args.prefix,
            bucket=args.bucket,
            output=out_json
        )

        if args.download:
            dest = Path(args.out) if args.out else rewritten_path.parent / "images"
            download_images(rewritten_path, dest)

        return

    # Default mode: download only
    dest = Path(args.out) if args.out else coco_path.parent / "images"
    download_images(coco_path, dest)


if __name__ == "__main__":
    main()
