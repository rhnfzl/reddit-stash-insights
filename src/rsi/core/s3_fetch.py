"""Fetch reddit-stash files from AWS S3 to local cache."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_from_s3(
    bucket: str,
    prefix: str = "reddit/",
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download reddit-stash files from S3 to a local cache directory.

    Returns the local path to the cached files (ready to pass to scanner).
    Only downloads files that are new or modified since last fetch.
    """
    try:
        import boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for S3 support. Install with: pip install boto3"
        )

    if cache_dir is None:
        from rsi.config import Settings
        cache_dir = Settings().s3_cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_reddit_dir = cache_dir / prefix.rstrip("/").split("/")[-1]
    local_reddit_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    downloaded = 0
    skipped = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip directory markers
            if key.endswith("/"):
                continue

            # Only download markdown and JSON files
            if not (key.endswith(".md") or key.endswith(".json")):
                continue

            # Compute local path relative to prefix
            relative = key[len(prefix):]
            local_path = local_reddit_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if local file exists and is same size
            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                skipped += 1
                continue

            logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
            s3.download_file(bucket, key, str(local_path))
            downloaded += 1

    logger.info("S3 fetch complete: %d downloaded, %d skipped", downloaded, skipped)
    return local_reddit_dir
