"""One-time, idempotent dataset download.

Idempotency has two independent layers:
  1. Filesystem: if the raw CSV is already cached locally, it is not
     re-downloaded.
  2. Database (see cli.py): if a file with this exact SHA-256 has already
     been loaded into Postgres (recorded in ingestion_runs), the load step
     is skipped even if the cache was deleted and re-downloaded fresh.
"""

from __future__ import annotations

from pathlib import Path

import requests

from churn.config import PROJECT_ROOT, DataConfig
from churn.utils.hashing import sha256_of_file
from churn.utils.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT_SECONDS = 30


class DownloadError(RuntimeError):
    pass


def download_dataset(data_config: DataConfig) -> tuple[Path, str]:
    """Return (path, sha256) of the cached raw CSV, downloading it if
    necessary. Tries the primary URL first, then the documented fallback
    mirror (same file — verified same byte size at plan time; see
    PLAN.md)."""
    target = PROJECT_ROOT / data_config.source.raw_cache_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        logger.info("dataset_cache_hit", path=str(target))
        return target, sha256_of_file(target)

    for label, url in (
        ("primary", data_config.source.primary_url),
        ("fallback", data_config.source.fallback_url),
    ):
        try:
            logger.info("dataset_download_attempt", source=label, url=url)
            response = requests.get(url, timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("dataset_download_failed", source=label, error=str(exc))
            continue

        target.write_bytes(response.content)
        checksum = sha256_of_file(target)
        logger.info(
            "dataset_download_succeeded",
            source=label,
            sha256=checksum,
            bytes=len(response.content),
        )
        return target, checksum

    raise DownloadError("Both primary and fallback dataset URLs failed; see logs above.")
