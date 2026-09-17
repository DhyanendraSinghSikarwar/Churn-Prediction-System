"""Checksum helpers used by the ingestion pipeline for idempotency."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_of_file(path: Path) -> str:
    """Stream a file through SHA-256 rather than loading it whole into
    memory — the dataset is small today, but ingestion code shouldn't
    assume that stays true."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
