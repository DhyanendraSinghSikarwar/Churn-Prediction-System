"""Entry point: `python -m churn.ingestion.cli [--force]`.

Runs the full one-time ingestion pipeline: download (idempotent) -> skip
if this exact file content was already loaded (idempotent) -> synthesize
the temporal layer -> bulk load -> record the run.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from churn.config import get_app_config
from churn.db.engine import get_engine
from churn.ingestion.download import download_dataset
from churn.ingestion.load import load_all
from churn.ingestion.synthesize import synthesize_all
from churn.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _already_loaded(conn: Connection, sha256: str) -> bool:
    result = conn.execute(
        text("SELECT 1 FROM ingestion_runs WHERE sha256 = :sha256"), {"sha256": sha256}
    ).first()
    return result is not None


def run(force: bool = False) -> None:
    cfg = get_app_config()
    engine = get_engine()

    path, sha256 = download_dataset(cfg.data)

    if not force:
        with engine.connect() as conn:
            if _already_loaded(conn, sha256):
                logger.info("ingestion_skipped_already_loaded", sha256=sha256)
                return

    raw = pd.read_csv(path)
    if len(raw) != cfg.data.source.expected_row_count:
        raise ValueError(
            f"Row count mismatch: expected {cfg.data.source.expected_row_count}, "
            f"got {len(raw)}. Refusing to load — the source file may have changed."
        )

    tables = synthesize_all(raw, cfg, seed=cfg.synthetic.random_seed)

    with engine.begin() as conn:
        counts = load_all(conn, tables)
        conn.execute(
            text(
                "INSERT INTO ingestion_runs (source_url, sha256, row_count) "
                "VALUES (:url, :sha256, :row_count)"
            ),
            {
                "url": cfg.data.source.primary_url,
                "sha256": sha256,
                "row_count": counts["customers"],
            },
        )

    logger.info("ingestion_complete", counts=counts)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Idempotent churn dataset ingestion")
    parser.add_argument("--force", action="store_true", help="Reload even if already loaded")
    args = parser.parse_args(sys.argv[1:])
    run(force=args.force)


if __name__ == "__main__":
    main()
