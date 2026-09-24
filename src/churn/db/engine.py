"""Database engine construction. One process-wide engine, created lazily
and cached so repeated calls (across modules) reuse the same connection
pool instead of opening a new one each time."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from churn.config import get_settings

# A stopped/restarting Postgres container doesn't always refuse a new
# connection immediately -- during `docker compose stop`/`start`, the
# port can sit in a state where the OS's default TCP connect timeout
# applies instead (tens of seconds). Without this, api/main.py's
# graceful-degradation story (a DB-down /health check should 503 fast,
# not hang) silently breaks -- found by actually stopping the container
# mid-Phase-6-verification, not by inspection.
_CONNECT_TIMEOUT_SECONDS = 5


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
        connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS},
    )
