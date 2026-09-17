"""Database engine construction. One process-wide engine, created lazily
and cached so repeated calls (across modules) reuse the same connection
pool instead of opening a new one each time."""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from churn.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.database_url, pool_pre_ping=True, future=True)
