from unittest.mock import MagicMock

import pandas as pd
import pytest

from churn.ingestion.cli import _already_loaded, _validate_row_count


def test_validate_row_count_accepts_a_matching_count() -> None:
    raw = pd.DataFrame({"a": [1, 2, 3]})
    _validate_row_count(raw, expected=3)  # does not raise


def test_validate_row_count_rejects_a_mismatched_count() -> None:
    raw = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="Row count mismatch"):
        _validate_row_count(raw, expected=7043)


def test_already_loaded_true_when_a_matching_sha256_exists() -> None:
    conn = MagicMock()
    conn.execute.return_value.first.return_value = (1,)
    assert _already_loaded(conn, "deadbeef") is True


def test_already_loaded_false_when_no_matching_sha256() -> None:
    conn = MagicMock()
    conn.execute.return_value.first.return_value = None
    assert _already_loaded(conn, "deadbeef") is False
