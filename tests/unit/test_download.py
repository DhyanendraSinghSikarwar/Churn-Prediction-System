from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from churn.config import DataConfig
from churn.ingestion.download import DownloadError, download_dataset
from churn.utils.hashing import sha256_of_file


def _data_config(raw_cache_path: str) -> DataConfig:
    return DataConfig.model_validate(
        {
            "snapshot_date": "2024-06-01",
            "panel_start_date": "2023-01-01",
            "panel_end_date": "2024-06-01",
            "prediction_horizon_days": 30,
            "train_months": 12,
            "validation_months": 3,
            "test_months": 3,
            "source": {
                "primary_url": "https://example.com/primary.csv",
                "fallback_url": "https://example.com/fallback.csv",
                "expected_row_count": 3,
                "raw_cache_path": raw_cache_path,
            },
        }
    )


def test_download_dataset_returns_cached_file_without_a_network_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("churn.ingestion.download.PROJECT_ROOT", tmp_path)
    cache_path = tmp_path / "cached.csv"
    cache_path.write_text("a,b\n1,2\n", encoding="utf-8")
    cfg = _data_config("cached.csv")

    with patch("churn.ingestion.download.requests.get") as mock_get:
        path, sha = download_dataset(cfg)

    mock_get.assert_not_called()
    assert path == cache_path
    assert sha == sha256_of_file(cache_path)


def test_download_dataset_succeeds_on_the_primary_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("churn.ingestion.download.PROJECT_ROOT", tmp_path)
    cfg = _data_config("new.csv")
    response = MagicMock(content=b"a,b\n1,2\n")
    response.raise_for_status.return_value = None

    with patch("churn.ingestion.download.requests.get", return_value=response) as mock_get:
        path, _ = download_dataset(cfg)

    assert mock_get.call_count == 1
    assert path.read_bytes() == b"a,b\n1,2\n"


def test_download_dataset_falls_back_when_the_primary_url_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("churn.ingestion.download.PROJECT_ROOT", tmp_path)
    cfg = _data_config("new.csv")
    fallback_response = MagicMock(content=b"x,y\n1,2\n")
    fallback_response.raise_for_status.return_value = None

    def side_effect(url: str, timeout: int) -> MagicMock:
        if url == cfg.source.primary_url:
            raise requests.ConnectionError("primary is down")
        return fallback_response

    with patch("churn.ingestion.download.requests.get", side_effect=side_effect) as mock_get:
        path, _ = download_dataset(cfg)

    assert mock_get.call_count == 2
    assert path.read_bytes() == b"x,y\n1,2\n"


def test_download_dataset_raises_when_both_urls_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("churn.ingestion.download.PROJECT_ROOT", tmp_path)
    cfg = _data_config("new.csv")

    with (
        patch(
            "churn.ingestion.download.requests.get",
            side_effect=requests.ConnectionError("down"),
        ),
        pytest.raises(DownloadError),
    ):
        download_dataset(cfg)
