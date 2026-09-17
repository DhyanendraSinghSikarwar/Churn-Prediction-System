from pathlib import Path

from churn.utils.hashing import sha256_of_bytes, sha256_of_file

_SHA256_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_of_bytes_matches_known_vector() -> None:
    assert sha256_of_bytes(b"") == _SHA256_EMPTY


def test_sha256_of_file_matches_bytes(tmp_path: Path) -> None:
    content = b"hello churn"
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(content)

    assert sha256_of_file(file_path) == sha256_of_bytes(content)
