from pathlib import Path

import pytest


@pytest.fixture
def local_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated synthetic local-only input root for contract tests."""
    path = tmp_path / "private-local-data"
    path.mkdir()
    monkeypatch.setenv("CHLIGHT_LOCAL_DATA_DIR", str(path))
    return path
