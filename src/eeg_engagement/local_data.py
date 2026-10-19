"""Paths and fail-fast checks for data kept outside tracked repository files."""

from __future__ import annotations

import os
from pathlib import Path


LOCAL_DATA_ENV = "CHLIGHT_LOCAL_DATA_DIR"


class LocalInputError(ValueError):
    """A local-only input is present but does not satisfy its contract."""


def local_data_root(repository_root: Path) -> Path:
    """Return the configured local-only data directory.

    The default is ``<repository>/data/local``. Relative environment values
    are resolved from the repository root so commands behave the same from
    the repository root and from an IDE.
    """
    root = Path(repository_root).resolve()
    configured = os.environ.get(LOCAL_DATA_ENV, "").strip()
    path = Path(configured).expanduser() if configured else root / "data" / "local"
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def local_data_path(repository_root: Path, *parts: str) -> Path:
    """Resolve a path under the local-only data directory."""
    base = local_data_root(repository_root)
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Local data paths must stay under CHLIGHT_LOCAL_DATA_DIR")
    return path


def require_local_input(path: Path, description: str, *, directory: bool = False) -> Path:
    """Fail clearly when a required local-only input is absent."""
    path = Path(path)
    present = path.is_dir() if directory else path.is_file()
    if not present:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(
            f"Missing local-only {description} ({kind}). Set {LOCAL_DATA_ENV} "
            "or pass the corresponding path option. See docs/DATA_PROTOCOL.md."
        )
    return path


def resolve_local_input(
    repository_root: Path,
    value: str | Path,
    *,
    description: str = "EDF",
) -> Path:
    """Resolve one local-only file reference without allowing repository escapes.

    Relative references can point under ``<repository>/data/local`` or the
    configured ``CHLIGHT_LOCAL_DATA_DIR``. Absolute references are accepted
    only when they stay inside one of those roots. Other tracked repository
    paths are never used as a fallback for missing private inputs.
    """
    root = Path(repository_root).resolve()
    local_root = local_data_root(root)
    raw = Path(value).expanduser()
    candidates = [raw.resolve()] if raw.is_absolute() else [
        (root / raw).resolve(),
        (local_root / raw).resolve(),
    ]
    allowed_roots = ((root / "data" / "local").resolve(), local_root)
    for candidate in candidates:
        if any(candidate.is_relative_to(allowed) for allowed in allowed_roots) and candidate.is_file():
            return candidate
    if not any(
        candidate.is_relative_to(allowed)
        for candidate in candidates
        for allowed in allowed_roots
    ):
        raise LocalInputError(
            f"Local-only {description} path must stay under repository data/local or "
            f"CHLIGHT_LOCAL_DATA_DIR. See docs/DATA_PROTOCOL.md."
        )
    raise FileNotFoundError(
        f"Missing local-only {description} referenced by the active manifest. Supply "
        f"the file under CHLIGHT_LOCAL_DATA_DIR or correct its local manifest/reference. "
        "See docs/DATA_PROTOCOL.md."
    )


# Backwards-compatible name; all path resolution goes through the shared helper.
resolve_edf_input = resolve_local_input
