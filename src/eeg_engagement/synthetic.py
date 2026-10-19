"""Seeded synthetic EDF writer; never reads human recordings."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from eeg_engagement.calibrated_highlow import v1


def synthetic_edf(path: Path, seed: int, *, channels=v1.CHANNELS, seconds=90, fs=128,
                  flat=False, amplitude=15.0) -> None:
    """Minimal standard EDF, with real header and quantized microvolt signals."""
    n = len(channels)
    field = lambda value, width: str(value).encode("ascii").ljust(width)
    header = b"".join(field(value, width) for value, width in (
        ("0", 8), ("synthetic", 80), ("test fixture", 80), ("01.01.00", 8), ("00.00.00", 8),
        (256 + n * 256, 8), ("", 44), (seconds, 8), (1, 8), (n, 4)))
    fields = [(channels, 16), ([""] * n, 80), (["uV"] * n, 8), ([-200] * n, 8),
              ([200] * n, 8), ([-32768] * n, 8), ([32767] * n, 8), ([""] * n, 80),
              ([fs] * n, 8), ([""] * n, 32)]
    header += b"".join(field(value, width) for values, width in fields for value in values)
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(n, seconds * fs)) * amplitude
    # Different covariance patterns make a meaningful save/reload prediction fixture.
    signal[:n // 2] *= 1.8 if seed % 2 else 0.7
    if flat:
        signal[0] = 0
    digital = np.clip(np.rint((signal + 200) * 65535 / 400 - 32768), -32768, 32767).astype("<i2")
    path.write_bytes(header + digital.reshape(n, seconds, fs).transpose(1, 0, 2).tobytes())
