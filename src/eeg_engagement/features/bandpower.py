"""Welch log-absolute and relative bandpower features."""

import numpy as np
from scipy.signal import welch

from eeg_engagement.preprocessing import make_windows, preprocess_eeg


# Insertion order is part of the frozen 240-feature contract.
BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 43.0),
}


def extract_bandpower_features(
    window_data: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] = BANDS,
    *,
    welch_nperseg_sec: float = 2.0,
) -> np.ndarray:
    """Extract ordered log absolute and relative power by channel/band."""
    features: list[float] = []
    for channel_data in window_data:
        freqs, psd = welch(
            channel_data,
            fs=sfreq,
            nperseg=min(len(channel_data), int(sfreq * welch_nperseg_sec)),
        )
        total_idx = (freqs >= 1.0) & (freqs <= 43.0)
        total_power = float(np.trapezoid(psd[total_idx], freqs[total_idx])) + 1e-12
        for low, high in bands.values():
            band_idx = (freqs >= low) & (freqs < high)
            power = float(np.trapezoid(psd[band_idx], freqs[band_idx])) + 1e-12
            features.extend((float(np.log(power)), float(power / total_power)))
    return np.asarray(features, dtype=np.float64)


def extract_segment_features(
    data: np.ndarray,
    sfreq: float,
    start_sec: float,
    end_sec: float,
    bands: dict[str, tuple[float, float]] = BANDS,
    *,
    window_sec: float = 4.0,
    step_sec: float = 2.0,
    l_freq: float = 0.5,
    h_freq: float = 43.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter a manifest interval and produce ordered feature windows."""
    start = int(round(start_sec * sfreq))
    stop = int(round(end_sec * sfreq))
    if start < 0 or stop > data.shape[1] or stop <= start:
        raise ValueError(
            f"Manifest interval [{start_sec}, {end_sec}) is outside EDF "
            f"({data.shape[1] / sfreq:.3f} s)"
        )
    filtered = preprocess_eeg(data[:, start:stop], sfreq, l_freq=l_freq, h_freq=h_freq)
    features: list[np.ndarray] = []
    starts: list[float] = []
    for window, offset_sec in make_windows(
        filtered,
        sfreq,
        window_sec=window_sec,
        step_sec=step_sec,
    ):
        features.append(extract_bandpower_features(window, sfreq, bands))
        starts.append(start_sec + offset_sec)
    return np.asarray(features), np.asarray(starts, dtype=np.float64)
