"""Frozen bootstrap preprocessing operations."""

from collections.abc import Iterator

import mne
import numpy as np


def preprocess_eeg(
    segment_data: np.ndarray,
    sfreq: float,
    *,
    l_freq: float = 0.5,
    h_freq: float = 43.0,
) -> np.ndarray:
    """Apply the frozen MNE FIR bandpass to one logical segment."""
    return mne.filter.filter_data(
        segment_data,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )


def make_windows(
    segment_data: np.ndarray,
    sfreq: float,
    *,
    window_sec: float = 4.0,
    step_sec: float = 2.0,
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield fixed-size windows and offsets without crossing boundaries."""
    window_size = int(round(window_sec * sfreq))
    step_size = int(round(step_sec * sfreq))
    if segment_data.shape[1] < window_size:
        raise ValueError("Segment is shorter than one frozen 4-second window")
    for start in range(0, segment_data.shape[1] - window_size + 1, step_size):
        yield segment_data[:, start : start + window_size], start / sfreq
