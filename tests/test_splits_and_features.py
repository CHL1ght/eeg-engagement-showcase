"""Synthetic checks of grouping, window boundaries, and spectral representation."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eeg_engagement.benchmarking.data_scope import (
    ScopeRecord, make_cross_day_splits, make_directional_subject_splits,
)
from eeg_engagement.benchmarking.models import build_b0
from eeg_engagement.features import extract_bandpower_features, extract_segment_features
from eeg_engagement.preprocessing import make_windows


def record(group, day, subject="SYNTHETIC_A"):
    return ScopeRecord(
        session_id=group, group_id=group, source_recording_id=group,
        subject_id=subject, recorded_date=day, label="high", task="synthetic",
        protocol="synthetic", source_kind="synthetic", channel_set="full24",
        source_root=Path("."), source_path="unused.edf",
        activity_start_s=0, activity_end_s=90,
    )


def test_cross_day_keeps_all_recordings_from_the_test_day_together():
    records = [record("a", "2000-01-01"), record("b", "2000-01-01"), record("c", "2000-01-02")]
    splits = make_cross_day_splits(records)
    assert len(splits) == 2
    assert splits[0].test_session_ids == ("a", "b")
    assert splits[0].train_session_ids == ("c",)
    with pytest.raises(ValueError, match="spans multiple dates"):
        make_cross_day_splits([*records, replace(records[0], session_id="a2", recorded_date="2000-01-02")])


def test_subject_split_rejects_mixed_identity_inside_one_physical_recording():
    records = [record("a", "2000-01-01"), record("b", "2000-01-02", "SYNTHETIC_B")]
    splits = make_directional_subject_splits(records)
    assert len(splits) == 2
    assert splits[0].train_group_ids == splits[1].test_group_ids
    with pytest.raises(ValueError, match="multiple subjects"):
        make_directional_subject_splits([*records, replace(records[0], session_id="a2", subject_id="SYNTHETIC_B")])


def test_windows_stay_inside_the_supplied_interval():
    signal = np.zeros((24, 75 * 128))
    windows = list(make_windows(signal, 128))
    assert len(windows) == 36
    assert windows[-1][1] == 70
    assert windows[-1][0].shape == (24, 512)
    with pytest.raises(ValueError, match="shorter"):
        list(make_windows(signal[:, :511], 128))


def test_known_alpha_sinusoid_and_240_feature_order():
    t = np.arange(512) / 128
    window = np.tile(20e-6 * np.sin(2 * np.pi * 10 * t), (24, 1))
    features = extract_bandpower_features(window, 128).reshape(24, 5, 2)
    assert features.size == 240 and np.isfinite(features).all()
    assert np.all(features[:, :, 1].argmax(axis=1) == 2)
    assert np.all(features[:, 2, 1] > 0.9)
    with pytest.raises(ValueError, match="outside"):
        extract_segment_features(window, 128, 0, 10)


def test_b0_learned_preprocessing_fits_only_supplied_training_data():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(20, 12))
    test = rng.normal(loc=1000, size=(5, 12))
    model = build_b0(7).fit(train, ["high", "low"] * 10)
    before = model.named_steps["scaler"].mean_.copy()
    probabilities = model.predict_proba(test)
    np.testing.assert_allclose(before, train.mean(axis=0))
    np.testing.assert_array_equal(before, model.named_steps["scaler"].mean_)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
