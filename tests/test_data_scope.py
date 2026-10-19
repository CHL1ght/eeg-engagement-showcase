from pathlib import Path

from eeg_engagement.benchmarking.data_scope import (
    ScopeRecord,
    assert_scope_splits_disjoint,
    make_grouped_loso_splits,
)
LABELS = ("focus", "unfocus")


def _record(session_id: str, group_id: str, label: str) -> ScopeRecord:
    return ScopeRecord(
        session_id=session_id,
        group_id=group_id,
        source_recording_id=group_id,
        subject_id="SYNTHETIC",
        recorded_date="2000-01-01",
        label=label,
        task="test",
        protocol="test",
        source_kind="synthetic",
        channel_set="full24",
        source_root=Path("."),
        source_path=f"{session_id}.edf",
        activity_start_s=0.0,
        activity_end_s=600.0,
    )


def test_grouped_loso_keeps_logical_segments_from_one_physical_recording_together() -> None:
    records = [
        _record("g1_unfocus", "g1", "unfocus"),
        _record("g1_focus", "g1", "focus"),
        _record("g2_focus", "g2", "focus"),
        _record("g3_unfocus", "g3", "unfocus"),
    ]
    splits = make_grouped_loso_splits(records)
    assert_scope_splits_disjoint(splits)
    first = next(split for split in splits if split.test_group_ids == ("g1",))
    assert first.test_session_ids == ("g1_focus", "g1_unfocus")
    assert not set(first.train_session_ids) & set(first.test_session_ids)


def test_grouped_split_generation_is_deterministic_and_binary() -> None:
    records = [
        _record("b", "b", "focus"),
        _record("a", "a", "unfocus"),
        _record("c", "c", "focus"),
    ]
    forward = [split.as_dict() for split in make_grouped_loso_splits(records)]
    reverse = [split.as_dict() for split in make_grouped_loso_splits(list(reversed(records)))]
    assert forward == reverse
    assert set(LABELS) == {record.label for record in records}
