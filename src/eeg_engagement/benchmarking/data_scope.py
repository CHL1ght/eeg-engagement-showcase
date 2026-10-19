"""Data structures and split algorithms only; no dataset or manifest loaders.

Callers must separately establish recording eligibility and label semantics.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


FULL24_CHANNELS = (
    "P3-Pz",
    "C3-Pz",
    "F3-Pz",
    "Fz-Pz",
    "F4-Pz",
    "C4-Pz",
    "P4-Pz",
    "Cz-Pz",
    "CM-Pz",
    "A1-Pz",
    "Fp1-Pz",
    "Fp2-Pz",
    "T3-Pz",
    "T5-Pz",
    "O1-Pz",
    "O2-Pz",
    "X3-Pz",
    "X2-Pz",
    "F7-Pz",
    "F8-Pz",
    "X1-Pz",
    "A2-Pz",
    "T6-Pz",
    "T4-Pz",
)


@dataclass(frozen=True)
class ScopeRecord:
    """One logical labeled interval with an explicit physical group."""

    session_id: str
    group_id: str
    source_recording_id: str
    subject_id: str
    recorded_date: str
    label: str
    task: str
    protocol: str
    source_kind: str
    channel_set: str
    source_root: Path
    source_path: str
    activity_start_s: float
    activity_end_s: float
    source_hash: str = ""
    provenance_role: str = ""
    manifest_path: str = ""
    feedback_exposure: str = ""

    @property
    def path(self) -> Path:
        return (self.source_root / self.source_path).resolve()

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "group_id": self.group_id,
            "source_recording_id": self.source_recording_id,
            "subject_id": self.subject_id,
            "recorded_date": self.recorded_date,
            "label": self.label,
            "task": self.task,
            "protocol": self.protocol,
            "source_kind": self.source_kind,
            "channel_set": self.channel_set,
            "source_root": str(self.source_root),
            "source_path": self.source_path,
            "activity_start_s": self.activity_start_s,
            "activity_end_s": self.activity_end_s,
            "source_hash": self.source_hash,
            "provenance_role": self.provenance_role,
            "manifest_path": self.manifest_path,
            "feedback_exposure": self.feedback_exposure,
        }


@dataclass(frozen=True)
class ScopeSplit:
    """A grouped train/test split; IDs are logical sessions, groups are physical recordings."""

    split_id: str
    train_session_ids: tuple[str, ...]
    test_session_ids: tuple[str, ...]
    evaluation_scope: str
    train_group_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "train_session_ids": list(self.train_session_ids),
            "test_session_ids": list(self.test_session_ids),
            "evaluation_scope": self.evaluation_scope,
            "train_group_ids": list(self.train_group_ids),
            "test_group_ids": list(self.test_group_ids),
        }


def group_map(records: Sequence[ScopeRecord]) -> dict[str, tuple[ScopeRecord, ...]]:
    grouped: dict[str, list[ScopeRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_id, []).append(record)
    return {group: tuple(sorted(items, key=lambda item: item.session_id)) for group, items in grouped.items()}


def _make_split(
    records: Sequence[ScopeRecord],
    train_groups: Iterable[str],
    test_groups: Iterable[str],
    *,
    split_id: str,
    evaluation_scope: str,
) -> ScopeSplit:
    groups = group_map(records)
    train_group_ids = tuple(sorted(set(train_groups)))
    test_group_ids = tuple(sorted(set(test_groups)))
    if not train_group_ids or not test_group_ids:
        raise ValueError(f"Empty train/test groups for {split_id}")
    if set(train_group_ids) & set(test_group_ids):
        raise ValueError(f"Group overlap for {split_id}")
    unknown = (set(train_group_ids) | set(test_group_ids)) - set(groups)
    if unknown:
        raise ValueError(f"Unknown groups in {split_id}: {sorted(unknown)}")
    train_ids = tuple(record.session_id for group in train_group_ids for record in groups[group])
    test_ids = tuple(record.session_id for group in test_group_ids for record in groups[group])
    if set(train_ids) & set(test_ids):
        raise ValueError(f"Logical session overlap for {split_id}")
    return ScopeSplit(
        split_id=split_id,
        train_session_ids=train_ids,
        test_session_ids=test_ids,
        evaluation_scope=evaluation_scope,
        train_group_ids=train_group_ids,
        test_group_ids=test_group_ids,
    )


def assert_scope_splits_disjoint(splits: Sequence[ScopeSplit]) -> None:
    if not splits:
        raise ValueError("At least one grouped split is required")
    tested: list[str] = []
    for split in splits:
        if set(split.train_group_ids) & set(split.test_group_ids):
            raise ValueError(f"Group overlap detected in {split.split_id}")
        tested.extend(split.test_group_ids)
    if len(tested) != len(set(tested)):
        raise ValueError("A complete physical group is tested more than once")


def make_grouped_loso_splits(records: Sequence[ScopeRecord], scope: str = "within_subject_cross_session") -> tuple[ScopeSplit, ...]:
    groups = tuple(sorted(group_map(records)))
    if len(groups) < 2:
        raise ValueError("At least two complete physical groups are required")
    splits = tuple(
        _make_split(
            records,
            (group for group in groups if group != test_group),
            (test_group,),
            split_id=f"loso_group_{test_group}",
            evaluation_scope=scope,
        )
        for test_group in groups
    )
    assert_scope_splits_disjoint(splits)
    return splits


def make_cross_day_splits(records: Sequence[ScopeRecord]) -> tuple[ScopeSplit, ...]:
    groups = group_map(records)
    group_dates = {group: {record.recorded_date for record in items} for group, items in groups.items()}
    if any(len(dates) != 1 for dates in group_dates.values()):
        raise ValueError("A complete group spans multiple dates; cross-day protocol is ambiguous")
    dates = tuple(sorted({next(iter(dates)) for dates in group_dates.values()}))
    if len(dates) < 2:
        raise ValueError("Cross-day evaluation requires at least two dates")
    splits = tuple(
        _make_split(
            records,
            (group for group, group_dates_ in group_dates.items() if next(iter(group_dates_)) != date),
            (group for group, group_dates_ in group_dates.items() if next(iter(group_dates_)) == date),
            split_id=f"cross_day_{date}",
            evaluation_scope="cross_day",
        )
        for date in dates
    )
    assert_scope_splits_disjoint(splits)
    return splits


def make_directional_subject_splits(records: Sequence[ScopeRecord]) -> tuple[ScopeSplit, ...]:
    subjects = tuple(sorted({record.subject_id for record in records}))
    if len(subjects) != 2:
        raise ValueError(f"Directional subject evaluation requires exactly two subjects, found {subjects}")
    groups = group_map(records)
    group_subjects = {group: {record.subject_id for record in items} for group, items in groups.items()}
    if any(len(subject_set) != 1 for subject_set in group_subjects.values()):
        raise ValueError("A physical group contains multiple subjects")
    splits = []
    for source, target in ((subjects[0], subjects[1]), (subjects[1], subjects[0])):
        train = [group for group, subject_set in group_subjects.items() if source in subject_set]
        test = [group for group, subject_set in group_subjects.items() if target in subject_set]
        splits.append(
            _make_split(
                records,
                train,
                test,
                split_id=f"{source}_to_{target}",
                evaluation_scope="directional_cross_subject",
            )
        )
    return tuple(splits)
