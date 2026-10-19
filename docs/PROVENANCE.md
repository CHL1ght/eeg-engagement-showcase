# Provenance and attribution

This is a curated source release of the current EEGEngagement project. It has
fresh Git history and no fork relationship to the private research repository.
No private commit IDs, participant source hashes, or historical Git objects are
included.

## Extraction record

| Released module | Treatment |
|---|---|
| `calibrated_highlow/v1.py` | Core calibration, validation, artifact loading and prediction retained. A fresh repository with no commit records `git_sha: null` instead of failing; input/model behavior is unchanged. |
| `benchmarking/models.py` | Existing B0 and B1 factories extracted; unused deep-model factories removed and dependency-install message points to the public requirements file. |
| `benchmarking/data_scope.py` | Channel constant, record/split data classes, and grouped session/day/subject split functions extracted. All real-data and historical manifest adapters omitted. |
| `local_data.py`, `preprocessing/`, `features/` | Existing local-input guard, FIR/windowing, and Welch feature implementation retained. |
| `synthetic.py` | Existing test EDF writer moved to a reusable synthetic-only module. |
| Two calibration/prediction CLI scripts | Existing CLI entry points retained. |
| Personal calibration tests | Synthetic cases retained; fixture EDF writer imported from `synthetic.py`; dates replaced with invented dates; private real-EDF opt-in test omitted. |
| Grouping tests | Synthetic grouping cases retained with invented identity/date and source kind. A non-executable protected-role assertion omitted. |
| Demo, extra tests, CI, audit, docs | Added for this public release. |

The source history inspected for these components records their addition in
the active project's initialization, benchmark, local-input, and personal
calibration changes. Neither the old project history nor its data is imported.
The standard algorithms and dependency implementations are not claimed as
original research contributions.

## Earlier reference repository

[CHL1ght/EEGAttention](https://github.com/CHL1ght/EEGAttention) is an earlier
reference fork of [gennwolf/EEGAttention](https://github.com/gennwolf/EEGAttention).
Its README credits Genn with “Code created by Genn” and links an external
Google Drive folder. This release preserves that credit as historical context.
No upstream notebook, originaldata directory, model/result tree, image, or
Drive material is included or re-licensed. Earlier PCA/SVM methodological
lineage is acknowledged; it is not presented as a newly invented algorithm.

No new license is assigned to these project files. Dependencies and any
separately acquired upstream materials retain their original rights.
