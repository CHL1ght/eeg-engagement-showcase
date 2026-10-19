# Release inventory and verification

Release scope: a fresh, curated source repository with synthetic-only execution.
The existing private research repository and local private data are outside the
release. No old Git references or objects are copied.

## Full code-file inventory

- `src/eeg_engagement/__init__.py`
- `src/eeg_engagement/local_data.py`
- `src/eeg_engagement/synthetic.py`
- `src/eeg_engagement/benchmarking/__init__.py`
- `src/eeg_engagement/benchmarking/models.py`
- `src/eeg_engagement/benchmarking/data_scope.py`
- `src/eeg_engagement/calibrated_highlow/__init__.py`
- `src/eeg_engagement/calibrated_highlow/v1.py`
- `src/eeg_engagement/preprocessing/__init__.py`
- `src/eeg_engagement/preprocessing/signal.py`
- `src/eeg_engagement/features/__init__.py`
- `src/eeg_engagement/features/bandpower.py`
- `scripts/calibrate_personal_activity_highlow.py`
- `scripts/predict_personal_activity_highlow.py`
- `scripts/synthetic_demo.py`
- `scripts/audit_release.py`
- `tests/conftest.py`
- `tests/test_data_scope.py`
- `tests/test_personal_activity_highlow.py`
- `tests/test_splits_and_features.py`

The remaining files are this repository's README, NOTICE, four architecture /
data / provenance / audit documents, package/dependency configuration, Git
ignore/line-ending rules, and one synthetic-test GitHub Actions workflow.
`scripts/audit_release.py` contains the complete 31-file allowlist.

## Withheld categories

- Raw human EDF/CSV/DSI, source tables, participant identifiers and acquisition metadata.
- Real manifests, participant-linked hashes, training/test memberships and stimulus records.
- Research model artifacts, row-level predictions and historical benchmark files.
- Internal discussions, prompt exchanges, meeting notes, task lists and private links.
- Credentials, environment files, passwords, machine-specific paths and live dashboards.
- Historical Git objects, branches, tags, submodules and LFS objects.
- Third-party notebooks, datasets, images, saved models and Drive materials.

Every published file must be a small UTF-8 text file in the explicit allowlist.
Runtime files remain ignored. The screening command checks common credential
patterns, private-key headers, machine-specific paths, external Drive links and
email addresses. A separate private pre-publication check covers known project
identifiers. Screening is not proof against every possible secret; manual
review of each released file and the staged tree is also required.

## Reproduce validation

```bash
python -m pytest -q
python scripts/synthetic_demo.py
python -m compileall -q src scripts tests
python scripts/audit_release.py
git diff --check
```

These tests exercise actual MNE EDF parsing and preprocessing, B1 fitting,
save/reload parity, prediction with all fit methods forbidden, bad input/role/
hash isolation, corrupt model rejection before deserialization, version
preservation, CLI subprocesses, group/day/subject boundaries and spectral feature
order. Every waveform is generated locally. There is no opt-in real-data test
or skipped private-data dependency in the release.

Initial isolated local verification used Python 3.12.14 and the published
dependency pins: **29 passed, zero skipped**. The demo produced 144 calibration
windows and 36 held-out prediction windows with exact reload parity and zero
prediction fit calls. All 20 Python files passed `py_compile`; the 31-file
privacy allowlist, staged-byte comparison, and `git diff --cached --check`
passed. `pip check` reported no broken requirements. The test run emitted
dependency deprecation warnings from joblib/NumPy and the retained SVM
probability API; these did not fail the checks.

Synthetic checks cannot reproduce or substantiate real EEG accuracy, deployment
readiness, or the private Streamlit historical replay. See the workflow runs
for remote CI results rather than inferring their status from this document.
