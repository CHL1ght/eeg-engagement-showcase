"""Versioned B1 High/Low calibration and prediction-only EDF inference.

Private plans, artifacts and predictions stay beneath the configured local root.
Joblib artifacts must be locally trusted: checksums detect corruption, not an
attacker who can replace both the model and its manifest.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import joblib
import mne
import numpy as np
from sklearn.pipeline import Pipeline

from eeg_engagement.benchmarking.data_scope import FULL24_CHANNELS
from eeg_engagement.benchmarking.models import build_b1
from eeg_engagement.local_data import local_data_root
from eeg_engagement.preprocessing import make_windows, preprocess_eeg

SCHEMA = "calibrated_highlow/v1"
LABELS = ["high", "low"]
CHANNELS = tuple(FULL24_CHANNELS)
SEED = 7


class Unavailable(ValueError):
    """Input/artifact cannot safely satisfy the fixed prototype contract."""


def contract() -> dict[str, Any]:
    return {
        "channels": list(CHANNELS), "reference": "native DSI24 Pz channel labels",
        "sampling_rate_hz": 128.0, "filter_hz": [0.5, 43.0],
        "interval_s": [15.0, 90.0], "window_s": 4.0, "step_s": 2.0,
        "representation": "24x512 float32 volts; OAS covariance; riemann tangent",
        "tangent_feature_count": 300, "seed": SEED,
        "classifier": "balanced LogisticRegression(lbfgs,max_iter=2000)",
        "edf_weight": "each EDF total classifier sample_weight = 1",
        "vote": "window majority; ties mean P(high) >= 0.5",
        "qc": {"amplitude_v": 100e-6, "warning_fraction": 0.01,
               "flat_std_v": 1e-12},
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dependencies() -> dict[str, str]:
    import platform
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name)
        for name in ("mne", "numpy", "scipy", "scikit-learn", "pyriemann", "joblib")}}


def _git_revision(root: Path) -> str | None:
    """A fresh source checkout may not yet contain a commit."""
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def private_path(root: Path, value: str | Path) -> Path:
    """Resolve under CHLIGHT_LOCAL_DATA_DIR; reject tracked output locations."""
    base = local_data_root(root)
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not path.is_relative_to(base):
        raise Unavailable("Private files must remain under CHLIGHT_LOCAL_DATA_DIR")
    if path.is_relative_to(root.resolve()):
        relative = path.relative_to(root.resolve()).as_posix()
        ignored = subprocess.run(["git", "check-ignore", "--quiet", "--no-index", "--", relative],
                                 cwd=root, capture_output=True)
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "--", relative],
                                 cwd=root, capture_output=True)
        if ignored.returncode != 0 or tracked.returncode == 0:
            raise Unavailable("Private path inside repository must be ignored and untracked")
    return path


def write_private_json(root: Path, path: str | Path, value: dict) -> Path:
    out = private_path(root, path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return out


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise Unavailable("User/activity identifiers must use letters, digits, _ or -")
    return value


def model_base(root: Path, user: str, activity: str) -> Path:
    return private_path(root, Path("personal_activity_models") / safe_name(user) / safe_name(activity))


def prepare_edf(path: Path) -> tuple[np.ndarray, dict]:
    """Read exact reference/channel labels, then the unchanged Issue #8 recipe."""
    if path.suffix.casefold() != ".edf" or not path.is_file():
        raise Unavailable("Input must be an existing EDF")
    raw = mne.io.read_raw_edf(str(path), preload=True, infer_types=True, verbose="ERROR")
    source_fs = float(raw.info["sfreq"])
    if (not np.isfinite(source_fs) or source_fs < 128
            or not np.isclose(source_fs, round(source_fs), atol=1e-6, rtol=0)):
        raise Unavailable("Source sampling rate must be integral and at least 128 Hz")
    if raw.n_times / source_fs < 90:
        raise Unavailable("EDF must cover the fixed [15,90) second interval")
    normalize = lambda name: " ".join(name.strip().casefold().split())
    names = [normalize(name) for name in raw.ch_names]
    if len(names) != len(set(names)):
        raise Unavailable("Duplicate normalized channel names")
    if any(normalize(name) not in names for name in CHANNELS):
        raise Unavailable("Missing required DSI24/Pz-reference channel labels")
    actual = [raw.ch_names[names.index(normalize(name))] for name in CHANNELS]
    channel_types = dict(zip(raw.ch_names, raw.get_channel_types()))
    if any(channel_types[name] != "eeg" for name in actual):
        raise Unavailable("All required channels must be recognized as EEG")
    raw.pick(actual)
    raw.reorder_channels(actual)
    if not np.isclose(source_fs, 128):
        raw.resample(128, npad="auto", verbose="ERROR")
    data = raw.get_data()[:, 15 * 128:90 * 128]
    if data.shape != (24, 75 * 128) or not np.isfinite(data).all():
        raise Unavailable("Invalid length or non-finite EEG samples")
    if np.any(np.std(data, axis=1) <= 1e-12):
        raise Unavailable("Flat channel in analysis interval; prediction unavailable")
    filtered = preprocess_eeg(data, 128, l_freq=0.5, h_freq=43)
    windows = np.asarray([w for w, _ in make_windows(filtered, 128, window_sec=4, step_sec=2)],
                         dtype=np.float32)
    if windows.shape != (36, 24, 512) or not np.isfinite(windows).all():
        raise Unavailable("Invalid preprocessed window representation")
    if np.any(np.std(windows, axis=2) <= 1e-12):
        raise Unavailable("Flat channel within a window; prediction unavailable")
    fractions = [float(np.mean(np.abs(data[:, i * 256:i * 256 + 512]) > 100e-6))
                 for i in range(36)]
    warnings = ["four_calibration_edfs_only", "operational_labels_not_psychological_truth",
                "reference_verified_by_channel_labels_only", "stimulus_boundary_is_edf_relative_proxy"]
    if any(f > 0.01 for f in fractions):
        warnings.append("amplitude_over_100uV_above_1_percent")
    return windows, {"source_sampling_rate_hz": source_fs,
                     "resampled": not np.isclose(source_fs, 128),
                     "amplitude_over_100uV_fraction_by_window": fractions,
                     "warnings": warnings}


def validate_plan(root: Path, plan: dict) -> tuple[list[dict], list[dict]]:
    """Freeze exact physical files and explicit local experiment roles before fit."""
    if (not isinstance(plan, dict) or plan.get("schema") != SCHEMA
            or plan.get("authorization") != "issue9_retrospective_prototype"):
        raise Unavailable("Explicit Issue #9 retrospective prototype plan required")
    user, activity = safe_name(plan.get("user", "")), safe_name(plan.get("activity", ""))
    if activity != "Video-On":
        raise Unavailable("v0.1 supports Video-On calibration only")
    train, reserved = plan.get("calibration", []), plan.get("reserved_test", [])
    if (not isinstance(train, list) or not isinstance(reserved, list)
            or any(not isinstance(r, dict) for r in train + reserved)):
        raise Unavailable("Calibration and reserved test records must be lists of objects")
    if len(train) != 4 or Counter(r.get("label") for r in train) != {"high": 2, "low": 2}:
        raise Unavailable("Exactly four calibration EDFs (2 High + 2 Low) required")
    if not reserved:
        raise Unavailable("Reserved independent test EDF hashes must be declared before fitting")
    hashes = []
    for rows, role in ((train, "calibration"), (reserved, "reserved_test")):
        for row in rows:
            if (row.get("role") != role or row.get("subject_id") != user
                    or row.get("activity") != activity):
                raise Unavailable("Wrong session role, subject or activity")
            if row.get("usage_policy") not in {"undecided", "bootstrap_train_source"}:
                raise Unavailable("Protected or unknown data role cannot be used in this prototype plan")
            if not row.get("date") or row["date"] == "unknown":
                raise Unavailable("Known recording date required")
            path = Path(row["edf"]).resolve()
            if not path.is_file() or path.suffix.casefold() != ".edf" or sha256(path) != row.get("sha256"):
                raise Unavailable("Declared physical EDF hash or file does not match")
            hashes.append(row["sha256"])
    if len(hashes) != len(set(hashes)):
        raise Unavailable("Duplicate physical EDF or calibration/test hash overlap")
    if {r["date"] for r in train} & {r["date"] for r in reserved}:
        raise Unavailable("Calibration and reserved test dates must be disjoint")
    train_sources = {r.get("source_video_id") for r in train} - {None, ""}
    test_sources = {r.get("source_video_id") for r in reserved} - {None, ""}
    if train_sources & test_sources:
        raise Unavailable("Known video source overlaps calibration and reserved test")
    return train, reserved


def calibrate(root: Path, plan_path: str | Path) -> Path:
    plan_file = private_path(root, plan_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    train, reserved = validate_plan(root, plan)
    arrays, quality = [], []
    # Only the four calibration signals are opened. Reserved files are hashed only.
    for row in train:
        windows, qc = prepare_edf(Path(row["edf"]))
        arrays.append(windows)
        quality.append(qc)
        if sha256(Path(row["edf"])) != row["sha256"]:
            raise Unavailable("Calibration EDF changed during loading")
    x = np.concatenate(arrays)
    y = np.concatenate([np.full(len(a), row["label"], dtype=object) for a, row in zip(arrays, train)])
    weights = np.concatenate([np.full(len(a), 1 / len(a)) for a in arrays])
    pipeline = build_b1(SEED)
    pipeline.fit(x, y, logistic_regression__sample_weight=weights)
    if list(pipeline.classes_) != LABELS or pipeline[-1].n_features_in_ != 300:
        raise Unavailable("Unexpected fitted class order or tangent dimension")
    manifest = {
        "schema": SCHEMA, "prototype_version": "0.1", "model_id": "B1",
        "user": plan["user"], "activity": plan["activity"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dependencies": dependencies(), "contract": contract(), "classes": LABELS,
        "scope_id": plan.get("scope_id"), "plan_sha256": sha256(plan_file),
        "authorization": plan["authorization"], "calibration": train,
        "reserved_test": reserved, "training_quality": quality,
        "training_window_counts": [len(a) for a in arrays], "insufficient_for_deployment_claim": True,
        "source_independence": "known IDs isolated; missing stimulus IDs unverified",
        "git_sha": _git_revision(root),
        "implementation_sha256": sha256(Path(__file__)),
        "source_sha256": {name: sha256(root / name) for name in (
            "src/eeg_engagement/calibrated_highlow/v1.py",
            "src/eeg_engagement/benchmarking/models.py",
            "src/eeg_engagement/preprocessing/signal.py", "pyproject.toml")},
    }
    base = model_base(root, plan["user"], plan["activity"])
    base.mkdir(parents=True, exist_ok=True)
    # Atomic directory reservation avoids concurrent version overwrite; gaps survive failures.
    for number in range(1, 10000):
        directory = base / f"v{number:04d}"
        try:
            directory.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise Unavailable("Model version namespace exhausted")
    manifest["version"] = directory.name
    bundle = {"pipeline": pipeline, "schema": SCHEMA, "contract": contract(),
              "classes": LABELS, "training_hashes": [r["sha256"] for r in train]}
    joblib.dump(bundle, directory / "model.joblib", compress=3)
    manifest["model_sha256"] = sha256(directory / "model.joblib")
    write_private_json(root, directory / "manifest.json", manifest)  # completion marker written last
    return directory


def load_model(root: Path, user: str, activity: str, version: str) -> tuple[Any, dict]:
    if not re.fullmatch(r"v[0-9]{4}", version):
        raise Unavailable("Model version must be explicit, such as v0001")
    directory = model_base(root, user, activity) / version
    directory = private_path(root, directory)
    manifest_file = private_path(root, directory / "manifest.json")
    model_file = private_path(root, directory / "model.joblib")
    if not manifest_file.is_file() or not model_file.is_file():
        raise Unavailable("No completed calibration model at the requested version")
    m = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(m, dict):
        raise Unavailable("Invalid model manifest object")
    if (m.get("schema") != SCHEMA or m.get("contract") != contract() or m.get("model_id") != "B1"
            or m.get("classes") != LABELS or m.get("user") != user or m.get("activity") != activity
            or m.get("version") != version):
        raise Unavailable("Incompatible model manifest/schema/user/activity/version")
    if m.get("dependencies") != dependencies():
        raise Unavailable("Dependency versions differ; use the recorded environment")
    train = m.get("calibration", [])
    reserved = m.get("reserved_test", [])
    if (not isinstance(train, list) or not isinstance(reserved, list) or not reserved
            or any(not isinstance(r, dict) or not re.fullmatch(r"[a-f0-9]{64}", str(r.get("sha256", "")))
                   for r in train + reserved)):
        raise Unavailable("Invalid saved calibration/test record schema")
    if (len(train) != 4 or Counter(r.get("label") for r in train) != {"high": 2, "low": 2}
            or len({r.get("sha256") for r in train}) != 4
            or {r.get("sha256") for r in train} & {r.get("sha256") for r in m.get("reserved_test", [])}):
        raise Unavailable("Invalid saved calibration labels or hash isolation")
    if sha256(model_file) != m.get("model_sha256"):
        raise Unavailable("Model checksum mismatch; refusing to deserialize")
    # Only our local, trusted model store is accepted; never load a downloaded joblib.
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("pipeline"), Pipeline):
        raise Unavailable("Serialized artifact is not a calibration pipeline bundle")
    p = bundle["pipeline"]
    if tuple(p.named_steps) != ("covariance", "tangent_space", "logistic_regression"):
        raise Unavailable("Serialized pipeline steps do not match B1")
    if (bundle.get("schema") != SCHEMA or bundle.get("contract") != contract()
            or bundle.get("classes") != LABELS
            or bundle.get("training_hashes") != [r["sha256"] for r in m["calibration"]]
            or list(p.classes_) != LABELS or p[-1].n_features_in_ != 300
            or p.named_steps["tangent_space"].tsupdate
            or p.named_steps["tangent_space"].reference_.shape != (24, 24)
            or p.named_steps["covariance"].estimator != "oas"
            or p.named_steps["tangent_space"].metric != "riemann"
            or p[-1].class_weight != "balanced" or p[-1].solver != "lbfgs"
            or p[-1].max_iter != 2000 or p[-1].random_state != SEED):
        raise Unavailable("Serialized pipeline incompatible with manifest")
    return p, m


def predict_edf(root: Path, user: str, activity: str, version: str, edf: Path) -> dict:
    pipeline, manifest = load_model(root, user, activity, version)
    edf = Path(edf).resolve()
    if not edf.is_file():
        raise Unavailable("EDF input is unavailable")
    identity = sha256(edf)
    if identity in {r["sha256"] for r in manifest["calibration"]}:
        raise Unavailable("Calibration EDF cannot be presented as independent prediction")
    windows, quality = prepare_edf(edf)
    predicted = pipeline.predict(windows)
    high_score = pipeline.predict_proba(windows)[:, LABELS.index("high")]
    # sklearn's positive raw binary margin favors classes_[1] = low.
    high_margin = -pipeline.decision_function(windows)
    if not np.isfinite(high_score).all() or not np.isfinite(high_margin).all():
        raise Unavailable("Non-finite model outputs")
    if sha256(edf) != identity:
        raise Unavailable("EDF changed during prediction")
    high_votes = int(np.sum(predicted == "high"))
    session = ("high" if float(high_score.mean()) >= 0.5 else "low") if high_votes * 2 == len(windows) else (
        "high" if high_votes * 2 > len(windows) else "low")
    return {
        "schema": SCHEMA, "status": "AVAILABLE_WITH_WARNINGS", "mode": "offline_edf_replay",
        "user": user, "activity": activity, "version": version, "model_sha256": manifest["model_sha256"],
        "edf_sha256": identity, "fit_calls": 0, "session_prediction": session,
        "high_vote_fraction": high_votes / len(windows), "n_windows": len(windows),
        "score_semantics": "LR model score, not validated psychological engagement percentage",
        "quality": quality, "contract": contract(),
        "scope_check": "declared user/activity; EDF does not establish subject or task identity",
        "windows": [{"index": i, "start_s": 15 + 2 * i, "end_s": 19 + 2 * i,
                     "prediction": str(label), "high_model_score": float(high_score[i]),
                     "high_margin": float(high_margin[i]),
                     "quality_warnings": ["amplitude_over_100uV_above_1_percent"]
                     if quality["amplitude_over_100uV_fraction_by_window"][i] > 0.01 else []}
                    for i, label in enumerate(predicted)],
    }
