"""Actual synthetic EDF roundtrip and private-artifact/leakage contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from eeg_engagement.calibrated_highlow import v1
from eeg_engagement.synthetic import synthetic_edf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plan(local_data_dir):
    records = []
    for i in range(5):
        edf = local_data_dir / f"opaque-{i}.edf"
        synthetic_edf(edf, i + 1)
        records.append({"edf": str(edf), "sha256": v1.sha256(edf),
                        "role": "calibration" if i < 4 else "reserved_test",
                        "subject_id": "TEST", "activity": "Video-On", "usage_policy": "undecided",
                        "date": "2000-01-01" if i < 4 else "2000-01-02",
                        "source_video_id": "", **({"label": "high" if i % 2 else "low"} if i < 4 else {})})
    value = {"schema": v1.SCHEMA, "authorization": "issue9_retrospective_prototype",
             "user": "TEST", "activity": "Video-On", "scope_id": "synthetic",
             "calibration": records[:4], "reserved_test": records[4:]}
    file = v1.write_private_json(ROOT, "plan.json", value)
    return value, file


def forbid_fit(*args, **kwargs):
    raise AssertionError("Prediction must never fit or fit_transform")


def test_actual_edf_training_save_reload_prediction_parity(plan, monkeypatch):
    value, file = plan
    directory = v1.calibrate(ROOT, file)
    pipeline, manifest = v1.load_model(ROOT, "TEST", "Video-On", directory.name)
    edf = Path(value["reserved_test"][0]["edf"])
    windows, _ = v1.prepare_edf(edf)
    before = pipeline.predict_proba(windows)
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    for cls in (Pipeline, Covariances, TangentSpace, LogisticRegression):
        monkeypatch.setattr(cls, "fit", forbid_fit)
        if hasattr(cls, "fit_transform"):
            monkeypatch.setattr(cls, "fit_transform", forbid_fit)
    result = v1.predict_edf(ROOT, "TEST", "Video-On", directory.name, edf)
    np.testing.assert_array_equal([w["high_model_score"] for w in result["windows"]], before[:, 0])
    assert result["fit_calls"] == 0 and result["n_windows"] == 36
    assert result["windows"][0]["start_s"] == 15 and result["windows"][-1]["end_s"] == 89
    assert manifest["classes"] == ["high", "low"]
    assert manifest["training_window_counts"] == [36] * 4
    assert len(manifest["source_sha256"]) == 4
    assert result["model_sha256"] == v1.sha256(directory / "model.joblib")
    for row in result["windows"]:
        assert (row["high_margin"] > 0) == (row["prediction"] == "high")
    with pytest.raises(v1.Unavailable, match="Calibration EDF"):
        v1.predict_edf(ROOT, "TEST", "Video-On", directory.name, Path(value["calibration"][0]["edf"]))


def test_new_version_preserves_existing_bytes_and_only_opens_calibration(plan, monkeypatch):
    value, file = plan
    original = v1.prepare_edf
    seen = []
    def reader(path):
        seen.append(str(path))
        return original(path)
    monkeypatch.setattr(v1, "prepare_edf", reader)
    first = v1.calibrate(ROOT, file)
    old = {p.name: p.read_bytes() for p in first.iterdir()}
    second = v1.calibrate(ROOT, file)
    assert (first.name, second.name) == ("v0001", "v0002")
    assert old == {p.name: p.read_bytes() for p in first.iterdir()}
    assert seen == [r["edf"] for r in value["calibration"]] * 2


@pytest.mark.parametrize("damage", ["labels", "duplicate", "test_overlap", "hash", "missing",
                                        "role", "reference", "unknown_role", "date", "source", "empty"])
def test_plan_rejects_bad_scope_before_fit(plan, monkeypatch, damage):
    value, file = plan
    broken = copy.deepcopy(value)
    if damage == "labels": broken["calibration"][0]["label"] = "high"
    if damage == "duplicate": broken["calibration"][1].update(edf=broken["calibration"][0]["edf"], sha256=broken["calibration"][0]["sha256"])
    if damage == "test_overlap": broken["reserved_test"][0].update(edf=broken["calibration"][0]["edf"], sha256=broken["calibration"][0]["sha256"])
    if damage == "hash": broken["calibration"][0]["sha256"] = "0" * 64
    if damage == "missing": broken["calibration"][0]["edf"] = str(file.parent / "missing.edf")
    if damage == "role": broken["calibration"][0]["role"] = "reserved_test"
    if damage == "reference": broken["calibration"][0]["usage_policy"] = "prediction_only_reference"
    if damage == "unknown_role": broken["calibration"][0].pop("usage_policy")
    if damage == "date": broken["reserved_test"][0]["date"] = broken["calibration"][0]["date"]
    if damage == "source":
        broken["reserved_test"][0]["source_video_id"] = "shared"
        broken["calibration"][0]["source_video_id"] = "shared"
    if damage == "empty": broken["calibration"] = []
    monkeypatch.setattr(v1, "build_b1", forbid_fit)
    with pytest.raises(v1.Unavailable):
        v1.validate_plan(ROOT, broken)


@pytest.mark.parametrize("kind", ["missing_channel", "reference", "short", "sampling", "flat"])
def test_edf_compatibility_and_quality_rejection(local_data_dir, kind):
    edf = local_data_dir / "bad.edf"
    options = {"channels": v1.CHANNELS[:-1]} if kind == "missing_channel" else {}
    if kind == "reference": options["channels"] = tuple(s.replace("-Pz", "-Cz") for s in v1.CHANNELS)
    if kind == "short": options["seconds"] = 89
    if kind == "sampling": options["fs"] = 64
    if kind == "flat": options["flat"] = True
    synthetic_edf(edf, 9, **options)
    with pytest.raises(v1.Unavailable):
        v1.prepare_edf(edf)


def test_resampling_and_amplitude_warning(local_data_dir):
    edf = local_data_dir / "source300.edf"
    synthetic_edf(edf, 12, fs=300, amplitude=80)
    windows, qc = v1.prepare_edf(edf)
    assert windows.shape == (36, 24, 512) and qc["resampled"]
    assert "amplitude_over_100uV_above_1_percent" in qc["warnings"]


def test_private_paths_and_no_overwrite(local_data_dir):
    for value in (ROOT / "benchmarks/reports/private.json", "../escape.json"):
        with pytest.raises(v1.Unavailable):
            v1.write_private_json(ROOT, value, {})
    out = v1.write_private_json(ROOT, "prediction.json", {"original": True})
    with pytest.raises(FileExistsError):
        v1.write_private_json(ROOT, out, {})
    assert json.loads(out.read_text()) == {"original": True}
    with pytest.raises(v1.Unavailable):
        v1.model_base(ROOT, "../bootstrap_baseline", "Video-On")


def test_load_rejects_corruption_and_incompatible_metadata_before_unpickle(plan, monkeypatch):
    _, file = plan
    directory = v1.calibrate(ROOT, file)
    metadata_file = directory / "manifest.json"
    original = json.loads(metadata_file.read_text())
    monkeypatch.setattr(v1.joblib, "load", forbid_fit)
    for field, bad in (("schema", "unknown"), ("user", "OTHER"), ("classes", ["low", "high"]),
                       ("contract", {}), ("dependencies", {}), ("version", "v9999")):
        changed = copy.deepcopy(original)
        changed[field] = bad
        metadata_file.write_text(json.dumps(changed))
        with pytest.raises(v1.Unavailable):
            v1.load_model(ROOT, "TEST", "Video-On", directory.name)
    metadata_file.write_text(json.dumps(original))
    with (directory / "model.joblib").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(v1.Unavailable, match="checksum"):
        v1.load_model(ROOT, "TEST", "Video-On", directory.name)


def test_cli_separate_process_roundtrip_and_failure(plan):
    value, file = plan
    py = sys.executable
    train = subprocess.run([py, str(ROOT / "scripts/calibrate_personal_activity_highlow.py"),
                            "--plan", str(file)], cwd=ROOT, capture_output=True)
    assert train.returncode == 0, train.stderr.decode()
    out = file.parent / "cli_prediction.json"
    command = [py, str(ROOT / "scripts/predict_personal_activity_highlow.py"), "--user", "TEST",
               "--version", "v0001", "--edf", value["reserved_test"][0]["edf"], "--output", str(out), "--replay"]
    predicted = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert predicted.returncode == 0, predicted.stderr.decode()
    assert json.loads(out.read_text())["n_windows"] == 36
    assert len(predicted.stdout.decode().splitlines()) == 38
    original = out.read_bytes()
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode == 2
    assert out.read_bytes() == original
    command[command.index("v0001")] = "v9999"
    command[command.index(str(out))] = str(file.parent / "unavailable.json")
    assert subprocess.run(command, cwd=ROOT, capture_output=True).returncode == 2
    assert json.loads((file.parent / "unavailable.json").read_text())["status"] == "UNAVAILABLE"
