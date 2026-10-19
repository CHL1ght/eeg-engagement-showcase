"""Generate synthetic EDFs, calibrate, reload, and predict without private inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from eeg_engagement.calibrated_highlow import v1
from eeg_engagement.synthetic import synthetic_edf


def run() -> dict:
    """Use an isolated temporary store, even if private-data env vars are set."""
    old_root = os.environ.get("CHLIGHT_LOCAL_DATA_DIR")
    try:
        with TemporaryDirectory(prefix="eeg-synthetic-") as temporary:
            base = Path(temporary).resolve()
            os.environ["CHLIGHT_LOCAL_DATA_DIR"] = str(base)
            rows = []
            for i in range(5):
                path = base / f"synthetic-{i}.edf"
                synthetic_edf(path, i + 1)
                rows.append({
                    "edf": str(path), "sha256": v1.sha256(path),
                    "role": "calibration" if i < 4 else "reserved_test",
                    "subject_id": "SYNTHETIC", "activity": "Video-On",
                    "usage_policy": "undecided", "source_video_id": "",
                    "date": "2000-01-01" if i < 4 else "2000-01-02",
                    **({"label": "high" if i % 2 else "low"} if i < 4 else {}),
                })
            plan = {
                "schema": v1.SCHEMA, "authorization": "issue9_retrospective_prototype",
                "user": "SYNTHETIC", "activity": "Video-On", "scope_id": "synthetic",
                "calibration": rows[:4], "reserved_test": rows[4:],
            }
            plan_path = v1.write_private_json(ROOT, "plan.json", plan)
            directory = v1.calibrate(ROOT, plan_path)
            pipeline, manifest = v1.load_model(ROOT, "SYNTHETIC", "Video-On", directory.name)
            test_edf = Path(rows[4]["edf"])
            windows, _ = v1.prepare_edf(test_edf)
            before = pipeline.predict_proba(windows)[:, 0]
            reference = pipeline.named_steps["tangent_space"].reference_.copy()
            result = v1.predict_edf(ROOT, "SYNTHETIC", "Video-On", directory.name, test_edf)
            np.testing.assert_array_equal(before, [w["high_model_score"] for w in result["windows"]])
            np.testing.assert_array_equal(reference, pipeline.named_steps["tangent_space"].reference_)
            return {
                "input": "seeded synthetic EDF only; no human EEG",
                "calibration_edfs": 4,
                "calibration_windows": sum(manifest["training_window_counts"]),
                "held_out_edfs": 1, "prediction_windows": result["n_windows"],
                "prediction_fit_calls": result["fit_calls"],
                "save_reload_prediction_parity": True,
                "synthetic_session_prediction": result["session_prediction"],
                "interpretation": "engineering demonstration; no real-data accuracy claim",
            }
    finally:
        if old_root is None:
            os.environ.pop("CHLIGHT_LOCAL_DATA_DIR", None)
        else:
            os.environ["CHLIGHT_LOCAL_DATA_DIR"] = old_root


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
