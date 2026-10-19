"""Train only the four EDFs in an explicit private calibration plan."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_engagement.calibrated_highlow.v1 import Unavailable, calibrate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="private JSON path relative to CHLIGHT_LOCAL_DATA_DIR")
    args = parser.parse_args()
    try:
        directory = calibrate(ROOT, args.plan)
    except (Unavailable, OSError, ValueError, KeyError) as error:
        raise SystemExit(f"UNAVAILABLE: {error}") from error
    print(f"Saved B1 calibration version {directory.name}; 4 EDFs, 144 windows; model store is local-only")


if __name__ == "__main__":
    main()
