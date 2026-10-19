"""Load a trusted local calibration version; predict an independent EDF; no fit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eeg_engagement.calibrated_highlow.v1 import Unavailable, predict_edf, private_path, write_private_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--activity", default="Video-On")
    parser.add_argument("--version", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--edf", type=Path, help="explicit independent EDF; label is never inferred from its name")
    inputs.add_argument("--reserved-plan", help="private plan; select one reserved EDF without reading its truth")
    parser.add_argument("--reserved-index", type=int, default=0)
    parser.add_argument("--output", required=True, help="new private JSON beneath local data root")
    parser.add_argument("--replay", action="store_true", help="print windows in EDF-time order (offline, no realtime sleep)")
    args = parser.parse_args()
    try:
        out = private_path(ROOT, args.output)
        if out.exists():
            raise FileExistsError("Refusing to overwrite existing prediction output")
        edf = args.edf
        if args.reserved_plan:
            plan = json.loads(private_path(ROOT, args.reserved_plan).read_text(encoding="utf-8"))
            if plan["user"] != args.user or plan["activity"] != args.activity:
                raise Unavailable("Reserved plan user/activity does not match request")
            if args.reserved_index < 0 or args.reserved_index >= len(plan["reserved_test"]):
                raise Unavailable("Reserved EDF index out of range")
            record = plan["reserved_test"][args.reserved_index]
            from eeg_engagement.calibrated_highlow.v1 import sha256
            edf = Path(record["edf"])
            if sha256(edf) != record["sha256"]:
                raise Unavailable("Reserved EDF checksum mismatch")
        result = predict_edf(ROOT, args.user, args.activity, args.version, edf)
    except (Unavailable, OSError, ValueError, KeyError, IndexError) as error:
        # No output is written if its path is unsafe or already occupied.
        try:
            write_private_json(ROOT, args.output, {"status": "UNAVAILABLE", "reason": str(error), "fit_calls": 0})
        except (Unavailable, OSError):
            pass
        print(f"UNAVAILABLE: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    write_private_json(ROOT, args.output, result)
    print(f"{result['status']}: {result['session_prediction'].title()} | High votes {result['high_vote_fraction']:.3f} | {result['n_windows']} windows | fit_calls=0")
    print("Quality: " + ", ".join(result["quality"]["warnings"]))
    if args.replay:
        for window in result["windows"]:
            print(f"[{window['start_s']:5.1f},{window['end_s']:5.1f})s {window['prediction'].title():4s} high_margin={window['high_margin']:+.5f}")


if __name__ == "__main__":
    main()
