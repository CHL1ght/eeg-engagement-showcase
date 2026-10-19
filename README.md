# EEGEngagement · engineering showcase

An executable subset of an EEG engagement research project: signal processing,
physical-recording split boundaries, and versioned personal High/Low calibration.
The runnable example generates every input from seeded random numbers.

**Start here:** [core calibration implementation](src/eeg_engagement/calibrated_highlow/v1.py)
· [synthetic integration tests](tests/test_personal_activity_highlow.py)
· [architecture](docs/ARCHITECTURE.md)
· [release inventory and exclusions](docs/RELEASE_AUDIT.md)

## What the code demonstrates

- MNE FIR filtering at 0.5–43 Hz, explicit 24-channel order, and 4-second windows
  with a 2-second step; Welch log-absolute/relative bandpower is also included.
- Two classical factories: StandardScaler → PCA → SVM, and OAS covariance →
  Riemannian tangent space → balanced logistic regression.
- Calibration on exactly two High and two Low recordings, with each recording
  contributing equal total classifier weight. Reserved signals are hashed but
  never opened for training.
- Versioned model directories, dependency/contract checks, corruption detection
  before deserialization, and prediction that never fits or updates the model.
- Physical recordings kept together in session/day/subject splits, rather than
  splitting overlapping windows between train and test.

This repository contains genuine project implementation, narrowed to remove
private dataset loaders and session-specific integration. See
[provenance and attribution](docs/PROVENANCE.md) for the extraction boundaries.

## Run locally

Use Python 3.12 and Git. From a fresh clone:

```bash
git clone https://github.com/CHL1ght/eeg-engagement-showcase.git
cd eeg-engagement-showcase
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/synthetic_demo.py
python -m pytest -q
```

The demo writes five artificial EDF files into an isolated temporary directory,
trains on four (144 windows), saves/reloads a new B1 model, and predicts the
fifth (36 windows). It checks identical probabilities after reload and removes
the temporary inputs and models on exit. It overrides any configured private
data directory during execution, then restores the environment.

The output identifies synthetic inputs, 144 calibration windows, 36 prediction
windows, zero prediction fit calls, and successful save/reload parity. These are
software checks, not a real EEG accuracy benchmark.

## Small public API

```python
from eeg_engagement.benchmarking.models import build_b0, build_b1
from eeg_engagement.preprocessing import make_windows, preprocess_eeg
from eeg_engagement.features import extract_bandpower_features
from eeg_engagement.benchmarking.data_scope import make_grouped_loso_splits
from eeg_engagement.calibrated_highlow.v1 import (
    contract, validate_plan, calibrate, load_model, predict_edf,
)
```

The two CLI entry points are retained for users supplying their own compatible,
consented local EDFs and an explicit plan. Their full path is shown below:

```bash
python scripts/calibrate_personal_activity_highlow.py --help
python scripts/predict_personal_activity_highlow.py --help
```

See [input and reproducibility boundaries](docs/DATA_PROTOCOL.md) before using
those entry points. The synthetic demo needs no external dataset or private plan.

## Evidence boundary

High/Low are operational labels; model scores are not psychological engagement
percentages. The generated classes have artificial covariance differences and
do not validate neural mechanisms, cross-day generalization, deployment
accuracy, four-state decoding, or real-time monitoring.

The private project's Streamlit page replays historical predictions using
private models and result files. It is **not a publicly reproducible real-data
demo** and is excluded here. No participant EEG, trained research model,
recording manifest, per-window research prediction, or research benchmark
result is distributed. No frozen research model is modified by this showcase.

## Attribution and rights

Earlier reference work: [CHL1ght/EEGAttention](https://github.com/CHL1ght/EEGAttention),
a fork of [gennwolf/EEGAttention](https://github.com/gennwolf/EEGAttention), whose
README credits Genn. That historical project and its linked third-party
materials are separate from this showcase. Its notebooks, datasets, images,
and model files are not redistributed here.

The included project code is published for inspection without adding a new
license grant. Dependencies retain their own licenses; see [NOTICE](NOTICE.md).
