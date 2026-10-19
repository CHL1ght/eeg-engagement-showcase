# Inputs and reproducibility boundaries

The default example and all published tests generate artificial EDF waveforms.
They never open human recordings. Dates and identities in fixtures are invented.
There are no stored input datasets, model binaries, or fixture manifests.

`CHLIGHT_LOCAL_DATA_DIR` names a local-only runtime store; its default is
`data/local` beneath the checkout. The public demo temporarily points this
setting to its own temporary directory. Tests use pytest temporary directories.
Do not point either workflow at existing research inputs or artifacts.

For the retained standalone calibration/prediction CLI, compatible user-owned
inputs and a plan must be supplied separately. A plan uses the original
`calibrated_highlow/v1` schema, `Video-On` activity identifier, and historical
`issue9_retrospective_prototype` scope token. That token is a software schema
value, not consent or ethical approval. The demo shows the complete generated
plan structure in `scripts/synthetic_demo.py`.

Exactly four calibration records are required, two per operational class.
Reserved test hashes must be declared before fitting; calibration/test physical
hashes and dates must be disjoint. Known source-video identifiers must also be
disjoint. Missing source identifiers leave content independence unverified.
Protected reference and unknown roles are rejected by the calibration API.
These checks do not verify a person's identity, task, or psychological state.

Runtime EDFs, plans, manifests, models, and prediction JSON belong in the ignored
local store. Version creation refuses overwrite. Dependency versions are
recorded at fit time and must match on reload. Never deserialize an untrusted
joblib file.

Without private EEG and private historical artifacts, this release cannot
reproduce real participant calibration, historical benchmark scores, or the
original Streamlit historical replay. The runnable synthetic workflow verifies
engineering behavior only. Device streaming and prospective real-time
validation are outside the released implementation.
