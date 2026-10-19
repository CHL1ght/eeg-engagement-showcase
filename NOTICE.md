# Attribution and rights notice

EEGEngagement showcase code is published for inspection. No new license grant
is made by this release. Public availability does not imply permission to
redistribute or re-license separately acquired research data or third-party
materials.

Earlier reference project: [gennwolf/EEGAttention](https://github.com/gennwolf/EEGAttention),
credited to Genn in its README, and its historical
[CHL1ght fork](https://github.com/CHL1ght/EEGAttention). No notebooks, model files,
datasets, images, or external Drive materials from that project are bundled.

This project calls the installed APIs of the following third-party libraries;
their source and assets are not vendored, and their licenses remain applicable:

- [MNE-Python](https://github.com/mne-tools/mne-python): EDF reading, filtering and resampling.
- [NumPy](https://github.com/numpy/numpy) and [SciPy](https://github.com/scipy/scipy): numerical arrays and Welch power spectra.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn): pipelines, PCA, SVM and logistic regression.
- [pyRiemann](https://github.com/pyRiemann/pyRiemann): OAS covariance and Riemannian tangent-space implementation.
- [joblib](https://github.com/joblib/joblib): trusted local artifact serialization.
- [pytest](https://github.com/pytest-dev/pytest): synthetic contract and integration tests.

GitHub Actions uses the maintained `actions/checkout` and `actions/setup-python`
actions. Their code is fetched by the workflow and is not bundled in this release.
