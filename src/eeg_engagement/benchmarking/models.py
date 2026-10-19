"""Classical model factories extracted from the research implementation."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def build_b0(seed: int) -> Pipeline:
    """Build a new classical benchmark copy of the legacy PCA+SVM recipe."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, svd_solver="full", random_state=seed)),
            (
                "svc",
                SVC(
                    C=10.0,
                    class_weight="balanced",
                    gamma="scale",
                    kernel="rbf",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_b1(seed: int) -> Pipeline:
    """Build pyRiemann OAS covariance -> Riemannian tangent -> balanced LR."""

    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except ImportError as error:  # pragma: no cover - exercised in uninstalled environments
        raise RuntimeError(
            "B1 requires pyriemann; install the pinned benchmark dependencies from "
            "requirements.txt"
        ) from error
    return Pipeline(
        [
            ("covariance", Covariances(estimator="oas")),
            ("tangent_space", TangentSpace(metric="riemann")),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )
