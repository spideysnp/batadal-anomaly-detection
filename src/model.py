import numpy as np
from sklearn.ensemble import IsolationForest

from src.evaluate import batadal_score


def fit(X_train, random_state: int = 42) -> IsolationForest:
    """Fit an isolation forest on normal operation data."""

    model = IsolationForest(random_state=random_state)
    model.fit(X_train)
    return model


def score(model: IsolationForest, X) -> np.ndarray:
    """Return anomaly scores, higher means more anomalous."""

    return -model.score_samples(X)


def pick_threshold(
    normal_scores,
    tune_scores,
    tune_labels,
    max_false_alarm: float = 0.1,
) -> float:
    """Return the best scoring threshold within the false alarm budget."""

    candidates = np.quantile(
        normal_scores, np.linspace(1 - max_false_alarm, 0.999, 100)
    )

    best_threshold = float(candidates[0])
    best_s = -1.0

    for threshold in candidates:
        flags = (tune_scores > threshold).astype(int)
        s = batadal_score(tune_labels, flags)["s"]

        if s > best_s:
            best_s = s
            best_threshold = float(threshold)

    return best_threshold
