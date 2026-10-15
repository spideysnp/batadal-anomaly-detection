import numpy as np

from src.model import pick_threshold


def test_threshold_respects_false_alarm_budget():
    rng = np.random.default_rng(0)
    normal_scores = rng.normal(size=1000)
    tune_scores = rng.normal(size=200)
    tune_labels = np.zeros(200, dtype=int)
    tune_labels[50:60] = 1

    threshold = pick_threshold(normal_scores, tune_scores, tune_labels)

    assert threshold >= np.quantile(normal_scores, 0.9)
    assert (normal_scores > threshold).mean() <= 0.1