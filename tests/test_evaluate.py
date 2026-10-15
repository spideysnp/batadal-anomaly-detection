import pandas as pd
import pytest

from src.evaluate import (
    attack_windows,
    batadal_score,
    detection_delays,
    pointwise_metrics,
    report_to_flags,
)


def test_attack_windows_finds_each_run():
    assert attack_windows([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]


def test_attack_windows_empty_when_no_attacks():
    assert attack_windows([0, 0, 0]) == []


def test_delay_counts_hours_from_window_start():
    assert detection_delays([0, 1, 1, 1, 0], [0, 0, 1, 0, 0]) == [1]


def test_delay_is_zero_when_flagged_at_start():
    assert detection_delays([0, 1, 1], [0, 1, 0]) == [0]


def test_missed_window_returns_none():
    assert detection_delays([0, 1, 1, 0], [0, 0, 0, 0]) == [None]


def test_flag_before_window_does_not_count():
    assert detection_delays([0, 0, 1, 1], [1, 0, 0, 0]) == [None]


def test_delays_handle_multiple_windows():
    assert detection_delays([1, 1, 0, 1, 1], [0, 1, 0, 0, 0]) == [1, None]


def test_pointwise_metrics_basic():
    m = pointwise_metrics([0, 1, 1, 0], [0, 1, 0, 1])
    assert m["tpr"] == pytest.approx(0.5)
    assert m["tnr"] == pytest.approx(0.5)


def test_pointwise_metrics_no_predictions():
    m = pointwise_metrics([0, 1], [0, 0])
    assert m["tpr"] == 0.0
    assert m["tnr"] == 1.0


def test_rules_example_scores_0_819():
    y_true = [1 if 60 <= h < 100 else 0 for h in range(168)]
    y_pred = [1 if 68 <= h < 95 else 0 for h in range(168)]
    result = batadal_score(y_true, y_pred)
    assert result["s_ttd"] == pytest.approx(0.8)
    assert result["s_cm"] == pytest.approx(0.838, abs=1e-3)
    assert result["s"] == pytest.approx(0.819, abs=1e-3)


def test_missed_attack_scores_zero_ttd():
    result = batadal_score([0, 1, 1, 0], [0, 0, 0, 0])
    assert result["s_ttd"] == 0.0
    assert result["s_cm"] == pytest.approx(0.5)
    assert result["s"] == pytest.approx(0.25)


def test_score_without_attacks_raises():
    with pytest.raises(ValueError):
        batadal_score([0, 0], [0, 0])


def test_report_to_flags_includes_both_endpoints(tmp_path):
    report = tmp_path / "report.csv"
    report.write_text("1, 01-01-2017 02, 01-01-2017 04, T1\n")
    index = pd.date_range("2017-01-01", periods=6, freq="h")

    flags = report_to_flags(report, index)

    assert flags.tolist() == [0, 0, 1, 1, 1, 0]