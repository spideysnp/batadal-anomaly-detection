from pathlib import Path

import numpy as np
import pandas as pd

Labels = list[int] | pd.Series | np.ndarray


def attack_windows(y_true: Labels) -> list[tuple[int, int]]:
    """Return (start, end) positions of each run of 1s, both ends inclusive."""

    runs = []
    start = None

    for i, value in enumerate(y_true):
        if value == 1 and start is None:
            start = i

        elif value == 0 and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(y_true) - 1))

    return runs


def detection_delays(y_true: Labels, y_pred: Labels) -> list[int | None]:
    """Return hours to first detection per attack window, None if missed."""

    y_true = list(y_true)
    y_pred = list(y_pred)
    delays = []

    for start, end in attack_windows(y_true):
        window = y_pred[start : end + 1]
        delay = next(
            (i for i, value in enumerate(window) if value == 1),
            None,
        )
        delays.append(delay)

    return delays


def pointwise_metrics(y_true: Labels, y_pred: Labels) -> dict[str, float]:
    """Return true positive rate and true negative rate."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0

    return {"tpr": tpr, "tnr": tnr}


def batadal_score(
    y_true: Labels,
    y_pred: Labels,
    gamma: float = 0.5,
) -> dict[str, float]:
    """Return S_TTD, S_CM, and the combined BATADAL score S."""

    windows = attack_windows(y_true)
    if not windows:
        raise ValueError("y_true contains no attack windows")

    delays = detection_delays(y_true, y_pred)

    ratios = [
        1.0 if delay is None else delay / (end - start + 1)
        for (start, end), delay in zip(windows, delays)
    ]
    s_ttd = 1 - sum(ratios) / len(ratios)

    m = pointwise_metrics(y_true, y_pred)
    s_cm = (m["tpr"] + m["tnr"]) / 2

    s = gamma * s_ttd + (1 - gamma) * s_cm

    return {"s_ttd": s_ttd, "s_cm": s_cm, "s": s}


def report_to_flags(report_path: str | Path, index: pd.DatetimeIndex) -> pd.Series:
    """Return an hourly 0/1 track for the events in a BATADAL report."""

    report = pd.read_csv(
        report_path,
        header=None,
        names=["id", "start", "end", "devices"],
        skipinitialspace=True,
    ).dropna(subset=["start"])

    starts = pd.to_datetime(report["start"], format="%d-%m-%Y %H")
    ends = pd.to_datetime(report["end"], format="%d-%m-%Y %H")

    flags = pd.Series(0, index=index)

    for start, end in zip(starts, ends):
        flags.loc[start:end] = 1

    return flags