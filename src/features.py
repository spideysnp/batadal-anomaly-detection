import pandas as pd


def add_time_features(X: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Return sensor columns plus hourly change, rolling deviation, and hour of day."""

    diffs = X.diff().fillna(0).add_suffix("_diff")
    rolling_mean = X.rolling(window, min_periods=1).mean()
    devs = (X - rolling_mean).add_suffix("_dev")
    hour = pd.Series(pd.DatetimeIndex(X.index).hour, index=X.index, name="hour")

    return pd.concat([X, diffs, devs, hour], axis=1)