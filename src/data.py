import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a BATADAL CSV indexed by hourly DATETIME."""

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
    df = df.set_index("DATETIME").sort_index()

    return df


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return sensor columns without ATT_FLAG."""

    return df.drop(columns=["ATT_FLAG"], errors="ignore")


def get_labels(df: pd.DataFrame) -> pd.Series:
    """Return 1 for attack hours, 0 otherwise."""

    is_attack = (df["ATT_FLAG"] == 1).astype(int)

    return is_attack
