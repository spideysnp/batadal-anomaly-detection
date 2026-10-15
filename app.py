from pathlib import Path

import pandas as pd
import streamlit as st

from src.data import get_features, get_labels, load_csv
from src.evaluate import (
    attack_windows,
    batadal_score,
    detection_delays,
    pointwise_metrics,
    report_to_flags,
)
from src.model import fit, pick_threshold, score

root = Path(__file__).parent


@st.cache_resource
def run_pipeline():
    normal = load_csv(root / "data" / "train_normal.csv")
    attacks = load_csv(root / "data" / "train_attacks.csv")
    test = load_csv(root / "data" / "test.csv")

    model = fit(get_features(normal))
    threshold = pick_threshold(
        score(model, get_features(normal)),
        score(model, get_features(attacks)),
        get_labels(attacks),
    )

    test_scores = score(model, get_features(test))
    y_test = report_to_flags(root / "labels" / "test_attacks.csv", test.index)
    flags = (test_scores > threshold).astype(int)

    return test.index, test_scores, threshold, y_test, flags


st.title("SCADA Attack Detection on a Water Distribution Network")

try:
    index, test_scores, threshold, y_test, flags = run_pipeline()
except FileNotFoundError as err:
    st.error(f"Missing {err.filename or err}. See the Data section of README.md.")
    st.stop()

result = batadal_score(y_test, flags)
rates = pointwise_metrics(y_test, flags)

st.caption(
    "Isolation forest trained on normal operation. Threshold capped at a 10% "
    "false alarm budget. Scored once on the held out test set with the official "
    "BATADAL metric."
)

col1, col2, col3 = st.columns(3)
col1.metric(
    "S",
    f"{result['s']:.3f}",
    # 0.75 is S for a detector that flags every hour
    f"{result['s'] - 0.75:+.3f} vs flag everything",
)
col2.metric("TPR", f"{rates['tpr']:.2f}")
col3.metric("TNR", f"{rates['tnr']:.2f}")

low, high = test_scores.min(), test_scores.max()
chart = pd.DataFrame(
    {
        "anomaly score": test_scores,
        "threshold": threshold,
        "attack": low + y_test.values * (high - low),
    },
    index=index,
)
st.line_chart(chart)

windows = attack_windows(y_test)
delays = detection_delays(y_test, flags)
delay_table = pd.DataFrame(
    {
        "attack start": [index[start] for start, _ in windows],
        "duration (hours)": [end - start + 1 for start, end in windows],
        "hours to first flag": delays,
    }
)

st.subheader("Detection delay per attack")
st.dataframe(delay_table, hide_index=True)