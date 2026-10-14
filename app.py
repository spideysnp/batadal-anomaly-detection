import pandas as pd
import streamlit as st

from src.data import load_csv, get_features, get_labels
from src.evaluate import batadal_score, pointwise_metrics, report_to_flags
from src.model import fit, score, pick_threshold


@st.cache_resource
def run_pipeline():
    normal = load_csv("data/train_normal.csv")
    attacks = load_csv("data/train_attacks.csv")
    test = load_csv("data/test.csv")

    model = fit(get_features(normal))
    threshold = pick_threshold(
        score(model, get_features(normal)),
        score(model, get_features(attacks)),
        get_labels(attacks),
    )

    test_scores = score(model, get_features(test))
    y_test = report_to_flags("labels/test_attacks.csv", test.index)
    flags = (test_scores > threshold).astype(int)

    return test.index, test_scores, threshold, y_test, flags


index, test_scores, threshold, y_test, flags = run_pipeline()
result = batadal_score(y_test, flags)
rates = pointwise_metrics(y_test, flags)

st.title("SCADA Attack Detection on the BATADAL Water Network")
st.caption(
    "Isolation forest trained on normal operation. Threshold capped at a 10% "
    "false alarm budget. Scored once on the held out test set with the official "
    "BATADAL metric."
)

col1, col2, col3 = st.columns(3)
col1.metric("S", f"{result['s']:.3f}", f"{result['s'] - 0.75:+.3f} vs flag everything")
col2.metric("TPR", f"{rates['tpr']:.2f}")
col3.metric("TNR", f"{rates['tnr']:.2f}")

chart = pd.DataFrame(
    {
        "anomaly score": test_scores,
        "threshold": threshold,
        "attack": y_test.values * test_scores.max(),
    },
    index=index,
)
st.line_chart(chart)