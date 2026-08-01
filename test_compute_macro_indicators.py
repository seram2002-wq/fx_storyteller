import json
from pathlib import Path

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from compute_macro_indicators import (
    build_japan_policy_rate_summary,
    build_weekly_feature_summary,
    compute_rate_diffs,
)


def test_build_weekly_feature_summary_uses_latest_row(tmp_path):
    csv_path = tmp_path / "weekly_fx_features.csv"
    pd.DataFrame(
        [
            {"week_ending": "2024-01-05", "usd_krw": 1300.0, "us_kr_policy_spread": 0.5},
            {"week_ending": "2024-01-12", "usd_krw": 1320.0, "us_kr_policy_spread": 0.7},
        ]
    ).to_csv(csv_path, index=False)

    summary = build_weekly_feature_summary(csv_path)

    assert summary["latest_week"] == "2024-01-12"
    assert summary["usd_krw"]["latest"] == 1320.0
    assert summary["usd_krw"]["change"] == 20.0
    assert summary["us_kr_policy_spread"]["latest"] == 0.7


def test_build_japan_policy_rate_summary_uses_latest_values():
    df = pd.DataFrame(
        [
            {"date": "2024-01-01", "policy_rate": 0.1},
            {"date": "2024-01-08", "policy_rate": 0.2},
        ]
    )

    summary = build_japan_policy_rate_summary(df)

    assert summary["latest"] == 0.2
    assert summary["prev"] == 0.1
    assert summary["change"] == 0.1
    assert summary["trend"] == "상승"


def test_compute_rate_diffs_uses_weekly_feature_fallback(tmp_path):
    weekly_path = tmp_path / "weekly_fx_features.csv"
    pd.DataFrame(
        [
            {"week_ending": "2024-01-05", "kr_policy_rate": 2.5, "us_policy_rate": 3.5},
            {"week_ending": "2024-01-12", "kr_policy_rate": 2.6, "us_policy_rate": 3.4},
        ]
    ).to_csv(weekly_path, index=False)

    result = compute_rate_diffs({}, weekly_csv_path=weekly_path)

    assert result["USD/KRW"]["금리차_pp"] == -0.8
