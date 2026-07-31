# -*- coding: utf-8 -*-
"""VRP 자료를 논문의 핵심 회귀식으로 분석한다.

복잡한 계량 패키지를 추가하지 않도록 선형 회귀계수와 Newey-West HAC
표준오차를 NumPy로 직접 계산한다. 설명변수를 도구변수로 쓰는 선형 GMM의
간소화 형태이며, 논문의 핵심 방향성과 유의성을 확인하는 용도다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
GROUP_COLUMNS = ["maturity", "strike"]


def normal_p_value(statistic: float) -> float:
    """표준정규분포의 양측 p-value."""
    return float(math.erfc(abs(statistic) / math.sqrt(2.0)))


def safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fit_hac_ols(
    frame: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    requested_maxlags: int = 252,
) -> dict[str, Any]:
    """OLS 계수와 Bartlett kernel Newey-West 공분산을 계산한다."""
    columns = [dependent, *regressors]
    clean = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    n_observations = len(clean)
    n_parameters = len(regressors) + 1
    minimum = max(10, n_parameters + 5)

    if n_observations < minimum:
        return {
            "status": "skipped",
            "reason": f"관측치 부족: {n_observations}건(최소 {minimum}건 필요)",
            "n": n_observations,
        }

    y = clean[dependent].to_numpy(dtype=float)
    if regressors:
        x_values = clean[regressors].to_numpy(dtype=float)
        design = np.column_stack([np.ones(n_observations), x_values])
    else:
        design = np.ones((n_observations, 1), dtype=float)

    if np.linalg.matrix_rank(design) < n_parameters:
        return {
            "status": "skipped",
            "reason": "설명변수 사이에 완전한 선형관계가 있습니다.",
            "n": n_observations,
        }

    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ y
    residual = y - design @ beta
    score = design * residual[:, None]

    maxlags = min(requested_maxlags, n_observations - 1)
    meat = score.T @ score
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        covariance_at_lag = score[lag:].T @ score[:-lag]
        meat += weight * (covariance_at_lag + covariance_at_lag.T)

    covariance = xtx_inverse @ meat @ xtx_inverse
    if n_observations > n_parameters:
        covariance *= n_observations / (n_observations - n_parameters)

    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    names = ["constant", *regressors]
    coefficients: dict[str, Any] = {}
    for index, name in enumerate(names):
        estimate = safe_number(beta[index])
        standard_error = safe_number(standard_errors[index])
        if standard_error and standard_error > 0:
            statistic = float(beta[index] / standard_errors[index])
            p_value = normal_p_value(statistic)
        else:
            statistic = None
            p_value = None
        coefficients[name] = {
            "estimate": estimate,
            "std_error": standard_error,
            "z_stat": safe_number(statistic),
            "p_value": safe_number(p_value),
        }

    total_sum_squares = float(np.sum((y - y.mean()) ** 2))
    residual_sum_squares = float(np.sum(residual ** 2))
    r_squared = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares > 0 else None

    return {
        "status": "ok",
        "n": n_observations,
        "maxlags": maxlags,
        "r_squared": safe_number(r_squared),
        "coefficients": coefficients,
    }


def add_beta_equals_one_test(result: dict[str, Any], variable: str = "iv") -> None:
    if result.get("status") != "ok":
        return
    coefficient = result["coefficients"].get(variable)
    if not coefficient:
        return
    estimate = coefficient.get("estimate")
    standard_error = coefficient.get("std_error")
    if estimate is None or not standard_error:
        return
    statistic = (estimate - 1.0) / standard_error
    result["beta_equals_one_test"] = {
        "null_hypothesis": f"{variable} coefficient = 1",
        "z_stat": safe_number(statistic),
        "p_value": safe_number(normal_p_value(statistic)),
        "rejected_at_5pct": bool(normal_p_value(statistic) < 0.05),
    }


def group_records(frame: pd.DataFrame):
    return frame.groupby(GROUP_COLUMNS, dropna=False, sort=True)


def descriptive_results(data: pd.DataFrame) -> list[dict[str, Any]]:
    results = []
    for (maturity, strike), group in group_records(data):
        fit = fit_hac_ols(group, "vrp", [])
        mean_vrp = safe_number(pd.to_numeric(group["vrp"], errors="coerce").mean())
        record: dict[str, Any] = {
            "maturity": str(maturity),
            "strike": str(strike),
            "n": int(group["vrp"].notna().sum()),
            "mean_vrp": mean_vrp,
            "mean_lvrp": safe_number(pd.to_numeric(group["lvrp"], errors="coerce").mean()),
        }
        if fit.get("status") == "ok":
            constant = fit["coefficients"]["constant"]
            record["hac_std_error"] = constant["std_error"]
            record["hac_p_value"] = constant["p_value"]
        results.append(record)
    return results


def grouped_regressions(
    data: pd.DataFrame,
    dependent: str,
    regressors: list[str],
) -> list[dict[str, Any]]:
    results = []
    for (maturity, strike), group in group_records(data):
        fit = fit_hac_ols(group, dependent, regressors)
        fit["maturity"] = str(maturity)
        fit["strike"] = str(strike)
        results.append(fit)
    return results


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"분석 파일이 없습니다: {path}\n먼저 python prepare_vrp_data.py를 실행하세요."
        )
    data = pd.read_csv(path)
    required = {"date", "maturity", "strike", "iv", "rv", "vrp", "lvrp"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {missing}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data.dropna(subset=["date"]).sort_values("date")


def analyze(data: pd.DataFrame) -> dict[str, Any]:
    usable = data.dropna(subset=["iv", "rv", "vrp", "lvrp"]).copy()
    if usable.empty:
        raise ValueError("분석 가능한 VRP가 없습니다. IV 날짜 이후의 환율 기간을 확인하세요.")

    time_variability = grouped_regressions(usable, "rv", ["iv"])
    for result in time_variability:
        add_beta_equals_one_test(result)

    source_values = (
        sorted(data["volatility_source"].dropna().astype(str).unique().tolist())
        if "volatility_source" in data.columns
        else ["unknown"]
    )
    is_proxy = any(source.startswith("sv_proxy") for source in source_values)

    output: dict[str, Any] = {
        "method": {
            "estimator": "linear regression with Newey-West HAC covariance",
            "interpretation": "논문 GMM의 간소화 구현",
            "requested_maxlags": 252,
            "volatility_source": source_values,
            "metric_name": "sv_volatility_gap_proxy" if is_proxy else "option_vrp",
            "warning": (
                "실제 옵션 내재변동성이 아닌 stochvol 추정치 기반 프록시"
                if is_proxy
                else None
            ),
        },
        "sample": {
            "rows": len(usable),
            "start": usable["date"].min().strftime("%Y-%m-%d"),
            "end": usable["date"].max().strftime("%Y-%m-%d"),
        },
        "descriptive": descriptive_results(usable),
        "time_variability_rv_on_iv": time_variability,
    }

    traditional = ["rks", "vks", "d_cs"]
    if set(traditional).issubset(usable.columns):
        output["traditional_risk_lvrp"] = grouped_regressions(
            usable, "lvrp", traditional
        )
    else:
        output["traditional_risk_lvrp"] = {
            "status": "skipped",
            "missing_columns": [column for column in traditional if column not in usable.columns],
        }

    for delta in (10, 25):
        regressors = [f"rr_{delta}", f"bf_{delta}"]
        key = f"jump_risk_{delta}delta_lvrp"
        if set(regressors).issubset(usable.columns):
            output[key] = grouped_regressions(usable, "lvrp", regressors)
        else:
            output[key] = {
                "status": "skipped",
                "missing_columns": [column for column in regressors if column not in usable.columns],
            }

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원/달러 VRP 핵심 분석")
    parser.add_argument("--input", type=Path, default=BASE_DIR / "vrp_data.csv")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "vrp_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    results = analyze(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    print(f"분석 표본: {results['sample']['rows']:,}건")
    print(f"분석 기간: {results['sample']['start']} ~ {results['sample']['end']}")
    if results["method"]["metric_name"] == "sv_volatility_gap_proxy":
        print("[주의] 실제 옵션 VRP가 아니라 SV 기반 변동성 격차 프록시 분석입니다.")
    print(f"저장 완료 -> {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"[오류] {error}") from error
