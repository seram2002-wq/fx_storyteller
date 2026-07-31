# -*- coding: utf-8 -*-
"""원/달러 기대변동성과 환율 가격으로 VRP 또는 프록시 분석 자료를 만든다.

필수 입력
---------
fx_prices.csv
    date, USD/KRW 열을 가진 일별 환율 파일

기대변동성 입력(둘 중 하나)
-------------------------
fx_option_iv.csv
    date, maturity, strike, iv 열을 가진 옵션 내재변동성 파일

fx_option_iv.csv가 없으면 기존 sv_volatility.json의 USD/KRW 일별 sigma를
자동으로 연율화하여 1개월 ATM 기대변동성 프록시로 사용한다. 이 값은 실제
옵션 내재변동성이 아니므로 결과에도 sv_proxy라고 명시한다.

가장 단순한 1개월 ATM 분석만 할 때는 fx_option_iv.csv에 date, iv만
넣어도 된다. 이 경우 maturity=1m, strike=ATM으로 처리한다.

선택 입력
--------
vrp_factors.csv
    date와 rks, vks, d_cs 중 사용 가능한 열을 가진 위험요인 파일

출력
----
vrp_data.csv
    IV, 미래 실현변동성(RV), VRP, LVRP, RR, BF를 합친 분석 자료
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MATURITY_DAYS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
VALID_STRIKES = {"10P", "25P", "ATM", "25C", "10C"}


def normalize_maturity(value: object) -> str:
    text = (
        str(value).strip().lower().replace("개월", "m").replace("month", "m").replace(" ", "")
    )
    aliases = {
        "1": "1m", "1m": "1m",
        "3": "3m", "3m": "3m",
        "6": "6m", "6m": "6m",
        "12": "12m", "12m": "12m", "1y": "12m",
    }
    if text not in aliases:
        raise ValueError(f"지원하지 않는 만기입니다: {value!r}")
    return aliases[text]


def normalize_strike(value: object) -> str:
    text = str(value).strip().upper().replace("-DELTA", "").replace(" ", "")
    aliases = {
        "10PUT": "10P", "25PUT": "25P", "50": "ATM", "50D": "ATM",
        "ATM": "ATM", "25CALL": "25C", "10CALL": "10C",
    }
    text = aliases.get(text, text)
    if text not in VALID_STRIKES:
        raise ValueError(f"지원하지 않는 행사가격 구분입니다: {value!r}")
    return text


def load_prices(path: Path, requested_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"환율 파일이 없습니다: {path}")

    prices = pd.read_csv(path)
    if "date" not in prices.columns:
        raise ValueError(f"{path.name}에 date 열이 필요합니다.")

    candidates = [requested_column, "USD/KRW", "USD.KRW", "USDKRW", "USDKRW=X"]
    price_column = next((column for column in candidates if column in prices.columns), None)
    if price_column is None:
        raise ValueError(
            f"원/달러 가격 열을 찾지 못했습니다. 실제 열: {list(prices.columns)}"
        )

    prices = prices[["date", price_column]].rename(columns={price_column: "spot"})
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["spot"] = pd.to_numeric(prices["spot"], errors="coerce")
    prices = (
        prices.dropna(subset=["date", "spot"])
        .loc[lambda frame: frame["spot"] > 0]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(prices) < 23:
        raise ValueError("미래 1개월 RV 계산에는 최소 23개 이상의 가격 관측치가 필요합니다.")
    return prices


def load_iv(path: Path, unit: str) -> pd.DataFrame:
    if not path.exists():
        example = "date,maturity,strike,iv\n2026-01-02,1m,ATM,12.5"
        raise FileNotFoundError(
            f"옵션 내재변동성 파일이 없습니다: {path}\n"
            "다음 형식으로 파일을 준비하세요. IV는 0.125 또는 12.5 모두 지원합니다.\n"
            f"{example}"
        )

    iv = pd.read_csv(path)
    iv.columns = [str(column).strip().lower() for column in iv.columns]
    required = {"date", "iv"}
    if not required.issubset(iv.columns):
        raise ValueError(f"{path.name}에는 최소 date, iv 열이 필요합니다.")

    if "maturity" not in iv.columns:
        iv["maturity"] = "1m"
    if "strike" not in iv.columns:
        iv["strike"] = "ATM"

    iv["date"] = pd.to_datetime(iv["date"], errors="coerce")
    iv["iv"] = pd.to_numeric(iv["iv"], errors="coerce")
    iv = iv.dropna(subset=["date", "iv"]).copy()
    iv = iv.loc[iv["iv"] > 0].copy()
    if iv.empty:
        raise ValueError("유효한 양(+)의 IV 값이 없습니다.")

    iv["maturity"] = iv["maturity"].map(normalize_maturity)
    iv["strike"] = iv["strike"].map(normalize_strike)

    if unit == "percent" or (unit == "auto" and float(iv["iv"].median()) > 1.5):
        iv["iv"] = iv["iv"] / 100.0

    if float(iv["iv"].max()) > 3:
        raise ValueError("IV 단위를 확인하세요. 소수(0.125) 또는 퍼센트(12.5) 형식이어야 합니다.")

    result = (
        iv.groupby(["date", "maturity", "strike"], as_index=False)["iv"]
        .mean()
        .sort_values(["date", "maturity", "strike"])
    )
    result["volatility_source"] = "option_implied_volatility"
    return result


def load_sv_proxy(path: Path, requested_pair: str = "USD/KRW") -> pd.DataFrame:
    """stochvol의 일별 sigma(퍼센트)를 연율화한 기대변동성 프록시로 변환한다."""
    if not path.exists():
        raise FileNotFoundError(
            f"옵션 IV와 SV 파일이 모두 없습니다. 필요한 파일: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    normalized_target = requested_pair.replace("/", "").replace(".", "").upper()
    pair_key = next(
        (
            key
            for key in raw
            if str(key).replace("/", "").replace(".", "").upper() == normalized_target
        ),
        None,
    )
    if pair_key is None:
        raise ValueError(
            f"{path.name}에서 {requested_pair} 항목을 찾지 못했습니다. 실제 항목: {list(raw)}"
        )

    records = raw[pair_key]
    if isinstance(records, dict):
        records = [records]
    sv = pd.DataFrame(records)
    if not {"date", "sigma"}.issubset(sv.columns):
        raise ValueError(f"{path.name}의 {pair_key} 항목에 date, sigma가 필요합니다.")

    sv["date"] = pd.to_datetime(sv["date"], errors="coerce")
    sv["sigma"] = pd.to_numeric(sv["sigma"], errors="coerce")
    sv = sv.dropna(subset=["date", "sigma"]).loc[lambda frame: frame["sigma"] > 0].copy()
    if sv.empty:
        raise ValueError(f"{path.name}에 유효한 {pair_key} sigma 값이 없습니다.")

    # estimate_sv_volatility.R은 100 * diff(log(price))로 수익률을 만들기 때문에
    # sigma의 단위는 '일별 퍼센트'다. 이를 연율 소수 단위로 맞춘다.
    sv["iv"] = sv["sigma"] / 100.0 * np.sqrt(252.0)
    sv["maturity"] = "1m"
    sv["strike"] = "ATM"
    sv["volatility_source"] = "sv_proxy_annualized_daily_sigma"
    return (
        sv[["date", "maturity", "strike", "iv", "volatility_source"]]
        .drop_duplicates("date", keep="last")
        .sort_values("date")
    )


def add_forward_realized_volatility(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    returns = np.log(result["spot"]).diff().to_numpy(dtype=float)
    n_rows = len(result)

    for maturity, horizon in MATURITY_DAYS.items():
        rv = np.full(n_rows, np.nan, dtype=float)
        for index in range(n_rows):
            end = index + horizon + 1
            if end > n_rows:
                break
            future_returns = returns[index + 1:end]
            if len(future_returns) == horizon and np.isfinite(future_returns).all():
                rv[index] = float(np.std(future_returns, ddof=1) * np.sqrt(252.0))
        result[f"rv_{maturity}"] = rv

    return result


def add_rr_bf(data: pd.DataFrame) -> pd.DataFrame:
    surface = data.pivot_table(
        index=["date", "maturity"], columns="strike", values="iv", aggfunc="mean"
    )

    for delta in (10, 25):
        call_name, put_name = f"{delta}C", f"{delta}P"
        if {call_name, put_name}.issubset(surface.columns):
            surface[f"rr_{delta}"] = surface[call_name] - surface[put_name]
            if "ATM" in surface.columns:
                surface[f"bf_{delta}"] = (
                    (surface[call_name] + surface[put_name]) / 2.0 - surface["ATM"]
                )

    extra_columns = [column for column in surface.columns if str(column).startswith(("rr_", "bf_"))]
    if not extra_columns:
        return data

    factors = surface[extra_columns].reset_index()
    return data.merge(factors, on=["date", "maturity"], how="left")


def merge_optional_factors(data: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        if path is not None:
            print(f"[안내] 선택 파일 {path.name}이 없어 RKS/VKS/ΔCS 분석은 건너뜁니다.")
        return data

    factors = pd.read_csv(path)
    factors.columns = [str(column).strip().lower() for column in factors.columns]
    if "date" not in factors.columns:
        raise ValueError(f"{path.name}에 date 열이 필요합니다.")
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    factors = factors.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return data.merge(factors, on="date", how="left")


def build_dataset(prices: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    prices_with_rv = add_forward_realized_volatility(prices)
    data = iv.merge(prices_with_rv, on="date", how="left")
    data["horizon_days"] = data["maturity"].map(MATURITY_DAYS)
    data["rv"] = np.nan

    for maturity in MATURITY_DAYS:
        mask = data["maturity"].eq(maturity)
        data.loc[mask, "rv"] = data.loc[mask, f"rv_{maturity}"]

    data["vrp"] = data["iv"] - data["rv"]
    valid = (data["iv"] > 0) & (data["rv"] > 0)
    data["lvrp"] = np.where(valid, np.log(data["iv"] / data["rv"]), np.nan)
    proxy_mask = data.get(
        "volatility_source", pd.Series("option_implied_volatility", index=data.index)
    ).astype(str).str.startswith("sv_proxy")
    data["is_proxy"] = proxy_mask
    data["metric_label"] = np.where(
        proxy_mask,
        "SV 기반 변동성 격차 프록시",
        "옵션 내재변동성 위험 프리미엄",
    )
    data = add_rr_bf(data)

    helper_columns = [f"rv_{maturity}" for maturity in MATURITY_DAYS]
    return data.drop(columns=helper_columns).sort_values(["date", "maturity", "strike"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="원/달러 변동성 위험 프리미엄 자료 생성")
    parser.add_argument("--prices", type=Path, default=BASE_DIR / "fx_prices.csv")
    parser.add_argument("--iv", type=Path, default=BASE_DIR / "fx_option_iv.csv")
    parser.add_argument("--sv", type=Path, default=BASE_DIR / "sv_volatility.json")
    parser.add_argument("--factors", type=Path, default=BASE_DIR / "vrp_factors.csv")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "vrp_data.csv")
    parser.add_argument("--price-column", default="USD/KRW")
    parser.add_argument("--iv-unit", choices=["auto", "decimal", "percent"], default="auto")
    parser.add_argument(
        "--volatility-source",
        choices=["auto", "option", "sv"],
        default="auto",
        help="auto는 옵션 IV 파일이 있으면 option, 없으면 sv 프록시를 사용",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = load_prices(args.prices, args.price_column)
    if args.volatility_source == "option":
        iv = load_iv(args.iv, args.iv_unit)
    elif args.volatility_source == "sv":
        iv = load_sv_proxy(args.sv)
    elif args.iv.exists():
        iv = load_iv(args.iv, args.iv_unit)
    else:
        print(f"[안내] {args.iv.name}이 없어 {args.sv.name}의 SV 프록시를 사용합니다.")
        iv = load_sv_proxy(args.sv)
    data = build_dataset(prices, iv)
    data = merge_optional_factors(data, args.factors)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False, encoding="utf-8-sig")

    usable = int(data["vrp"].notna().sum())
    print(f"환율 가격: {len(prices):,}건")
    print(f"기대변동성 입력: {len(iv):,}건")
    print(f"VRP 계산 가능: {usable:,}건")
    if bool(data["is_proxy"].any()):
        print("[주의] 결과는 실제 옵션 VRP가 아니라 SV 기반 변동성 격차 프록시입니다.")
    print(f"저장 완료 -> {args.output.resolve()}")
    if usable == 0:
        print("[확인] 옵션 날짜 이후에 만기만큼의 환율 데이터가 존재해야 RV가 계산됩니다.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"[오류] {error}") from error
