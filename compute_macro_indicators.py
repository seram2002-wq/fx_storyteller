# -*- coding: utf-8 -*-
"""
macro_data.json (원자료) -> processed_macro.json (파생지표)

collect_macro_data.py가 받아온 원자료는 그대로 쓰기엔 단위/주기가 제각각이라
(금리는 %, 유가는 $/배럴, CPI는 지수...) 아래 파생지표로 가공한다.

만드는 파생지표:
    1) 금리차 (한국 - 상대국), 통화쌍 기준 (USD/KRW, JPY/KRW, EUR/KRW)
    2) CPI 전년동월비(%) 변화 추이 (이미 %로 오는 경우 그대로, 지수로 오면 계산)
    3) 환율 이동평균(5일/20일) + 피벗포인트 (전일 고/저/종가 기반)
    4) 각 지표의 "최신값 + 최근 변화 방향" 요약 (5단계 프롬프트에 바로 넣기 좋은 형태)

이 스크립트는 collect_macro_data.py가 만든 macro_data.json이 있어야 동작한다.
환율 자체의 고가/저가/종가는 compute_impact_weights.py와 동일한 티커로 별도 조회한다
(피벗포인트 계산에 고가/저가가 필요한데, 그 스크립트는 종가만 받아오기 때문).

사전 준비:
    pip install pandas yfinance
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
MACRO_INPUT_PATH = BASE_DIR / "macro_data.json"
OUTPUT_PATH = BASE_DIR / "processed_macro.json"

FX_TICKERS = {
    "USD/KRW": "USDKRW=X",
    "JPY/KRW": "JPYKRW=X",
    "EUR/KRW": "EURKRW=X",
}

# 통화쌍 <-> 어느 나라 금리와 비교할지
PAIR_TO_RATE_KEY = {
    "USD/KRW": "미국_기준금리",
    "JPY/KRW": "일본_기준금리",
    "EUR/KRW": "유럽_기준금리",
}

END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=90)  # 이동평균/피벗은 최근 3개월이면 충분


# ============================================================
# 1. 로드
# ============================================================

def load_macro_data() -> dict:
    if not MACRO_INPUT_PATH.exists():
        raise RuntimeError(
            f"{MACRO_INPUT_PATH}가 없습니다. 먼저 collect_macro_data.py를 실행하세요."
        )
    with MACRO_INPUT_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_macro_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        raise ValueError("날짜 값이 비었습니다")

    text = str(value).strip()
    if not text:
        raise ValueError("날짜 값이 비었습니다")

    # ECOS는 YYYYMMDD, YYYYMM, YYYYQn 형태를 자주 사용
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d")
    if re.fullmatch(r"\d{6}", text):
        return pd.to_datetime(text, format="%Y%m")
    if re.fullmatch(r"\d{4}Q[1-4]", text):
        return pd.to_datetime(text, format="%YQ%q")
    if re.fullmatch(r"\d{4}", text):
        return pd.to_datetime(text, format="%Y")

    try:
        return pd.to_datetime(text)
    except Exception:
        # 마지막 수단: 문자열에서 숫자형 패턴만 남긴 뒤 파싱
        digits = re.sub(r"\D", "", text)
        if len(digits) == 8:
            return pd.to_datetime(digits, format="%Y%m%d")
        if len(digits) == 6:
            return pd.to_datetime(digits, format="%Y%m")
        raise


def series_to_df(series: list[dict]) -> pd.Series:
    """[{'date':..., 'value':...}] -> pandas Series (index=날짜)"""
    if not series:
        return pd.Series(dtype=float)
    df = pd.DataFrame(series)
    df["date"] = df["date"].apply(_parse_macro_date)
    return df.set_index("date")["value"].sort_index()


def latest_value_and_trend(s: pd.Series) -> dict:
    """최신값 + 최근 변화 방향 요약. 5단계 프롬프트에 바로 넣기 좋은 형태."""
    if s.empty:
        return {"latest": None, "prev": None, "change": None, "trend": "데이터없음"}

    latest = float(s.iloc[-1])
    prev = float(s.iloc[-2]) if len(s) >= 2 else None
    change = round(latest - prev, 4) if prev is not None else None
    if change is None:
        trend = "데이터부족"
    elif change > 0:
        trend = "상승"
    elif change < 0:
        trend = "하락"
    else:
        trend = "보합"

    return {
        "latest": round(latest, 4),
        "latest_date": s.index[-1].strftime("%Y-%m-%d"),
        "prev": round(prev, 4) if prev is not None else None,
        "change": change,
        "trend": trend,
    }


# ============================================================
# 2. 금리차 계산
# ============================================================

def compute_rate_diffs(macro: dict) -> dict:
    fred = macro.get("fred", {})
    ecos = macro.get("ecos", {})

    kr_rate = series_to_df(ecos.get("한국_기준금리", []))
    result = {}

    for pair, rate_key in PAIR_TO_RATE_KEY.items():
        other_rate = series_to_df(fred.get(rate_key, []))
        if kr_rate.empty or other_rate.empty:
            result[pair] = {"latest_diff_pp": None, "note": "데이터 부족"}
            continue

        # 날짜 주기가 다르므로(한국 일간 vs 미국/일본/유럽 월간) 각각의 최신값끼리 비교
        kr_latest = float(kr_rate.iloc[-1])
        other_latest = float(other_rate.iloc[-1])
        diff = round(kr_latest - other_latest, 3)

        result[pair] = {
            "한국_기준금리": kr_latest,
            f"{rate_key.split('_')[0]}_기준금리": other_latest,
            "금리차_pp": diff,  # 양수면 한국이 더 높음, 음수면 상대국이 더 높음
            "해석": "한국 금리가 더 높음" if diff > 0 else ("한국 금리가 더 낮음" if diff < 0 else "동일"),
        }

    return result


# ============================================================
# 3. 환율 이동평균 + 피벗포인트
# ============================================================

def fetch_fx_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_fx_technical(pair: str, ticker: str) -> dict:
    start = START_DATE.strftime("%Y-%m-%d")
    end = (END_DATE + timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch_fx_ohlc(ticker, start, end)

    if df.empty or len(df) < 2:
        return {"note": "데이터 부족"}

    close = df["Close"]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None

    # 피벗포인트: 전일 고/저/종가 기준 (가장 흔한 클래식 공식)
    prev_high = float(df["High"].iloc[-2])
    prev_low = float(df["Low"].iloc[-2])
    prev_close = float(df["Close"].iloc[-2])
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high

    latest_close = float(close.iloc[-1])

    return {
        "latest_close": round(latest_close, 3),
        "ma5": round(float(ma5), 3) if pd.notna(ma5) else None,
        "ma20": round(float(ma20), 3) if ma20 is not None and pd.notna(ma20) else None,
        "ma_signal": "단기 상승추세(5일선>20일선)" if (ma20 is not None and pd.notna(ma20) and ma5 > ma20)
                     else ("단기 하락추세(5일선<20일선)" if ma20 is not None and pd.notna(ma20) else "판단불가(데이터부족)"),
        "pivot": round(pivot, 3),
        "resistance_r1": round(r1, 3),
        "support_s1": round(s1, 3),
        "position_vs_pivot": "피벗 상단(강세권)" if latest_close > pivot else "피벗 하단(약세권)",
    }


# ============================================================
# 실행
# ============================================================

def main():
    macro = load_macro_data()

    print("=== 금리차 계산 ===")
    rate_diffs = compute_rate_diffs(macro)
    for pair, info in rate_diffs.items():
        print(f"{pair}: {info}")

    print("\n=== 환율 기술적 지표 계산 ===")
    fx_technical = {}
    for pair, ticker in FX_TICKERS.items():
        fx_technical[pair] = compute_fx_technical(pair, ticker)
        print(f"{pair}: {fx_technical[pair]}")

    print("\n=== 기타 매크로 지표 요약 (최신값 + 추세) ===")
    other_summary = {}
    for source_key in ["fred", "ecos"]:
        for name, series in macro.get(source_key, {}).items():
            if "기준금리" in name:
                continue  # 이미 금리차에서 다룸
            other_summary[name] = latest_value_and_trend(series_to_df(series))

    for name, series in macro.get("market", {}).items():
        other_summary[name] = latest_value_and_trend(series_to_df(series))

    other_summary["외국인_코스피_순매수"] = latest_value_and_trend(
        series_to_df(macro.get("foreign_netflow_kospi", []))
    )
    other_summary["구글트렌드_환율관심도"] = latest_value_and_trend(
        series_to_df(macro.get("google_trends", []))
    )

    for name, info in other_summary.items():
        print(f"{name}: {info}")

    processed = {
        "computed_at": datetime.now().isoformat(),
        "rate_diffs": rate_diffs,
        "fx_technical": fx_technical,
        "other_indicators": other_summary,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료 -> {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
