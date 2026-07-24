# -*- coding: utf-8 -*-
"""
processed_news.jsonl + 실제 환율 시계열 -> impact_weights.json

목적:
    "이 카테고리(금리/무역/지정학/환율) 뉴스가 뜬 날, 실제로 환율이
    평소보다 더 많이 움직였는가?"를 통계적으로 계산해서
    카테고리별 '자체 영향도 가중치'를 만든다.

    이 가중치는 LLM이 준 confidence(주관적 확신도)와는 다르게,
    과거 실측 데이터로부터 나온 정량적 지표라서
    "왜 이 가중치를 썼는가"를 데이터로 설명할 수 있다.

방법론 (요약):
    1) Yahoo Finance에서 USD/KRW, JPY/KRW, EUR/KRW의 일별 종가를 받아온다.
    2) 일별 등락률(%)을 계산한다.
    3) 카테고리별로: "그 카테고리 뉴스가 발행된 날"의 평균 |등락률|을 구한다.
    4) 그걸 "전체 기간 평균 |등락률|"(기준선, baseline)과 비교해서
       ratio = (뉴스가 있던 날 평균 변동폭) / (전체 평균 변동폭)
       ratio가 1보다 크면 "그 카테고리 뉴스가 있는 날 환율이 평소보다 더 움직인다"는 뜻.
    5) ratio를 0~1 사이로 정규화해서 최종 가중치로 저장한다.

한계 (정직하게 밝혀둘 것):
    - NewsAPI 무료플랜 특성상 수집 기간이 짧아(약 1개월) 표본 수가 작다.
      통계적으로 엄밀한 인과관계 증명이 아니라, "탐색적 지표"로 봐야 한다.
    - 뉴스 발행 시각과 환율 종가 마감 시각의 시차(타임존)를 정교하게
      맞추지 않고, "같은 날짜"로 단순화해서 비교한다.
    - 표본이 적은 카테고리(예: 지정학)는 결과가 불안정할 수 있다.
    발표 자료에는 이 한계를 같이 언급하는 걸 추천한다 (정직하게 밝히는 게
    오히려 "설명가능성" 측면에서 좋은 인상을 준다).

사전 준비:
    pip install yfinance pandas
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

INPUT_PATH = Path("processed_news.jsonl")
OUTPUT_PATH = Path("impact_weights.json")

# Yahoo Finance 티커. KRW 기준 통화쌍.
TICKERS = {
    "USD/KRW": "USDKRW=X",
    "JPY/KRW": "JPYKRW=X",
    "EUR/KRW": "EURKRW=X",
}

CATEGORIES = ["금리", "무역", "지정학", "환율"]


# ============================================================
# 1. 뉴스 로드
# ============================================================

def load_relevant_news(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("is_relevant"):
                items.append(d)
    return items


def to_date(published_at: str) -> str | None:
    """'2026-07-23T10:11:57Z' -> '2026-07-23'"""
    if not published_at:
        return None
    try:
        return published_at[:10]
    except Exception:
        return None


# ============================================================
# 2. 환율 시계열 다운로드 + 일별 등락률 계산
# ============================================================

def fetch_fx_returns(ticker: str, start: str, end: str) -> pd.Series:
    """
    start~end 기간의 일별 종가를 받아서 % 등락률 Series를 반환.
    index: 날짜(YYYY-MM-DD 문자열), value: 그날의 |전일대비 등락률|(%)
    """
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        # yfinance 최신 버전은 티커 1개여도 MultiIndex 컬럼(DataFrame)을 반환할 때가 있음
        close = close.iloc[:, 0]

    pct_change = close.pct_change().abs() * 100  # 절대값 등락률(%)
    pct_change = pct_change.dropna()
    pct_change.index = pct_change.index.strftime("%Y-%m-%d")
    return pct_change


def build_fx_return_table(news_items: list[dict]) -> dict[str, pd.Series]:
    """뉴스에 등장한 통화쌍만 골라서 필요한 기간만큼 다운로드"""
    dates = [to_date(n.get("published_at", "")) for n in news_items]
    dates = [d for d in dates if d]
    if not dates:
        raise RuntimeError("뉴스에서 유효한 날짜를 찾지 못했습니다.")

    start = (min(dates))
    # 등락률 계산에 전일 데이터가 필요하므로 시작일을 며칠 더 앞당김
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=7)
    end_dt = datetime.strptime(max(dates), "%Y-%m-%d") + timedelta(days=1)

    fx_returns = {}
    for pair, ticker in TICKERS.items():
        series = fetch_fx_returns(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        fx_returns[pair] = series
        print(f"[{pair}] 환율 데이터 {len(series)}일치 확보")

    return fx_returns


# ============================================================
# 3. 카테고리별 영향도 ratio 계산
# ============================================================

def compute_category_ratios(news_items: list[dict], fx_returns: dict[str, pd.Series]) -> dict:
    # 통화쌍별 baseline(전체 기간 평균 |등락률|)
    baselines = {pair: series.mean() for pair, series in fx_returns.items() if len(series) > 0}

    # 카테고리별로 "그 카테고리 뉴스가 뜬 날 & 해당 통화쌍"의 등락률을 모은다
    category_changes: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    category_sample_count: dict[str, int] = {c: 0 for c in CATEGORIES}

    for item in news_items:
        category = item.get("category")
        if category not in CATEGORIES:
            continue
        date = to_date(item.get("published_at", ""))
        if not date:
            continue

        pairs = item.get("currency_pairs") or []
        if not pairs:
            # 통화쌍이 명시 안 된 뉴스는 3개 통화쌍 전체 평균 변동으로 대체
            pairs = list(TICKERS.keys())

        for pair in pairs:
            series = fx_returns.get(pair)
            if series is None or date not in series.index:
                continue
            category_changes[category].append(series.loc[date])
            category_sample_count[category] += 1

    results = {}
    overall_baseline = sum(baselines.values()) / len(baselines) if baselines else 0

    for category in CATEGORIES:
        changes = category_changes[category]
        n = category_sample_count[category]
        if n == 0 or overall_baseline == 0:
            avg_change = 0.0
            ratio = 0.0
        else:
            avg_change = sum(changes) / len(changes)
            ratio = avg_change / overall_baseline

        results[category] = {
            "sample_size": n,
            "avg_abs_change_pct": round(avg_change, 4),
            "baseline_avg_abs_change_pct": round(overall_baseline, 4),
            "raw_ratio": round(ratio, 4),
        }

    return results


def normalize_weights(ratios: dict) -> dict:
    """raw_ratio를 0.2~1.0 범위로 min-max 정규화 (0으로 완전히 죽이지 않기 위해 하한 0.2)"""
    raw_values = [v["raw_ratio"] for v in ratios.values() if v["sample_size"] > 0]
    if not raw_values:
        for v in ratios.values():
            v["weight"] = 0.5  # 표본이 아예 없으면 중립값
        return ratios

    lo, hi = min(raw_values), max(raw_values)
    for v in ratios.values():
        if v["sample_size"] == 0:
            v["weight"] = 0.5  # 표본 부족 카테고리는 중립값으로 대체
            continue
        if hi == lo:
            v["weight"] = 0.6
        else:
            normalized = (v["raw_ratio"] - lo) / (hi - lo)
            v["weight"] = round(0.2 + normalized * 0.8, 3)  # 0.2~1.0 범위로 스케일
    return ratios


# ============================================================
# 실행
# ============================================================

def main():
    news_items = load_relevant_news(INPUT_PATH)
    print(f"관련 뉴스 {len(news_items)}건 로드")

    fx_returns = build_fx_return_table(news_items)
    ratios = compute_category_ratios(news_items, fx_returns)
    weights = normalize_weights(ratios)

    print("\n=== 카테고리별 영향도 분석 결과 ===")
    for category, info in weights.items():
        print(f"{category}: weight={info['weight']}  "
              f"(raw_ratio={info['raw_ratio']}, 표본={info['sample_size']}건)")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료 -> {OUTPUT_PATH.resolve()}")
    print("\n[주의] 표본이 적은 카테고리(특히 지정학)는 결과가 불안정할 수 있습니다.")
    print("발표 자료에는 표본 크기와 함께 이 지표를 '탐색적 분석'으로 소개하는 걸 권장합니다.")


if __name__ == "__main__":
    main()
