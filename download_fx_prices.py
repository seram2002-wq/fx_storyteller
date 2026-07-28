# -*- coding: utf-8 -*-
"""
[SV 모형 1단계] 환율 원자료(종가) 다운로드 -> CSV 저장
------------------------------------------------------------
확률적 변동성(SV) 모형은 R의 stochvol 패키지로 추정하기 때문에,
Python에서는 "가격 데이터를 받아서 CSV로 넘겨주는 역할"만 한다.

이 스크립트가 만드는 fx_prices.csv를 R 스크립트(estimate_sv_volatility.R)가
읽어서 로그수익률 계산 + SV 모형 추정을 수행한다.

사전 준비:
    pip install yfinance pandas
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

NEWS_PATH = Path("processed_news.jsonl")
OUTPUT_PATH = Path("fx_prices.csv")

TICKERS = {
    "USD/KRW": "USDKRW=X",
    "JPY/KRW": "JPYKRW=X",
    "EUR/KRW": "EURKRW=X",
}


def load_news_dates() -> tuple[str, str]:
    """processed_news.jsonl에서 뉴스 발행 기간을 읽어와 필요한 다운로드 범위를 정한다"""
    dates = []
    with NEWS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            published = d.get("published_at", "")
            if published:
                dates.append(published[:10])

    if not dates:
        raise RuntimeError("processed_news.jsonl에서 유효한 날짜를 찾지 못했습니다.")

    # SV 모형은 자기상관을 학습하는 모형이라 데이터가 짧으면 추정이 불안정하다.
    # 뉴스 기간보다 넉넉하게(최소 6개월) 앞당겨서 받는다.
    start = min(dates)
    start_dt = datetime.strptime(start, "%Y-%m-%d") - timedelta(days=180)
    end_dt = datetime.strptime(max(dates), "%Y-%m-%d") + timedelta(days=1)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def download_prices(start: str, end: str) -> pd.DataFrame:
    frames = {}
    for pair, ticker in TICKERS.items():
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            print(f"[경고] {pair} 데이터를 받지 못했습니다.")
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]  # yfinance 최신 버전 MultiIndex 컬럼 대응

        frames[pair] = close
        print(f"[{pair}] {len(close)}일치 종가 확보")

    combined = pd.DataFrame(frames)
    combined.index.name = "date"
    return combined


def main():
    start, end = load_news_dates()
    print(f"다운로드 기간: {start} ~ {end}")

    prices = download_prices(start, end)
    prices.to_csv(OUTPUT_PATH)
    print(f"저장 완료 -> {OUTPUT_PATH.resolve()}")
    print("다음 단계: Rscript estimate_sv_volatility.R 실행")


if __name__ == "__main__":
    main()
