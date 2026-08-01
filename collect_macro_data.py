# -*- coding: utf-8 -*-
"""
매크로 지표 수집 -> macro_data.json

수집 대상 (9개):
    1) 한/미/일/유럽 기준금리          - ECOS(한국) + FRED(미국/일본/유럽)
    2) 물가상승률(CPI)                 - ECOS(한국) + FRED(미국/일본/유럽)
    3) 무역수지                        - ECOS(한국만; 미/일/유럽은 범위 밖으로 두고 필요시 확장)
    4) GDP 성장률                      - ECOS(한국) + FRED(미국/일본/유럽)
    5) VIX(변동성지수)                 - yfinance
    6) 미국 국채금리(10년물)           - yfinance
    7) 외국인 코스피 순매수/순매도     - pykrx (비공식, KRX 스크레이핑 기반 무료 패키지)
    8) 유가(WTI)                       - yfinance
    9) Google Trends 검색량            - pytrends (비공식 Google Trends 래퍼)

신뢰도 안내 (정직하게):
    - ECOS(한국은행), FRED(세인트루이스 연준)는 공식 기관 API라서 신뢰도가 높음.
    - pykrx, pytrends는 "비공식" 라이브러리로, 해당 사이트 구조가 바뀌면 깨질 수 있음.
      발표 자료에는 이 부분을 "참고용 보조 지표"로 소개하는 걸 권장.
    - 기준금리/CPI/GDP의 FRED 시리즈 코드 중 "TODO 확인" 표시가 있는 것들은
      국가별로 갱신 주기가 다르거나 시리즈가 개편될 수 있으니, 실행 전에
      fred_search_series() 함수로 한 번 검색해서 실제 존재/최신 여부를 확인할 것.

사전 준비:
    pip install requests python-dotenv pandas yfinance pykrx pytrends

    .env 파일에 아래 두 키를 추가:
        FRED_API_KEY=발급받은_키        (https://fred.stlouisfed.org/docs/api/api_key.html, 무료)
        ECOS_API_KEY=발급받은_키        (https://ecos.bok.or.kr/api/ , 무료)

사용법:
    python collect_macro_data.py
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from pykrx import stock as pkstock
from pykrx import bond as pkbond

import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

FRED_API_KEY = os.environ.get("FRED_API_KEY")
ECOS_API_KEY = os.environ.get("ECOS_API_KEY")

OUTPUT_PATH = BASE_DIR / "macro_data.json"

# 조회 기간 (기본: 최근 1년치. 뉴스 수집 기간이 짧아도 지표는 넉넉히 받아서
# 이동평균/추세 계산에 여유를 둔다)
END_DATE = datetime.today()
START_DATE = END_DATE - timedelta(days=365)


# ============================================================
# 0. FRED 공용 유틸
# ============================================================

FRED_SERIES_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_SEARCH_URL = "https://api.stlouisfed.org/fred/series/search"


def fetch_fred_series(series_id: str, start: str, end: str) -> list[dict]:
    """FRED 시리즈 하나를 [{'date': ..., 'value': ...}, ...] 형태로 반환.
    값이 없는 날(FRED에서 '.'으로 표시)은 건너뛴다."""
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY가 .env에 없습니다.")

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(FRED_SERIES_URL, params=params, timeout=20)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    out = []
    for o in obs:
        if o["value"] == ".":
            continue
        out.append({"date": o["date"], "value": float(o["value"])})
    return out


def fred_search_series(keyword: str, limit: int = 10) -> list[dict]:
    """FRED에서 키워드로 시리즈를 검색. 코드가 맞는지 확인하거나
    새로운 지표를 찾을 때 이 함수를 먼저 호출해볼 것.
    예: fred_search_series(
    "CPALTT01USM657N")  # 미국 CPI 전년동월비(%)
    """
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY가 .env에 없습니다.")

    params = {
        "search_text": keyword,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
    }
    resp = requests.get(FRED_SEARCH_URL, params=params, timeout=20)
    resp.raise_for_status()
    series = resp.json().get("seriess", [])
    print(f"[FRED] '{keyword}' 검색 결과 {len(series)}건")

    return [
        {"id": s["id"], "title": s["title"], "frequency": s["frequency"],
         "last_updated": s["last_updated"]}
        for s in series
    ]
    


# FRED 시리즈 코드. 확인된 것과 확인이 더 필요한 것을 구분해둠.
FRED_SERIES = {
    "미국_기준금리": "FEDFUNDS",        # 확인됨: 미 연준 실효 기준금리(월간)
    "미국_국채10년물": "DGS10",          # 확인됨: 10년물 국채 수익률(일간) - yfinance(^TNX)와 대조 가능
    "미국_CPI": "CPIAUCSL",       # TODO 확인: OECD MEI 기반 미국 CPI 전년동월비(%)
    "미국_GDP성장률": "NAEXKP01USQ657S",  # TODO 확인: OECD MEI 기반 미국 실질GDP 성장률(분기)
    "일본_기준금리": "INTDSRJPM193N",    # TODO 확인: 일본 할인율. BOJ 정책금리(단기금리목표)와 다를 수 있음
    "일본_CPI": "JPNPCPIPCPPPT",       # TODO 확인
    "일본_GDP성장률": "NAEXKP01JPQ657S", # TODO 확인
    "유럽_기준금리": "ECBDFR",           # TODO 확인: ECB 예금금리(Deposit Facility Rate)
    "유럽_CPI": "CP0000EU272020M086NEST",       # TODO 확인: 유로존 CPI 전년동월비(%)
    "유럽_GDP성장률": "CLVMNACSCAB1GQEU272020", # TODO 확인
    "한국_GDP성장률": "NGDPRSAXDCKRQ"     # TODO 확인 
}


def collect_fred_indicators() -> dict:
    start = START_DATE.strftime("%Y-%m-%d")
    end = END_DATE.strftime("%Y-%m-%d")
    result = {}
    for name, series_id in FRED_SERIES.items():
        try:
            result[name] = fetch_fred_series(series_id, start, end)
            print(f"[FRED] {name} ({series_id}) - {len(result[name])}건 확보")
        except Exception as e:
            print(f"[FRED] {name} ({series_id}) 실패: {e}")
            result[name] = []
        time.sleep(0.2)  # FRED 요청 제한 완충
    return result


# ============================================================
# 1. ECOS(한국은행) 공용 유틸 - 한국 지표 전담
# ============================================================

def fetch_ecos_series(stat_code: str, item_code1: str, period: str,
                       start: str, end: str) -> list[dict]:
    """
    period: "D"(일간) / "M"(월간) / "Q"(분기) / "A"(연간)
    start/end 포맷은 period에 맞춰야 함 (D: YYYYMMDD, M: YYYYMM, Q: YYYYQ#, A: YYYY)
    """
    if not ECOS_API_KEY:
        raise RuntimeError("ECOS_API_KEY가 .env에 없습니다.")

    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}"
        f"/json/kr/1/1000/{stat_code}/{period}/{start}/{end}/{item_code1}"
    )
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if "StatisticSearch" not in data:
        # 인증키 오류, 코드 오류 등은 여기로 옴 (data에 RESULT 에러 메시지가 담김)
        print(f"[ECOS] 응답에 데이터 없음: {data}")
        return []

    rows = data["StatisticSearch"].get("row", [])
    return [{"date": r["TIME"], "value": float(r["DATA_VALUE"])} for r in rows]


# 한국은행 ECOS 통계코드 (실행 전 https://ecos.bok.or.kr 에서 최신 여부 확인 권장)
ECOS_TARGETS = {
    "한국_기준금리": {"stat_code": "722Y001", "item_code1": "0101000", "period": "D"},
    "한국_CPI": {"stat_code": "901Y009", "item_code1": "0", "period": "M"},
    "한국_무역수지": {"stat_code": "301Y013", "item_code1": "*AA0", "period": "M"},  # TODO 확인: item_code 구조
    
    }


def collect_ecos_indicators() -> dict:
    result = {}
    for name, cfg in ECOS_TARGETS.items():
        period = cfg["period"]
        if period == "D":
            start, end = START_DATE.strftime("%Y%m%d"), END_DATE.strftime("%Y%m%d")
        elif period == "M":
            start, end = START_DATE.strftime("%Y%m"), END_DATE.strftime("%Y%m")
        elif period == "Q":
            start = f"{START_DATE.year}Q{(START_DATE.month - 1)//3 + 1}"
            end = f"{END_DATE.year}Q{(END_DATE.month - 1)//3 + 1}"
        else:
            start, end = str(START_DATE.year), str(END_DATE.year)

        try:
            result[name] = fetch_ecos_series(
                cfg["stat_code"], cfg["item_code1"], period, start, end
            )
            print(f"[ECOS] {name} - {len(result[name])}건 확보")
        except Exception as e:
            print(f"[ECOS] {name} 실패: {e}")
            result[name] = []
        time.sleep(0.2)
    return result


# ============================================================
# 2. yfinance - VIX / 미국채10년물 / 유가(WTI)
# ============================================================

MARKET_TICKERS = {
    "VIX": "^VIX",
    "미국국채10년물_야후": "^TNX",  # FRED DGS10과 대조용 (야후는 10배수로 나올 때가 있어 /10 필요할 수 있음)
    "유가_WTI": "CL=F",
}


def collect_market_indicators() -> dict:
    start = START_DATE.strftime("%Y-%m-%d")
    end = (END_DATE + timedelta(days=1)).strftime("%Y-%m-%d")
    result = {}
    for name, ticker in MARKET_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                result[name] = []
                continue
            close = df["Close"]
            if hasattr(close, "iloc") and close.ndim == 2:
                close = close.iloc[:, 0]
            close.index = close.index.strftime("%Y-%m-%d")
            result[name] = [{"date": d, "value": round(float(v), 4)} for d, v in close.items()]
            print(f"[yfinance] {name} ({ticker}) - {len(result[name])}건 확보")
        except Exception as e:
            print(f"[yfinance] {name} ({ticker}) 실패: {e}")
            result[name] = []
    return result


# ============================================================
# 3. pykrx - 외국인 코스피 순매수/순매도 (비공식, 선택 사항)
# ============================================================

def collect_foreign_netflow() -> list[dict]:
    """
    pykrx는 KRX 정보데이터시스템을 스크레이핑하는 비공식 무료 라이브러리.
    설치 안 되어 있거나 KRX 쪽 구조가 바뀌면 실패할 수 있어 try/except로 감싼다.
    """
    if not os.environ.get("KRX_ID") or not os.environ.get("KRX_PW"):
        print("[pykrx] KRX_ID/KRX_PW 환경 변수가 없어 외국인 순매수 지표는 건너뜀.")
        return []

    try:
        from pykrx import stock
    except ImportError:
        print("[pykrx] 설치 안 됨 (pip install pykrx). 이 지표는 건너뜀.")
        return []

    start = START_DATE.strftime("%Y%m%d")
    end = END_DATE.strftime("%Y%m%d")
    try:
        # 코스피 전체 투자자별 순매수 거래대금 (외국인 컬럼만 사용)
        df = stock.get_market_trading_value_by_date(start, end, "KOSPI")
        if "외국인합계" not in df.columns:
            print(f"[pykrx] 예상 컬럼 없음, 실제 컬럼: {list(df.columns)}")
            return []
        series = df["외국인합계"]
        series.index = series.index.strftime("%Y-%m-%d")
        out = [{"date": d, "value": int(v)} for d, v in series.items()]
        print(f"[pykrx] 외국인 코스피 순매수 - {len(out)}건 확보")
        return out
    except Exception as e:
        message = str(e).strip()
        if "KRX_ID" in message or "KRX_PW" in message or "로그인 실패" in message:
            print("[pykrx] KRX 로그인 정보가 없어 외국인 순매수 지표는 건너뜀.")
        else:
            print(f"[pykrx] 수집 실패: {message}")
        return []


# ============================================================
# 4. pytrends - Google Trends (비공식, 선택 사항, 우선순위 낮음)
# ============================================================

def collect_google_trends(keywords: list[str] = None) -> list[dict]:
    keywords = keywords or ["환율", "달러 환율"]
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print("[pytrends] 설치 안 됨 (pip install pytrends). 이 지표는 건너뜀.")
        return []

    try:
        pytrends = TrendReq(hl="ko-KR", tz=540)
        pytrends.build_payload(keywords, timeframe="today 12-m", geo="KR")
        df = pytrends.interest_over_time()
        if df.empty:
            return []
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        df["평균관심도"] = df.mean(axis=1)
        out = [
            {"date": idx.strftime("%Y-%m-%d"), "value": round(float(v), 2)}
            for idx, v in df["평균관심도"].items()
        ]
        print(f"[pytrends] 검색 관심도 - {len(out)}건 확보")
        return out
    except Exception as e:
        print(f"[pytrends] 수집 실패 (Google Trends는 요청이 잦으면 잘 막힘): {e}")
        return []


# ============================================================
# 실행
# ============================================================

def main():
    print(f"수집 기간: {START_DATE.date()} ~ {END_DATE.date()}\n")

    macro_data = {
        "collected_at": datetime.now().isoformat(),
        "period": {"start": START_DATE.strftime("%Y-%m-%d"), "end": END_DATE.strftime("%Y-%m-%d")},
        "fred": collect_fred_indicators(),
        "ecos": collect_ecos_indicators(),
        "market": collect_market_indicators(),
        "foreign_netflow_kospi": collect_foreign_netflow(),
        "google_trends": collect_google_trends(),
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(macro_data, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료 -> {OUTPUT_PATH.resolve()}")
    print("\n[확인 필요] FRED_SERIES / ECOS_TARGETS 안의 'TODO 확인' 코드들은")
    print("실행 결과 건수가 0이거나 이상하면 fred_search_series()로 재검색해서 교체할 것.")


if __name__ == "__main__":
    # 확인용
    # fred_search_series("CPALTT01EZM657N")

    main()



