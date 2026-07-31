# -*- coding: utf-8 -*-
"""
자산별 개인화 요약 생성 (RAG 파이프라인의 마지막 단계)
--------------------------------------------------------
입력:
    processed_news.jsonl  - 카테고리/방향/통화쌍/confidence 태깅된 뉴스
    impact_weights.json   - 카테고리별 자체 영향도 가중치 (통계 기반)
    user_assets.json      - 사용자 자산 프로필 (+ 실시간 환율)

과정:
    1) 자산마다 관련 뉴스를 매칭 (통화쌍 일치 > 카테고리/민감도 일치)
    2) score = confidence × impact_weight × match_bonus 로 관련도 점수 계산
    3) 자산별 상위 N개 뉴스만 골라서 Groq LLM에 전달
    4) "당신의 OO 자산에는 이런 의미예요" 형태의 개인화 문장 생성

출력:
    personalized_summary.json

사전 준비:
    pip install requests python-dotenv
    .env 파일에 GROQ_API_KEY=발급받은_키
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from volatility_context import (
    build_all_volatility_contexts,
    format_volatility_context,
    normalize_pair,
)

load_dotenv()

NEWS_PATH = Path("processed_news.jsonl")
WEIGHTS_PATH = Path("impact_weights.json")
ASSETS_PATH = Path("user_assets.json")
OUTPUT_PATH = Path("personalized_summary.json")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

TOP_N_NEWS_PER_ASSET = 3
DEFAULT_IMPACT_WEIGHT = 0.5  # impact_weights.json에 없는 카테고리는 중립값

# 자산의 통화쌍 -> 그 외화의 발행국. 뉴스의 country가 이 나라이거나
# "한국"(원화 자체 이슈, 모든 통화쌍의 원화쪽 다리에 영향)이면 매칭 인정
COUNTRY_OF_PAIR = {
    "USD/KRW": "미국",
    "JPY/KRW": "일본",
    "EUR/KRW": "유럽",
}


# ============================================================
# 1. 데이터 로드
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2. 자산 <-> 뉴스 매칭 + 스코어링
# ============================================================

def match_score(asset: dict, news: dict, impact_weights: dict) -> float:
    """
    관련도 점수 = confidence × impact_weight × match_bonus
    - match_bonus 1.0 : 뉴스의 currency_pairs에 자산의 currency_pair가 정확히 포함
    - match_bonus 0.6 : 통화쌍은 안 겹치지만 카테고리가 자산의 sensitivity에 포함
                        *그리고* 뉴스의 country가 그 자산의 외화 발행국이거나 "한국"인 경우만
                        (예: 엔화 적금인데 country가 "미국"인 뉴스는 카테고리만 같아도 매칭 안 함)
    - 둘 다 아니면 매칭 안 됨 (score 계산 안 함)
    """
    confidence = news.get("confidence", 0.0)
    category = news.get("category")
    country = news.get("country", "기타")
    weight_info = impact_weights.get(category, {})
    impact_weight = weight_info.get("weight", DEFAULT_IMPACT_WEIGHT)

    news_pairs = set(news.get("currency_pairs") or [])
    asset_pair = asset.get("currency_pair")
    asset_sensitivity = set(asset.get("sensitivity") or [])
    asset_country = COUNTRY_OF_PAIR.get(asset_pair)

    if asset_pair and asset_pair in news_pairs:
        match_bonus = 1.0
    elif category in asset_sensitivity and (country == asset_country or country == "한국"):
        match_bonus = 0.6
    else:
        return 0.0  # 매칭 안 됨 (카테고리는 같아도 관련 없는 나라 이슈)

    return round(confidence * impact_weight * match_bonus, 4)


def find_matched_news(asset: dict, news_items: list[dict], impact_weights: dict) -> list[dict]:
    scored = []
    for news in news_items:
        if not news.get("is_relevant"):
            continue
        score = match_score(asset, news, impact_weights)
        if score > 0:
            scored.append({**news, "match_score": score})

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:TOP_N_NEWS_PER_ASSET]


# ============================================================
# 3. Groq LLM으로 개인화 문장 생성
# ============================================================

SYSTEM_PROMPT = """당신은 사용자의 자산 상황을 고려해 환율 뉴스를 쉽게 설명해주는
금융 어시스턴트입니다. 아래 뉴스들과 사용자 자산 정보를 보고, 이 뉴스가
사용자의 자산에 어떤 의미인지 자연스러운 한국어 3~5문장으로 설명하세요.

반드시 지켜야 할 방향성 규칙 (틀리기 쉬우니 특히 주의):
1. "원화 약세(환율 상승, 예: 1,400원 -> 1,450원)"가 되면, 달러/엔화/유로 등
   외화로 표시된 자산(예금, 주식)의 "원화 환산 가치"는 오히려 상승할 가능성이
   있습니다. 반대로 "원화 강세(환율 하락)"가 되면 원화 환산 가치는 하락할
   가능성이 있습니다. 이 방향을 절대 거꾸로 설명하지 마세요.
2. 예금 상품의 "적용 금리(이자율)"는 그 예금이 표시된 통화를 발행한 국가의
   기준금리에 영향을 받습니다. 예를 들어 엔화 적금의 금리는 일본은행 정책에
   영향받는 것이지, 한국은행 기준금리가 엔화 적금의 이자율 자체를 바꾸지는
   않습니다. 다만 한국은행 정책은 원/엔 환율을 통해 그 적금의 "원화 환산
   평가금액"에는 영향을 줄 수 있습니다. 이자율에 대한 영향과 환산 평가금액에
   대한 영향을 혼동해서 설명하지 마세요.
3. 무역 분쟁/관세 뉴스처럼 특정국 통화 약세 요인이 있으면, 그 나라 통화
   자산과 원화 자산에 미치는 영향 방향이 다를 수 있다는 점을 고려하세요.
4. 변동성 정보가 제공되면 연율화 변동성 수치(%), 과거 백분위 또는 구간,
   월간 환산 변동성(%)을 구체적인 숫자로 반드시 한 번 이상 언급하세요.
5. 현재 환율 기준 월간 통계적 변동 폭(±원)이 제공되면 그 숫자도 언급하세요.
6. 변동성은 환율의 상승·하락 방향이 아니라 움직임의 크기입니다. 변동성이
   높다는 이유만으로 환율 상승 또는 하락을 예측하지 마세요.
7. 변동성 수치는 실제 옵션 내재변동성이 아니라 SV 기반 프록시라는 점과,
   ±원은 확정 범위가 아닌 통계적 환산값이라는 점을 짧게 밝히세요.
8. 변동성 정보에 기준일이 있으면 기준일을 함께 언급하고, 오래된 정보라는
   경고가 있으면 현재 수치처럼 표현하지 마세요.
9. 숫자를 보고서처럼 나열하지 마세요. 먼저 "평소보다 환율 움직임이 큰 편"
   또는 "한 달 기준 약 ±45원 정도의 통계적 움직임에 해당"처럼 쉬운 말로
   의미를 설명한 뒤 괄호나 이어지는 문장에서 핵심 수치를 제시하세요.
10. "연율화", "백분위", "%p", "SV" 같은 용어를 단독으로 쓰지 말고,
    각각 "1년 기준으로 환산한 움직임 크기", "과거 100일 중 몇 일보다 큰지",
    "변동성 차이", "과거 환율 움직임으로 추정한 값"이라는 뜻을 풀어주세요.
11. 변동성 관련 숫자는 가장 이해하기 쉬운 2~3개를 중심으로 설명하고,
    모든 통계값을 억지로 한 문장에 나열하지 마세요.
12. 마지막 문장은 반드시 "행동 제안:"으로 시작하는 한 문장으로 작성하세요.
    제공된 변동성 정보의 허용되는 행동 제안을 자산 상황에 맞게 쉽게 바꾸되,
    환율 확인, 예산 여유 확보, 환전·송금 시점 분산, 환율 노출 확인처럼 위험을
    관리하는 행동만 제시하세요. 매수·매도, 특정 환율 방향에 대한 베팅,
    수익을 보장하는 표현은 사용하지 마세요.

일반 규칙:
- 반드시 주어진 뉴스 내용에 근거해서만 설명하세요. 뉴스에 없는 내용을 지어내지 마세요.
- 투자 조언(사라, 팔아라)을 하지 말고, 사실과 그 의미만 담백하게 설명하세요.
- "~일 수 있어요", "~에는 큰 변화가 없어요" 처럼 단정적이지 않은 톤을 쓰세요.
- 출력은 설명 문장만 출력하고, 다른 부연설명이나 따옴표는 붙이지 마세요.
"""


def build_user_prompt(
    asset: dict,
    matched_news: list[dict],
    volatility_context: dict | None = None,
) -> str:
    news_block = "\n".join(
        f"- [{n['category']}/{n.get('country', '기타')}/{n['direction']}] {n['title']} : {n.get('reason', '')}"
        for n in matched_news
    )
    asset_country = COUNTRY_OF_PAIR.get(asset.get("currency_pair"), "알수없음")
    volatility_block = format_volatility_context(volatility_context)
    volatility_section = (
        f"\n\n[해당 통화쌍의 변동성 수치]\n{volatility_block}"
        if volatility_block
        else ""
    )
    return (
        f"[사용자 자산]\n"
        f"자산명: {asset['name']} ({asset['asset_type']})\n"
        f"통화: {asset['currency_pair']} (외화 발행국: {asset_country})\n"
        f"평가금액: 약 {asset['amount_krw']:,}원\n"
        f"현재 환율: {asset.get('current_rate', '정보없음')}\n\n"
        f"[관련 뉴스 (형식: [카테고리/관련국가/방향] 제목 : 근거)]\n{news_block}"
        f"{volatility_section}\n\n"
        f"위 자산에 뉴스와 변동성 수치가 어떤 의미인지 설명해주세요."
    )


def call_groq(
    api_key: str,
    asset: dict,
    matched_news: list[dict],
    volatility_context: dict | None = None,
    max_retries: int = 3,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 300,
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(asset, matched_news, volatility_context),
            },
        ],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait_s = 5 * (attempt + 1)
                print(f"[{asset['name']}] 요청 한도 초과(429). {wait_s}초 대기 후 재시도")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            print(f"[{asset['name']}] 요청 오류 (시도 {attempt+1}/{max_retries}): {e}")
            time.sleep(1.5 * (attempt + 1))

    return "(생성 실패 - Groq 응답을 받지 못했습니다)"


# ============================================================
# 전체 파이프라인
# ============================================================

def process_all():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(".env 파일에 GROQ_API_KEY=발급받은_키 를 추가하세요.")

    news_items = load_jsonl(NEWS_PATH)
    impact_weights = load_json(WEIGHTS_PATH)
    profile = load_json(ASSETS_PATH)
    assets = profile.get("assets", [])
    volatility_contexts = build_all_volatility_contexts()

    print(f"뉴스 {len(news_items)}건 / 자산 {len(assets)}건 로드 완료\n")

    results = []
    for asset in assets:
        matched = find_matched_news(asset, news_items, impact_weights)
        asset_pair = normalize_pair(asset.get("currency_pair"))
        volatility_context = volatility_contexts.get(asset_pair)

        if not matched and not volatility_context:
            summary_text = f"최근 {asset['name']}에 직접적으로 영향을 줄 만한 뉴스는 없었어요."
        else:
            summary_text = call_groq(
                api_key,
                asset,
                matched,
                volatility_context=volatility_context,
            )

        print(f"[{asset['name']}] 매칭 뉴스 {len(matched)}건")
        print(f"  -> {summary_text}\n")

        results.append({
            "asset_id": asset["asset_id"],
            "asset_name": asset["name"],
            "matched_news": [
                {"id": n["id"], "title": n["title"], "match_score": n["match_score"]}
                for n in matched
            ],
            "volatility_context": volatility_context,
            "personalized_summary": summary_text,
        })

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 -> {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    process_all()
