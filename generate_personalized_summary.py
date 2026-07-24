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

load_dotenv()

NEWS_PATH = Path("processed_news.jsonl")
WEIGHTS_PATH = Path("impact_weights.json")
ASSETS_PATH = Path("user_assets.json")
OUTPUT_PATH = Path("personalized_summary.json")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

TOP_N_NEWS_PER_ASSET = 3
DEFAULT_IMPACT_WEIGHT = 0.5  # impact_weights.json에 없는 카테고리는 중립값


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
    - 둘 다 아니면 매칭 안 됨 (score 계산 안 함)
    """
    confidence = news.get("confidence", 0.0)
    category = news.get("category")
    weight_info = impact_weights.get(category, {})
    impact_weight = weight_info.get("weight", DEFAULT_IMPACT_WEIGHT)

    news_pairs = set(news.get("currency_pairs") or [])
    asset_pair = asset.get("currency_pair")
    asset_sensitivity = set(asset.get("sensitivity") or [])

    if asset_pair and asset_pair in news_pairs:
        match_bonus = 1.0
    elif category in asset_sensitivity:
        match_bonus = 0.6
    else:
        return 0.0  # 매칭 안 됨

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
사용자의 자산에 어떤 의미인지 자연스러운 한국어 1~3문장으로 설명하세요.

규칙:
- 반드시 주어진 뉴스 내용에 근거해서만 설명하세요. 뉴스에 없는 내용을 지어내지 마세요.
- 투자 조언(사라, 팔아라)을 하지 말고, 사실과 그 의미만 담백하게 설명하세요.
- "~일 수 있어요", "~에는 큰 변화가 없어요" 처럼 단정적이지 않은 톤을 쓰세요.
- 출력은 설명 문장만 출력하고, 다른 부연설명이나 따옴표는 붙이지 마세요.
"""


def build_user_prompt(asset: dict, matched_news: list[dict]) -> str:
    news_block = "\n".join(
        f"- [{n['category']}/{n['direction']}] {n['title']} : {n.get('reason', '')}"
        for n in matched_news
    )
    return (
        f"[사용자 자산]\n"
        f"자산명: {asset['name']} ({asset['asset_type']})\n"
        f"통화: {asset['currency_pair']}\n"
        f"평가금액: 약 {asset['amount_krw']:,}원\n"
        f"현재 환율: {asset.get('current_rate', '정보없음')}\n\n"
        f"[관련 뉴스]\n{news_block}\n\n"
        f"위 자산에 위 뉴스들이 어떤 의미인지 설명해주세요."
    )


def call_groq(api_key: str, asset: dict, matched_news: list[dict], max_retries: int = 3) -> str:
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
            {"role": "user", "content": build_user_prompt(asset, matched_news)},
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

    print(f"뉴스 {len(news_items)}건 / 자산 {len(assets)}건 로드 완료\n")

    results = []
    for asset in assets:
        matched = find_matched_news(asset, news_items, impact_weights)

        if not matched:
            summary_text = f"최근 {asset['name']}에 직접적으로 영향을 줄 만한 뉴스는 없었어요."
        else:
            summary_text = call_groq(api_key, asset, matched)

        print(f"[{asset['name']}] 매칭 뉴스 {len(matched)}건")
        print(f"  -> {summary_text}\n")

        results.append({
            "asset_id": asset["asset_id"],
            "asset_name": asset["name"],
            "matched_news": [
                {"id": n["id"], "title": n["title"], "match_score": n["match_score"]}
                for n in matched
            ],
            "personalized_summary": summary_text,
        })

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 -> {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    process_all()
