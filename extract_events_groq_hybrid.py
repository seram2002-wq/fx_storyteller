# -*- coding: utf-8 -*-
"""
collected_news.jsonl -> processed_news.jsonl

Groq(무료 API)를 사용하는 하이브리드 파이프라인:
    1단계 (무료, 규칙 기반): 키워드/정규식으로 관련 없는 기사를 먼저 걸러낸다.
        관련 없다고 판단된 기사는 여기서 끝 -> LLM 호출 없음.
    2단계 (Groq 무료 API): 1단계를 통과한 기사만 Groq의 오픈소스 모델
        (Llama 3.3 70B)에 보내서 카테고리/방향/통화쌍/근거를 재판단한다.

    Groq는 신용카드 없이 가입만으로 무료 API 키를 받을 수 있고,
    분당 30회 / 하루 14,400회 요청까지는 과금 없이 사용 가능하다.
    (요금제·한도는 Groq 정책에 따라 바뀔 수 있으니 console.groq.com에서 최신 정보 확인 권장)

사전 준비:
    1) https://console.groq.com/keys 에서 무료 회원가입 후 API 키 발급 (신용카드 불필요)
    2) pip install requests python-dotenv
    3) 이 파일과 같은 폴더에 .env 파일을 만들고 아래 한 줄을 적어둘 것
       GROQ_API_KEY=gsk_...
       (코드에 키를 직접 적지 않는다! .env는 .gitignore에도 등록해둘 것)

사용법:
    python extract_events_hybrid_groq.py
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()  # 같은 폴더의 .env 파일을 읽어서 os.environ에 값을 채워줌

INPUT_PATH = Path("collected_news.jsonl")
OUTPUT_PATH = Path("processed_news.jsonl")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# 무료 티어에서 품질이 좋은 모델. 요청 한도를 더 아끼고 싶으면
# "llama-3.1-8b-instant"로 바꿔도 된다 (더 가볍고 빠름, 품질은 약간 낮음).
MODEL = "llama-3.3-70b-versatile"


# ============================================================
# 1단계: 규칙 기반 필터 (무료, API 호출 없음)
# ============================================================

CATEGORY_MIN_HITS = {
    "금리": 1,
    "무역": 1,
    "지정학": 2,  # war/conflict 등 일상 단어와 겹치는 키워드가 많아 기준을 높게 잡음
    "환율": 1,
}

CASE_SENSITIVE_KEYWORDS: set[str] = set()

EVENT_KEYWORDS = {
    "금리": [
        "금리", "기준금리", "연준", "FOMC", "Federal Reserve", "Fed Chair",
        "Fed's", "Fed policy", "Fed rate", "기준금리 인상", "기준금리 인하",
        "금리 동결", "금리 인상", "금리 인하", "rate hike", "rate cut", "interest rate",
    ],
    "무역": [
        "관세", "무역분쟁", "무역전쟁", "무역협상", "무역합의", "tariff",
        "trade war", "trade deal", "trade dispute",
    ],
    "지정학": [
        "제재", "지정학", "전쟁", "분쟁", "긴장 고조", "sanctions", "geopolitical",
        "conflict", "war",
    ],
    "환율": [
        "환율", "원달러", "달러당", "엔달러", "환율 변동", "환율 전망",
        "exchange rate", "currency market",
    ],
}

CURRENCY_PAIR_MAP = {
    "달러": "USD/KRW",
    "미국 달러": "USD/KRW",
    "엔화": "JPY/KRW",
    "엔": "JPY/KRW",
    "위안": "CNY/KRW",
    "유로": "EUR/KRW",
    "파운드": "GBP/KRW",
}

_PUNCT_STRIP = "\"'.,()·“”‘’·…\u200b"
_PATTERN_CACHE: dict[str, re.Pattern] = {}


def _keyword_pattern(kw: str) -> re.Pattern:
    flags = 0 if kw in CASE_SENSITIVE_KEYWORDS else re.IGNORECASE
    return re.compile(r"\b" + re.escape(kw), flags)


def _get_pattern(kw: str) -> re.Pattern:
    if kw not in _PATTERN_CACHE:
        _PATTERN_CACHE[kw] = _keyword_pattern(kw)
    return _PATTERN_CACHE[kw]


def count_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if _get_pattern(kw).search(text))


def detect_category(text: str) -> tuple[str, int]:
    best_category, best_hits = "기타", 0
    for category, keywords in EVENT_KEYWORDS.items():
        hits = count_hits(text, keywords)
        if hits > best_hits:
            best_category, best_hits = category, hits
    return best_category, best_hits


def detect_currency_pairs(text: str) -> list[str]:
    tokens = text.split()
    pairs = set()
    for word, pair in CURRENCY_PAIR_MAP.items():
        for token in tokens:
            if token.strip(_PUNCT_STRIP).startswith(word):
                pairs.add(pair)
                break
    return sorted(pairs)


def rule_based_prefilter(article: dict) -> tuple[bool, str, int]:
    text = " ".join([
        article.get("title", ""),
        article.get("summary", ""),
        article.get("content", ""),
    ])
    category, hits = detect_category(text)
    min_hits_required = CATEGORY_MIN_HITS.get(category, 1)
    is_relevant = hits >= min_hits_required
    return is_relevant, category, hits


# ============================================================
# 2단계: Groq 무료 API로 정교화 (1단계를 통과한 기사만 대상)
# ============================================================

SYSTEM_PROMPT = """당신은 환율/금리/무역/지정학 뉴스를 판별하고 구조화하는 애널리스트입니다.
아래 기사 하나를 보고 반드시 JSON 객체 하나만 출력하세요. 다른 설명, 서론, 코드블록 표시(```)는 절대 포함하지 마세요.

판단 기준:
- 기사가 실제로 "환율 변동에 영향을 줄 수 있는" 금리 정책, 무역 분쟁/관세, 지정학적 리스크, 또는 직접적인 환율 뉴스인지 판단하세요.
- 단순히 금액 표현에 "원", "달러" 같은 단어가 들어갔다고 해서 관련 있다고 판단하지 마세요.
- 애매하면 is_relevant를 false로 하고 confidence를 낮게 주세요.
- country는 이 뉴스가 다루는 정책/이벤트의 주체 국가입니다. 예: 미 연준 금리 발표 -> "미국",
  한국은행 금리 발표 -> "한국", 일본은행 -> "일본", ECB -> "유럽". 특정 국가로 판단하기 어려우면
  "기타"로 표시하세요. 이 필드는 이후 단계에서 "이 뉴스가 어느 통화에 영향을 주는지" 정확히
  매칭하는 데 사용되므로 신중하게 판단하세요.

출력 JSON 스키마 (이 형식만 출력):
{
  "is_relevant": true or false,
  "category": "금리" | "무역" | "지정학" | "환율" | "기타",
  "country": "한국" | "미국" | "일본" | "유럽" | "기타",
  "direction": "호재" | "악재" | "중립",
  "currency_pairs": ["USD/KRW", "JPY/KRW", ...],
  "confidence": 0.0 ~ 1.0,
  "reason": "한 문장 이내의 판단 근거"
}
"""


def build_user_prompt(article: dict, hinted_category: str) -> str:
    return (
        f"[1차 규칙 필터가 후보로 추정한 카테고리: {hinted_category} — 참고만 하고 직접 재판단할 것]\n"
        f"제목: {article.get('title', '')}\n"
        f"요약: {article.get('summary', '')}\n"
        f"본문 일부: {article.get('content', '')[:1000]}\n"
    )


def call_groq(api_key: str, article: dict, hinted_category: str,
               max_retries: int = 3) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 300,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(article, hinted_category)},
        ],
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 429:
                # 무료 티어 분당/일일 한도 초과. 잠시 쉬었다가 재시도.
                wait_s = 5 * (attempt + 1)
                print(f"[{article.get('id')}] 요청 한도 초과(429). {wait_s}초 대기 후 재시도")
                time.sleep(wait_s)
                continue

            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"].strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(raw_text)

        except json.JSONDecodeError:
            print(f"[{article.get('id')}] JSON 파싱 실패 (시도 {attempt+1}/{max_retries}): {raw_text[:200]}")
        except requests.RequestException as e:
            print(f"[{article.get('id')}] 요청 오류 (시도 {attempt+1}/{max_retries}): {e}")
        time.sleep(1.5 * (attempt + 1))

    return {
        "is_relevant": True,  # 1단계에서 이미 관련 있다고 판정했으므로 안전하게 True 유지
        "category": hinted_category,
        "country": "기타",
        "direction": "중립",
        "currency_pairs": [],
        "confidence": 0.0,
        "reason": "Groq 호출 실패 — 1차 필터 결과로 대체",
    }


# ============================================================
# 전체 파이프라인
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_all() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "환경변수 GROQ_API_KEY가 설정되어 있지 않습니다. "
            ".env 파일에 GROQ_API_KEY=발급받은_키 를 추가하세요. "
            "키는 https://console.groq.com/keys 에서 무료로 발급받을 수 있습니다."
        )

    articles = load_jsonl(INPUT_PATH)
    print(f"총 {len(articles)}건 로드 완료")

    # --- 1단계: 규칙 기반 필터 (무료) ---
    candidates = []
    results = []
    for article in articles:
        is_relevant, category, hits = rule_based_prefilter(article)
        if is_relevant:
            candidates.append((article, category))
        else:
            results.append({
                **article,
                "is_relevant": False,
                "category": "기타",
                "country": "기타",
                "direction": "중립",
                "currency_pairs": [],
                "confidence": 0.0,
                "reason": "1차 규칙 필터에서 제외 (이벤트 키워드 매칭 부족)",
            })

    print(f"1단계(규칙) 통과: {len(candidates)}건 / 전체 {len(articles)}건 "
          f"-> 이 {len(candidates)}건만 Groq 호출")

    # --- 2단계: Groq 무료 API로 정교화 ---
    relevant_count = 0
    for i, (article, hinted_category) in enumerate(candidates, 1):
        extracted = call_groq(api_key, article, hinted_category)
        results.append({**article, **extracted})
        if extracted["is_relevant"]:
            relevant_count += 1
        if i % 10 == 0:
            print(f"  Groq 처리 중: {i}/{len(candidates)}")
        time.sleep(0.3)  # 무료 티어 분당 요청 한도(RPM)를 넉넉하게 지키기 위한 완충

    save_jsonl(results, OUTPUT_PATH)
    print(f"\n완료: 전체 {len(articles)}건 중 Groq 호출 {len(candidates)}건, "
          f"최종 관련 기사 {relevant_count}건")
    print(f"저장 -> {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    process_all()
