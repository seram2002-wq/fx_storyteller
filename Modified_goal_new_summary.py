import os
import json
import requests
from dotenv import load_dotenv

from goal_news_search import search_goal_news

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "llama-3.1-8b-instant"

MACRO_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_macro.json")

# 뉴스의 국가 표기를 processed_macro.json의 통화쌍 키로 매핑
COUNTRY_TO_CURRENCY = {
    "미국": "USD/KRW",
    "일본": "JPY/KRW",
    "유럽": "EUR/KRW",
    "유로존": "EUR/KRW",
    "유로": "EUR/KRW",
}

# 뉴스의 통화/국가와 무관하게 항상 참고할 수 있는 공통 글로벌 지표
COMMON_INDICATOR_KEYS = [
    "VIX",
    "유가_WTI",
    "미국_국채10년물",
    "미국국채10년물_야후",
    "한국_CPI",
    "한국_GDP성장률",
]

SYSTEM_PROMPT = """
당신은 환율 뉴스를 사용자의 목적에 맞게 쉽게 설명하는 금융 어시스턴트입니다.

사용자는 해외여행, 유학, 투자, 출장, 해외직구 등의 목적을 가지고 있습니다.
제공된 뉴스와 거시 지표를 종합하여 이 목적에 어떤 영향을 줄 수 있는지 자연스럽게 설명하세요.

규칙

1. 뉴스를 하나씩 요약하지 말고, 여러 뉴스를 하나의 흐름으로 종합해서 설명하세요.

2. 설명 순서는 다음을 따르세요.
- 뉴스의 핵심 내용
- 환율 또는 해당 국가 경제에 미칠 가능성
- 사용자의 목적에 미칠 영향
- 참고하면 좋을 점

3. 반드시 뉴스 내용과 함께 제공된 거시 지표만을 근거로 설명하세요.
뉴스와 거시 지표에 없는 사실이나 경제 시나리오를 만들어 설명하지 마세요.
대학 입학, 취업, 생활비, 등록금 등은 뉴스에 직접 언급된 경우가 아니라면 설명하지 마세요.

4. 환율 방향(강세·약세, 상승·하락)은 뉴스 또는 거시 지표에 근거가 있을 때만 설명하세요.
뉴스와 거시 지표만으로 판단하기 어렵다면
"이번 뉴스와 지표만으로는 환율 방향을 단정하기 어렵습니다."
라고 설명하세요.

5. 여러 단계의 추론을 하지 마세요.
예를 들어
관세 → 물가 → 금리 → 환율 → 생활비
처럼 뉴스나 지표에 없는 인과관계를 이어서 설명하면 안 됩니다.
환율과 직접 관련 있는 영향만 설명하세요.

6. 투자나 환전을 추천하지 마세요.
"지금 환전하세요", "매수하세요" 같은 표현은 사용하지 않습니다.
대신
"환율 변화를 함께 확인해 보는 것이 좋습니다."
처럼 중립적으로 설명하세요.

7. 금융 용어는 쉬운 한국어로 설명하세요. 피벗, VIX, CPI, GDP 등은 그대로 쓰되, 일반인이 이해하기 어려운 용어는 쉬운 말로 풀어서 설명하세요.

8. 함께 제공되는 거시 지표(금리차, 이동평균 신호, 피벗 대비 위치, VIX, 유가, 국채금리 등)는
뉴스 내용을 뒷받침하는 배경 정보로만 사용하세요.
거시 지표만 보고 뉴스에 없는 새로운 원인이나 예측을 만들어내지 마세요.
지표 값을 나열하지 말고, 뉴스와 자연스럽게 엮어서 설명하세요.
데이터가 부족하거나 없는 지표는 언급하지 마세요.

출력 형식

- 제목이나 번호 없이 하나의 자연스러운 문단으로 작성하세요.
- 4~6문장으로 작성하세요.
- 같은 내용을 반복하지 마세요.
- "~할 수 있습니다", "~가능성이 있습니다"처럼 단정하지 않은 표현을 사용하세요.
"""

def load_macro_data(path=MACRO_FILE_PATH):
    """processed_macro.json을 읽어 dict로 반환. 파일이 없으면 None."""

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_macro_context(macro, news):
    """뉴스의 국가 정보를 바탕으로 관련된 거시 지표만 골라 텍스트로 구성."""

    if not macro:
        return ""

    rate_diffs = macro.get("rate_diffs", {})
    fx_technical = macro.get("fx_technical", {})
    other_indicators = macro.get("other_indicators", {})

    currencies = set()
    for n in news:
        country = n.get("country") or n.get("country_name") or "기타"
        currency = COUNTRY_TO_CURRENCY.get(country)
        if currency:
            currencies.add(currency)

    if not currencies:
        currencies = {"USD/KRW"}

    text = ""

    for currency in currencies:
        rd = rate_diffs.get(currency)
        ft = fx_technical.get(currency)

        lines = []

        if rd and rd.get("금리차_pp") is not None:
            lines.append(f"금리차 {rd.get('금리차_pp')}pp ({rd.get('해석', '')})")

        if ft:
            if ft.get("ma_signal"):
                lines.append(f"이동평균 신호 : {ft.get('ma_signal')}")
            if ft.get("position_vs_pivot"):
                lines.append(f"피벗 대비 위치 : {ft.get('position_vs_pivot')}")

        if lines:
            text += f"\n[{currency} 관련 지표]\n- " + "\n- ".join(lines) + "\n"

    common_lines = []
    for key in COMMON_INDICATOR_KEYS:
        ind = other_indicators.get(key)
        if ind and ind.get("latest") is not None:
            common_lines.append(f"{key} : {ind.get('latest')} ({ind.get('trend', '')})")

    if common_lines:
        text += "\n[공통 글로벌 지표]\n- " + "\n- ".join(common_lines) + "\n"

    return text


def build_prompt(goal, news, macro_context=""):

    text = ""

    for n in news:
        title = n.get("title", "(제목 없음)")
        summary = n.get("summary", "")
        reason = n.get("reason", "")
        country = n.get("country") or n.get("country_name") or "기타"
        category = n.get("category", "기타")

        text += f"""
제목 : {title}

요약 : {summary}

근거 : {reason}

국가 : {country}

카테고리 : {category}

"""

    macro_section = f"""
참고할 거시 지표
{macro_context}
""" if macro_context else ""

    return f"""
사용자 목적

{goal}

관련 뉴스

{text}
{macro_section}
"""

def validate_groq_api_key(api_key):
    if not api_key:
        raise RuntimeError("GROQ_API_KEY가 설정되지 않았습니다. Groq 콘솔에서 발급한 gsk_... 형태의 키를 .env에 넣어주세요.")
    if not api_key.startswith("gsk_"):
        raise RuntimeError("GROQ_API_KEY 형식이 올바르지 않습니다. Groq 콘솔의 gsk_... 키를 사용해야 합니다.")
    return api_key


def generate_summary(goal, news, macro_context=""):

    api_key = os.getenv("GROQ_API_KEY")
    validate_groq_api_key(api_key)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    macro = load_macro_data()
    if not macro_context:
        macro_context = build_macro_context(macro, news)

    payload = {

        "model": MODEL,

        "temperature":0.3,

        "messages":[

            {
                "role":"system",
                "content":SYSTEM_PROMPT
            },

            {
                "role":"user",
                "content":build_prompt(goal, news, macro_context)
            }

        ]

    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if response.status_code == 401:
            raise RuntimeError("Groq 인증에 실패했습니다. .env의 GROQ_API_KEY가 유효한 Groq 키인지 확인하세요.")

        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Groq 요청 중 오류가 발생했습니다: {exc}") from exc

    return response.json()["choices"][0]["message"]["content"]

def main():

    goal = input("목적을 입력하세요 : ")

    context, news = search_goal_news(goal)

    if not news:
        print("현재 목적과 직접적으로 관련된 환율 뉴스가 없습니다.")
        return

    print("\n===== 관련 뉴스 =====\n")

    for i, n in enumerate(news, 1):
        print(f"제목 : {n['title']}")


        print()

    print("===== AI 분석 =====\n")

    try:
        summary = generate_summary(goal, news, macro_context=build_macro_context(load_macro_data(), news))
    except RuntimeError as exc:
        print(f"오류: {exc}")
        return

    print(summary)

if __name__ == "__main__":
    main()
