import os
import requests
from dotenv import load_dotenv

from goal_news_search import search_goal_news

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """
당신은 환율 뉴스를 사용자의 목적에 맞게 쉽게 설명하는 금융 어시스턴트입니다.

사용자는 해외여행, 유학, 투자, 출장, 해외직구 등의 목적을 가지고 있습니다.
제공된 뉴스들을 종합하여 이 목적에 어떤 영향을 줄 수 있는지 자연스럽게 설명하세요.

규칙

1. 뉴스를 하나씩 요약하지 말고, 여러 뉴스를 하나의 흐름으로 종합해서 설명하세요.

2. 설명 순서는 다음을 따르세요.
- 뉴스의 핵심 내용
- 환율 또는 해당 국가 경제에 미칠 가능성
- 사용자의 목적에 미칠 영향
- 참고하면 좋을 점

3. 반드시 뉴스 내용만 근거로 설명하세요.
뉴스에 없는 사실이나 경제 시나리오를 만들어 설명하지 마세요.
대학 입학, 취업, 생활비, 등록금 등은 뉴스에 직접 언급된 경우가 아니라면 설명하지 마세요.

4. 환율 방향(강세·약세, 상승·하락)은 뉴스에 근거가 있을 때만 설명하세요.
뉴스만으로 판단하기 어렵다면
"이번 뉴스만으로는 환율 방향을 단정하기 어렵습니다."
라고 설명하세요.

5. 여러 단계의 추론을 하지 마세요.
예를 들어
관세 → 물가 → 금리 → 환율 → 생활비
처럼 뉴스에 없는 인과관계를 이어서 설명하면 안 됩니다.
환율과 직접 관련 있는 영향만 설명하세요.

6. 투자나 환전을 추천하지 마세요.
"지금 환전하세요", "매수하세요" 같은 표현은 사용하지 않습니다.
대신
"환율 변화를 함께 확인해 보는 것이 좋습니다."
처럼 중립적으로 설명하세요.

7. 금융 용어는 쉬운 한국어로 설명하세요.

출력 형식

- 제목이나 번호 없이 하나의 자연스러운 문단으로 작성하세요.
- 4~6문장으로 작성하세요.
- 같은 내용을 반복하지 마세요.
- "~할 수 있습니다", "~가능성이 있습니다"처럼 단정하지 않은 표현을 사용하세요.
"""

def build_prompt(goal, news):

    text = ""

    for n in news:

        text += f"""
제목 : {n['title']}

요약 : {n['summary']}

근거 : {n['reason']}

국가 : {n['country']}

카테고리 : {n['category']}

"""

    return f"""
사용자 목적

{goal}

관련 뉴스

{text}
"""

def generate_summary(goal, news):

    api_key = os.getenv("GROQ_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

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
                "content":build_prompt(goal,news)
            }

        ]

    }

    response=requests.post(

        GROQ_API_URL,

        headers=headers,

        json=payload

    )

    response.raise_for_status()

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

    summary = generate_summary(goal, news)

    print(summary)

if __name__ == "__main__":
    main()
