import json
from pathlib import Path

NEWS_PATH = Path("processed_news.jsonl")

NEWS_PATH = Path("processed_news.jsonl")

MIN_SCORE = 0.5      # 최소 관련도
TOP_K = 5            # 최대 뉴스 개수

COUNTRY_ALIAS = {

    "일본": {
        "country": "일본",
        "currency_pair": "JPY/KRW",
        "keywords": [
            "일본",
            "엔화",
            "여행",
            "관광",
            "숙박"
        ]
    },

    "미국": {
        "country": "미국",
        "currency_pair": "USD/KRW",
        "keywords": [
            "미국",
            "달러",
            "주식",
            "투자"
        ]
    },

    "유럽": {
        "country": "유럽",
        "currency_pair": "EUR/KRW",
        "keywords": [
            "유럽",
            "독일",
            "프랑스",
            "이탈리아",
            "유로"
        ]
    }
}



GOAL_TYPES = {

    "여행": [
        "여행",
        "관광",
        "휴가",
        "숙박",
        "항공"
    ],

    "유학": [
        "유학",
        "교환학생",
        "학비",
        "등록금",
        "생활비"
    ],

    "투자": [
        "투자",
        "주식",
        "ETF",
        "펀드"
    ],

    "출장": [
        "출장",
        "업무",
        "비즈니스"
    ],

    "직구": [
        "직구",
        "쇼핑",
        "구매"
    ]
}



def analyze_goal(goal):

    result = {

        "goal": goal,

        "intent": "기타",

        "country": "기타",

        "currency_pair": None,

        "keywords": []

    }
    

    # 국가 분석

    for key, info in COUNTRY_ALIAS.items():

        for word in info["keywords"]:

            if word in goal:

                result["country"] = info["country"]

                result["currency_pair"] = info["currency_pair"]

                result["keywords"].extend(
                    info["keywords"]
                )

                break


        if result["currency_pair"]:
            break



    # 목적 분석

    for intent, words in GOAL_TYPES.items():

        for word in words:

            if word in goal:

                result["intent"] = intent

                result["keywords"].extend(words)

                break


        if result["intent"] != "기타":
            break



    result["keywords"] = list(
        set(result["keywords"])
    )


    return result


# ===========================
# 뉴스 로드
# ===========================
def load_news():

    news=[]

    with open(
        NEWS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            news.append(
                json.loads(line)
            )

    return news



def calculate_score(context, news):

    score = 0


    pair = context["currency_pair"]

    keywords = context["keywords"]


    # 통화 매칭

    if pair and pair in (
        news.get("currency_pairs") or []
    ):

        score += 1



    text = (

        news.get("title","")

        +

        news.get("reason","")

    )


    # 키워드 매칭

    for keyword in keywords:

        if keyword in text:

            score += 0.2

    return score



def find_related_news(context):

    news_list = load_news()
    results = []

    for news in news_list:

        score = calculate_score(
            context,
            news
        )

        if score >= MIN_SCORE:

            results.append(
                {
                    **news,
                    "score": round(score, 3)
                }
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # 중복 제거
    unique = []
    seen = set()

    for news in results:

        if news["title"] in seen:
            continue

        seen.add(news["title"])
        unique.append(news)

    return unique[:TOP_K]


def search_goal_news(goal):

    context = analyze_goal(goal)

    news = find_related_news(context)

    return context, news