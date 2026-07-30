import streamlit as st

from Modified_goal_new_summary import (
    generate_summary,
    build_macro_context,
    load_macro_data,
)

from goal_news_search import search_goal_news


st.set_page_config(
    page_title="환율 뉴스 AI",
    page_icon="💱",
    layout="wide"
)

st.title("💱 목적 기반 환율 뉴스 AI")

st.write(
    """
사용 목적을 입력하면

- 관련 환율 뉴스를 찾아주고
- AI가 종합해서 쉽게 설명해줍니다.
"""
)

example_goals = [
    "미국 주식 투자",
    "일본 여행",
    "유럽 출장",
    "미국 유학",
    "해외직구"
]

goal = st.selectbox(
    "목적 선택",
    example_goals
)

custom_goal = st.text_input(
    "또는 직접 입력",
    placeholder="예) 미국 ETF 투자"
)

if custom_goal.strip():
    goal = custom_goal


if st.button("AI 분석하기", use_container_width=True):

    with st.spinner("관련 뉴스를 찾는 중입니다..."):

        context, news = search_goal_news(goal)

    if not news:

        st.warning("관련 뉴스를 찾지 못했습니다.")
        st.stop()

    st.success(f"{len(news)}개의 관련 뉴스를 찾았습니다.")

    st.divider()

    st.subheader("📰 관련 뉴스")

    for idx, n in enumerate(news, 1):

        with st.expander(f"{idx}. {n['title']}"):

            st.write("**요약**")
            st.write(n["summary"])

            st.write("**국가**")
            st.write(n.get("country", "-"))

            st.write("**카테고리**")
            st.write(n.get("category", "-"))

            st.write("**선정 이유**")
            st.info(n.get("reason", ""))

    st.divider()

    st.subheader("🤖 AI 종합 분석")

    with st.spinner("AI가 분석 중입니다..."):

        macro = load_macro_data()

        macro_context = build_macro_context(
            macro,
            news
        )

        summary = generate_summary(
            goal,
            news,
            macro_context
        )

    for n in news:

        st.container(border=True)

        st.markdown(f"### 📰 {n['title']}")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("국가", n.get("country","-"))

        with c2:
            st.metric("카테고리", n.get("category","-"))

        st.write(n["summary"])

        st.caption(n.get("reason",""))

        st.divider()

    with st.sidebar:

        st.header("서비스 소개")

        st.write("""
        목적에 맞는 환율 뉴스를 검색하여
        AI가 이해하기 쉽게 설명합니다.
        """)

        st.markdown(
    """
<div class="stAlert">
✔ 투자<br>
✔ 여행<br>
✔ 유학<br>
✔ 출장<br>
✔ 해외직구
</div>
""",
    unsafe_allow_html=True,
)
    st.success(summary)