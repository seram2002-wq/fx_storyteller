# -*- coding: utf-8 -*-
"""
[SV 모형 3단계] sv_volatility.json + processed_news.jsonl -> impact_weights.json
--------------------------------------------------------------------------------
기존 compute_impact_weights.py와 로직은 같다 (카테고리별 뉴스가 있는 날의
평균 변동성을 baseline과 비교해서 가중치 산출). 차이는 "변동성"을 구하는
방법뿐이다.

    기존 버전: 단순 |전일대비 등락률|(%) 사용
    이 버전:   R stochvol로 추정한 SV 모형의 조건부 표준편차(sigma_t) 사용
              -> 논문(천도현 외, 2017)이 사용한 것과 같은 방법론이라, GARCH나
                 단순 등락률보다 극단치(급등락) 구간을 더 정교하게 반영한다.

사전 준비:
    estimate_sv_volatility.R을 먼저 실행해서 sv_volatility.json을 만들어둘 것
"""

import json
from pathlib import Path

NEWS_PATH = Path("processed_news.jsonl")
SV_PATH = Path("sv_volatility.json")
OUTPUT_PATH = Path("impact_weights.json")

CATEGORIES = ["금리", "무역", "지정학", "환율"]


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
    if not published_at:
        return None
    return published_at[:10]


def load_sv_volatility(path: Path) -> dict[str, dict[str, float]]:
    """
    sv_volatility.json -> { "USD/KRW": {"2026-07-01": 0.42, ...}, ... } 형태로 변환
    (R에서 리스트로 저장한 걸 date->sigma 딕셔너리로 재구성)
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    result = {}
    for pair, records in raw.items():
        # jsonlite가 단일 레코드일 때 list가 아니라 dict로 저장하는 경우 방어
        if isinstance(records, dict):
            records = [records]
        result[pair] = {r["date"]: r["sigma"] for r in records}

    return result


def compute_category_ratios(news_items: list[dict], sv_data: dict) -> dict:
    # 통화쌍별 baseline(전체 기간 평균 sigma)
    baselines = {}
    for pair, date_sigma in sv_data.items():
        values = list(date_sigma.values())
        if values:
            baselines[pair] = sum(values) / len(values)

    overall_baseline = sum(baselines.values()) / len(baselines) if baselines else 0

    category_changes: dict[str, list[float]] = {c: [] for c in CATEGORIES}
    category_sample_count: dict[str, int] = {c: 0 for c in CATEGORIES}

    for item in news_items:
        category = item.get("category")
        if category not in CATEGORIES:
            continue
        date = to_date(item.get("published_at", ""))
        if not date:
            continue

        pairs = item.get("currency_pairs") or list(sv_data.keys())

        for pair in pairs:
            date_sigma = sv_data.get(pair)
            if not date_sigma or date not in date_sigma:
                continue
            category_changes[category].append(date_sigma[date])
            category_sample_count[category] += 1

    results = {}
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
            "avg_sv_sigma": round(avg_change, 4),
            "baseline_avg_sv_sigma": round(overall_baseline, 4),
            "raw_ratio": round(ratio, 4),
            "method": "stochastic_volatility(stochvol)",
        }

    return results


def normalize_weights(ratios: dict) -> dict:
    raw_values = [v["raw_ratio"] for v in ratios.values() if v["sample_size"] > 0]
    if not raw_values:
        for v in ratios.values():
            v["weight"] = 0.5
        return ratios

    lo, hi = min(raw_values), max(raw_values)
    for v in ratios.values():
        if v["sample_size"] == 0:
            v["weight"] = 0.5
            continue
        if hi == lo:
            v["weight"] = 0.6
        else:
            normalized = (v["raw_ratio"] - lo) / (hi - lo)
            v["weight"] = round(0.2 + normalized * 0.8, 3)
    return ratios


def main():
    news_items = load_relevant_news(NEWS_PATH)
    print(f"관련 뉴스 {len(news_items)}건 로드")

    sv_data = load_sv_volatility(SV_PATH)
    for pair, date_sigma in sv_data.items():
        print(f"[{pair}] SV 변동성 {len(date_sigma)}일치 로드")

    ratios = compute_category_ratios(news_items, sv_data)
    weights = normalize_weights(ratios)

    print("\n=== 카테고리별 영향도 분석 결과 (SV 모형 기반) ===")
    for category, info in weights.items():
        print(f"{category}: weight={info['weight']}  "
              f"(raw_ratio={info['raw_ratio']}, 표본={info['sample_size']}건)")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료 -> {OUTPUT_PATH.resolve()}")
    print("이 impact_weights.json은 generate_personalized_summary.py에서 그대로 사용 가능합니다.")


if __name__ == "__main__":
    main()
