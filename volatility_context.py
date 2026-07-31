# -*- coding: utf-8 -*-
"""SV 변동성 결과를 LLM 프롬프트에 넣기 좋은 수치와 문장으로 변환한다."""

from __future__ import annotations

import csv
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
SV_PATH = BASE_DIR / "sv_volatility.json"
VRP_DATA_PATH = BASE_DIR / "vrp_data.csv"
LIVE_RATES_PATH = BASE_DIR / "live_fx_rates.json"
ASSETS_PATH = BASE_DIR / "user_assets.json"
FX_PRICES_PATH = BASE_DIR / "fx_prices.csv"

SUPPORTED_PAIRS = ("USD/KRW", "JPY/KRW", "EUR/KRW")


def normalize_pair(value: object) -> str:
    text = str(value or "").upper().replace(".", "/").replace("-", "/")
    compact = text.replace("/", "")
    aliases = {
        "USDKRW": "USD/KRW",
        "JPYKRW": "JPY/KRW",
        "EURKRW": "EUR/KRW",
    }
    return aliases.get(compact, text)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_spot_rates() -> dict[str, float]:
    """실시간 API 결과, 사용자 자산, 가격 CSV 순서로 최신 환율을 찾는다."""
    rates: dict[str, float] = {}

    if LIVE_RATES_PATH.exists():
        try:
            raw = load_json(LIVE_RATES_PATH).get("rates", {})
            for pair, value in raw.items():
                number = safe_float(value)
                if number and number > 0:
                    rates[normalize_pair(pair)] = number
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    if ASSETS_PATH.exists():
        try:
            assets = load_json(ASSETS_PATH).get("assets", [])
            for asset in assets:
                pair = normalize_pair(asset.get("currency_pair"))
                number = safe_float(asset.get("current_rate"))
                if pair not in rates and number and number > 0:
                    rates[pair] = number
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    if FX_PRICES_PATH.exists():
        try:
            with FX_PRICES_PATH.open("r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            for row in reversed(rows):
                for pair in SUPPORTED_PAIRS:
                    if pair in rates:
                        continue
                    number = safe_float(row.get(pair))
                    if number and number > 0:
                        rates[pair] = number
        except OSError:
            pass

    return rates


def load_completed_vrp_gaps() -> dict[str, dict[str, float | str]]:
    """미래 RV가 확보되어 계산이 끝난 프록시 격차의 평균과 최신값을 읽는다."""
    if not VRP_DATA_PATH.exists():
        return {}

    grouped: dict[str, list[tuple[str, float]]] = {}
    try:
        with VRP_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                pair = normalize_pair(row.get("currency_pair") or "USD/KRW")
                gap = safe_float(row.get("vrp"))
                row_date = str(row.get("date") or "")[:10]
                if pair in SUPPORTED_PAIRS and gap is not None and row_date:
                    grouped.setdefault(pair, []).append((row_date, gap))
    except OSError:
        return {}

    result: dict[str, dict[str, float | str]] = {}
    for pair, values in grouped.items():
        values.sort(key=lambda item: item[0])
        result[pair] = {
            "historical_mean_gap_pct_point": round(
                sum(value for _, value in values) / len(values) * 100.0, 3
            ),
            "latest_completed_gap_pct_point": round(values[-1][1] * 100.0, 3),
            "latest_completed_gap_date": values[-1][0],
        }
    return result


def percentile_rank(values: list[float], latest: float) -> float:
    return 100.0 * sum(value <= latest for value in values) / len(values)


def volatility_regime(percentile: float) -> str:
    if percentile >= 80:
        return "높음"
    if percentile <= 20:
        return "낮음"
    return "보통"


def percentile_explanation(percentile: float) -> str:
    count = round(percentile)
    if percentile >= 80:
        comparison = "평소보다 움직임이 큰 편"
    elif percentile >= 50:
        comparison = "과거 중간 수준보다 움직임이 다소 큰 편"
    elif percentile > 20:
        comparison = "과거 중간 수준보다 움직임이 다소 작은 편"
    else:
        comparison = "평소보다 움직임이 작은 편"
    return (
        f"과거 관측일 100일을 놓고 보면 약 {count}일보다 변동성이 큰 수준으로, "
        f"{comparison}입니다."
    )


def change_explanation(change: float | None) -> str | None:
    if change is None:
        return None
    if change > 0.25:
        return "5거래일 전보다 환율의 예상 움직임 폭이 커졌습니다."
    if change < -0.25:
        return "5거래일 전보다 환율의 예상 움직임 폭이 작아졌습니다."
    return "5거래일 전과 비교해 환율의 예상 움직임 폭은 비슷한 수준입니다."


def action_guidance(regime: str) -> str:
    if regime == "높음":
        return (
            "큰 금액의 환전·송금·결제가 예정돼 있다면 한 번에 결정하기보다 "
            "환율을 여러 차례 확인하고 예산에 여유를 두는 방식이 적절합니다."
        )
    if regime == "낮음":
        return (
            "현재 움직임이 작더라도 계속 안정적이라고 단정하지 말고, 실제 "
            "환전·송금·결제일이 가까워질 때 환율을 다시 확인하는 것이 적절합니다."
        )
    return (
        "예정된 환전·송금·결제일 전후로 환율을 확인하고, 통계적 변동 폭만큼 "
        "원화 예산에 여유를 두는 방식이 적절합니다."
    )


def days_since(date_text: str) -> int | None:
    try:
        observed = datetime.strptime(date_text[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    return (date.today() - observed).days


def build_all_volatility_contexts() -> dict[str, dict[str, Any]]:
    if not SV_PATH.exists():
        return {}

    try:
        raw = load_json(SV_PATH)
    except (OSError, json.JSONDecodeError):
        return {}

    spot_rates = load_spot_rates()
    gap_contexts = load_completed_vrp_gaps()
    contexts: dict[str, dict[str, Any]] = {}

    for raw_pair, raw_records in raw.items():
        pair = normalize_pair(raw_pair)
        if pair not in SUPPORTED_PAIRS:
            continue
        records = raw_records if isinstance(raw_records, list) else [raw_records]
        observations: list[tuple[str, float]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            sigma = safe_float(record.get("sigma"))
            observed_date = str(record.get("date") or "")[:10]
            if sigma and sigma > 0 and observed_date:
                annualized_pct = sigma * math.sqrt(252.0)
                observations.append((observed_date, annualized_pct))

        if not observations:
            continue
        observations.sort(key=lambda item: item[0])
        annualized_values = [value for _, value in observations]
        latest_date, latest_annualized = observations[-1]
        percentile = percentile_rank(annualized_values, latest_annualized)
        monthly_pct = latest_annualized / math.sqrt(12.0)
        five_day_change = (
            latest_annualized - observations[-6][1] if len(observations) >= 6 else None
        )
        spot = spot_rates.get(pair)
        estimated_move = spot * monthly_pct / 100.0 if spot else None
        age = days_since(latest_date)

        context: dict[str, Any] = {
            "currency_pair": pair,
            "reference_date": latest_date,
            "annualized_volatility_pct": round(latest_annualized, 2),
            "monthly_volatility_pct": round(monthly_pct, 2),
            "five_day_change_pct_point": (
                round(five_day_change, 2) if five_day_change is not None else None
            ),
            "historical_percentile": round(percentile, 1),
            "regime": volatility_regime(percentile),
            "percentile_explanation": percentile_explanation(percentile),
            "five_day_change_explanation": change_explanation(five_day_change),
            "spot_rate": round(spot, 4) if spot else None,
            "estimated_monthly_move_rate": round(estimated_move, 2) if estimated_move else None,
            "data_age_days": age,
            "is_stale": bool(age is not None and age > 14),
            "is_proxy": True,
            "source": "stochvol SV 기반 프록시",
        }
        context.update(gap_contexts.get(pair, {}))
        context["action_guidance"] = action_guidance(context["regime"])
        contexts[pair] = context

    return contexts


def format_volatility_context(context: dict[str, Any] | None) -> str:
    if not context:
        return ""

    lines = [
        f"통화쌍: {context['currency_pair']}",
        f"기준일: {context['reference_date']}",
        f"SV 기반 연율화 변동성: {context['annualized_volatility_pct']:.2f}%",
        f"월간 환산 변동성: {context['monthly_volatility_pct']:.2f}%",
        f"과거 관측치 대비 위치: {context['historical_percentile']:.1f}백분위 ({context['regime']} 구간)",
        f"쉬운 해석: {context['percentile_explanation']}",
    ]
    change = context.get("five_day_change_pct_point")
    if change is not None:
        sign = "+" if change >= 0 else ""
        lines.append(f"5거래일 전 대비 변화: {sign}{change:.2f}%p")
    spot = context.get("spot_rate")
    move = context.get("estimated_monthly_move_rate")
    if spot is not None and move is not None:
        lines.append(
            f"참고 환율 {spot:,.4f}원 기준 월간 통계적 변동 폭 환산: 약 ±{move:,.2f}원"
        )
        lines.append(
            f"쉬운 해석: 환율 방향을 예측하는 값은 아니지만, 한 달 단위 움직임의 크기를 현재 환율로 바꾸면 대략 ±{move:,.2f}원에 해당합니다."
        )
    change_text = context.get("five_day_change_explanation")
    if change_text:
        lines.append(f"쉬운 해석: {change_text}")
    action_text = context.get("action_guidance")
    if action_text:
        lines.append(f"허용되는 행동 제안: {action_text}")
    mean_gap = context.get("historical_mean_gap_pct_point")
    if mean_gap is not None:
        lines.append(f"계산 완료 표본의 평균 SV-실현변동성 격차: {mean_gap:+.3f}%p")
        if mean_gap < 0:
            lines.append(
                f"쉬운 해석: 과거에는 SV 추정 변동성이 이후 실제 변동성보다 평균 {abs(mean_gap):.3f}%p 낮았습니다."
            )
        elif mean_gap > 0:
            lines.append(
                f"쉬운 해석: 과거에는 SV 추정 변동성이 이후 실제 변동성보다 평균 {mean_gap:.3f}%p 높았습니다."
            )
    latest_gap = context.get("latest_completed_gap_pct_point")
    latest_gap_date = context.get("latest_completed_gap_date")
    if latest_gap is not None and latest_gap_date:
        lines.append(
            f"가장 최근 계산 완료 격차({latest_gap_date}): {latest_gap:+.3f}%p"
        )
    if context.get("is_stale"):
        lines.append(f"주의: 최신 SV 기준일로부터 {context.get('data_age_days')}일이 지났음")
    lines.extend(
        [
            "해석 제한: 이 수치는 실제 옵션 내재변동성이 아닌 SV 기반 프록시임",
            "해석 제한: 변동성은 방향이 아니라 움직임의 크기이며, ±원은 확정 범위가 아닌 통계적 환산값임",
        ]
    )
    return "\n".join(f"- {line}" for line in lines)


def select_context_for_goal(
    goal: str,
    news: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    text = goal.lower()
    keyword_pairs = (
        (("미국", "달러", "미주"), "USD/KRW"),
        (("일본", "엔화", "엔", "도쿄", "오사카"), "JPY/KRW"),
        (("유럽", "유로", "독일", "프랑스", "이탈리아"), "EUR/KRW"),
    )
    for keywords, pair in keyword_pairs:
        if any(keyword in text for keyword in keywords):
            return contexts.get(pair)

    for item in news:
        for pair in item.get("currency_pairs") or []:
            normalized = normalize_pair(pair)
            if normalized in contexts:
                return contexts[normalized]
    return contexts.get("USD/KRW")
