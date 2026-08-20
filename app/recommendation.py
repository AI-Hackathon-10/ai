"""사용자 나이·성별·증상으로 복용 가능 여부와 주의사항을 개인화한다."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.drug_detail_models import DrugDetailApiItem

logger = logging.getLogger(__name__)

SYMPTOM_TYPE_MAP: dict[str, str] = {
    "HEADACHE": "두통",
    "FEVER": "발열",
    "COUGH": "기침",
    "SORE_THROAT": "인후통",
    "RUNNY_NOSE": "콧물",
    "NASAL_CONGESTION": "코막힘",
    "ABDOMINAL_PAIN": "복통",
    "INDIGESTION": "소화불량",
    "DIARRHEA": "설사",
    "CONSTIPATION": "변비",
    "HEARTBURN": "속쓰림",
    "NAUSEA_OR_VOMITING": "구토/메스꺼움",
    "MUSCLE_PAIN": "근육통",
    "MENSTRUAL_CRAMPS": "생리통",
    "TOOTHACHE": "치통",
    "ALLERGY": "알레르기",
    "ITCHY_SKIN": "피부 가려움",
    "BODY_ACHES": "몸살",
    "DIZZINESS": "어지러움",
    "CHILLS": "오한",
}

_KST = timezone(timedelta(hours=9))
_ELDERLY_AGE = 65
_DURATION_CAUTION_DAYS = 3

_PEDIATRIC_CLAUSE = re.compile(r",?\s*만?\s*\d+\s*세\s*미만의?\s*소아[는은]?")
_PREGNANCY_CLAUSE = re.compile(r",?\s*임부(?:\s*또는\s*수유부)?|,?\s*수유부")
_ELDERLY_CLAUSE = re.compile(r",?\s*고령자(?:\(노인\))?|,?\s*노인")
_MIN_AGE_UNDER = re.compile(r"만?\s*(\d+)\s*세\s*미만")
_MIN_AGE_OVER = re.compile(r"만?\s*(\d+)\s*세\s*이상")


@dataclass(frozen=True)
class RecommendationDraft:
    status: str
    score: float
    confidence: str
    reason: str
    caution: Optional[str]


async def build_recommendation(
    *,
    identification,
    detail: DrugDetailApiItem,
    symptoms: list[str],
    gender: Optional[str] = None,
    birth_date: Optional[date] = None,
    symptom_started_at: Optional[datetime] = None,
    as_of: Optional[date] = None,
) -> RecommendationDraft:
    today = as_of or datetime.now(_KST).date()
    age = _age_years(birth_date, today) if birth_date else None
    gender_key = (gender or "").upper() or None
    display_symptoms = [_display_symptom(s) for s in symptoms if s]
    duration_days = _symptom_days(symptom_started_at, today)

    # LLM 호출 시도
    try:
        from app.recommendation_llm import generate_recommendation_via_llm

        return await generate_recommendation_via_llm(
            detail=detail,
            symptoms=display_symptoms,
            age=age,
            gender_key=gender_key,
            duration_days=duration_days,
        )
    except Exception:
        logger.warning("LLM 추천 생성 실패, 규칙 기반 폴백", exc_info=True)

    # 기존 규칙 기반 로직 (폴백)
    matched = [s for s in display_symptoms if s and detail.efficacy and s in detail.efficacy]
    min_age = _min_age(detail.precautions, detail.usage_method, detail.warning)
    age_blocked = age is not None and min_age is not None and age < min_age
    long_duration = duration_days is not None and duration_days >= _DURATION_CAUTION_DAYS
    pregnancy_relevant = gender_key == "FEMALE" and (age is None or age >= 12)
    elderly_relevant = age is not None and age >= _ELDERLY_AGE
    personal_flags = bool(pregnancy_relevant or elderly_relevant or long_duration)

    if age_blocked:
        status, factor, confidence = "NOT_RECOMMENDED", 0.2, "HIGH"
    elif not matched:
        status, factor, confidence = "NOT_RECOMMENDED", 0.4, "LOW"
    elif personal_flags or (display_symptoms and len(matched) < len(display_symptoms)):
        status, factor, confidence = "RECOMMENDED", 0.9, "MEDIUM"
    else:
        status, factor, confidence = "RECOMMENDED", 0.94, "HIGH"

    score = round((identification.score if identification else 1.0) * factor, 2)
    reason = _reason(
        status=status,
        age=age,
        gender_key=gender_key,
        display_symptoms=display_symptoms,
        matched=matched,
        min_age=min_age,
        age_blocked=age_blocked,
    )
    caution = _caution(
        detail=detail,
        keep_pediatric=age_blocked or (age is not None and age < 18),
        keep_pregnancy=pregnancy_relevant,
        keep_elderly=elderly_relevant,
        long_duration=long_duration,
    )
    return RecommendationDraft(
        status=status,
        score=score,
        confidence=confidence,
        reason=reason,
        caution=caution,
    )


def _display_symptom(value: str) -> str:
    return SYMPTOM_TYPE_MAP.get(value, value)


def _age_years(birth_date: date, today: date) -> int:
    years = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years


def _min_age(*texts: Optional[str]) -> Optional[int]:
    joined = "\n".join(text for text in texts if text)
    ages = [int(n) for n in _MIN_AGE_UNDER.findall(joined)]
    ages.extend(int(n) for n in _MIN_AGE_OVER.findall(joined))
    return max(ages) if ages else None


def _symptom_days(started_at: Optional[datetime], today: date) -> Optional[int]:
    if started_at is None:
        return None
    started = started_at.astimezone(_KST).date() if started_at.tzinfo else started_at.date()
    return (today - started).days


def _gender_ko(gender_key: Optional[str]) -> Optional[str]:
    if gender_key == "MALE":
        return "남성"
    if gender_key == "FEMALE":
        return "여성"
    return None


def _who(age: Optional[int], gender_key: Optional[str]) -> str:
    gender_ko = _gender_ko(gender_key)
    if age is not None and gender_ko:
        return f"{age}세 {gender_ko}"
    if age is not None:
        return f"{age}세"
    if gender_ko:
        return gender_ko
    return "사용자"


def _symptom_text(display_symptoms: list[str]) -> str:
    return ", ".join(display_symptoms) if display_symptoms else "입력한 증상"


def _reason(
    *,
    status: str,
    age: Optional[int],
    gender_key: Optional[str],
    display_symptoms: list[str],
    matched: list[str],
    min_age: Optional[int],
    age_blocked: bool,
) -> str:
    who = _who(age, gender_key)
    symptoms_ko = _symptom_text(display_symptoms)
    if age_blocked and min_age is not None:
        return f"{who}는 이 약의 복용 연령(만 {min_age}세 이상)에 해당하지 않습니다."
    if status == "NOT_RECOMMENDED":
        return f"{who}의 {symptoms_ko} 증상은 이 약의 효능과 일치하지 않습니다."
    if matched and len(matched) < len(display_symptoms):
        matched_ko = ", ".join(matched)
        return (
            f"{who}의 {symptoms_ko} 중 {matched_ko}은(는) 이 약의 효능과 맞습니다. "
            "나머지 증상은 이 약으로 해결되지 않을 수 있어 복용 전 확인이 필요합니다."
        )
    return (
        f"{who}의 {symptoms_ko} 증상은 이 약의 효능과 맞습니다. "
        "연령 금기에도 해당하지 않아 복용을 고려할 수 있습니다."
    )


def _caution(
    *,
    detail: DrugDetailApiItem,
    keep_pediatric: bool,
    keep_pregnancy: bool,
    keep_elderly: bool,
    long_duration: bool,
) -> Optional[str]:
    chunks: list[str] = []
    for raw in (detail.warning, detail.precautions):
        if not raw:
            continue
        for paragraph in raw.split("\n"):
            filtered = _filter_paragraph(
                paragraph,
                keep_pediatric=keep_pediatric,
                keep_pregnancy=keep_pregnancy,
                keep_elderly=keep_elderly,
            )
            if filtered:
                chunks.append(filtered)
    if long_duration:
        chunks.append("증상이 3일 이상 지속되고 있습니다. 호전되지 않으면 의사 또는 약사와 상의하십시오.")
    if not chunks:
        return detail.precautions
    return "\n".join(chunks)


def _filter_paragraph(
    paragraph: str,
    *,
    keep_pediatric: bool,
    keep_pregnancy: bool,
    keep_elderly: bool,
) -> str:
    text = paragraph.strip()
    if not text:
        return ""
    if not keep_pediatric:
        text = _PEDIATRIC_CLAUSE.sub("", text)
    if not keep_pregnancy:
        text = _PREGNANCY_CLAUSE.sub("", text)
    if not keep_elderly:
        text = _ELDERLY_CLAUSE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"^[,\s]+", "", text)
    text = re.sub(r"(사람|환자)\s+이 약", r"\1은 이 약", text)
    text = text.strip()
    if not text or text in {"는 의사 또는 약사와 상의하십시오.", "은 의사 또는 약사와 상의하십시오."}:
        return ""
    # 개인 태그를 모두 빼면 주어가 비는 문장만 남는 경우 버린다.
    if re.fullmatch(r"[는은]?\s*의사 또는 약사와 상의하십시오\.?", text):
        return ""
    return text
