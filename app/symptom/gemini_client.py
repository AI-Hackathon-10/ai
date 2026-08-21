"""
증상 기반 추천에서 쓰는 Gemini 호출 전담 함수 모음.

⚠️ SDK 호출 시그니처는 app/vision/gemini_client.py와 동일한 패턴
(client.models.generate_content + GenerateContentConfig(response_schema=...))을
그대로 따른다. google-genai 최신 문서에서 정확한 호출 방식이 바뀌었다면 이
파일 내부만 교체하면 된다 — service.py는 app.symptom.types의 Protocol
시그니처에만 의존한다.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Optional

from google import genai
from google.genai import types
from langsmith import traceable

from app.recommendation import SYMPTOM_TYPE_MAP
from app.symptom.schema import (
    DRUG_CANDIDATE_RESPONSE_SCHEMA,
    DRUG_CANDIDATE_SYSTEM_PROMPT,
    SYMPTOM_EXTRACTION_RESPONSE_SCHEMA,
    SYMPTOM_EXTRACTION_SYSTEM_PROMPT,
    build_drug_candidate_prompt,
    build_symptom_extraction_prompt,
)
from app.symptom.types import SymptomExtraction

MODEL_NAME = "gemini-3.5-flash"


class SymptomGeminiError(Exception):
    """증상 추천 관련 Gemini 호출/응답 파싱 오류."""


@traceable(run_type="llm", name="gemini_symptom_extraction")
async def extract_symptoms_and_onset(
    *,
    symptom_text: str,
    onset_text: str,
    reference_time: datetime,
    gender: Optional[str] = None,
    birth_date: Optional[date] = None,
) -> SymptomExtraction:
    """증상/발생시각 STT 텍스트 + 사용자 정보 -> 구조화된 SymptomExtraction.

    발화에서 명확히 확인되지 않는 내용은 만들어내지 않는다(스키마의 enum 제약
    + 프롬프트 규칙으로 강제). 응답 파싱/모델 호출 자체가 실패하면
    SymptomGeminiError를 던진다(라우터가 503으로 변환) — 하지만 onset_at 하나가
    이상한 문자열이라 파싱만 실패하는 경우는 예외를 던지지 않고 그냥 None으로
    떨어뜨린다. 발생 시각 하나 때문에 전체 추천 흐름을 막지 않기 위해서다.
    """
    client = genai.Client()

    today = datetime.now().date()
    age: Optional[int] = None
    if birth_date is not None:
        age = today.year - birth_date.year - (
            1 if (today.month, today.day) < (birth_date.month, birth_date.day) else 0
        )
    gender_ko = {"MALE": "남성", "FEMALE": "여성"}.get((gender or "").upper())

    prompt = build_symptom_extraction_prompt(
        symptom_text=symptom_text,
        onset_text=onset_text,
        reference_time_iso=reference_time.isoformat(),
        age=age,
        gender_ko=gender_ko,
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                system_instruction=SYMPTOM_EXTRACTION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=SYMPTOM_EXTRACTION_RESPONSE_SCHEMA,
            ),
        )
    except Exception as e:  # noqa: BLE001 - SDK 예외 타입이 버전마다 달라 광범위하게 감쌈
        raise SymptomGeminiError(f"증상 추출 Gemini 호출 실패: {e}") from e

    try:
        raw = json.loads(response.text)
    except (ValueError, AttributeError) as e:
        raise SymptomGeminiError(f"증상 추출 응답 파싱 실패: {e}") from e

    return SymptomExtraction(
        symptoms=[s for s in raw.get("symptoms", []) if s in SYMPTOM_TYPE_MAP],
        onset_at=_parse_datetime_or_none(raw.get("onset_at")),
        onset_confidence=raw.get("onset_confidence") or "UNKNOWN",
    )


@traceable(run_type="llm", name="gemini_drug_candidates")
async def suggest_candidate_drug_names(
    *,
    symptoms: list[str],
    gender: Optional[str],
    birth_date: Optional[date],
    as_of: Optional[date] = None,
) -> list[str]:
    """증상(+나이/성별) -> OTC 제품명 후보 최대 3개.

    ⚠️ 여기서 나온 이름은 아직 검증되지 않은 '검색어 후보'다. 실존 여부와
    상세정보는 반드시 app/symptom/service.py에서 e약은요 공식 API로 재확인한
    뒤에만 사용자에게 노출해야 한다 — 이 함수의 반환값을 그대로 최종 결과로
    쓰지 말 것.
    """
    client = genai.Client()

    today = as_of or datetime.now().date()
    age: Optional[int] = None
    if birth_date is not None:
        age = today.year - birth_date.year - (
            1 if (today.month, today.day) < (birth_date.month, birth_date.day) else 0
        )
    gender_ko = {"MALE": "남성", "FEMALE": "여성"}.get((gender or "").upper())
    symptoms_ko = [SYMPTOM_TYPE_MAP.get(s, s) for s in symptoms]

    prompt = build_drug_candidate_prompt(symptoms_ko=symptoms_ko, age=age, gender_ko=gender_ko)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                system_instruction=DRUG_CANDIDATE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=DRUG_CANDIDATE_RESPONSE_SCHEMA,
            ),
        )
    except Exception as e:  # noqa: BLE001
        raise SymptomGeminiError(f"약 후보 제안 Gemini 호출 실패: {e}") from e

    try:
        raw = json.loads(response.text)
    except (ValueError, AttributeError) as e:
        raise SymptomGeminiError(f"약 후보 제안 응답 파싱 실패: {e}") from e

    candidates = raw.get("candidates", [])
    # 방어적으로 문자열이 아니거나 빈 값인 후보는 걸러낸다.
    return [c.strip() for c in candidates if isinstance(c, str) and c.strip()][:3]


def _parse_datetime_or_none(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
