"""OpenAI LLM을 이용한 개인화 복용 추천 생성."""
from __future__ import annotations

import json
import logging
from typing import Optional

from langsmith import traceable
from openai import AsyncOpenAI

from app.config import settings
from app.drug_detail_models import DrugDetailApiItem
from app.recommendation import RecommendationDraft

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"


class RecommendationLLMError(Exception):
    """LLM 추천 생성 관련 오류."""


_SYSTEM_PROMPT = """\
당신은 의약품 복용 추천 전문가입니다. 사용자 정보와 의약품 상세정보를 바탕으로
해당 의약품의 복용 적합성을 판단합니다.

다음 JSON 형식으로만 응답하십시오:
- status: "RECOMMENDED" 또는 "NOT_RECOMMENDED"
- score: 0.0~1.0 사이의 추천 점수 (소수점 둘째 자리까지)
- confidence: "HIGH", "MEDIUM", "LOW" 중 하나
- reason: 한국어로 작성한 추천/비추천 사유 (1~3문장)
- caution: 한국어로 작성한 복용 시 주의사항 (없으면 null)

판단 기준:
1. 의약품 효능과 사용자 증상의 일치 여부
2. 사용자 연령에 따른 금기 사항 (소아, 고령자)
3. 성별에 따른 주의 (임부/수유부 가능성)
4. 증상 지속 기간에 따른 전문의 상담 권고
5. 의약품의 경고/주의사항/부작용/상호작용 정보"""

_RECOMMENDATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["RECOMMENDED", "NOT_RECOMMENDED"],
        },
        "score": {
            "type": "number",
        },
        "confidence": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"],
        },
        "reason": {
            "type": "string",
        },
        "caution": {
            "type": ["string", "null"],
        },
    },
    "required": ["status", "score", "confidence", "reason", "caution"],
    "additionalProperties": False,
}


def _build_user_prompt(
    detail: DrugDetailApiItem,
    symptoms: list[str],
    age: Optional[int],
    gender_key: Optional[str],
    duration_days: Optional[int],
) -> str:
    gender_ko = {"MALE": "남성", "FEMALE": "여성"}.get(gender_key or "", "미상")
    age_text = f"{age}세" if age is not None else "미상"
    symptoms_text = ", ".join(symptoms) if symptoms else "없음"
    duration_text = f"{duration_days}일" if duration_days is not None else "미상"

    parts = [
        "## 사용자 정보",
        f"- 나이: {age_text}",
        f"- 성별: {gender_ko}",
        f"- 증상: {symptoms_text}",
        f"- 증상 지속일수: {duration_text}",
        "",
        "## 의약품 정보",
        f"- 의약품명: {detail.item_name or '미상'}",
    ]
    for label, value in (
        ("효능", detail.efficacy),
        ("용법", detail.usage_method),
        ("경고", detail.warning),
        ("주의사항", detail.precautions),
        ("상호작용", detail.interactions),
        ("부작용", detail.side_effects),
    ):
        if value:
            parts.append(f"- {label}: {value}")

    return "\n".join(parts)


@traceable(run_type="llm", name="recommendation_llm_call")
async def generate_recommendation_via_llm(
    detail: DrugDetailApiItem,
    symptoms: list[str],
    age: Optional[int],
    gender_key: Optional[str],
    duration_days: Optional[int],
) -> RecommendationDraft:
    """OpenAI를 호출하여 개인화 복용 추천을 생성한다.

    실패 시 RecommendationLLMError를 발생시킨다.
    """
    client = AsyncOpenAI()
    model_name = settings.recommend_model or _DEFAULT_MODEL
    user_prompt = _build_user_prompt(detail, symptoms, age, gender_key, duration_days)

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendation_response",
                    "strict": True,
                    "schema": _RECOMMENDATION_RESPONSE_SCHEMA,
                },
            },
        )
    except Exception as e:  # noqa: BLE001
        raise RecommendationLLMError(f"OpenAI 추천 LLM 호출 실패: {e}") from e

    try:
        data = json.loads(response.choices[0].message.content)
    except (ValueError, AttributeError, IndexError) as e:
        raise RecommendationLLMError(f"OpenAI 추천 LLM 응답 파싱 실패: {e}") from e

    score = max(0.0, min(1.0, round(float(data["score"]), 2)))

    return RecommendationDraft(
        status=data["status"],
        score=score,
        confidence=data["confidence"],
        reason=data["reason"],
        caution=data.get("caution"),
    )
