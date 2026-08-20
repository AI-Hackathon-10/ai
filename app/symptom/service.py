"""
사진 없이 '증상 음성 입력'만으로 OTC 후보 약을 추천하는 오케스트레이션.

⚠️ 신뢰성 원칙(기획서: "AI가 약 자체를 추측하는 대신 Vision AI + 식약처 공식
데이터를 결합해 신뢰성을 높인다"): 이 서비스는 LLM을 딱 두 곳에서만 쓴다.
  1) STT 텍스트 -> 구조화된 증상/발생시각 추출 (extract_fn)
  2) 증상 -> OTC 제품명 '후보' 제안 (suggest_fn)
후보 이름은 반드시 e약은요 공식 API(detail_service)로 조회해서 실존 + 상세정보가
확인된 것만 최종 결과에 포함한다(_verify_candidates). 공식 DB에서 못 찾은 후보,
효능 정보가 없는 후보, 연령/효능 필터에서 걸러진 후보는 전부 버린다 — 확인 안 된
약 이름을 사용자에게 보여주지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from app.drug_detail_service import DrugDetailService
from app.identify_result import Identification, Official, Recommendation, official_from_detail
from app.recommendation import build_recommendation
from app.symptom.types import ExtractSymptomsFn, SuggestCandidateDrugsFn


@dataclass(frozen=True)
class CandidateDrug:
    item_seq: Optional[str]
    item_name: Optional[str]
    image_url: Optional[str]
    official: Official
    recommendation: Recommendation


@dataclass(frozen=True)
class SymptomRecommendResult:
    parsed_symptoms: list[str]
    symptom_started_at: Optional[datetime]
    onset_confidence: str
    candidates: list[CandidateDrug] = field(default_factory=list)
    notice: Optional[str] = None  # 증상을 못 읽었거나 후보를 하나도 못 찾았을 때 안내 문구


# 사진으로 확정된 알약이 아니라 AI가 증상만으로 제안한 후보이므로, identification
# score를 사진 스캔(확정, 1.0)보다 낮은 0.7로 잡는다. build_recommendation()은
# 이 score만 최종 recommendation.score 계산에 곱하고(score = identification.score
# * factor), recommendation.confidence는 별도로 증상 일치도/연령 등에 따라
# 자체 산정하므로 여기서 confidence를 낮게 줘도 출력 confidence에는 반영되지
# 않는다 — "AI 제안이라 신뢰도가 낮다"는 사실은 recommendation.score 하락과
# CandidateDrugResponse.source="AI_SUGGESTED" 두 곳으로 프론트에 전달된다.
_AI_SUGGESTED_IDENTIFICATION = Identification(confidence="MEDIUM", score=0.7)

_NO_SYMPTOM_NOTICE = "말씀하신 내용에서 증상을 확인하지 못했습니다. 다시 말씀해 주세요."
_NO_CANDIDATE_NOTICE = (
    "증상에 맞는 일반의약품을 공식 데이터베이스에서 확인하지 못했습니다. "
    "약사 또는 의사와 상담해 주세요."
)


async def recommend_drugs_from_voice(
    *,
    symptom_text: str,
    onset_text: str,
    gender: Optional[str],
    birth_date: Optional[date],
    detail_service: DrugDetailService,
    extract_fn: ExtractSymptomsFn,
    suggest_fn: SuggestCandidateDrugsFn,
    reference_time: Optional[datetime] = None,
) -> SymptomRecommendResult:
    now = reference_time or datetime.now().astimezone()

    extraction = await extract_fn(
        symptom_text=symptom_text,
        onset_text=onset_text,
        reference_time=now,
    )

    if not extraction.symptoms:
        return SymptomRecommendResult(
            parsed_symptoms=[],
            symptom_started_at=None,
            onset_confidence="UNKNOWN",
            candidates=[],
            notice=_NO_SYMPTOM_NOTICE,
        )

    candidate_names = await suggest_fn(
        symptoms=extraction.symptoms,
        gender=gender,
        birth_date=birth_date,
    ) 
    print(f"[DEBUG] Gemini가 제안한 후보 약 이름: {candidate_names}")

    candidates = await _verify_candidates(
        candidate_names=candidate_names,
        symptoms=extraction.symptoms,
        gender=gender,
        birth_date=birth_date,
        symptom_started_at=extraction.onset_at,
        detail_service=detail_service,
    )

    return SymptomRecommendResult(
        parsed_symptoms=extraction.symptoms,
        symptom_started_at=extraction.onset_at,
        onset_confidence=extraction.onset_confidence,
        candidates=candidates,
        notice=None if candidates else _NO_CANDIDATE_NOTICE,
    )


async def _verify_candidates(
    *,
    candidate_names: list[str],
    symptoms: list[str],
    gender: Optional[str],
    birth_date: Optional[date],
    symptom_started_at: Optional[datetime],
    detail_service: DrugDetailService,
) -> list[CandidateDrug]:
    verified: list[CandidateDrug] = []
    seen_item_seq: set[str] = set()

    for name in candidate_names:
        detail = await detail_service.get_detail_by_name(name)
        if detail is None or not detail.efficacy:
            print(f"[DEBUG] '{name}' -> 공식 DB에 없거나 효능 정보 없음 (detail={detail})")
            continue  # 공식 DB에 없거나 효능 정보가 없는 후보는 신뢰할 수 없으므로 버린다
        if detail.item_seq:
            if detail.item_seq in seen_item_seq:
                continue
            seen_item_seq.add(detail.item_seq)

        draft = await build_recommendation(
            identification=_AI_SUGGESTED_IDENTIFICATION,
            detail=detail,
            symptoms=symptoms,
            gender=gender,
            birth_date=birth_date,
            symptom_started_at=symptom_started_at,
        )
        if draft.status == "NOT_RECOMMENDED":
            print(f"[DEBUG] '{name}' -> 공식 DB에는 있지만 NOT_RECOMMENDED 처리됨: {draft.reason}")
            continue  # 연령 금기/효능 불일치로 판정된 후보는 애초에 노출하지 않는다

        verified.append(
            CandidateDrug(
                item_seq=detail.item_seq,
                item_name=detail.item_name,
                image_url=detail.item_image,
                official=official_from_detail(detail),
                recommendation=Recommendation(
                    status=draft.status,
                    score=draft.score,
                    confidence=draft.confidence,
                    reason=draft.reason,
                    caution=draft.caution,
                ),
            )
        )

    return verified
