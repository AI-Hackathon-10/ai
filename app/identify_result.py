"""판별 파이프라인 결과를 백엔드가 받는 최종 result 항목으로 변환한다."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from datetime import date, datetime

from app.drug_detail_models import DrugDetailApiItem
from app.models import MatchStatus, PillCandidate, VisionExtractionResult
from app.pipeline import IdentifyFromDbOutcome
from app.recommendation import SYMPTOM_TYPE_MAP, build_recommendation

_SHAPE_MAP = {
    "원형": "ROUND",
    "타원형": "OVAL",
    "장방형": "OBLONG",
    "삼각형": "TRIANGLE",
    "사각형": "RECTANGLE",
    "마름모형": "DIAMOND",
    "오각형": "PENTAGON",
    "육각형": "HEXAGON",
    "팔각형": "OCTAGON",
    "반원형": "SEMICIRCLE",
    "기타": "OTHER",
}

_COLOR_MAP = {
    "하양": "WHITE",
    "노랑": "YELLOW",
    "주황": "ORANGE",
    "분홍": "PINK",
    "빨강": "RED",
    "갈색": "BROWN",
    "연두": "LIGHT_GREEN",
    "초록": "GREEN",
    "청록": "TEAL",
    "파랑": "BLUE",
    "남색": "NAVY",
    "자주": "MAGENTA",
    "보라": "PURPLE",
    "회색": "GRAY",
    "검정": "BLACK",
    "투명": "TRANSPARENT",
}


class Identification(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    confidence: str
    score: float


class Recommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    score: float
    confidence: str
    reason: str
    caution: Optional[str] = None


class Features(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    front_imprint: Optional[str] = Field(default=None, alias="frontImprint")
    back_imprint: Optional[str] = Field(default=None, alias="backImprint")
    shape: Optional[str] = None
    color: Optional[str] = None
    score_line: bool = Field(default=False, alias="scoreLine")


class Official(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    item_seq: Optional[str] = Field(default=None, alias="itemSeq")
    item_name: Optional[str] = Field(default=None, alias="itemName")
    efficacy: Optional[str] = None
    use_method: Optional[str] = Field(default=None, alias="useMethod")
    warning: Optional[str] = None
    caution: Optional[str] = None
    interaction: Optional[str] = None
    side_effect: Optional[str] = Field(default=None, alias="sideEffect")
    storage: Optional[str] = None
    image_url: Optional[str] = Field(default=None, alias="imageUrl")


class IdentifyResultItem(BaseModel):
    """백엔드가 result 리스트의 원소로 받는 알약 1건."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    ok: bool
    item_seq: Optional[str] = Field(default=None, alias="itemSeq")
    item_name: Optional[str] = Field(default=None, alias="itemName")
    image_url: Optional[str] = Field(default=None, alias="imageUrl")
    identification: Identification
    recommendation: Optional[Recommendation] = None
    features: Optional[Features] = None
    official: Optional[Official] = None
    document: Optional[str] = None


def needs_official_detail(outcome: IdentifyFromDbOutcome) -> bool:
    match = outcome.match_result
    return (
        not outcome.vision_failed
        and match is not None
        and match.status == MatchStatus.SINGLE_MATCH
        and bool(match.candidates)
        and bool(match.candidates[0].item_seq)
    )


def build_identify_result(
    *,
    id: str,
    outcome: IdentifyFromDbOutcome,
    detail: Optional[DrugDetailApiItem] = None,
    symptoms: Optional[list[str]] = None,
    gender: Optional[str] = None,
    birth_date: Optional[date] = None,
    symptom_started_at: Optional[datetime] = None,
) -> IdentifyResultItem:
    vision = outcome.vision_result
    candidate = _confirmed_candidate(outcome)
    ok = candidate is not None
    identification = _identification(ok)
    official = official_from_detail(detail) if detail else None
    recommendation = _recommendation(
        ok,
        identification,
        detail,
        symptoms or [],
        gender=gender,
        birth_date=birth_date,
        symptom_started_at=symptom_started_at,
    )

    return IdentifyResultItem(
        id=id,
        ok=ok,
        item_seq=candidate.item_seq if candidate else None,
        item_name=candidate.item_name if candidate else None,
        image_url=_image_url(candidate, detail),
        identification=identification,
        recommendation=recommendation,
        features=_features(vision, candidate),
        official=official,
        document=_document(candidate, official, recommendation) if ok else None,
    )


def _confirmed_candidate(outcome: IdentifyFromDbOutcome) -> Optional[PillCandidate]:
    match = outcome.match_result
    if outcome.vision_failed or match is None:
        return None
    if match.status != MatchStatus.SINGLE_MATCH or not match.candidates:
        return None
    return match.candidates[0]


def _identification(ok: bool) -> Identification:
    if not ok:
        return Identification(confidence="LOW", score=0.0)
    return Identification(confidence="HIGH", score=1.0)


def _image_url(candidate: Optional[PillCandidate], detail: Optional[DrugDetailApiItem]) -> Optional[str]:
    if candidate and candidate.item_image:
        return candidate.item_image
    if detail and detail.item_image:
        return detail.item_image
    return None


def _features(
    vision: Optional[VisionExtractionResult],
    candidate: Optional[PillCandidate],
) -> Optional[Features]:
    if vision is None and candidate is None:
        return None
    return Features(
        front_imprint=_first(vision.print_front if vision else None, candidate.print_front if candidate else None),
        back_imprint=_first(vision.print_back if vision else None, candidate.print_back if candidate else None),
        shape=_map_code(
            _first(vision.drug_shape if vision else None, candidate.drug_shape if candidate else None),
            _SHAPE_MAP,
        ),
        color=_map_code(
            _first(vision.color_class1 if vision else None, candidate.color_class1 if candidate else None),
            _COLOR_MAP,
        ),
        score_line=_has_score_line(vision, candidate),
    )


def official_from_detail(detail: DrugDetailApiItem) -> Official:
    """e약은요 상세정보(DrugDetailApiItem) -> 응답용 Official.

    app.symptom.service 에서도 그대로 재사용한다(사진 스캔 플로우와 증상 기반
    추천 플로우가 같은 '공식 데이터만 노출' 규칙을 공유해야 하므로, 공개
    함수로 분리해뒀다).
    """
    return Official(
        item_seq=detail.item_seq,
        item_name=detail.item_name,
        efficacy=detail.efficacy,
        use_method=detail.usage_method,
        warning=detail.warning,
        caution=detail.precautions,
        interaction=detail.interactions,
        side_effect=detail.side_effects,
        storage=detail.storage_method,
        image_url=detail.item_image,
    )


def _recommendation(
    ok: bool,
    identification: Identification,
    detail: Optional[DrugDetailApiItem],
    symptoms: list[str],
    *,
    gender: Optional[str] = None,
    birth_date: Optional[date] = None,
    symptom_started_at: Optional[datetime] = None,
) -> Optional[Recommendation]:
    if not ok or not symptoms or detail is None or not detail.efficacy:
        return None
    if not gender and birth_date is None and symptom_started_at is None:
        display_symptoms = [SYMPTOM_TYPE_MAP.get(s, s) for s in symptoms]
        matched = any(symptom and symptom in detail.efficacy for symptom in display_symptoms)
        if not matched:
            return Recommendation(
                status="NOT_RECOMMENDED",
                score=round(identification.score * 0.4, 2),
                confidence="LOW",
                reason="현재 입력한 증상과 해당 의약품의 효능이 일치하지 않습니다.",
                caution=detail.precautions,
            )
        return Recommendation(
            status="RECOMMENDED",
            score=round(identification.score * 0.94, 2),
            confidence=identification.confidence,
            reason="현재 입력한 증상과 해당 의약품의 효능이 일치합니다.",
            caution=detail.precautions,
        )
    draft = build_recommendation(
        identification=identification,
        detail=detail,
        symptoms=symptoms,
        gender=gender,
        birth_date=birth_date,
        symptom_started_at=symptom_started_at,
    )
    return Recommendation(
        status=draft.status,
        score=draft.score,
        confidence=draft.confidence,
        reason=draft.reason,
        caution=draft.caution,
    )


def _document(
    candidate: Optional[PillCandidate],
    official: Optional[Official],
    recommendation: Optional[Recommendation] = None,
) -> Optional[str]:
    name = (official.item_name if official and official.item_name else None) or (
        candidate.item_name if candidate else None
    )
    parts: list[str] = []
    if name:
        parts.append(name)
    if recommendation and recommendation.reason:
        parts.append(f"개인판단: {recommendation.reason}")
    if official:
        for label, value in (
            ("효능", official.efficacy),
            ("용법", official.use_method),
            ("경고", official.warning),
            ("주의", official.caution),
            ("상호작용", official.interaction),
            ("부작용", official.side_effect),
            ("보관", official.storage),
        ):
            if value:
                parts.append(f"{label}: {value}")
    return "\n".join(parts) if parts else None


def _has_score_line(vision: Optional[VisionExtractionResult], candidate: Optional[PillCandidate]) -> bool:
    if vision is not None and vision.score_line is not None:
        return vision.score_line
    if candidate is None:
        return False
    return any(_is_present_line(value) for value in (candidate.line_front, candidate.line_back))


def _is_present_line(value: Optional[str]) -> bool:
    if value is None:
        return False
    stripped = value.strip()
    return stripped not in {"", "없음"}


def _map_code(value: Optional[str], mapping: dict[str, str]) -> Optional[str]:
    if not value:
        return None
    return mapping.get(value, value)


def _first(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return value
    return None
