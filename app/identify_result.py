"""판별 파이프라인 결과를 백엔드가 받는 최종 result 항목으로 변환한다."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.drug_detail_models import DrugDetailApiItem
from app.models import MatchStatus, PillCandidate, VisionExtractionResult
from app.pipeline import IdentifyFromDbOutcome

# SymptomType enum → 한글 displayName 매핑 (백엔드 SymptomType.java 와 동기화)
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
) -> IdentifyResultItem:
    vision = outcome.vision_result
    candidate = _confirmed_candidate(outcome)
    ok = candidate is not None
    identification = _identification(ok, vision)
    official = _official(detail) if detail else None

    return IdentifyResultItem(
        id=id,
        ok=ok,
        item_seq=candidate.item_seq if candidate else None,
        item_name=candidate.item_name if candidate else None,
        image_url=_image_url(candidate, detail),
        identification=identification,
        recommendation=_recommendation(ok, identification, detail, symptoms or []),
        features=_features(vision, candidate),
        official=official,
        document=_document(candidate, official) if ok else None,
    )


def _confirmed_candidate(outcome: IdentifyFromDbOutcome) -> Optional[PillCandidate]:
    match = outcome.match_result
    if outcome.vision_failed or match is None:
        return None
    if match.status != MatchStatus.SINGLE_MATCH or not match.candidates:
        return None
    return match.candidates[0]


def _identification(ok: bool, vision: Optional[VisionExtractionResult]) -> Identification:
    if not ok:
        return Identification(confidence="LOW", score=0.0)
    score = vision.overall_confidence if vision and vision.overall_confidence is not None else 1.0
    return Identification(confidence=_confidence_label(score), score=score)


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    return "LOW"


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


def _official(detail: DrugDetailApiItem) -> Official:
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
) -> Optional[Recommendation]:
    if not ok or not symptoms or detail is None or not detail.efficacy:
        return None
    # enum 값(HEADACHE)이면 한글로 변환, 이미 한글이면 그대로 사용
    display_symptoms = [SYMPTOM_TYPE_MAP.get(s, s) for s in symptoms]
    matched = any(symptom and symptom in detail.efficacy for symptom in display_symptoms)
    if not matched:
        return Recommendation(
            status="NOT_RECOMMENDED",
            score=round(identification.score * 0.5, 2),
            confidence=identification.confidence,
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


def _document(candidate: Optional[PillCandidate], official: Optional[Official]) -> Optional[str]:
    name = (official.item_name if official and official.item_name else None) or (
        candidate.item_name if candidate else None
    )
    parts: list[str] = []
    if name:
        parts.append(name)
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
    values = []
    if vision:
        values.extend([vision.line_front, vision.line_back])
    if candidate:
        values.extend([candidate.line_front, candidate.line_back])
    return any(_is_present_line(value) for value in values)


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
