"""사진 없이 증상 음성 입력만으로 OTC 후보를 추천하는 오케스트레이션을 검증한다.

실제 Gemini/e약은요 API는 호출하지 않는다 — app.symptom.types의 Protocol
시그니처(ExtractSymptomsFn/SuggestCandidateDrugsFn)만 지키는 가짜 함수와, 가짜
DrugDetailService를 주입해서 순수 로직(공식 DB 검증, 연령/효능 필터링)만
검증한다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.drug_detail_models import DrugDetailApiItem
from app.symptom.service import recommend_drugs_from_voice
from app.symptom.types import SymptomExtraction

_KST = timezone(timedelta(hours=9))
_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=_KST)

_TYLENOL_DETAIL = DrugDetailApiItem(
    itemSeq="202106092",
    itemName="타이레놀정500밀리그람(아세트아미노펜)",
    efcyQesitm="이 약은 감기로 인한 발열 및 동통(통증), 두통, 신경통, 근육통, 월경통에 사용합니다",
    useMethodQesitm="만 12세 이상 소아 및 성인은 1회 1~2정씩 복용합니다.",
    atpnQesitm="임부 또는 수유부, 고령자(노인)는 의사 또는 약사와 상의하십시오.",
    itemImage="http://example.com/tylenol.jpg",
)

_ADULT_ONLY_DETAIL = DrugDetailApiItem(
    itemSeq="999",
    itemName="성인용해열진통제",
    efcyQesitm="이 약은 발열, 두통에 사용합니다",
    useMethodQesitm="만 12세 이상 소아 및 성인만 복용하십시오.",
    atpnQesitm="12세 미만의 소아는 이 약을 복용하지 마십시오.",
)


class _FakeDetailService:
    """item_name -> DrugDetailApiItem 매핑을 흉내내는 가짜 e약은요 서비스."""

    def __init__(self, catalog: dict[str, Optional[DrugDetailApiItem]]):
        self._catalog = catalog
        self.queried_names: list[str] = []

    async def get_detail_by_name(self, item_name: str):
        self.queried_names.append(item_name)
        return self._catalog.get(item_name)


def _extract_fn(symptoms: list[str], onset_at: Optional[datetime] = None, onset_confidence: str = "HIGH"):
    async def _fn(*, symptom_text: str, onset_text: str, reference_time: datetime):
        return SymptomExtraction(symptoms=symptoms, onset_at=onset_at, onset_confidence=onset_confidence)

    return _fn


def _suggest_fn(names: list[str]):
    async def _fn(*, symptoms: list[str], gender: Optional[str], birth_date):
        return names

    return _fn


async def test_증상을_하나도_못_읽으면_후보_조회_없이_안내만_반환한다():
    detail_service = _FakeDetailService({})

    result = await recommend_drugs_from_voice(
        symptom_text="그냥 좀 힘들어요",
        onset_text="",
        gender="FEMALE",
        birth_date=date(1995, 1, 1),
        detail_service=detail_service,
        extract_fn=_extract_fn(symptoms=[]),
        suggest_fn=_suggest_fn(["아무약"]),
        reference_time=_NOW,
    )

    assert result.candidates == []
    assert result.notice is not None
    assert detail_service.queried_names == []  # 증상이 없으니 약 후보 조회 자체를 하지 않는다


async def test_공식_DB에_없는_후보는_버리고_실존_확인된_후보만_남긴다():
    detail_service = _FakeDetailService({"타이레놀정500밀리그람(아세트아미노펜)": _TYLENOL_DETAIL})

    result = await recommend_drugs_from_voice(
        symptom_text="머리가 아프고 열이 나요",
        onset_text="3시간 전부터요",
        gender="MALE",
        birth_date=date(1995, 12, 30),
        detail_service=detail_service,
        extract_fn=_extract_fn(symptoms=["HEADACHE", "FEVER"], onset_at=_NOW - timedelta(hours=3)),
        suggest_fn=_suggest_fn(["타이레놀정500밀리그람(아세트아미노펜)", "존재하지않는가짜약"]),
        reference_time=_NOW,
    )

    assert result.notice is None
    assert len(result.candidates) == 1
    assert result.candidates[0].item_name == "타이레놀정500밀리그람(아세트아미노펜)"
    assert result.candidates[0].recommendation.status == "RECOMMENDED"
    # 사진 스캔 확정(score=1.0)이 아니라 AI 제안(identification.score=0.7)에서
    # 출발하므로, 최종 점수도 그만큼 낮게 시작한다 (0.7 * 0.94 = 0.658 -> 0.66).
    assert result.candidates[0].recommendation.score == 0.66
    assert "존재하지않는가짜약" not in [c.item_name for c in result.candidates]


async def test_연령_금기에_걸리는_후보는_추천_목록에서_제외된다():
    # build_recommendation()은 "N세 미만 복용 금지" 같은 최소 연령 금기만
    # 인식한다(recommendation.py의 _MIN_AGE_UNDER/_MIN_AGE_OVER 참고). 사용자가
    # 그 최소 연령에 못 미치면 NOT_RECOMMENDED로 걸러져야 한다.
    detail_service = _FakeDetailService({"성인용해열진통제": _ADULT_ONLY_DETAIL})

    result = await recommend_drugs_from_voice(
        symptom_text="머리가 아프고 열이 나요",
        onset_text="",
        gender="MALE",
        birth_date=date(2022, 1, 1),  # 만 4세 -> "12세 미만 복용 금지"에 걸림
        detail_service=detail_service,
        extract_fn=_extract_fn(symptoms=["HEADACHE", "FEVER"]),
        suggest_fn=_suggest_fn(["성인용해열진통제"]),
        reference_time=_NOW,
    )

    assert result.candidates == []
    assert result.notice is not None


async def test_같은_item_seq는_중복_없이_한_번만_담는다():
    detail_service = _FakeDetailService(
        {"타이레놀정500밀리그람(아세트아미노펜)": _TYLENOL_DETAIL, "타이레놀": _TYLENOL_DETAIL}
    )

    result = await recommend_drugs_from_voice(
        symptom_text="머리가 아프고 열이 나요",
        onset_text="",
        gender="MALE",
        birth_date=date(1995, 12, 30),
        detail_service=detail_service,
        extract_fn=_extract_fn(symptoms=["HEADACHE", "FEVER"]),
        suggest_fn=_suggest_fn(["타이레놀정500밀리그람(아세트아미노펜)", "타이레놀"]),
        reference_time=_NOW,
    )

    assert len(result.candidates) == 1
