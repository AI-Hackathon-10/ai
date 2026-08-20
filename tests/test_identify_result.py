"""판별 결과를 백엔드가 받는 camelCase result 리스트 항목으로 변환하는 로직을 검증한다."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.drug_detail_models import DrugDetailApiItem
from app.identify_result import build_identify_result
from app.models import MatchStatus, PillCandidate, PillMatchResult, VisionExtractionResult
from app.pipeline import IdentifyFromDbOutcome


def _tylenol_outcome() -> IdentifyFromDbOutcome:
    return IdentifyFromDbOutcome(
        vision_failed=False,
        vision_result=VisionExtractionResult(
            print_front="TYLENOL",
            print_back="500",
            color_class1="하양",
            drug_shape="장방형",
            score_line=True,
        ),
        match_result=PillMatchResult(
            status=MatchStatus.SINGLE_MATCH,
            candidates=[
                PillCandidate(
                    item_seq="199703222",
                    item_name="타이레놀정500밀리그람",
                    item_image="https://nedrug.mfds.go.kr/pbp/cmn/itemImage.do?itemSeq=199703222",
                    print_front="TYLENOL",
                    print_back="500",
                    drug_shape="장방형",
                    color_class1="하양",
                )
            ],
            query_level_used="DIRECT_MATCH",
        ),
    )


def _tylenol_detail() -> DrugDetailApiItem:
    return DrugDetailApiItem(
        item_seq="199703222",
        item_name="타이레놀정500밀리그람",
        item_image="https://nedrug.mfds.go.kr/pbp/cmn/itemImage.do?itemSeq=199703222",
        efficacy="감기로 인한 발열 및 동통, 두통, 신경통 완화",
        usage_method="성인 1회 1~2정",
        warning="매일 세잔 이상 술을 마시는 사람은 복용 전 의사와 상의하십시오.",
        precautions="다른 아세트아미노펜 함유 의약품과 함께 복용하지 마십시오.",
        interactions="다른 해열진통제와 함께 복용하지 마십시오.",
        side_effects="발진, 가려움",
        storage_method="실온보관",
    )


@pytest.mark.asyncio
async def test_단일_매칭이면_ok_True와_camelCase_필드를_채운다():
    item = await build_identify_result(
        id="1",
        outcome=_tylenol_outcome(),
        detail=_tylenol_detail(),
        symptoms=["HEADACHE", "FEVER"],
    )
    payload = item.model_dump(by_alias=True)

    assert payload["id"] == "1"
    assert payload["ok"] is True
    assert payload["matchStatus"] == "SINGLE_MATCH"
    assert payload["visionResult"] is not None
    assert payload["visionResult"]["print_front"] == "TYLENOL"
    assert payload["candidates"] is None
    assert payload["itemSeq"] == "199703222"
    assert payload["itemName"] == "타이레놀정500밀리그람"
    assert payload["imageUrl"].startswith("https://nedrug.mfds.go.kr/")
    assert payload["identification"] == {"confidence": "HIGH", "score": 1.0}
    assert payload["recommendation"]["status"] == "RECOMMENDED"
    assert payload["recommendation"]["confidence"] == "HIGH"
    assert "증상" in payload["recommendation"]["reason"]
    assert "아세트아미노펜" in payload["recommendation"]["caution"]
    assert payload["features"] == {
        "frontImprint": "TYLENOL",
        "backImprint": "500",
        "shape": "OBLONG",
        "color": "WHITE",
        "scoreLine": True,
    }
    assert payload["official"]["itemSeq"] == "199703222"
    assert payload["official"]["efficacy"].startswith("감기로 인한")
    assert payload["official"]["useMethod"] == "성인 1회 1~2정"
    assert payload["official"]["caution"].startswith("다른 아세트아미노펜")
    assert payload["official"]["interaction"].startswith("다른 해열진통제")
    assert payload["official"]["sideEffect"] == "발진, 가려움"
    assert payload["official"]["storage"] == "실온보관"
    assert "타이레놀정500밀리그람" in payload["document"]


@pytest.mark.asyncio
async def test_Vision_실패면_ok_False이고_식별_필드는_비운다():
    item = await build_identify_result(
        id="2",
        outcome=IdentifyFromDbOutcome(vision_failed=True),
    )
    payload = item.model_dump(by_alias=True)

    assert payload["id"] == "2"
    assert payload["ok"] is False
    assert payload["matchStatus"] is None
    assert payload["visionResult"] is None
    assert payload["candidates"] is None
    assert payload["itemSeq"] is None
    assert payload["itemName"] is None
    assert payload["imageUrl"] is None
    assert payload["identification"] == {"confidence": "LOW", "score": 0.0}
    assert payload["recommendation"] is None
    assert payload["official"] is None
    assert payload["document"] is None


@pytest.mark.asyncio
async def test_증상이_효능과_안_맞으면_NOT_RECOMMENDED다():
    item = await build_identify_result(
        id="1",
        outcome=_tylenol_outcome(),
        detail=_tylenol_detail(),
        symptoms=["DIARRHEA"],
    )
    payload = item.model_dump(by_alias=True)

    assert payload["ok"] is True
    assert payload["matchStatus"] == "SINGLE_MATCH"
    assert payload["recommendation"]["status"] == "NOT_RECOMMENDED"
    assert "일치하지 않습니다" in payload["recommendation"]["reason"]


@pytest.mark.asyncio
async def test_복수_후보면_확정하지_않고_ok_False다():
    outcome = IdentifyFromDbOutcome(
        vision_failed=False,
        vision_result=VisionExtractionResult(color_class1="하양", drug_shape="원형"),
        match_result=PillMatchResult(
            status=MatchStatus.MULTIPLE_MATCHES,
            candidates=[
                PillCandidate(item_seq="1", item_name="약A"),
                PillCandidate(item_seq="2", item_name="약B"),
            ],
            query_level_used="SHAPE_AND_COLOR",
        ),
    )
    payload = (await build_identify_result(id="3", outcome=outcome)).model_dump(by_alias=True)

    assert payload["ok"] is False
    assert payload["matchStatus"] == "MULTIPLE_MATCHES"
    assert payload["visionResult"] is not None
    assert payload["candidates"] is not None
    assert len(payload["candidates"]) == 2
    assert payload["itemSeq"] is None
    assert payload["features"]["shape"] == "ROUND"
    assert payload["features"]["color"] == "WHITE"
    assert payload["features"]["scoreLine"] is False


@pytest.mark.asyncio
async def test_NO_MATCH이고_완화후보가_있으면_candidates를_반환한다():
    outcome = IdentifyFromDbOutcome(
        vision_failed=False,
        vision_result=VisionExtractionResult(
            print_front="SJ",
            print_back="V 8",
            color_class1="주황",
            drug_shape="원형",
            score_line=True,
        ),
        match_result=PillMatchResult(
            status=MatchStatus.NO_MATCH,
            candidates=[
                PillCandidate(
                    item_seq="200810473",
                    item_name="엑스반정80밀리그램(발사르탄)",
                    similarity_score=0.775,
                ),
            ],
            query_level_used="RELAXED_MATCH",
        ),
    )
    payload = (await build_identify_result(id="4", outcome=outcome)).model_dump(by_alias=True)

    assert payload["ok"] is False
    assert payload["matchStatus"] == "NO_MATCH"
    assert payload["candidates"] is not None
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["item_seq"] == "200810473"
    assert payload["candidates"][0]["similarity_score"] == pytest.approx(0.775)
    assert payload["itemSeq"] is None


@pytest.mark.asyncio
async def test_NO_MATCH이고_완화후보가_없으면_candidates는_None이다():
    outcome = IdentifyFromDbOutcome(
        vision_failed=False,
        vision_result=VisionExtractionResult(color_class1="회색", drug_shape="기타"),
        match_result=PillMatchResult(
            status=MatchStatus.NO_MATCH,
            candidates=[],
            query_level_used="DIRECT_MATCH",
        ),
    )
    payload = (await build_identify_result(id="5", outcome=outcome)).model_dump(by_alias=True)

    assert payload["ok"] is False
    assert payload["matchStatus"] == "NO_MATCH"
    assert payload["candidates"] is None
