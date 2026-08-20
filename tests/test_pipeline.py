"""
Step 1~4 전체 파이프라인 오케스트레이션(app/pipeline.py)을 mock으로 검증한다.
Vision AI, 낱알식별 매칭, e약은요 조회를 모두 가짜로 대체해서 실제 이미지·API 키
없이도 "SINGLE_MATCH일 때만 Step 4가 자동 실행되는가"를 확인한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.drug_detail_models import DrugDetailApiItem
from app.models import MatchStatus, PillCandidate, PillMatchResult, VisionExtractionResult
from app.pipeline import identify_from_db, identify_pill


@pytest.mark.asyncio
async def test_SINGLE_MATCH이면_상세정보까지_자동으로_조회한다(monkeypatch):
    vision_result = VisionExtractionResult(color_class1="하양", drug_shape="장방형")

    async def fake_run_vision_extraction(front, back, **kwargs):
        return vision_result

    monkeypatch.setattr("app.pipeline.run_vision_extraction", fake_run_vision_extraction)

    matching_service = AsyncMock()
    matching_service.match.return_value = PillMatchResult(
        status=MatchStatus.SINGLE_MATCH,
        candidates=[PillCandidate(item_seq="199900001", item_name="타이레놀정500mg")],
        query_level_used="STRICT_WITH_PRINT_BOTH",
    )

    detail_service = AsyncMock()
    detail_service.get_detail.return_value = DrugDetailApiItem(
        item_seq="199900001", item_name="타이레놀정500mg"
    )

    outcome = await identify_pill(b"front", b"back", matching_service, detail_service)

    assert outcome.vision_failed is False
    assert outcome.match_result.status == MatchStatus.SINGLE_MATCH
    assert outcome.detail is not None
    detail_service.get_detail.assert_awaited_once_with("199900001")


@pytest.mark.asyncio
async def test_MULTIPLE_MATCHES면_상세정보를_자동_조회하지_않는다(monkeypatch):
    vision_result = VisionExtractionResult(color_class1="하양", drug_shape="원형")

    async def fake_run_vision_extraction(front, back, **kwargs):
        return vision_result

    monkeypatch.setattr("app.pipeline.run_vision_extraction", fake_run_vision_extraction)

    matching_service = AsyncMock()
    matching_service.match.return_value = PillMatchResult(
        status=MatchStatus.MULTIPLE_MATCHES,
        candidates=[
            PillCandidate(item_seq="1", item_name="약A"),
            PillCandidate(item_seq="2", item_name="약B"),
        ],
        query_level_used="SHAPE_AND_COLOR",
    )

    detail_service = AsyncMock()

    outcome = await identify_pill(b"front", b"back", matching_service, detail_service)

    assert outcome.match_result.status == MatchStatus.MULTIPLE_MATCHES
    assert outcome.detail is None
    detail_service.get_detail.assert_not_awaited()


@pytest.mark.asyncio
async def test_Vision_추출_실패하면_매칭_자체를_시도하지_않는다(monkeypatch):
    async def fake_run_vision_extraction(front, back, **kwargs):
        return None  # 재시도를 다 써도 색상/모양을 못 뽑은 경우

    monkeypatch.setattr("app.pipeline.run_vision_extraction", fake_run_vision_extraction)

    matching_service = AsyncMock()
    detail_service = AsyncMock()

    outcome = await identify_pill(b"front", b"back", matching_service, detail_service)

    assert outcome.vision_failed is True
    assert outcome.match_result is None
    matching_service.match.assert_not_awaited()
    detail_service.get_detail.assert_not_awaited()


@pytest.mark.asyncio
async def test_identify_from_db는_vision_결과로_테이블_매칭만_한다(monkeypatch):
    vision_result = VisionExtractionResult(
        print_front="YH",
        print_front_confidence=0.9,
        color_class1="노랑",
        drug_shape="원형",
    )

    async def fake_run_vision_extraction(front, back, **kwargs):
        return vision_result

    monkeypatch.setattr("app.pipeline.run_vision_extraction", fake_run_vision_extraction)

    matching_service = AsyncMock()
    matching_service.match.return_value = PillMatchResult(
        status=MatchStatus.SINGLE_MATCH,
        candidates=[PillCandidate(item_seq="200808877", item_name="페라트라정", print_front="YH")],
        query_level_used="STRICT_WITH_PRINT_FRONT_ONLY",
    )

    outcome = await identify_from_db(b"front", b"back", matching_service)

    assert outcome.vision_failed is False
    assert outcome.vision_result is vision_result
    assert outcome.match_result.status == MatchStatus.SINGLE_MATCH
    matching_service.match.assert_awaited_once_with(vision_result)
