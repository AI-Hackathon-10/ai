"""
실제 Vision AI / 실제 낱알식별 API 없이도 매칭 로직(레벨 완화 전략)을 검증한다.

식약처 v03 요청변수에는 각인/색/모양이 없다. 매칭은 카탈로그(응답 목록)를
가져온 뒤 PRINT_FRONT 등 응답 필드로 로컬 필터한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.models import MatchStatus, PillApiItem, VisionExtractionResult
from app.pill_matching_service import PillMatchingService


def item(
    item_seq: str,
    item_name: str,
    *,
    print_front: str | None = None,
    print_back: str | None = None,
    drug_shape: str | None = None,
    color_class1: str | None = None,
    color_class2: str | None = None,
) -> PillApiItem:
    return PillApiItem(
        item_seq=item_seq,
        item_name=item_name,
        entp_name="제조사",
        chart="흰색 장방형 정제",
        print_front=print_front,
        print_back=print_back,
        drug_shape=drug_shape,
        color_class1=color_class1,
        color_class2=color_class2,
    )


def service_with_catalog(*items: PillApiItem) -> PillMatchingService:
    client = AsyncMock()
    client.get_catalog.return_value = list(items)
    return PillMatchingService(client)


@pytest.mark.asyncio
async def test_앞뒤_각인_모두_신뢰도_높고_1건이면_STRICT_WITH_PRINT_BOTH_레벨에서_SINGLE_MATCH():
    matching = service_with_catalog(
        item(
            "199900001",
            "타이레놀정500mg",
            print_front="TY",
            print_back="500",
            drug_shape="장방형",
            color_class1="하양",
        ),
        item("2", "다른약", print_front="AB", print_back="12", drug_shape="원형", color_class1="노랑"),
    )
    vision = VisionExtractionResult(
        print_front="TY",
        print_front_confidence=0.85,
        print_back="500",
        print_back_confidence=0.8,
        color_class1="하양",
        drug_shape="장방형",
        form_code_name="정제",
        overall_confidence=0.9,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "STRICT_WITH_PRINT_BOTH"
    assert result.candidates[0].item_seq == "199900001"


@pytest.mark.asyncio
async def test_앞면만_신뢰도_높으면_앞면_각인만으로_STRICT_WITH_PRINT_FRONT_ONLY_시도():
    matching = service_with_catalog(
        item("1", "약A", print_front="TY", print_back="XXX", drug_shape="장방형", color_class1="하양"),
    )
    vision = VisionExtractionResult(
        print_front="TY",
        print_front_confidence=0.9,
        print_back="흐릿함",
        print_back_confidence=0.3,
        color_class1="하양",
        drug_shape="장방형",
        form_code_name="정제",
        overall_confidence=0.7,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "STRICT_WITH_PRINT_FRONT_ONLY"


@pytest.mark.asyncio
async def test_둘다_신뢰도_높아도_BOTH가_0건이면_FRONT_ONLY로_구제된다():
    matching = service_with_catalog(
        item(
            "199900001",
            "타이레놀정500mg",
            print_front="TY",
            print_back="500",
            drug_shape="장방형",
            color_class1="하양",
        ),
    )
    vision = VisionExtractionResult(
        print_front="TY",
        print_front_confidence=0.9,
        print_back="5O0",
        print_back_confidence=0.75,
        color_class1="하양",
        drug_shape="장방형",
        form_code_name="정제",
        overall_confidence=0.85,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "STRICT_WITH_PRINT_FRONT_ONLY"


@pytest.mark.asyncio
async def test_앞뒤_모두_각인_신뢰도가_낮으면_각인_없이_바로_SHAPE_AND_COLOR_레벨로_건너뛴다():
    matching = service_with_catalog(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="노랑"),
        item("2", "약B", print_front="BB", drug_shape="원형", color_class1="노랑"),
    )
    vision = VisionExtractionResult(
        print_front="흐릿함",
        print_front_confidence=0.2,
        print_back="흐릿함2",
        print_back_confidence=0.1,
        color_class1="노랑",
        drug_shape="원형",
        form_code_name="정제",
        overall_confidence=0.4,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.MULTIPLE_MATCHES
    assert result.query_level_used == "SHAPE_AND_COLOR"
    assert len(result.candidates) == 2


@pytest.mark.asyncio
async def test_모든_레벨에서_0건이면_NO_MATCH():
    matching = service_with_catalog(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(
        color_class1="회색",
        drug_shape="기타",
        form_code_name="정제",
        overall_confidence=0.4,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.candidates == []


@pytest.mark.asyncio
async def test_마지막_레벨_COLOR_ONLY_에서도_후보가_너무_많으면_TOO_MANY_CANDIDATES():
    matching = service_with_catalog(
        *[item(str(i), f"약{i}", color_class1="하양") for i in range(37)]
    )
    vision = VisionExtractionResult(
        color_class1="하양",
        form_code_name="정제",
        overall_confidence=0.5,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.TOO_MANY_CANDIDATES
