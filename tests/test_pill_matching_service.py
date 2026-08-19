"""
실제 Vision AI / 실제 낱알식별 API 없이도 매칭 로직(레벨 완화 전략)을 검증하기 위한 테스트.

아직 이미지 파이프라인이 없는 지금 단계에서, "Vision AI가 이런 값을 줬다고 가정하면
낱알식별 API가 이런 결과를 줬을 때 우리 서비스가 어떻게 판단하는가"를 mock으로 재현한다.

PillIdentificationApiClient.search() 를 AsyncMock으로 대체해서 실제 네트워크 호출 없이
"이 레벨의 파라미터로 부르면 N건이 나온다"는 시나리오만 주입한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.models import (
    MatchStatus,
    PillApiBody,
    PillApiHeader,
    PillApiItem,
    PillApiResponse,
    VisionExtractionResult,
)
from app.pill_matching_service import PillMatchingService


def fake_response(total_count: int, *item_seq_name_pairs: str) -> PillApiResponse:
    items = []
    for i in range(0, len(item_seq_name_pairs) - 1, 2):
        items.append(
            PillApiItem(
                item_seq=item_seq_name_pairs[i],
                item_name=item_seq_name_pairs[i + 1],
                entp_name="제조사",
                chart="흰색 장방형 정제",
            )
        )
    return PillApiResponse(
        header=PillApiHeader(result_code="00", result_msg="NORMAL SERVICE."),
        body=PillApiBody(items=items, total_count=total_count, page_no=1, num_of_rows=50),
    )


@pytest.mark.asyncio
async def test_앞뒤_각인_모두_신뢰도_높고_1건이면_STRICT_WITH_PRINT_BOTH_레벨에서_SINGLE_MATCH():
    client = AsyncMock()

    async def search_side_effect(params: dict[str, str]) -> PillApiResponse:
        if "PRINT_FRONT" in params and "PRINT_BACK" in params:
            return fake_response(1, "199900001", "타이레놀정500mg")
        return fake_response(0)

    client.search.side_effect = search_side_effect

    service = PillMatchingService(client)
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

    result = await service.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "STRICT_WITH_PRINT_BOTH"
    assert result.candidates[0].item_seq == "199900001"


@pytest.mark.asyncio
async def test_앞면만_신뢰도_높으면_앞면_각인만으로_STRICT_WITH_PRINT_FRONT_ONLY_시도():
    client = AsyncMock()

    async def search_side_effect(params: dict[str, str]) -> PillApiResponse:
        # 뒷면은 신뢰도가 낮으므로 PRINT_BACK 조건이 절대 들어가면 안 됨
        assert "PRINT_BACK" not in params
        if "PRINT_FRONT" in params:
            return fake_response(1, "1", "약A")
        return fake_response(0)

    client.search.side_effect = search_side_effect

    service = PillMatchingService(client)
    vision = VisionExtractionResult(
        print_front="TY",
        print_front_confidence=0.9,
        print_back="흐릿함",
        print_back_confidence=0.3,  # 임계값(0.6) 미만 -> 뒷면은 신뢰 불가
        color_class1="하양",
        drug_shape="장방형",
        form_code_name="정제",
        overall_confidence=0.7,
    )

    result = await service.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "STRICT_WITH_PRINT_FRONT_ONLY"


@pytest.mark.asyncio
async def test_둘다_신뢰도_높아도_BOTH가_0건이면_FRONT_ONLY로_구제된다():
    """앞면 각인은 실제로 맞았는데 뒷면 각인 하나가 틀려서 BOTH 레벨이 0건인 경우,
    각인 정보를 통째로 포기하지 않고 앞면만으로 다시 시도해서 구제할 수 있어야 한다."""
    client = AsyncMock()

    async def search_side_effect(params: dict[str, str]) -> PillApiResponse:
        if "PRINT_FRONT" in params and "PRINT_BACK" in params:
            return fake_response(0)  # 뒷면 각인이 미묘하게 틀려서 매칭 실패
        if "PRINT_FRONT" in params:  # front-only 레벨
            return fake_response(1, "199900001", "타이레놀정500mg")
        return fake_response(0)

    client.search.side_effect = search_side_effect

    service = PillMatchingService(client)
    vision = VisionExtractionResult(
        print_front="TY",
        print_front_confidence=0.9,
        print_back="5O0",  # 실제로는 "500"인데 오독됨 (신뢰도 자체는 높게 나왔다고 가정)
        print_back_confidence=0.75,
        color_class1="하양",
        drug_shape="장방형",
        form_code_name="정제",
        overall_confidence=0.85,
    )

    result = await service.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "STRICT_WITH_PRINT_FRONT_ONLY"


@pytest.mark.asyncio
async def test_앞뒤_모두_각인_신뢰도가_낮으면_각인_없이_바로_SHAPE_AND_COLOR_레벨로_건너뛴다():
    client = AsyncMock()

    async def search_side_effect(params: dict[str, str]) -> PillApiResponse:
        assert "PRINT_FRONT" not in params
        assert "PRINT_BACK" not in params
        if "DRUG_SHAPE" in params:
            return fake_response(2, "1", "약A", "2", "약B")
        return fake_response(0)

    client.search.side_effect = search_side_effect

    service = PillMatchingService(client)
    vision = VisionExtractionResult(
        print_front="흐릿함",
        print_front_confidence=0.2,  # 임계값(0.6) 미만
        print_back="흐릿함2",
        print_back_confidence=0.1,  # 임계값(0.6) 미만
        color_class1="노랑",
        drug_shape="원형",
        form_code_name="정제",
        overall_confidence=0.4,
    )

    result = await service.match(vision)

    assert result.status == MatchStatus.MULTIPLE_MATCHES
    assert result.query_level_used == "SHAPE_AND_COLOR"
    assert len(result.candidates) == 2


@pytest.mark.asyncio
async def test_모든_레벨에서_0건이면_NO_MATCH():
    client = AsyncMock()
    client.search.return_value = fake_response(0)

    service = PillMatchingService(client)
    vision = VisionExtractionResult(
        color_class1="회색",
        drug_shape="기타",
        form_code_name="정제",
        overall_confidence=0.4,
    )

    result = await service.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.candidates == []


@pytest.mark.asyncio
async def test_마지막_레벨_COLOR_ONLY_에서도_후보가_너무_많으면_TOO_MANY_CANDIDATES():
    client = AsyncMock()

    async def search_side_effect(params: dict[str, str]) -> PillApiResponse:
        if set(params.keys()) == {"COLOR_CLASS1"}:
            return fake_response(37)  # MAX_CANDIDATES(10) 초과
        return fake_response(0)

    client.search.side_effect = search_side_effect

    service = PillMatchingService(client)
    # 각인/모양 정보가 없어서 레벨 대부분이 사실상 color_class1 하나만 남는 극단적인 케이스
    vision = VisionExtractionResult(
        color_class1="하양",
        form_code_name="정제",
        overall_confidence=0.5,
    )

    result = await service.match(vision)

    assert result.status == MatchStatus.TOO_MANY_CANDIDATES
