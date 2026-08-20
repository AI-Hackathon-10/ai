"""
실제 Vision AI / 실제 낱알식별 API 없이도 매칭 로직(직접 비교)을 검증한다.

식약처 v03 요청변수에는 각인/색/모양이 없다. 매칭은 카탈로그(응답 목록)를
가져온 뒤 PRINT_FRONT 등 응답 필드로 로컬 필터한다. 조건은 단계적으로 완화하지
않는다 — Vision이 추출한 값(None이 아닌 필드만)을 한 번에 합쳐서 비교한다.
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
    line_front: str | None = None,
    line_back: str | None = None,
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
        line_front=line_front,
        line_back=line_back,
    )


def service_with_catalog(*items: PillApiItem) -> PillMatchingService:
    client = AsyncMock()
    client.get_catalog.return_value = list(items)
    return PillMatchingService(client)


@pytest.mark.asyncio
async def test_각인_모양_색상_모두_일치하는_1건이면_SINGLE_MATCH():
    matching = service_with_catalog(
        item(
            "199900001", "타이레놀정500mg",
            print_front="TY", print_back="500", drug_shape="장방형", color_class1="하양",
        ),
        item("2", "다른약", print_front="AB", print_back="12", drug_shape="원형", color_class1="노랑"),
    )
    vision = VisionExtractionResult(
        print_front="TY", print_back="500", color_class1="하양", drug_shape="장방형",
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "DIRECT_MATCH"
    assert result.candidates[0].item_seq == "199900001"


@pytest.mark.asyncio
async def test_각인을_판단못해도_모양_색상만으로_매칭을_시도한다():
    matching = service_with_catalog(
        item("1", "약A", print_front="TY", drug_shape="장방형", color_class1="하양"),
    )
    # print_front/print_back 모두 None (Vision이 판단 못 함)
    vision = VisionExtractionResult(color_class1="하양", drug_shape="장방형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH


@pytest.mark.asyncio
async def test_각인이_다르면_매칭되지_않는다():
    matching = service_with_catalog(
        item("1", "약A", print_front="AB", drug_shape="장방형", color_class1="하양"),
    )
    vision = VisionExtractionResult(print_front="TY", color_class1="하양", drug_shape="장방형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.candidates == []


@pytest.mark.asyncio
async def test_모양_색상만_같아도_여러건이면_MULTIPLE_MATCHES():
    matching = service_with_catalog(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="노랑"),
        item("2", "약B", print_front="BB", drug_shape="원형", color_class1="노랑"),
    )
    vision = VisionExtractionResult(color_class1="노랑", drug_shape="원형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.MULTIPLE_MATCHES
    assert result.query_level_used == "DIRECT_MATCH"
    assert len(result.candidates) == 2


@pytest.mark.asyncio
async def test_후보가_없으면_NO_MATCH():
    matching = service_with_catalog(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="회색", drug_shape="기타")

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.candidates == []


@pytest.mark.asyncio
async def test_후보가_너무_많으면_TOO_MANY_CANDIDATES():
    matching = service_with_catalog(
        *[item(str(i), f"약{i}", color_class1="하양") for i in range(37)]
    )
    vision = VisionExtractionResult(color_class1="하양")

    result = await matching.match(vision)

    assert result.status == MatchStatus.TOO_MANY_CANDIDATES


@pytest.mark.asyncio
async def test_분할선_있음으로_판단되면_분할선_없는_후보는_제외된다():
    matching = service_with_catalog(
        item("1", "분할선있음", drug_shape="원형", color_class1="하양", line_front="-"),
        item("2", "분할선없음", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="원형", score_line=True)

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.candidates[0].item_seq == "1"


@pytest.mark.asyncio
async def test_분할선_없음으로_판단되면_분할선_있는_후보는_제외된다():
    matching = service_with_catalog(
        item("1", "분할선있음", drug_shape="원형", color_class1="하양", line_back="-"),
        item("2", "분할선없음", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="원형", score_line=False)

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.candidates[0].item_seq == "2"


@pytest.mark.asyncio
async def test_분할선을_판단못하면_필터에서_아예_빠진다():
    matching = service_with_catalog(
        item("1", "분할선있음", drug_shape="원형", color_class1="하양", line_front="-"),
        item("2", "분할선없음", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="원형", score_line=None)

    result = await matching.match(vision)

    assert result.status == MatchStatus.MULTIPLE_MATCHES
    assert len(result.candidates) == 2


class _FinderClient:
    def __init__(self, items: list[PillApiItem]):
        self.items = items
        self.calls: list[dict[str, str]] = []
        self.get_catalog_called = False

    async def find(self, filters: dict[str, str], limit: int = 11) -> list[PillApiItem]:
        self.calls.append(dict(filters))
        return list(self.items)

    async def get_catalog(self) -> list[PillApiItem]:
        self.get_catalog_called = True
        return list(self.items)


@pytest.mark.asyncio
async def test_find가_있으면_카탈로그_전체가_아니라_필터_조회를_쓴다():
    client = _FinderClient(
        [item("200808877", "페라트라정", print_front="YH", drug_shape="원형", color_class1="노랑")]
    )
    matching = PillMatchingService(client)
    vision = VisionExtractionResult(print_front="YH", color_class1="노랑", drug_shape="원형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert client.get_catalog_called is False
    assert client.calls
    assert client.calls[0]["PRINT_FRONT"] == "YH"
    assert client.calls[0]["DRUG_SHAPE"] == "원형"
