"""e약은요 API를 mock으로 대체해 DrugDetailService(Step 4) 로직을 검증한다."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.drug_detail_models import (
    DrugDetailApiBody,
    DrugDetailApiHeader,
    DrugDetailApiItem,
    DrugDetailApiResponse,
)
from app.drug_detail_service import DrugDetailService


@pytest.mark.asyncio
async def test_상세정보가_있으면_첫번째_항목을_반환():
    client = AsyncMock()
    client.get_by_item_seq.return_value = DrugDetailApiResponse(
        header=DrugDetailApiHeader(result_code="00", result_msg="NORMAL SERVICE."),
        body=DrugDetailApiBody(
            items=[
                DrugDetailApiItem(
                    item_seq="199900001",
                    item_name="타이레놀정500mg",
                    efficacy="감기로 인한 발열 및 동통, 두통, 신경통 완화",
                    usage_method="성인 1회 1~2정, 1일 3~4회 복용",
                    storage_method="밀폐용기, 실온보관",
                )
            ],
            total_count=1,
        ),
    )

    service = DrugDetailService(client)
    detail = await service.get_detail("199900001")

    assert detail is not None
    assert detail.item_name == "타이레놀정500mg"
    assert detail.efficacy is not None
    client.get_by_item_seq.assert_awaited_once_with("199900001")


@pytest.mark.asyncio
async def test_상세정보가_없으면_None을_반환():
    client = AsyncMock()
    client.get_by_item_seq.return_value = DrugDetailApiResponse(
        header=DrugDetailApiHeader(result_code="00", result_msg="NORMAL SERVICE."),
        body=DrugDetailApiBody(items=[], total_count=0),
    )

    service = DrugDetailService(client)
    detail = await service.get_detail("000000000")

    assert detail is None
