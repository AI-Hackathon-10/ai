"""
Step 4: 낱알식별 매칭에서 확정된 ITEM_SEQ로 e약은요 상세정보를 조회한다.

⚠️ 모든 의약품이 e약은요에 등록되어 있는 건 아니다(일반의약품 위주이고, 전문의약품
등은 없을 수 있다). 0건이면 그냥 None을 반환하고, 호출부는 "상세정보 없음 — 포장지
표시사항 및 전문가 확인 필요"로 안내해야 한다. 이 함수 자체는 그 안내 문구를 만들지
않는다 — 그건 API 응답을 사용자에게 보여주는 쪽(프론트/라우터)의 책임이다.
"""
from __future__ import annotations

from typing import Optional

from app.drug_detail_client import DrugDetailApiClient
from app.drug_detail_models import DrugDetailApiItem


class DrugDetailService:
    def __init__(self, api_client: DrugDetailApiClient):
        self._api_client = api_client

    async def get_detail(self, item_seq: str) -> Optional[DrugDetailApiItem]:
        response = await self._api_client.get_by_item_seq(item_seq)
        if response.total_count == 0 or not response.body or not response.body.items:
            return None
        # itemSeq로 단건 조회했으므로 정상적인 경우 1건만 나온다.
        return response.body.items[0]

    async def get_detail_by_name(self, item_name: str) -> Optional[DrugDetailApiItem]:
        """증상 기반 추천에서 LLM이 제안한 제품명 후보를 공식 DB로 검증한다.

        같은 이름의 다른 용량/제형 제품이 여러 건 검색될 수 있는데, 이 서비스는
        일단 첫 번째 결과를 채택한다 — 정확한 단일 매칭이 중요해지면 item_name
        완전 일치 필터를 추가하는 걸 권장한다(활용가이드에서 부분일치 지원
        여부를 확인한 뒤). 0건이면 '실존 확인 실패'로 보고 None을 반환한다 —
        호출부(app/symptom/service.py)는 이 후보를 그냥 버려야 한다.
        """
        response = await self._api_client.get_by_item_name(item_name)
        if response.total_count == 0 or not response.body or not response.body.items:
            return None
        return response.body.items[0]
