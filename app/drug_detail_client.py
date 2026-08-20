"""
e약은요(DrbEasyDrugInfoService) API 호출 전담 클라이언트 (Step 4).

낱알식별 매칭 단계에서 확정된 ITEM_SEQ(품목일련번호)로 효능/사용법/주의사항/
상호작용/부작용/보관법을 조회한다.

⚠️ BASE_URL 의 오퍼레이션명(getDrbEasyDrugList 부분)은 통용되는 명칭 기준 추정치다.
data.go.kr 활용신청 상세페이지의 참고문서/요청 URL 예제에서 확인 후 교체할 것.
PillIdentificationApiClient 와 마찬가지로 인증키 이중 인코딩 문제를 겪지 않도록
정규화한다.
"""
from __future__ import annotations

import urllib.parse
from typing import Optional

import httpx

from app.drug_detail_models import DrugDetailApiResponse

# TODO: 실제 오퍼레이션명 확인 후 교체
BASE_URL = "https://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
NUM_OF_ROWS = 10
DEFAULT_TIMEOUT_SECONDS = 5.0


class DrugDetailApiError(Exception):
    """e약은요 API 호출/응답 관련 오류."""


def _normalize_service_key(raw_key: str) -> str:
    # PillIdentificationApiClient 와 동일한 이유로 동일하게 처리한다.
    # (data.go.kr의 "일반 인증키(Encoding)"를 그대로 넘기면 이중 인코딩되는 문제 방지)
    if "%" in raw_key:
        return urllib.parse.unquote(raw_key)
    return raw_key


class DrugDetailApiClient:
    def __init__(self, service_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self._service_key = _normalize_service_key(service_key)
        self._client = http_client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._owns_client = http_client is None

    async def get_by_item_seq(self, item_seq: str) -> DrugDetailApiResponse:
        return await self._fetch({"itemSeq": item_seq})

    async def get_by_item_name(self, item_name: str) -> DrugDetailApiResponse:
        return await self._fetch({"itemName": item_name})

    async def _fetch(self, extra_params: dict) -> DrugDetailApiResponse:
        query = {
            "serviceKey": self._service_key,
            "type": "json",
            "pageNo": 1,
            "numOfRows": NUM_OF_ROWS,
            **extra_params,
        }

        try:
            response = await self._client.get(BASE_URL, params=query)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise DrugDetailApiError(f"e약은요 API 호출 실패: {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            raise DrugDetailApiError(f"e약은요 API 응답 파싱 실패: {e}") from e

        parsed = DrugDetailApiResponse.model_validate(data)
        if parsed.header is not None and not parsed.header.is_success:
            raise DrugDetailApiError(f"e약은요 API 오류 응답: {parsed.header.result_msg}")

        return parsed

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
