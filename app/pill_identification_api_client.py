"""
식약처 의약품 낱알식별 정보 API 호출 전담 클라이언트.

쿼리 완화 전략(레벨링) 로직은 여기 두지 않고 PillMatchingService 에서만 다룬다.
이 클래스는 "params 를 받아 한 번 호출하고 파싱된 응답을 돌려준다"는 책임만 가진다.

BASE_URL 은 실제 확인된 값(https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03)
까지는 확정이다. ⚠️ 그 뒤에 붙는 오퍼레이션명(getMdcinGrnIdntfcInfoList03 부분)은 아직
추정치이니, data.go.kr 활용신청 상세페이지의 "참고문서"/요청 URL 예제에서 정확한
오퍼레이션명을 확인해 이 상수만 교체하면 된다.
"""
from __future__ import annotations

import urllib.parse

import httpx

from app.models import PillApiResponse

# TODO: 오퍼레이션명(...InfoService03/ 뒤에 붙는 부분)만 활용가이드 확인 후 교체
BASE_URL = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"
NUM_OF_ROWS = 50
DEFAULT_TIMEOUT_SECONDS = 5.0


class PillApiError(Exception):
    """낱알식별 API 호출/응답 관련 오류."""


def _normalize_service_key(raw_key: str) -> str:
    """
    data.go.kr는 인증키를 "일반 인증키(Encoding)"(이미 URL 인코딩된 형태)와
    "인증키(Decoding)"(원본 그대로) 두 가지로 제공한다.

    httpx 는 params 로 넘긴 값을 요청 시점에 자동으로 URL 인코딩한다. 그런데 이미
    인코딩된 키(예: ...%2BHdVDg...%3D%3D)를 그대로 넘기면 '%'가 다시 인코딩되어
    %2B -> %252B 처럼 두 번 인코딩된 값이 서버로 전송되고, 그 결과 인증키가 깨져
    SERVICE_KEY_IS_NOT_REGISTERED_ERROR 같은 오류가 난다.

    그래서 키에 '%'가 포함돼 있으면(=Encoding 키를 넣었다는 신호) 한 번 디코딩해서
    원본 형태로 되돌려 놓는다. 이렇게 하면 Encoding 키를 넣든 Decoding 키를 넣든
    항상 httpx가 정확히 한 번만 인코딩하게 되어 동일하게 동작한다.
    """
    if "%" in raw_key:
        return urllib.parse.unquote(raw_key)
    return raw_key


class PillIdentificationApiClient:
    def __init__(self, service_key: str, http_client: httpx.AsyncClient | None = None):
        self._service_key = _normalize_service_key(service_key)
        # 호출부(FastAPI 앱)에서 만든 공유 AsyncClient를 주입받을 수도 있게 열어둠 (커넥션 재사용)
        self._client = http_client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._owns_client = http_client is None

    async def search(self, params: dict[str, str]) -> PillApiResponse:
        """
        params 예: {"DRUG_SHAPE": "원형", "COLOR_CLASS1": "하양"}
        (data.go.kr 응답 필드명과 동일하게 대문자 스네이크케이스 키를 사용해야 한다.
        소문자로 보내면 API가 조건 없는 요청으로 취급해 사실상 필터링 없이 넓은
        범위의 결과를 그대로 반환하는 것으로 보인다 — 이 부분이 원인으로 의심되어
        수정했으나, 실제 서비스키로 라이브 API를 재확인하는 검증은 아직 못했다.)
        서비스키/type/페이징 파라미터는 여기서 자동으로 붙는다.
        """
        query = {
            "serviceKey": self._service_key,
            "type": "json",
            "pageNo": 1,
            "numOfRows": NUM_OF_ROWS,
            **params,
        }

        try:
            response = await self._client.get(BASE_URL, params=query)
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise PillApiError(f"낱알식별 API 호출 실패: {e}") from e

        try:
            data = response.json()
        except ValueError as e:
            raise PillApiError(f"낱알식별 API 응답 파싱 실패: {e}") from e

        parsed = PillApiResponse.model_validate(data)
        if parsed.header is not None and not parsed.header.is_success:
            raise PillApiError(f"낱알식별 API 오류 응답: {parsed.header.result_msg}")

        return parsed

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
