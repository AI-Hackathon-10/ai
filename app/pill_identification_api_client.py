"""
식약처 의약품 낱알식별 정보 API 호출 전담 클라이언트.

쿼리 완화 전략(레벨링) 로직은 여기 두지 않고 PillMatchingService 에서만 다룬다.
이 클래스는 공식 요청변수로 한 번 호출하고, 필요하면 전체 목록 카탈로그를 가져온다.

getMdcinGrnIdntfcInfoList03 공식 요청변수:
  serviceKey, pageNo, numOfRows, type,
  item_name, entp_name, item_seq, img_regist_ts, edi_code, bizrno

PRINT_FRONT / DRUG_SHAPE / COLOR_CLASS1 은 응답 필드이지 요청변수가 아니다.
대문자로 보내면 필터가 무시되고 전체 목록이 내려온다.
"""
from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import httpx

from app.models import PillApiItem, PillApiResponse

BASE_URL = "https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService03/getMdcinGrnIdntfcInfoList03"
NUM_OF_ROWS = 50
CATALOG_PAGE_SIZE = 100
CATALOG_TTL_SECONDS = 6 * 60 * 60
CATALOG_FETCH_CONCURRENCY = 5
DEFAULT_TIMEOUT_SECONDS = 5.0

OFFICIAL_REQUEST_PARAMS = frozenset(
    {
        "item_name",
        "entp_name",
        "item_seq",
        "img_regist_ts",
        "edi_code",
        "bizrno",
    }
)

_CACHE_PATH = Path(__file__).resolve().parent.parent / ".cache" / "pill_catalog.json"


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


def to_official_request_params(params: dict[str, str]) -> dict[str, str]:
    """호출부가 넘긴 키를 공식 요청변수(소문자)만 남긴다.

    ITEM_NAME 같은 대문자 별칭은 소문자로 바꾸고, PRINT_FRONT 처럼 스펙에 없는
    키는 버린다. 빈 문자열/"null" 도 보내지 않는다.
    """
    official: dict[str, str] = {}
    for key, raw_value in params.items():
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if not value or value.lower() == "null":
            continue
        mapped = key.strip().lower()
        if mapped in OFFICIAL_REQUEST_PARAMS:
            official[mapped] = value
    return official


class PillIdentificationApiClient:
    def __init__(self, service_key: str, http_client: Optional[httpx.AsyncClient] = None):
        self._service_key = _normalize_service_key(service_key)
        self._client = http_client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
        self._owns_client = http_client is None
        self._catalog: Optional[list[PillApiItem]] = None
        self._catalog_fetched_at: Optional[float] = None
        self._catalog_lock = asyncio.Lock()

    async def search(self, params: dict[str, str]) -> PillApiResponse:
        """공식 요청변수만 붙여 1페이지를 조회한다."""
        return await self._fetch_page(params, page_no=1, num_of_rows=NUM_OF_ROWS)

    async def get_catalog(self) -> list[PillApiItem]:
        """전체 낱알 목록. 메모리/디스크 캐시를 TTL 동안 재사용한다."""
        now = time.time()
        if self._catalog is not None and self._catalog_fetched_at is not None:
            if now - self._catalog_fetched_at < CATALOG_TTL_SECONDS:
                return self._catalog

        async with self._catalog_lock:
            now = time.time()
            if self._catalog is not None and self._catalog_fetched_at is not None:
                if now - self._catalog_fetched_at < CATALOG_TTL_SECONDS:
                    return self._catalog

            cached = _read_disk_catalog(now)
            if cached is not None:
                self._catalog = cached
                self._catalog_fetched_at = now
                return self._catalog

            items = await self._download_catalog()
            self._catalog = items
            self._catalog_fetched_at = time.time()
            _write_disk_catalog(items, self._catalog_fetched_at)
            return items

    async def _download_catalog(self) -> list[PillApiItem]:
        first = await self._fetch_page({}, page_no=1, num_of_rows=CATALOG_PAGE_SIZE)
        items = list(first.body.items) if first.body else []
        total = first.total_count
        if total <= len(items):
            return items

        last_page = (total + CATALOG_PAGE_SIZE - 1) // CATALOG_PAGE_SIZE
        semaphore = asyncio.Semaphore(CATALOG_FETCH_CONCURRENCY)

        async def fetch_page(page_no: int) -> list[PillApiItem]:
            async with semaphore:
                response = await self._fetch_page(
                    {}, page_no=page_no, num_of_rows=CATALOG_PAGE_SIZE
                )
                return list(response.body.items) if response.body else []

        pages = await asyncio.gather(
            *[fetch_page(page_no) for page_no in range(2, last_page + 1)]
        )
        for page_items in pages:
            items.extend(page_items)
        return items

    async def _fetch_page(
        self, params: dict[str, str], page_no: int, num_of_rows: int
    ) -> PillApiResponse:
        query = {
            "serviceKey": self._service_key,
            "type": "json",
            "pageNo": page_no,
            "numOfRows": num_of_rows,
            **to_official_request_params(params),
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


def _read_disk_catalog(now: float) -> Optional[list[PillApiItem]]:
    if not _CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fetched_at = payload.get("fetched_at")
    raw_items = payload.get("items")
    if not isinstance(fetched_at, (int, float)) or not isinstance(raw_items, list):
        return None
    if now - fetched_at >= CATALOG_TTL_SECONDS:
        return None
    return [PillApiItem.model_validate(row) for row in raw_items]


def _write_disk_catalog(items: list[PillApiItem], fetched_at: float) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": fetched_at,
            "items": [item.model_dump(by_alias=True) for item in items],
        }
        _CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        return
