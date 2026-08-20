"""
Vision AI가 1차로 추출한 각인·모양·색상·분할선을 하나의 조건으로 합쳐 낱알식별
카탈로그와 직접 비교한다.

식약처 getMdcinGrnIdntfcInfoList03 요청변수에는 품목명/업체명/품목코드 등만 있고
각인·색·모양은 없다. 그래서 API에 PRINT_FRONT 를 보내지 않고, 전체 목록을 받은 뒤
응답 필드(PRINT_FRONT, DRUG_SHAPE, COLOR_CLASS1 …)로 걸러낸다.

조건을 단계적으로 완화하는 로직은 두지 않는다 — Vision이 판단하지 못한 필드(None)는
조건에서 그냥 빠지고, 나머지 필드로 한 번만 조회한다. color_class1/drug_shape 가
둘 다 없는 경우(=조건을 하나도 못 만드는 경우)는 여기까지 오지 않는다 —
app/vision/graph.py 가 그 시점에 이미 판단 불가로 끊어낸다.
"""
from __future__ import annotations

from app.models import (
    MatchStatus,
    PillApiItem,
    PillCandidate,
    PillMatchResult,
    VisionExtractionResult,
)
from app.pill_identification_api_client import PillIdentificationApiClient

MAX_CANDIDATES = 10

_FILTER_ATTRS = {
    "PRINT_FRONT": "print_front",
    "PRINT_BACK": "print_back",
    "DRUG_SHAPE": "drug_shape",
    "COLOR_CLASS1": "color_class1",
}
_PRINT_FILTERS = {"PRINT_FRONT", "PRINT_BACK"}


class PillMatchingService:
    def __init__(self, api_client: PillIdentificationApiClient):
        self._api_client = api_client

    async def match(self, vision: VisionExtractionResult) -> PillMatchResult:
        filters = self._build_filters(vision)

        finder = getattr(self._api_client.__class__, "find", None)
        if callable(finder):
            matched = await self._api_client.find(filters, limit=MAX_CANDIDATES + 1)
        else:
            catalog = await self._api_client.get_catalog()
            matched = [item for item in catalog if _item_matches(item, filters)]

        total = len(matched)
        if total == 0:
            return PillMatchResult(status=MatchStatus.NO_MATCH, candidates=[], query_level_used="DIRECT_MATCH")
        if total > MAX_CANDIDATES:
            return PillMatchResult(
                status=MatchStatus.TOO_MANY_CANDIDATES, candidates=[], query_level_used="DIRECT_MATCH"
            )

        candidates = [PillCandidate.from_item(item) for item in matched]
        status = MatchStatus.SINGLE_MATCH if total == 1 else MatchStatus.MULTIPLE_MATCHES
        return PillMatchResult(status=status, candidates=candidates, query_level_used="DIRECT_MATCH")

    def _build_filters(self, v: VisionExtractionResult) -> dict[str, str]:
        filters: dict[str, str] = {}
        self._add_if_present(filters, "PRINT_FRONT", v.print_front)
        self._add_if_present(filters, "PRINT_BACK", v.print_back)
        self._add_if_present(filters, "DRUG_SHAPE", v.drug_shape)
        self._add_if_present(filters, "COLOR_CLASS1", v.color_class1)
        if v.score_line is not None:  # False도 "판단됨"이라 필터에 넣어야 함
            filters["SCORE_LINE"] = "true" if v.score_line else "false"
        return filters

    @staticmethod
    def _present(value: str | None) -> bool:
        return bool(value and value.strip())

    @classmethod
    def _add_if_present(cls, target: dict[str, str], key: str, value: str | None) -> None:
        if cls._present(value):
            target[key] = value


def _normalize_print(value: str) -> str:
    return "".join(value.split()).casefold()


def _has_score_line(item: PillApiItem) -> bool:
    """테이블에 SCORE_LINE 컬럼은 없다 — line_front/line_back 중 하나라도 값이
    있으면 분할선이 있는 것으로 본다(app/pill_identification_repository.py 의
    SQL 버전과 동일한 판단 기준)."""
    return bool((item.line_front and item.line_front.strip()) or (item.line_back and item.line_back.strip()))


def _item_matches(item: PillApiItem, filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        if key == "SCORE_LINE":
            if _has_score_line(item) != (expected == "true"):
                return False
            continue

        attr = _FILTER_ATTRS.get(key)
        if attr is None:
            continue
        actual = getattr(item, attr, None)
        if not actual or not str(actual).strip():
            return False
        if key in _PRINT_FILTERS:
            if _normalize_print(str(actual)) != _normalize_print(expected):
                return False
        elif str(actual).strip() != expected.strip():
            return False
    return True
