"""
Vision AI가 1차로 추출한 각인·모양·색상·분할선을 하나의 조건으로 합쳐
pill_identification 테이블과 직접 비교한다.

조건을 단계적으로 완화하는 로직은 두지 않는다 — Vision이 판단하지 못한 필드(None)는
조건에서 그냥 빠지고, 나머지 필드로 한 번만 조회한다. color_class1/drug_shape 가
둘 다 없는 경우(=조건을 하나도 못 만드는 경우)는 여기까지 오지 않는다 —
app/vision/graph.py 가 그 시점에 이미 판단 불가로 끊어낸다.
"""
from __future__ import annotations

from typing import Protocol

from app.models import (
    MatchStatus,
    PillApiItem,
    PillCandidate,
    PillMatchResult,
    VisionExtractionResult,
)

MAX_CANDIDATES = 10


class CatalogFinder(Protocol):
    async def find(self, filters: dict[str, str], limit: int = ...) -> list[PillApiItem]: ...


class PillMatchingService:
    def __init__(self, catalog_client: CatalogFinder):
        self._client = catalog_client

    async def match(self, vision: VisionExtractionResult) -> PillMatchResult:
        filters = self._build_filters(vision)
        matched = await self._client.find(filters, limit=MAX_CANDIDATES + 1)

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
