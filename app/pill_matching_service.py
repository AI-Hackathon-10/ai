"""
Vision AI 추출 결과 → pill_identification DB 테이블을 조회해 의약품을 매칭한다.

핵심 아이디어: Vision AI 결과를 곧바로 한 번에 검색하지 않고,
"가장 구체적인 조건 → 점점 완화되는 조건" 순서로 여러 레벨을 미리 만들어두고
위에서부터 하나씩 필터하며 첫 번째로 "합리적인 개수"의 결과가 나오는 레벨에서 멈춘다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.models import (
    MatchStatus,
    PillApiItem,
    PillCandidate,
    PillMatchResult,
    VisionExtractionResult,
)

PRINT_CONFIDENCE_THRESHOLD = 0.6
MAX_CANDIDATES = 10


class CatalogFinder(Protocol):
    async def find(self, filters: dict[str, str], limit: int = ...) -> list[PillApiItem]: ...


@dataclass
class _QueryLevel:
    name: str
    params: dict[str, str] = field(default_factory=dict)


class PillMatchingService:
    def __init__(self, catalog_client: CatalogFinder):
        self._client = catalog_client

    async def match(self, vision: VisionExtractionResult) -> PillMatchResult:
        levels = self._build_query_levels(vision)

        for index, level in enumerate(levels):
            if not level.params:
                continue

            matched = await self._client.find(level.params, limit=MAX_CANDIDATES + 1)
            total = len(matched)

            if total == 0:
                continue

            if total <= MAX_CANDIDATES:
                candidates = [PillCandidate.from_item(item) for item in matched]
                status = MatchStatus.SINGLE_MATCH if total == 1 else MatchStatus.MULTIPLE_MATCHES
                return PillMatchResult(status=status, candidates=candidates, query_level_used=level.name)

            is_last_level = index == len(levels) - 1
            if is_last_level:
                return PillMatchResult(
                    status=MatchStatus.TOO_MANY_CANDIDATES,
                    candidates=[],
                    query_level_used=level.name,
                )

        return PillMatchResult(
            status=MatchStatus.NO_MATCH,
            candidates=[],
            query_level_used="ALL_LEVELS_EXHAUSTED",
        )

    def _build_query_levels(self, v: VisionExtractionResult) -> list[_QueryLevel]:
        """레벨은 '구체적인 것 → 넓은 것' 순서로 정의한다."""
        levels: list[_QueryLevel] = []

        for print_fields, level_name in self._reliable_print_combinations(v):
            params: dict[str, str] = dict(print_fields)
            self._add_if_present(params, "DRUG_SHAPE", v.drug_shape)
            self._add_if_present(params, "COLOR_CLASS1", v.color_class1)
            self._add_if_present(params, "COLOR_CLASS2", v.color_class2)
            levels.append(_QueryLevel(level_name, params))

        p_shape_color: dict[str, str] = {}
        self._add_if_present(p_shape_color, "DRUG_SHAPE", v.drug_shape)
        self._add_if_present(p_shape_color, "COLOR_CLASS1", v.color_class1)
        self._add_if_present(p_shape_color, "COLOR_CLASS2", v.color_class2)
        levels.append(_QueryLevel("SHAPE_AND_COLOR", p_shape_color))

        p_shape_primary: dict[str, str] = {}
        self._add_if_present(p_shape_primary, "DRUG_SHAPE", v.drug_shape)
        self._add_if_present(p_shape_primary, "COLOR_CLASS1", v.color_class1)
        levels.append(_QueryLevel("SHAPE_AND_PRIMARY_COLOR", p_shape_primary))

        p_color_only: dict[str, str] = {}
        self._add_if_present(p_color_only, "COLOR_CLASS1", v.color_class1)
        levels.append(_QueryLevel("COLOR_ONLY", p_color_only))

        return levels

    def _reliable_print_combinations(
        self, v: VisionExtractionResult
    ) -> list[tuple[dict[str, str], str]]:
        front_reliable = (
            self._present(v.print_front)
            and v.print_front_confidence is not None
            and v.print_front_confidence >= PRINT_CONFIDENCE_THRESHOLD
        )
        back_reliable = (
            self._present(v.print_back)
            and v.print_back_confidence is not None
            and v.print_back_confidence >= PRINT_CONFIDENCE_THRESHOLD
        )

        combos: list[tuple[dict[str, str], str]] = []

        if front_reliable and back_reliable:
            combos.append(
                ({"PRINT_FRONT": v.print_front, "PRINT_BACK": v.print_back}, "STRICT_WITH_PRINT_BOTH")
            )
        if front_reliable:
            combos.append(({"PRINT_FRONT": v.print_front}, "STRICT_WITH_PRINT_FRONT_ONLY"))
        if back_reliable:
            combos.append(({"PRINT_BACK": v.print_back}, "STRICT_WITH_PRINT_BACK_ONLY"))

        return combos

    @staticmethod
    def _present(value: str | None) -> bool:
        return bool(value and value.strip())

    @classmethod
    def _add_if_present(cls, target: dict[str, str], key: str, value: str | None) -> None:
        if cls._present(value):
            target[key] = value
