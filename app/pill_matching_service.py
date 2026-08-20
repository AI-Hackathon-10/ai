"""
Vision AI 추출 결과 → 낱알식별 카탈로그를 로컬 필터해 의약품을 매칭한다.

식약처 getMdcinGrnIdntfcInfoList03 요청변수에는 품목명/업체명/품목코드 등만 있고
각인·색·모양은 없다. 그래서 API에 PRINT_FRONT 를 보내지 않고, 전체 목록을 받은 뒤
응답 필드(PRINT_FRONT, DRUG_SHAPE, COLOR_CLASS1 …)로 걸러낸다.

핵심 아이디어: Vision AI 결과를 곧바로 한 번에 검색하지 않고,
"가장 구체적인 조건 → 점점 완화되는 조건" 순서로 여러 레벨을 미리 만들어두고
위에서부터 하나씩 필터하며 첫 번째로 "합리적인 개수"의 결과가 나오는 레벨에서 멈춘다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models import (
    MatchStatus,
    PillApiItem,
    PillCandidate,
    PillMatchResult,
    VisionExtractionResult,
)
from app.pill_identification_api_client import PillIdentificationApiClient

PRINT_CONFIDENCE_THRESHOLD = 0.6
MAX_CANDIDATES = 10

_FILTER_ATTRS = {
    "PRINT_FRONT": "print_front",
    "PRINT_BACK": "print_back",
    "DRUG_SHAPE": "drug_shape",
    "COLOR_CLASS1": "color_class1",
    "COLOR_CLASS2": "color_class2",
}
_PRINT_FILTERS = {"PRINT_FRONT", "PRINT_BACK"}


@dataclass
class _QueryLevel:
    name: str
    params: dict[str, str] = field(default_factory=dict)


class PillMatchingService:
    def __init__(self, api_client: PillIdentificationApiClient):
        self._api_client = api_client

    async def match(self, vision: VisionExtractionResult) -> PillMatchResult:
        levels = self._build_query_levels(vision)
        finder = getattr(self._api_client.__class__, "find", None)
        catalog: list[PillApiItem] | None = None
        if not callable(finder):
            catalog = await self._api_client.get_catalog()

        for index, level in enumerate(levels):
            if not level.params:
                continue

            if callable(finder):
                matched = await self._api_client.find(level.params, limit=MAX_CANDIDATES + 1)
            else:
                assert catalog is not None
                matched = [item for item in catalog if _item_matches(item, level.params)]
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
        """레벨은 '구체적인 것 → 넓은 것' 순서로 정의한다.

        여기서 만드는 키(PRINT_FRONT, DRUG_SHAPE …)는 API 요청변수가 아니라
        카탈로그 응답 필드로 로컬 필터할 때 쓰는 이름이다.
        """
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


def _normalize_print(value: str) -> str:
    return "".join(value.split()).casefold()


def _item_matches(item: PillApiItem, filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
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
