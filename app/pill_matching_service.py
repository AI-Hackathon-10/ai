"""
Vision AI가 1차로 추출한 각인·모양·색상·분할선을 하나의 조건으로 합쳐
pill_identification 테이블과 직접 비교한다.

직접 비교에서 0건(NO_MATCH)이면 조건을 완화해서 재조회하고,
각 후보에 유사도 점수를 매겨 가장 비슷한 알약을 상위로 정렬해 반환한다.
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

# 유사도 가중치: 각인은 물리적으로 새겨져 있어 신뢰도가 높고, 색상은 주관적
_WEIGHTS = {
    "print_front": 0.35,
    "print_back": 0.25,
    "drug_shape": 0.2,
    "color_class1": 0.1,
    "score_line": 0.1,
}


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
            # ── 조건 완화 재조회 ──
            relaxed = await self._relaxed_search(vision, filters)
            if relaxed:
                candidates = self._score_and_sort(relaxed, vision)[:MAX_CANDIDATES]
                return PillMatchResult(
                    status=MatchStatus.NO_MATCH,
                    candidates=candidates,
                    query_level_used="RELAXED_MATCH",
                )
            return PillMatchResult(status=MatchStatus.NO_MATCH, candidates=[], query_level_used="DIRECT_MATCH")

        if total > MAX_CANDIDATES:
            return PillMatchResult(
                status=MatchStatus.TOO_MANY_CANDIDATES, candidates=[], query_level_used="DIRECT_MATCH"
            )

        candidates = [PillCandidate.from_item(item) for item in matched]
        status = MatchStatus.SINGLE_MATCH if total == 1 else MatchStatus.MULTIPLE_MATCHES
        return PillMatchResult(status=status, candidates=candidates, query_level_used="DIRECT_MATCH")

    async def _relaxed_search(
        self, vision: VisionExtractionResult, original_filters: dict[str, str]
    ) -> list[PillApiItem]:
        """조건 완화 전략으로 후보를 찾는다.

        1) 각인(PRINT_FRONT, PRINT_BACK)만으로 재조회
        2) 1에서도 0건이면 모양(DRUG_SHAPE)만으로 재조회
        3) 결과를 합쳐서 중복 제거 후 반환
        """
        seen_seqs: set[str | None] = set()
        combined: list[PillApiItem] = []
        limit = MAX_CANDIDATES + 1

        # 전략 1: 각인만으로 검색
        print_filters: dict[str, str] = {}
        self._add_if_present(print_filters, "PRINT_FRONT", vision.print_front)
        self._add_if_present(print_filters, "PRINT_BACK", vision.print_back)

        if print_filters and print_filters != original_filters:
            items = await self._client.find(print_filters, limit=limit)
            for it in items:
                if it.item_seq not in seen_seqs:
                    seen_seqs.add(it.item_seq)
                    combined.append(it)

        # 전략 2: 모양만으로 검색 (전략 1에서 결과가 부족한 경우)
        if len(combined) == 0 and self._present(vision.drug_shape):
            shape_filters = {"DRUG_SHAPE": vision.drug_shape}
            if shape_filters != original_filters:
                items = await self._client.find(shape_filters, limit=limit)
                for it in items:
                    if it.item_seq not in seen_seqs:
                        seen_seqs.add(it.item_seq)
                        combined.append(it)

        return combined

    def _score_and_sort(
        self, items: list[PillApiItem], vision: VisionExtractionResult
    ) -> list[PillCandidate]:
        """각 후보와 Vision 추출값을 필드별로 비교해 가중 합산 유사도를 계산한다."""
        scored: list[PillCandidate] = []
        for it in items:
            score = self._calc_similarity(it, vision)
            candidate = PillCandidate.from_item(it)
            candidate.similarity_score = round(score, 4)
            scored.append(candidate)
        scored.sort(key=lambda c: c.similarity_score or 0.0, reverse=True)
        return scored

    def _calc_similarity(self, item: PillApiItem, vision: VisionExtractionResult) -> float:
        total = 0.0

        # print_front
        total += _WEIGHTS["print_front"] * _compare_print(vision.print_front, item.print_front)

        # print_back
        total += _WEIGHTS["print_back"] * _compare_print(vision.print_back, item.print_back)

        # drug_shape
        total += _WEIGHTS["drug_shape"] * _compare_exact(vision.drug_shape, item.drug_shape)

        # color_class1
        total += _WEIGHTS["color_class1"] * _compare_exact(vision.color_class1, item.color_class1)

        # score_line
        total += _WEIGHTS["score_line"] * _compare_score_line(vision.score_line, item)

        return total

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
    """공백 제거 + 소문자화."""
    return "".join(value.split()).casefold()


def _compare_print(vision_val: str | None, db_val: str | None) -> float:
    """각인 비교: 정확 일치 → 1.0, 부분 포함 → 0.5, 불일치 → 0."""
    if not vision_val or not db_val:
        return 0.0
    v = _normalize_print(vision_val)
    d = _normalize_print(db_val)
    if v == d:
        return 1.0
    if v in d or d in v:
        return 0.5
    return 0.0


def _compare_exact(vision_val: str | None, db_val: str | None) -> float:
    """정확 일치 비교."""
    if not vision_val or not db_val:
        return 0.0
    return 1.0 if vision_val.strip() == db_val.strip() else 0.0


def _compare_score_line(vision_score_line: bool | None, item: PillApiItem) -> float:
    """분할선 비교."""
    if vision_score_line is None:
        return 0.0
    item_has_line = _item_has_score_line(item)
    return 1.0 if vision_score_line == item_has_line else 0.0


def _item_has_score_line(item: PillApiItem) -> bool:
    return bool(
        (item.line_front and item.line_front.strip() and item.line_front.strip() != "없음")
        or (item.line_back and item.line_back.strip() and item.line_back.strip() != "없음")
    )
