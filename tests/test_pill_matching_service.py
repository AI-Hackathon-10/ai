"""
실제 Vision AI / 실제 DB 없이도 매칭 로직(직접 비교)을 검증한다.

조건은 단계적으로 완화하지 않는다 — Vision이 추출한 값(None이 아닌 필드만)을
한 번에 합쳐서 비교한다. 신뢰도 점수는 쓰지 않는다.
"""
from __future__ import annotations

import pytest

from app.models import MatchStatus, PillApiItem, VisionExtractionResult
from app.pill_matching_service import PillMatchingService

_FILTER_ATTRS = {
    "PRINT_FRONT": "print_front",
    "PRINT_BACK": "print_back",
    "DRUG_SHAPE": "drug_shape",
    "COLOR_CLASS1": "color_class1",
}
_PRINT_FILTERS = {"PRINT_FRONT", "PRINT_BACK"}


def _normalize_print(value: str) -> str:
    return "".join(value.split()).casefold()


def _has_score_line(item: PillApiItem) -> bool:
    return bool(
        (item.line_front and item.line_front.strip())
        or (item.line_back and item.line_back.strip())
    )


def _item_matches(item: PillApiItem, filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        if key == "SCORE_LINE":
            if _has_score_line(item) != (str(expected).strip().lower() == "true"):
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


def item(
    item_seq: str,
    item_name: str,
    *,
    print_front: str | None = None,
    print_back: str | None = None,
    drug_shape: str | None = None,
    color_class1: str | None = None,
    line_front: str | None = None,
    line_back: str | None = None,
) -> PillApiItem:
    return PillApiItem(
        item_seq=item_seq,
        item_name=item_name,
        entp_name="제조사",
        chart="흰색 장방형 정제",
        print_front=print_front,
        print_back=print_back,
        drug_shape=drug_shape,
        color_class1=color_class1,
        line_front=line_front,
        line_back=line_back,
    )


class _InMemoryFinder:
    """테스트용: 메모리 내 아이템 목록을 로컬 필터로 find() 하는 클라이언트."""

    def __init__(self, items: list[PillApiItem]):
        self.items = items
        self.calls: list[dict[str, str]] = []

    async def find(self, filters: dict[str, str], limit: int = 11) -> list[PillApiItem]:
        self.calls.append(dict(filters))
        return [i for i in self.items if _item_matches(i, filters)][:limit]


def service_with_items(*items: PillApiItem) -> PillMatchingService:
    return PillMatchingService(_InMemoryFinder(list(items)))


@pytest.mark.asyncio
async def test_타이레놀_각인_모양_색이_일치하면_SINGLE_MATCH():
    matching = service_with_items(
        item(
            "199703222",
            "타이레놀정500밀리그람",
            print_front="TYLENOL",
            print_back="500",
            drug_shape="장방형",
            color_class1="하양",
        ),
        item("2", "다른약", print_front="AB", print_back="12", drug_shape="원형", color_class1="노랑"),
    )
    vision = VisionExtractionResult(
        print_front="TYLENOL",
        print_back="500",
        color_class1="하양",
        drug_shape="장방형",
        score_line=False,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.query_level_used == "DIRECT_MATCH"
    assert result.candidates[0].item_seq == "199703222"


@pytest.mark.asyncio
async def test_각인을_판단못해도_모양_색상만으로_매칭을_시도한다():
    matching = service_with_items(
        item("1", "약A", print_front="TY", drug_shape="장방형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="장방형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH


@pytest.mark.asyncio
async def test_각인이_다르면_직접_매칭되지_않고_완화검색으로_넘어간다():
    matching = service_with_items(
        item("1", "약A", print_front="AB", drug_shape="장방형", color_class1="하양"),
    )
    vision = VisionExtractionResult(print_front="TYLENOL", color_class1="하양", drug_shape="장방형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    # 각인 "TYLENOL"로 완화검색 → 0건, 모양 "장방형"으로 완화검색 → 1건
    assert len(result.candidates) == 1
    assert result.query_level_used == "RELAXED_MATCH"
    assert result.candidates[0].similarity_score is not None


@pytest.mark.asyncio
async def test_모양_색상만_같아도_여러건이면_MULTIPLE_MATCHES():
    matching = service_with_items(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="노랑"),
        item("2", "약B", print_front="BB", drug_shape="원형", color_class1="노랑"),
    )
    vision = VisionExtractionResult(color_class1="노랑", drug_shape="원형")

    result = await matching.match(vision)

    assert result.status == MatchStatus.MULTIPLE_MATCHES
    assert result.query_level_used == "DIRECT_MATCH"
    assert len(result.candidates) == 2


@pytest.mark.asyncio
async def test_후보가_없으면_NO_MATCH():
    matching = service_with_items(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="회색", drug_shape="기타")

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.candidates == []


@pytest.mark.asyncio
async def test_후보가_너무_많으면_TOO_MANY_CANDIDATES():
    matching = service_with_items(
        *[item(str(i), f"약{i}", color_class1="하양") for i in range(37)]
    )
    vision = VisionExtractionResult(color_class1="하양")

    result = await matching.match(vision)

    assert result.status == MatchStatus.TOO_MANY_CANDIDATES


@pytest.mark.asyncio
async def test_분할선_있음으로_판단되면_분할선_없는_후보는_제외된다():
    matching = service_with_items(
        item("1", "분할선있음", drug_shape="원형", color_class1="하양", line_front="-"),
        item("2", "분할선없음", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="원형", score_line=True)

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.candidates[0].item_seq == "1"


@pytest.mark.asyncio
async def test_분할선_없음으로_판단되면_분할선_있는_후보는_제외된다():
    matching = service_with_items(
        item("1", "분할선있음", drug_shape="원형", color_class1="하양", line_back="-"),
        item("2", "분할선없음", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="원형", score_line=False)

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert result.candidates[0].item_seq == "2"


@pytest.mark.asyncio
async def test_분할선을_판단못하면_필터에서_아예_빠진다():
    matching = service_with_items(
        item("1", "분할선있음", drug_shape="원형", color_class1="하양", line_front="-"),
        item("2", "분할선없음", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(color_class1="하양", drug_shape="원형", score_line=None)

    result = await matching.match(vision)

    assert result.status == MatchStatus.MULTIPLE_MATCHES
    assert len(result.candidates) == 2


@pytest.mark.asyncio
async def test_NO_MATCH시_각인으로_완화검색하면_유사후보를_반환한다():
    """색상만 다른 경우 → 각인 일치 후보가 similarity_score와 함께 반환."""
    matching = service_with_items(
        item(
            "200810473",
            "엑스반정80밀리그램(발사르탄)",
            print_front="SJ",
            print_back="V분할선8",
            drug_shape="원형",
            color_class1="분홍",
            line_back="-",
        ),
        item("999", "전혀다른약", print_front="ZZ", drug_shape="삼각형", color_class1="검정"),
    )
    # Vision: 색상이 "주황"으로 달라서 직접 매칭 실패
    vision = VisionExtractionResult(
        print_front="SJ",
        print_back="V 8",
        color_class1="주황",
        drug_shape="원형",
        score_line=True,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.query_level_used == "RELAXED_MATCH"
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.item_seq == "200810473"
    assert c.similarity_score is not None
    assert c.similarity_score > 0.5  # 각인·모양·분할선 일치하므로 높은 점수


@pytest.mark.asyncio
async def test_완화검색_결과도_없으면_빈_candidates를_반환한다():
    """완전히 매칭 안 되는 경우 → 기존과 동일하게 빈 리스트."""
    matching = service_with_items(
        item("1", "약A", print_front="AA", drug_shape="원형", color_class1="하양"),
    )
    vision = VisionExtractionResult(
        print_front="ZZZZZ",
        color_class1="회색",
        drug_shape="기타",
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.candidates == []
    assert result.query_level_used == "DIRECT_MATCH"


@pytest.mark.asyncio
async def test_완화검색_후보는_유사도_점수_내림차순으로_정렬된다():
    matching = service_with_items(
        # 각인 부분 일치 + 모양 불일치
        item("low", "낮은점수", print_front="SJ", drug_shape="삼각형", color_class1="검정"),
        # 각인 완전 일치 + 모양 일치
        item("high", "높은점수", print_front="SJ", drug_shape="원형", color_class1="분홍"),
    )
    vision = VisionExtractionResult(
        print_front="SJ",
        color_class1="주황",  # 둘 다 색상 불일치
        drug_shape="원형",
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.query_level_used == "RELAXED_MATCH"
    assert len(result.candidates) == 2
    assert result.candidates[0].item_seq == "high"
    assert result.candidates[1].item_seq == "low"
    assert result.candidates[0].similarity_score >= result.candidates[1].similarity_score


@pytest.mark.asyncio
async def test_엑스반정_색상불일치_케이스():
    """실제 사례: SJ/V분할선8/분홍 vs SJ/V8/주황 — 유사도 0.775 예상."""
    db_item = item(
        "200810473",
        "엑스반정80밀리그램(발사르탄)",
        print_front="SJ",
        print_back="V분할선8",
        drug_shape="원형",
        color_class1="분홍",
        line_back="-",
    )
    matching = service_with_items(db_item)

    vision = VisionExtractionResult(
        print_front="SJ",
        print_back="V 8",
        color_class1="주황",
        drug_shape="원형",
        score_line=True,
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.NO_MATCH
    assert result.query_level_used == "RELAXED_MATCH"
    assert len(result.candidates) == 1
    c = result.candidates[0]
    assert c.item_seq == "200810473"
    # print_front: "sj"=="sj" → 0.35
    # print_back: "v8" vs "v분할선8" → 연속 부분문자열 아님 → 0.0
    # drug_shape: "원형"=="원형" → 0.2
    # color_class1: "주황"≠"분홍" → 0.0
    # score_line: True==has_line("-") → 0.1
    # 합계: 0.65
    assert c.similarity_score == pytest.approx(0.65, abs=0.01)


@pytest.mark.asyncio
async def test_신뢰도_필드_없이도_find에_각인_필터가_전달된다():
    client = _InMemoryFinder(
        [item("199703222", "타이레놀정500밀리그람", print_front="TYLENOL", drug_shape="장방형", color_class1="하양")]
    )
    matching = PillMatchingService(client)
    vision = VisionExtractionResult(
        print_front="TYLENOL",
        color_class1="하양",
        drug_shape="장방형",
    )

    result = await matching.match(vision)

    assert result.status == MatchStatus.SINGLE_MATCH
    assert client.calls
    assert client.calls[0]["PRINT_FRONT"] == "TYLENOL"
    assert client.calls[0]["DRUG_SHAPE"] == "장방형"
    assert "COLOR_CLASS2" not in client.calls[0]
