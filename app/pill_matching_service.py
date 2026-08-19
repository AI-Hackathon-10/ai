"""
Vision AI 추출 결과 → 낱알식별 API 쿼리로 변환해 의약품을 매칭한다.

핵심 아이디어: Vision AI 결과를 곧바로 한 번에 검색하지 않고,
"가장 구체적인 조건 → 점점 완화되는 조건" 순서로 여러 레벨을 미리 만들어두고
위에서부터 하나씩 호출하며 첫 번째로 "합리적인 개수"의 결과가 나오는 레벨에서 멈춘다.

이렇게 하는 이유:
- 각인(印刻)은 Vision AI가 가장 잘못 읽기 쉬운 값이라, 처음부터 필수 조건으로 걸면
  실제로는 맞는 약인데도 각인 오독 때문에 0건이 나오는 경우가 많다.
- 앞면과 뒷면은 인식 난이도가 서로 다를 수 있다(한쪽만 흐릿하거나, 애초에 한쪽엔 각인이
  없는 경우가 흔함). 그래서 신뢰도를 앞/뒤 통합 하나로 묶지 않고 각각 따로 평가해서,
  "둘 다 넣었을 때는 실패해도 사실 한쪽은 맞았던" 케이스를 구제할 수 있게 했다.
- 반대로 색상 하나만으로 검색하면 후보가 수백 건이 나올 수 있어 그대로 사용자에게
  보여주면 의미가 없다. 그래서 "결과가 너무 많으면 그 레벨은 버리고 다음 축 조합을 시도"
  하도록 설계했다.
- 이 판단(레벨 순서, 몇 건이면 확정/선택/실패로 볼지)은 LLM에게 맡기지 않고 코드로
  고정한다. 매번 같은 규칙으로 동작해야 하는 결정론적 로직이기 때문이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.models import (
    MatchStatus,
    PillApiResponse,
    PillCandidate,
    PillMatchResult,
    VisionExtractionResult,
)
from app.pill_identification_api_client import PillIdentificationApiClient

# 이 신뢰도 미만이면 각인 값을 쿼리에 아예 포함시키지 않는다.
PRINT_CONFIDENCE_THRESHOLD = 0.6

# 후보가 이 개수 이하일 때만 "사용자가 고를 수 있는 후보 목록"으로 인정한다.
MAX_CANDIDATES = 10


@dataclass
class _QueryLevel:
    name: str
    params: dict[str, str] = field(default_factory=dict)


class PillMatchingService:
    def __init__(self, api_client: PillIdentificationApiClient):
        self._api_client = api_client

    async def match(self, vision: VisionExtractionResult) -> PillMatchResult:
        levels = self._build_query_levels(vision)

        for index, level in enumerate(levels):
            if not level.params:
                continue  # 이 레벨에 쓸 데이터가 아예 없으면 건너뜀 (예: 각인 신뢰도 낮아 레벨1 스킵)

            response = await self._api_client.search(level.params)
            total = response.total_count

            if total == 0:
                continue  # 더 완화된 다음 레벨로

            if total <= MAX_CANDIDATES:
                candidates = self._to_candidates(response)
                status = MatchStatus.SINGLE_MATCH if total == 1 else MatchStatus.MULTIPLE_MATCHES
                return PillMatchResult(status=status, candidates=candidates, query_level_used=level.name)

            # total > MAX_CANDIDATES: 이 레벨은 너무 넓다.
            # 마지막(가장 넓은) 레벨까지 왔는데도 너무 많다면 더 좁힐 방법이 없다는 뜻 →
            # 재촬영/추가 정보 요청으로 넘긴다.
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
        각 레벨은 Vision AI가 실제로 값을 준 필드만 사용하고, 값이 없으면 그 레벨 자체가
        비어 있는 채로 넘어가 match()에서 자동으로 스킵된다.

        각인은 앞/뒤 신뢰도를 따로 평가한다. 두 면의 인식 난이도가 다를 수 있기 때문에
        (한쪽만 흐릿하거나, 애초에 한쪽엔 각인이 없는 경우가 흔함) 종합 신뢰도 하나로
        묶어서 켜고 끄지 않고, 신뢰 가능한 조합을 '구체적인 것부터' 여러 개 만들어둔다.
        예를 들어 앞/뒤 모두 신뢰 가능하면 [둘다] -> [앞만] -> [뒤만] 순으로 시도하게 되는데,
        이렇게 하면 "둘 다 넣었을 때는 0건이지만 사실 앞면 각인은 맞았던" 케이스도
        각인 정보를 아예 포기하지 않고 구제할 수 있다.
        """
        levels: list[_QueryLevel] = []

        for print_fields, level_name in self._reliable_print_combinations(v):
            params: dict[str, str] = dict(print_fields)
            self._add_if_present(params, "DRUG_SHAPE", v.drug_shape)
            self._add_if_present(params, "COLOR_CLASS1", v.color_class1)
            self._add_if_present(params, "COLOR_CLASS2", v.color_class2)
            levels.append(_QueryLevel(level_name, params))

        # 각인 제외 — 모양 + 색상 2개
        p_shape_color: dict[str, str] = {}
        self._add_if_present(p_shape_color, "DRUG_SHAPE", v.drug_shape)
        self._add_if_present(p_shape_color, "COLOR_CLASS1", v.color_class1)
        self._add_if_present(p_shape_color, "COLOR_CLASS2", v.color_class2)
        levels.append(_QueryLevel("SHAPE_AND_COLOR", p_shape_color))

        # 모양 + 주색상만
        p_shape_primary: dict[str, str] = {}
        self._add_if_present(p_shape_primary, "DRUG_SHAPE", v.drug_shape)
        self._add_if_present(p_shape_primary, "COLOR_CLASS1", v.color_class1)
        levels.append(_QueryLevel("SHAPE_AND_PRIMARY_COLOR", p_shape_primary))

        # 주색상만 — 가장 넓음. 여기서도 후보가 너무 많으면 TOO_MANY_CANDIDATES 로 종료.
        p_color_only: dict[str, str] = {}
        self._add_if_present(p_color_only, "COLOR_CLASS1", v.color_class1)
        levels.append(_QueryLevel("COLOR_ONLY", p_color_only))

        return levels

    def _reliable_print_combinations(
        self, v: VisionExtractionResult
    ) -> list[tuple[dict[str, str], str]]:
        """각인 조합 후보를 '구체적인 것부터' 순서대로 만든다.

        - 앞/뒤 둘 다 신뢰 가능 -> [(둘 다, "STRICT_WITH_PRINT_BOTH"),
                                    (앞만, "STRICT_WITH_PRINT_FRONT_ONLY"),
                                    (뒤만, "STRICT_WITH_PRINT_BACK_ONLY")]
        - 앞만 신뢰 가능        -> [(앞만, "STRICT_WITH_PRINT_FRONT_ONLY")]
        - 뒤만 신뢰 가능        -> [(뒤만, "STRICT_WITH_PRINT_BACK_ONLY")]
        - 둘 다 신뢰 불가       -> [] (각인 없이 바로 SHAPE_AND_COLOR 로)
        """
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

    @staticmethod
    def _to_candidates(response: PillApiResponse) -> list[PillCandidate]:
        if response.body is None or not response.body.items:
            return []
        return [PillCandidate.from_item(item) for item in response.body.items]
