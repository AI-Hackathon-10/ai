"""
pill_identification 테이블에서 Vision 추출 조건으로 후보를 조회한다.

각인(PRINT_FRONT/PRINT_BACK)은 공백·대소문자를 무시하고,
색/모양은 TRIM 한 값으로 비교한다. 후보가 너무 많은지 판단하려면
MAX_CANDIDATES+1 건만 가져오면 되므로 LIMIT 을 건다.

SCORE_LINE 은 테이블에 그 이름의 컬럼이 없다 — line_front/line_back 중 하나라도
비어있지 않으면 분할선이 있는 것으로 본다.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Optional

from app.models import PillApiItem

MAX_FIND_LIMIT = 11

FILTER_COLUMNS = {
    "PRINT_FRONT": "print_front",
    "PRINT_BACK": "print_back",
    "DRUG_SHAPE": "drug_shape",
    "COLOR_CLASS1": "color_class1",
    "COLOR_CLASS2": "color_class2",
}
_PRINT_FILTERS = {"PRINT_FRONT", "PRINT_BACK"}
_SCORE_LINE_KEY = "SCORE_LINE"
_HAS_SCORE_LINE_SQL = "(TRIM(IFNULL(line_front, '')) != '' OR TRIM(IFNULL(line_back, '')) != '')"

SELECT_COLUMNS = (
    "item_seq",
    "item_name",
    "entp_name",
    "chart",
    "item_image",
    "print_front",
    "print_back",
    "drug_shape",
    "color_class1",
    "color_class2",
    "line_front",
    "line_back",
    "form_code_name",
)


def build_find_query(
    filters: dict[str, str], limit: int = MAX_FIND_LIMIT
) -> tuple[Optional[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for key, expected in filters.items():
        if key == _SCORE_LINE_KEY:
            value = str(expected).strip().lower()
            if value not in ("true", "false"):
                continue
            clauses.append(_HAS_SCORE_LINE_SQL if value == "true" else f"NOT {_HAS_SCORE_LINE_SQL}")
            continue

        column = FILTER_COLUMNS.get(key)
        if column is None:
            continue
        value = str(expected).strip()
        if not value:
            continue
        if key in _PRINT_FILTERS:
            clauses.append(f"REPLACE(LOWER(IFNULL({column}, '')), ' ', '') = %s")
            params.append("".join(value.split()).casefold())
        else:
            clauses.append(f"TRIM({column}) = %s")
            params.append(value)

    if not clauses:
        return None, []

    sql = (
        f"SELECT {', '.join(SELECT_COLUMNS)} FROM pill_identification "
        f"WHERE {' AND '.join(clauses)} LIMIT %s"
    )
    params.append(limit)
    return sql, params


def row_to_item(row: dict[str, Any]) -> PillApiItem:
    return PillApiItem.model_validate(row)


class PillIdentificationRepository:
    def __init__(self, connect: Callable[[], Any]):
        self._connect = connect

    def find(self, filters: dict[str, str], limit: int = MAX_FIND_LIMIT) -> list[PillApiItem]:
        sql, params = build_find_query(filters, limit)
        if sql is None:
            return []

        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
        finally:
            conn.close()

        return [row_to_item(row) for row in rows]


class MysqlPillCatalogClient:
    """PillMatchingService 가 쓰는 find() 인터페이스."""

    def __init__(self, repository: PillIdentificationRepository):
        self._repository = repository

    async def find(
        self, filters: dict[str, str], limit: int = MAX_FIND_LIMIT
    ) -> list[PillApiItem]:
        return await asyncio.to_thread(self._repository.find, filters, limit)
