"""
pill_identification 테이블 조회 SQL/매핑을 실제 MySQL 없이 검증한다.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.pill_identification_repository import (
    MAX_FIND_LIMIT,
    PillIdentificationRepository,
    build_find_query,
    row_to_item,
)


def test_각인_필터는_공백과_대소문자를_무시하는_SQL을_만든다():
    sql, params = build_find_query(
        {"PRINT_FRONT": "TY 500", "DRUG_SHAPE": "장방형", "COLOR_CLASS1": "하양"},
        limit=11,
    )

    assert "FROM pill_identification" in sql
    assert "REPLACE(LOWER(IFNULL(print_front, '')), ' ', '') = %s" in sql
    assert "TRIM(drug_shape) = %s" in sql
    assert "TRIM(color_class1) = %s" in sql
    assert "LIMIT %s" in sql
    assert params == ["ty500", "장방형", "하양", 11]


def test_빈_필터면_조회하지_않는다():
    sql, params = build_find_query({}, limit=11)

    assert sql is None
    assert params == []


def test_SCORE_LINE_true면_분할선_있는_행만_거르는_조건을_만든다():
    sql, params = build_find_query({"SCORE_LINE": "true"}, limit=11)

    assert "(TRIM(IFNULL(line_front, '')) != '' OR TRIM(IFNULL(line_back, '')) != '')" in sql
    assert "NOT" not in sql
    assert params == [11]


def test_SCORE_LINE_false면_NOT_조건을_만든다():
    sql, params = build_find_query({"SCORE_LINE": "false"}, limit=11)

    assert "NOT (TRIM(IFNULL(line_front, '')) != '' OR TRIM(IFNULL(line_back, '')) != '')" in sql
    assert params == [11]


def test_row_to_item은_테이블_컬럼을_PillApiItem으로_변환한다():
    item = row_to_item(
        {
            "item_seq": "200808877",
            "item_name": "페라트라정2.5밀리그램(레트로졸)",
            "entp_name": "업체",
            "chart": "노란색 원형 정제",
            "item_image": "http://example.com/p.png",
            "print_front": "YH",
            "print_back": "LT",
            "drug_shape": "원형",
            "color_class1": "노랑",
            "color_class2": None,
            "line_front": None,
            "line_back": None,
            "form_code_name": "나정",
        }
    )

    assert item.item_seq == "200808877"
    assert item.print_front == "YH"
    assert item.drug_shape == "원형"
    assert item.color_class1 == "노랑"


def test_find는_SQL을_실행하고_매핑된_항목을_돌려준다():
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        {
            "item_seq": "1",
            "item_name": "약A",
            "entp_name": "업체",
            "chart": None,
            "item_image": None,
            "print_front": "TY",
            "print_back": "500",
            "drug_shape": "장방형",
            "color_class1": "하양",
            "color_class2": None,
            "line_front": None,
            "line_back": None,
            "form_code_name": "나정",
        }
    ]
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False

    conn = MagicMock()
    conn.cursor.return_value = cursor

    repo = PillIdentificationRepository(connect=lambda: conn)
    items = repo.find({"PRINT_FRONT": "TY", "COLOR_CLASS1": "하양"})

    assert len(items) == 1
    assert items[0].item_seq == "1"
    assert items[0].print_front == "TY"
    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args.args
    assert "print_front" in sql
    assert params[-1] == MAX_FIND_LIMIT
    conn.close.assert_called_once()
