"""낱알식별 API 요청변수는 공식 스펙(소문자)만 나가야 한다."""
from __future__ import annotations

from app.pill_identification_api_client import to_official_request_params


def test_대문자_응답필드명은_요청에서_제거한다():
    params = to_official_request_params(
        {
            "PRINT_FRONT": "TYLENOL",
            "PRINT_BACK": "500",
            "DRUG_SHAPE": "장방형",
            "COLOR_CLASS1": "하양",
        }
    )
    assert params == {}


def test_공식_요청변수는_소문자로_정규화한다():
    params = to_official_request_params(
        {
            "ITEM_NAME": "타이레놀정500",
            "ENTP_NAME": "한국존슨앤드존슨판매",
            "ITEM_SEQ": "202106092",
            "item_seq": "202106092",
        }
    )
    assert params == {
        "item_name": "타이레놀정500",
        "entp_name": "한국존슨앤드존슨판매",
        "item_seq": "202106092",
    }


def test_빈값_null은_보내지_않는다():
    params = to_official_request_params(
        {
            "item_name": "  ",
            "edi_code": "null",
            "bizrno": "6828500435",
            "img_regist_ts": "20100326",
        }
    )
    assert params == {
        "bizrno": "6828500435",
        "img_regist_ts": "20100326",
    }
