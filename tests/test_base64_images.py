"""
app/base64_images.py 의 base64 디코딩 유틸리티를 검증한다.
"""
from __future__ import annotations

import base64

import pytest
from fastapi import HTTPException

from app.base64_images import decode_base64_image, decode_base64_image_or_400, InvalidBase64ImageError

_RAW_BYTES = b"fake-png-bytes"
_RAW_B64 = base64.b64encode(_RAW_BYTES).decode("ascii")


def test_data_uri에서_mime_type을_자동으로_추출한다():
    data_uri = f"data:image/png;base64,{_RAW_B64}"
    image_bytes, mime_type = decode_base64_image(data_uri, fallback_mime_type="image/jpeg")
    assert image_bytes == _RAW_BYTES
    assert mime_type == "image/png"


def test_순수_base64는_fallback_mime_type을_사용한다():
    image_bytes, mime_type = decode_base64_image(_RAW_B64, fallback_mime_type="image/png")
    assert image_bytes == _RAW_BYTES
    assert mime_type == "image/png"


def test_앞뒤_공백이나_줄바꿈이_섞여도_디코딩된다():
    noisy = f"  {_RAW_B64}\n"
    image_bytes, _ = decode_base64_image(noisy, fallback_mime_type="image/jpeg")
    assert image_bytes == _RAW_BYTES


def test_잘못된_base64는_InvalidBase64ImageError를_던진다():
    with pytest.raises(InvalidBase64ImageError):
        decode_base64_image("이건-base64가-아님!!!", fallback_mime_type="image/jpeg")


def test_or_400_래퍼는_None이나_빈문자열이면_None을_반환한다():
    assert decode_base64_image_or_400(None) is None
    assert decode_base64_image_or_400("") is None


def test_or_400_래퍼는_디코딩_실패시_400_HTTPException을_던진다():
    with pytest.raises(HTTPException) as exc_info:
        decode_base64_image_or_400("이건-base64가-아님!!!", field_name="front_image")
    assert exc_info.value.status_code == 400
    assert "front_image" in exc_info.value.detail
