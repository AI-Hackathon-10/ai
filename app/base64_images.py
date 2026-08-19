"""
base64로 인코딩된 이미지를 다루기 위한 유틸리티.

두 방향을 모두 지원한다.

1. 디코딩 (decode_base64_image / decode_base64_image_or_400): 프론트에서 이미지를
   multipart/form-data 파일 업로드가 아니라 JSON body 안에 base64 문자열로 넣어 보내는
   경우 사용한다. data URI("data:image/png;base64,...")와 순수 base64 문자열을 모두
   허용한다.
2. 인코딩 (encode_image_to_data_uri): /identify 는 multipart/form-data로 파일을 받지만,
   S3 저장이나 프론트 미리보기, 로그 등 base64 원본 문자열이 필요한 다른 용도를 위해
   업로드받은 bytes를 base64 data URI로도 함께 만들어둔다.
"""
from __future__ import annotations

import base64
import binascii
from typing import Optional, Tuple

from fastapi import HTTPException

_DATA_URI_PREFIX = "data:"


class InvalidBase64ImageError(ValueError):
    """base64 디코딩 실패 시 발생."""


def decode_base64_image(
    raw: str,
    fallback_mime_type: str = "image/jpeg",
    field_name: str = "image",
) -> Tuple[bytes, str]:
    """base64 문자열(또는 data URI) -> (이미지 bytes, mime_type).

    잘못된 base64 값이 들어오면 InvalidBase64ImageError 를 던진다. 라우터에서는
    이를 잡아 400 Bad Request로 변환해야 한다(사용자 입력 오류이지 서버 오류가
    아니므로 500이 아니라 400이 맞다).
    """
    mime_type = fallback_mime_type
    payload = raw

    if raw.startswith(_DATA_URI_PREFIX):
        # "data:image/png;base64,AAAA..." -> header="image/png", 뒤가 실제 데이터
        try:
            header, payload = raw.split(",", 1)
        except ValueError as e:
            raise InvalidBase64ImageError(
                f"{field_name}: data URI 형식이 올바르지 않습니다 (콤마로 구분된 base64 데이터가 없음)."
            ) from e
        # header 예: "data:image/png;base64"
        header_body = header[len(_DATA_URI_PREFIX):]
        if ";" in header_body:
            mime_type = header_body.split(";", 1)[0] or fallback_mime_type
        elif header_body:
            mime_type = header_body

    try:
        # 공백/줄바꿈이 섞여 들어오는 경우가 흔해서 미리 제거한다.
        image_bytes = base64.b64decode(payload.strip(), validate=False)
    except (binascii.Error, ValueError) as e:
        raise InvalidBase64ImageError(f"{field_name}: base64 디코딩에 실패했습니다 ({e}).") from e

    if not image_bytes:
        raise InvalidBase64ImageError(f"{field_name}: 디코딩된 이미지 데이터가 비어 있습니다.")

    return image_bytes, mime_type


def encode_image_to_data_uri(raw: bytes, mime_type: str) -> str:
    """이미지 bytes -> "data:<mime_type>;base64,<...>" 형태의 data URI 문자열.

    multipart로 업로드받은 원본 파일을 base64로도 같이 갖고 있어야 할 때(S3 저장,
    프론트 미리보기 응답, 로그/디버깅 등) 사용한다.
    """
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def decode_base64_image_or_400(
    raw: Optional[str],
    fallback_mime_type: str = "image/jpeg",
    field_name: str = "image",
) -> Optional[Tuple[bytes, str]]:
    """decode_base64_image 의 FastAPI 라우터용 래퍼.

    raw 가 None/빈 문자열이면 None을 반환하고(선택적 필드용), 디코딩 실패 시
    500이 아니라 400 HTTPException을 던진다.
    """
    if not raw:
        return None
    try:
        return decode_base64_image(raw, fallback_mime_type=fallback_mime_type, field_name=field_name)
    except InvalidBase64ImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
