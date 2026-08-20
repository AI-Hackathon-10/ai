"""
OpenAI Vision 호출 전담 함수 (Step 2: 사진 -> 검색 조건 추출).

Gemini 클라이언트(gemini_client.py)와 동일한 VisionCallFn 시그니처를 지키므로,
factory.py 에서 프로바이더 설정에 따라 교체 주입할 수 있다.
"""
from __future__ import annotations

import base64
import json
from typing import Optional

from langsmith import traceable
from openai import AsyncOpenAI

from app.config import settings
from app.vision.schema import PILL_VISION_RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt

_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIVisionError(Exception):
    """OpenAI 호출/응답 파싱 관련 오류."""


def _redact_image_bytes(inputs: dict) -> dict:
    """LangSmith 트레이스에 이미지 원본 바이트가 그대로 올라가지 않도록
    크기 정보만 남기고 지운다."""
    redacted = dict(inputs)
    for key in ("front_image_bytes", "back_image_bytes"):
        value = redacted.get(key)
        if isinstance(value, (bytes, bytearray)):
            redacted[key] = f"<{len(value)} bytes>"
    return redacted


def _image_to_data_url(image_bytes: bytes, mime_type: str) -> str:
    """이미지 바이트를 OpenAI Vision API가 받는 data URL 형식으로 변환한다."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


@traceable(run_type="llm", name="openai_vision_call", process_inputs=_redact_image_bytes)
async def call_openai_vision(
    front_image_bytes: bytes,
    back_image_bytes: Optional[bytes],
    front_mime_type: str = "image/jpeg",
    back_mime_type: Optional[str] = None,
) -> dict:
    """알약 앞/뒤 사진(bytes) -> Vision AI가 1차로 추출한 원시 dict.

    반환값은 app/vision/schema.py 의 PILL_VISION_RESPONSE_SCHEMA 형태를 따른다.
    """
    client = AsyncOpenAI()

    image_content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": _image_to_data_url(front_image_bytes, front_mime_type)},
        },
    ]
    if back_image_bytes:
        image_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(
                        back_image_bytes, back_mime_type or front_mime_type
                    )
                },
            },
        )
    image_content.append({"type": "text", "text": build_user_prompt()})

    model_name = settings.vision_model or _DEFAULT_OPENAI_MODEL

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": image_content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "pill_vision_response",
                    "strict": True,
                    "schema": PILL_VISION_RESPONSE_SCHEMA,
                },
            },
        )
    except Exception as e:  # noqa: BLE001
        raise OpenAIVisionError(f"OpenAI Vision 호출 실패: {e}") from e

    try:
        return json.loads(response.choices[0].message.content)
    except (ValueError, AttributeError, IndexError) as e:
        raise OpenAIVisionError(f"OpenAI Vision 응답 파싱 실패: {e}") from e
