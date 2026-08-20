"""
Gemini Vision 호출 전담 함수 (Step 2: 사진 -> 검색 조건 추출).

⚠️ SDK 호출 시그니처 확인 필요: google-genai SDK는 버전에 따라 API 표면이 계속
바뀐다. 아래는 `client.models.generate_content()` + `GenerateContentConfig(
response_mime_type=..., response_schema=...)` 패턴으로, 구조화된 출력을 요청하는
방법 중 오랫동안 안정적으로 문서화되어 온 방식이다. 실제 사용 시점의 google-genai
최신 문서(https://googleapis.github.io/python-genai/)에서 정확한 호출 방식을 한 번
확인한 뒤 이 함수 내부만 교체하면 된다. 그래프(graph.py)는 이 함수의 시그니처
(app.vision.types.GeminiVisionCallFn)만 지키면 되므로 내부 구현이 바뀌어도 영향받지
않는다.
"""
from __future__ import annotations

import json
from typing import Optional

from google import genai
from google.genai import types
from langsmith import traceable

from app.vision.schema import PILL_VISION_RESPONSE_SCHEMA, SYSTEM_PROMPT, build_user_prompt

MODEL_NAME = "gemini-3.7-flash"


class GeminiVisionError(Exception):
    """Gemini 호출/응답 파싱 관련 오류."""


def _redact_image_bytes(inputs: dict) -> dict:
    """LangSmith 트레이스에 이미지 원본 바이트(base64로 부풀려짐)가 그대로 올라가지
    않도록 크기 정보만 남기고 지운다."""
    redacted = dict(inputs)
    for key in ("front_image_bytes", "back_image_bytes"):
        value = redacted.get(key)
        if isinstance(value, (bytes, bytearray)):
            redacted[key] = f"<{len(value)} bytes>"
    return redacted


@traceable(run_type="llm", name="gemini_vision_call", process_inputs=_redact_image_bytes)
async def call_gemini_vision(
    front_image_bytes: bytes,
    back_image_bytes: Optional[bytes],
    retry_hint: Optional[str],
    front_mime_type: str = "image/jpeg",
    back_mime_type: Optional[str] = None,
) -> dict:
    """알약 앞/뒤 사진(bytes) -> Vision AI가 추출한 원시 dict.

    반환값은 app/vision/schema.py 의 PILL_VISION_RESPONSE_SCHEMA 형태를 따른다.
    이 함수는 파싱까지만 책임지고, 값의 최소 유효성(색상/모양 존재 여부)은
    graph.py 의 validate 노드가 판단한다.

    ⚠️ mime_type을 실제 업로드된 파일과 다르게(예: PNG인데 "image/jpeg") 보내면
    Gemini가 이미지를 제대로 못 읽거나 요청 자체가 실패할 수 있다. 반드시 업로드
    시점의 실제 mime type(base64 data URI에 포함된 값, 또는 클라이언트가 별도로 알려준
    실제 파일 형식)을 그대로 전달할 것.
    """
    client = genai.Client()

    parts = [types.Part.from_bytes(data=front_image_bytes, mime_type=front_mime_type)]
    if back_image_bytes:
        parts.append(
            types.Part.from_bytes(data=back_image_bytes, mime_type=back_mime_type or front_mime_type)
        )
    parts.append(types.Part.from_text(text=build_user_prompt(retry_hint)))

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=PILL_VISION_RESPONSE_SCHEMA,
            ),
        )
    except Exception as e:  # noqa: BLE001 - SDK 예외 타입이 버전마다 달라 광범위하게 감쌈
        raise GeminiVisionError(f"Gemini Vision 호출 실패: {e}") from e

    try:
        return json.loads(response.text)
    except (ValueError, AttributeError) as e:
        raise GeminiVisionError(f"Gemini Vision 응답 파싱 실패: {e}") from e
