"""
그래프(graph.py)와 실제 Gemini 호출(gemini_client.py) 사이의 계약(타입)만 정의한다.

이 파일이 google-genai 를 import하지 않는 게 중요하다. graph.py 는 이 파일에만
의존하게 만들어서, 그래프의 검증/재시도 로직을 테스트할 때 google-genai 패키지나
실제 API 키 없이도 순수하게 가짜 함수를 주입해서 검증할 수 있게 한다.
"""
from __future__ import annotations

from typing import Optional, Protocol, TypedDict

from app.models import VisionExtractionResult


class GeminiVisionCallFn(Protocol):
    """extract 노드가 의존하는 콜 시그니처.

    운영 코드에서는 app.vision.gemini_client.call_gemini_vision 을 넘기고,
    테스트에서는 이 시그니처만 지키는 가짜 함수를 넘겨서 그래프 로직만 검증한다.
    """

    async def __call__(
        self,
        front_image_bytes: bytes,
        back_image_bytes: Optional[bytes],
        front_mime_type: str = "image/jpeg",
        back_mime_type: Optional[str] = None,
    ) -> dict:
        ...


class VisionGraphState(TypedDict, total=False):
    front_image_bytes: bytes
    back_image_bytes: Optional[bytes]
    front_image_mime_type: str
    back_image_mime_type: Optional[str]
    raw_output: Optional[dict]
    result: Optional[VisionExtractionResult]
    failed: bool
