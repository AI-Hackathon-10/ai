"""
Vision 프로바이더 팩토리.

환경변수 VISION_PROVIDER 값에 따라 Gemini / OpenAI 클라이언트 함수를 반환한다.
새 프로바이더를 추가하려면 여기에 분기를 하나 더 넣으면 된다.
"""
from __future__ import annotations

from app.config import settings
from app.vision.types import VisionCallFn


def get_vision_call() -> VisionCallFn:
    """settings.vision_provider에 따라 적절한 Vision 호출 함수를 반환한다."""
    provider = settings.vision_provider.lower()

    if provider == "openai":
        from app.vision.openai_client import call_openai_vision

        return call_openai_vision
    else:  # 기본값: gemini
        from app.vision.gemini_client import call_gemini_vision

        return call_gemini_vision
