"""
FastAPI 라우터에서 호출할 진입점. 그래프를 한 번 컴파일해서 모듈 레벨에 캐싱해둔다
(요청마다 새로 빌드할 필요 없음).
"""
from __future__ import annotations

from typing import Optional

from app.models import VisionExtractionResult
from app.vision.gemini_client import call_gemini_vision
from app.vision.graph import build_vision_graph

_compiled_graph = build_vision_graph(call_gemini_vision)


async def run_vision_extraction(
    front_image_bytes: bytes,
    back_image_bytes: Optional[bytes] = None,
    front_mime_type: str = "image/jpeg",
    back_mime_type: Optional[str] = None,
) -> Optional[VisionExtractionResult]:
    """알약 사진(들) -> VisionExtractionResult.

    재시도를 다 써도 색상/모양을 못 뽑으면 None을 반환한다. 호출부(FastAPI 라우터)는
    이를 "재촬영 요청" 응답으로 변환해야 한다.

    ⚠️ mime_type은 반드시 실제 업로드된 파일의 Content-Type과 일치해야 한다
    (예: PNG 업로드인데 "image/jpeg"를 고정으로 넘기면 Gemini 호출이 실패하거나
    이미지를 제대로 못 읽는다).
    """
    final_state = await _compiled_graph.ainvoke(
        {
            "front_image_bytes": front_image_bytes,
            "back_image_bytes": back_image_bytes,
            "front_image_mime_type": front_mime_type,
            "back_image_mime_type": back_mime_type,
            "attempt": 0,
        }
    )
    if final_state.get("failed"):
        return None
    return final_state["result"]
