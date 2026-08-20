"""
Step 2: Gemini Vision 호출 -> 최소 사용 가능 여부 판단 -> 최종 결과 반환 체인을
LangGraph StateGraph로 구조화한다.

재질의(재시도)는 하지 않는다 — Vision은 딱 한 번만 호출되고, 판단하지 못한 필드는
confidence 점수 없이 그 자리에서 null로 받는다. color_class1/drug_shape 가 둘 다
없으면(=매칭 조건을 전혀 만들 수 없으면) 그 즉시 판단 불가로 끝낸다.

google-genai 를 직접 import하지 않는다(app.vision.types 의 Protocol에만 의존). 그래서
테스트에서는 실제 Gemini 호출 없이 가짜 vision_call 함수를 주입해서 라우팅 로직만
순수하게 확인할 수 있다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.models import VisionExtractionResult
from app.vision.types import VisionCallFn, VisionGraphState


def _has_minimum_usable_data(raw: dict) -> bool:
    """color_class1 이나 drug_shape 둘 다 없으면 PillMatchingService 가 걸 조건이
    하나도 없다. 이 경우만 판단 불가로 본다 — 각인이 없는 것 자체는 정상 케이스다."""
    return bool(raw.get("color_class1")) or bool(raw.get("drug_shape"))


def build_vision_graph(vision_call: VisionCallFn):
    """vision_call 을 주입받는 형태로 그래프를 빌드한다.
    운영 코드는 app.vision.run 에서 app.vision.factory.get_vision_call() 이 반환한
    함수를 넘겨 만들고, 테스트는 가짜 함수를 넘겨 그래프 로직만 검증한다."""

    async def extract_node(state: VisionGraphState) -> dict:
        raw = await vision_call(
            state["front_image_bytes"],
            state.get("back_image_bytes"),
            front_mime_type=state.get("front_image_mime_type", "image/jpeg"),
            back_mime_type=state.get("back_image_mime_type"),
        )
        return {"raw_output": raw}

    def finalize_success_node(state: VisionGraphState) -> dict:
        raw = state["raw_output"]
        result = VisionExtractionResult(
            print_front=raw.get("print_front"),
            print_back=raw.get("print_back"),
            color_class1=raw.get("color_class1"),
            drug_shape=raw.get("drug_shape"),
            score_line=raw.get("score_line"),
        )
        return {"result": result, "failed": False}

    def finalize_failure_node(state: VisionGraphState) -> dict:
        # 색상/모양조차 못 뽑은 경우 — 매칭 자체를 시도할 수 없는 상태. result 는
        # 비워두고 failed=True 로 표시해서, 호출부(FastAPI)가 "재촬영 요청"으로
        # 안내하게 한다.
        return {"result": None, "failed": True}

    def route_after_extract(state: VisionGraphState) -> str:
        raw = state.get("raw_output")
        if raw and _has_minimum_usable_data(raw):
            return "success"
        return "failure"

    graph = StateGraph(VisionGraphState)
    graph.add_node("extract", extract_node)
    graph.add_node("finalize_success", finalize_success_node)
    graph.add_node("finalize_failure", finalize_failure_node)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "success": "finalize_success",
            "failure": "finalize_failure",
        },
    )
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)

    return graph.compile()
