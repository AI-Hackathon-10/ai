"""
Step 2: Gemini Vision 호출 -> 결과 검증 -> (불충분하면) 재질의 -> 최종 결과 반환 체인을
LangGraph StateGraph로 구조화한다.

이 체인이 하는 검증은 "신뢰도 숫자가 얼마나 높은가"가 아니라 "매칭 자체를 시도할 수
있는 최소한의 데이터가 있는가"이다. 신뢰도 임계값 판단은 여기서 다루지 않고
PillMatchingService 가 담당한다 — 이 그래프의 책임은 "쓸 수 있는 결과를 만드는 것"까지다.

google-genai 를 직접 import하지 않는다(app.vision.types 의 Protocol에만 의존). 그래서
테스트에서는 실제 Gemini 호출 없이 가짜 vision_call 함수를 주입해서 검증/재시도
라우팅 로직만 순수하게 확인할 수 있다.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.models import VisionExtractionResult
from app.vision.types import GeminiVisionCallFn, VisionGraphState

MAX_ATTEMPTS = 2


def _has_minimum_usable_data(raw: dict) -> bool:
    """color_class1 이나 drug_shape 둘 다 없으면 PillMatchingService 의 어떤 쿼리
    레벨도 만들 수 없다(가장 넓은 COLOR_ONLY 레벨조차 비어버림). 이 경우만 재질의
    대상으로 본다 — 각인 신뢰도가 낮은 것 자체는 정상 케이스이므로 재시도하지 않는다."""
    return bool(raw.get("color_class1")) or bool(raw.get("drug_shape"))


def build_vision_graph(vision_call: GeminiVisionCallFn):
    """vision_call 을 주입받는 형태로 그래프를 빌드한다.
    운영 코드는 app.vision.run 에서 app.vision.gemini_client.call_gemini_vision 을
    넘겨 만들고, 테스트는 가짜 함수를 넘겨 그래프 로직만 검증한다."""

    async def extract_node(state: VisionGraphState) -> dict:
        raw = await vision_call(
            state["front_image_bytes"],
            state.get("back_image_bytes"),
            state.get("retry_hint"),
            front_mime_type=state.get("front_image_mime_type", "image/jpeg"),
            back_mime_type=state.get("back_image_mime_type"),
        )
        return {
            "raw_output": raw,
            "attempt": state.get("attempt", 0) + 1,
        }

    def validate_node(state: VisionGraphState) -> dict:
        raw = state.get("raw_output")
        if raw and _has_minimum_usable_data(raw):
            return {}
        return {
            "retry_hint": (
                "이전 응답에서 색상과 모양을 모두 판단하지 못했습니다. 사진을 다시 "
                "자세히 살펴보고, 정말 아무것도 판단할 수 없는 경우가 아니라면 최소한 "
                "색상(color_class1)과 모양(drug_shape)은 반드시 채워주세요."
            )
        }

    def finalize_success_node(state: VisionGraphState) -> dict:
        raw = state["raw_output"]
        result = VisionExtractionResult(
            print_front=raw.get("print_front"),
            print_front_confidence=raw.get("print_front_confidence"),
            print_back=raw.get("print_back"),
            print_back_confidence=raw.get("print_back_confidence"),
            color_class1=raw.get("color_class1"),
            color_class2=raw.get("color_class2"),
            drug_shape=raw.get("drug_shape"),
            line_front=raw.get("line_front"),
            line_back=raw.get("line_back"),
            form_code_name=raw.get("form_code_name"),
            overall_confidence=raw.get("overall_confidence"),
        )
        return {"result": result, "failed": False}

    def finalize_failure_node(state: VisionGraphState) -> dict:
        # 재시도를 다 써도 색상/모양조차 못 뽑은 경우 — 매칭 자체를 시도할 수 없는 상태.
        # result 는 비워두고 failed=True 로 표시해서, 호출부(FastAPI)가 "재촬영 요청"으로
        # 안내하게 한다.
        return {"result": None, "failed": True}

    def route_after_validate(state: VisionGraphState) -> str:
        raw = state.get("raw_output")
        if raw and _has_minimum_usable_data(raw):
            return "success"
        if state.get("attempt", 0) < MAX_ATTEMPTS:
            return "retry"
        return "failure"

    graph = StateGraph(VisionGraphState)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("finalize_success", finalize_success_node)
    graph.add_node("finalize_failure", finalize_failure_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "success": "finalize_success",
            "retry": "extract",
            "failure": "finalize_failure",
        },
    )
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)

    return graph.compile()
