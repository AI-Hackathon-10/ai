"""
실제 Gemini 호출 없이 LangGraph 체인(Step 2)의 검증/재시도 라우팅 로직만 검증한다.
vision_call 함수를 가짜로 주입해서 "이런 원시 응답이 오면 이렇게 라우팅된다"를 확인한다.
"""
from __future__ import annotations

import pytest

from app.vision.graph import MAX_ATTEMPTS, build_vision_graph


def _raw(**overrides) -> dict:
    base = {
        "print_front": None,
        "print_front_confidence": 0.0,
        "print_back": None,
        "print_back_confidence": 0.0,
        "color_class1": None,
        "color_class2": None,
        "drug_shape": None,
        "line_front": None,
        "line_back": None,
        "form_code_name": None,
        "overall_confidence": 0.0,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_첫_시도에서_색상과_모양이_모두_있으면_바로_성공():
    call_count = 0

    async def fake_vision_call(front, back, retry_hint, **kwargs):
        nonlocal call_count
        call_count += 1
        return _raw(
            print_front="TY",
            print_front_confidence=0.9,
            color_class1="하양",
            drug_shape="장방형",
            form_code_name="정제",
            overall_confidence=0.9,
        )

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke(
        {"front_image_bytes": b"front", "back_image_bytes": b"back", "attempt": 0}
    )

    assert call_count == 1
    assert final_state["failed"] is False
    assert final_state["result"].color_class1 == "하양"


@pytest.mark.asyncio
async def test_첫_시도에서_색상_모양_모두_비어있으면_재시도후_성공():
    call_count = 0

    async def fake_vision_call(front, back, retry_hint, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert retry_hint is None  # 첫 시도는 재시도 힌트가 없어야 함
            return _raw(overall_confidence=0.1)
        assert retry_hint is not None  # 2번째 시도부터는 재시도 힌트가 채워져 있어야 함
        return _raw(color_class1="회색", drug_shape="기타", form_code_name="정제", overall_confidence=0.5)

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke(
        {"front_image_bytes": b"front", "back_image_bytes": None, "attempt": 0}
    )

    assert call_count == 2
    assert final_state["failed"] is False
    assert final_state["result"].drug_shape == "기타"


@pytest.mark.asyncio
async def test_최대_재시도를_다_써도_실패하면_failed_True_이고_result는_None():
    call_count = 0

    async def fake_vision_call(front, back, retry_hint, **kwargs):
        nonlocal call_count
        call_count += 1
        return _raw()  # 색상/모양 계속 비어있음

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke(
        {"front_image_bytes": b"front", "back_image_bytes": None, "attempt": 0}
    )

    assert call_count == MAX_ATTEMPTS
    assert final_state["failed"] is True
    assert final_state["result"] is None


@pytest.mark.asyncio
async def test_색상만_있어도_모양이_없어도_사용가능한_데이터로_인정되어_재시도하지_않는다():
    call_count = 0

    async def fake_vision_call(front, back, retry_hint, **kwargs):
        nonlocal call_count
        call_count += 1
        return _raw(color_class1="하양", overall_confidence=0.3)  # drug_shape 는 없음

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke(
        {"front_image_bytes": b"front", "back_image_bytes": None, "attempt": 0}
    )

    assert call_count == 1
    assert final_state["failed"] is False
    assert final_state["result"].color_class1 == "하양"
    assert final_state["result"].drug_shape is None
