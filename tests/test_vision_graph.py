"""
실제 Gemini 호출 없이 LangGraph 체인(Step 2)의 라우팅 로직만 검증한다.
vision_call 함수를 가짜로 주입해서 "이런 원시 응답이 오면 이렇게 라우팅된다"를 확인한다.

재질의(재시도)는 없다 — Vision은 항상 한 번만 호출되고, color_class1/drug_shape 가
둘 다 없으면 그 자리에서 바로 판단 불가로 끝난다.
"""
from __future__ import annotations

import pytest

from app.vision.graph import build_vision_graph


def _raw(**overrides) -> dict:
    base = {
        "print_front": None,
        "print_back": None,
        "color_class1": None,
        "drug_shape": None,
        "score_line": None,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_색상과_모양이_모두_있으면_성공():
    call_count = 0

    async def fake_vision_call(front, back, **kwargs):
        nonlocal call_count
        call_count += 1
        return _raw(print_front="TY", color_class1="하양", drug_shape="장방형")

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke({"front_image_bytes": b"front", "back_image_bytes": b"back"})

    assert call_count == 1
    assert final_state["failed"] is False
    assert final_state["result"].color_class1 == "하양"


@pytest.mark.asyncio
async def test_색상_모양_모두_없으면_재시도없이_바로_실패():
    call_count = 0

    async def fake_vision_call(front, back, **kwargs):
        nonlocal call_count
        call_count += 1
        return _raw()  # 색상/모양 둘 다 비어있음

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke({"front_image_bytes": b"front", "back_image_bytes": None})

    assert call_count == 1
    assert final_state["failed"] is True
    assert final_state["result"] is None


@pytest.mark.asyncio
async def test_색상만_있어도_모양이_없어도_성공으로_인정된다():
    call_count = 0

    async def fake_vision_call(front, back, **kwargs):
        nonlocal call_count
        call_count += 1
        return _raw(color_class1="하양")  # drug_shape 는 없음

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke({"front_image_bytes": b"front", "back_image_bytes": None})

    assert call_count == 1
    assert final_state["failed"] is False
    assert final_state["result"].color_class1 == "하양"
    assert final_state["result"].drug_shape is None


@pytest.mark.asyncio
async def test_각인과_분할선도_결과에_그대로_담긴다():
    async def fake_vision_call(front, back, **kwargs):
        return _raw(
            print_front="TY", print_back="500", color_class1="하양", drug_shape="장방형", score_line=True
        )

    graph = build_vision_graph(fake_vision_call)
    final_state = await graph.ainvoke({"front_image_bytes": b"front", "back_image_bytes": b"back"})

    result = final_state["result"]
    assert result.print_front == "TY"
    assert result.print_back == "500"
    assert result.score_line is True
