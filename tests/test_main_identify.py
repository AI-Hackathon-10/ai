"""
app/main.py 의 POST /identify 엔드포인트를 TestClient로 검증한다. 이미지는 base64 JSON
body로 받는다(app/base64_images.py 의 decode_base64_image_or_400 경유). Vision/매칭/
상세조회는 모두 monkeypatch로 대체해서 실제 API 키 없이도 요청 파싱/디코딩/직렬화 로직만
순수하게 확인한다.
"""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.models import MatchStatus, PillCandidate, PillMatchResult, VisionExtractionResult
from app.pipeline import IdentifyFromDbOutcome, PillIdentificationOutcome
from app.vision.gemini_client import GeminiVisionError

_PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png-bytes"
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")


@pytest.fixture
def client():
    return TestClient(main_mod.app)


def test_data_uri로_보내면_mime_type을_자동으로_인식해서_전달한다(client, monkeypatch):
    async def fake_identify_pill(front_bytes, back_bytes, matching_service, detail_service, **kwargs):
        assert front_bytes == _PNG_BYTES
        assert kwargs["front_mime_type"] == "image/png"
        assert back_bytes is None
        return PillIdentificationOutcome(vision_failed=False)

    monkeypatch.setattr(main_mod, "identify_pill", fake_identify_pill)

    response = client.post("/identify", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 200
    body = response.json()
    assert body["vision_failed"] is False


def test_순수_base64와_mime_type을_같이_보내면_그대로_전달된다(client, monkeypatch):
    async def fake_identify_pill(front_bytes, back_bytes, matching_service, detail_service, **kwargs):
        assert kwargs["front_mime_type"] == "image/png"
        assert kwargs["back_mime_type"] == "image/png"
        return PillIdentificationOutcome(vision_failed=False)

    monkeypatch.setattr(main_mod, "identify_pill", fake_identify_pill)

    response = client.post(
        "/identify",
        json={
            "front_image": _PNG_B64,
            "front_mime_type": "image/png",
            "back_image": _PNG_B64,
            "back_mime_type": "image/png",
        },
    )

    assert response.status_code == 200


def test_front_image가_빈문자열이면_400을_반환한다(client):
    response = client.post("/identify", json={"front_image": ""})
    assert response.status_code == 400


def test_front_image가_잘못된_base64면_400을_반환한다(client):
    response = client.post("/identify", json={"front_image": "이건-base64가-아님!!!"})
    assert response.status_code == 400
    assert "front_image" in response.json()["detail"]


def test_vision_failed면_그대로_응답에_반영된다(client, monkeypatch):
    async def fake_identify_pill(*args, **kwargs):
        return PillIdentificationOutcome(vision_failed=True)

    monkeypatch.setattr(main_mod, "identify_pill", fake_identify_pill)

    response = client.post("/identify", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 200
    body = response.json()
    assert body["vision_failed"] is True
    assert body["match_result"] is None


def test_vision_extract는_매칭없이_vision_결과만_그대로_반환한다(client, monkeypatch):
    async def fake_run_vision_extraction(front_bytes, back_bytes, **kwargs):
        assert front_bytes == _PNG_BYTES
        assert kwargs["front_mime_type"] == "image/png"
        return VisionExtractionResult(
            print_front="TYLENOL",
            print_front_confidence=0.9,
            print_back="500",
            print_back_confidence=0.85,
            color_class1="하양",
            drug_shape="장방형",
            form_code_name="정제",
            overall_confidence=0.88,
        )

    monkeypatch.setattr(main_mod, "run_vision_extraction", fake_run_vision_extraction)

    response = client.post("/vision/extract", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 200
    body = response.json()
    assert body["vision_failed"] is False
    assert body["result"]["print_front"] == "TYLENOL"
    assert body["result"]["print_back"] == "500"
    assert body["result"]["color_class1"] == "하양"


def test_vision_extract는_추출_실패하면_vision_failed_True를_반환한다(client, monkeypatch):
    async def fake_run_vision_extraction(*args, **kwargs):
        return None

    monkeypatch.setattr(main_mod, "run_vision_extraction", fake_run_vision_extraction)

    response = client.post("/vision/extract", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 200
    body = response.json()
    assert body["vision_failed"] is True
    assert body["result"] is None


def test_vision_extract도_front_image가_빈문자열이면_400을_반환한다(client):
    response = client.post("/vision/extract", json={"front_image": ""})
    assert response.status_code == 400


def test_identify_db는_vision_추출값으로_테이블_매칭_결과를_반환한다(client, monkeypatch):
    vision = VisionExtractionResult(
        print_front="YH",
        print_front_confidence=0.9,
        color_class1="노랑",
        drug_shape="원형",
        overall_confidence=0.88,
    )

    async def fake_identify_from_db(front_bytes, back_bytes, matching_service, **kwargs):
        assert front_bytes == _PNG_BYTES
        assert matching_service is main_mod._db_matching_service
        return IdentifyFromDbOutcome(
            vision_failed=False,
            vision_result=vision,
            match_result=PillMatchResult(
                status=MatchStatus.SINGLE_MATCH,
                candidates=[
                    PillCandidate(
                        item_seq="200808877",
                        item_name="페라트라정2.5밀리그램(레트로졸)",
                        print_front="YH",
                        drug_shape="원형",
                        color_class1="노랑",
                    )
                ],
                query_level_used="STRICT_WITH_PRINT_FRONT_ONLY",
            ),
        )

    monkeypatch.setattr(main_mod, "identify_from_db", fake_identify_from_db)

    response = client.post("/identify/db", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 200
    body = response.json()
    assert body["vision_failed"] is False
    assert body["vision_result"]["print_front"] == "YH"
    assert body["match_result"]["status"] == "SINGLE_MATCH"
    assert body["match_result"]["candidates"][0]["item_seq"] == "200808877"
    assert body["match_result"]["candidates"][0]["print_front"] == "YH"


def test_identify_db는_추출_실패하면_매칭없이_vision_failed를_반환한다(client, monkeypatch):
    async def fake_identify_from_db(*args, **kwargs):
        return IdentifyFromDbOutcome(vision_failed=True)

    monkeypatch.setattr(main_mod, "identify_from_db", fake_identify_from_db)

    response = client.post("/identify/db", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 200
    body = response.json()
    assert body["vision_failed"] is True
    assert body["vision_result"] is None
    assert body["match_result"] is None


def test_identify_db도_front_image가_빈문자열이면_400을_반환한다(client):
    response = client.post("/identify/db", json={"front_image": ""})
    assert response.status_code == 400


def test_vision_extract는_Gemini_장애시_500이_아니라_503_JSON을_반환한다(client, monkeypatch):
    async def fake_run_vision_extraction(*args, **kwargs):
        raise GeminiVisionError(
            "Gemini Vision 호출 실패: 503 UNAVAILABLE. This model is currently experiencing high demand."
        )

    monkeypatch.setattr(main_mod, "run_vision_extraction", fake_run_vision_extraction)

    response = client.post("/vision/extract", json={"front_image": f"data:image/png;base64,{_PNG_B64}"})

    assert response.status_code == 503
    body = response.json()
    assert "Gemini" in body["detail"]
    assert "high demand" in body["detail"]


def test_vision_extract는_swagger_placeholder_뒷면값은_무시한다(client, monkeypatch):
    async def fake_run_vision_extraction(front_bytes, back_bytes, **kwargs):
        assert back_bytes is None
        assert not kwargs.get("back_mime_type") or kwargs["back_mime_type"].startswith("image/")
        return VisionExtractionResult(color_class1="하양", drug_shape="원형", overall_confidence=0.7)

    monkeypatch.setattr(main_mod, "run_vision_extraction", fake_run_vision_extraction)

    response = client.post(
        "/vision/extract",
        json={
            "front_image": f"data:image/png;base64,{_PNG_B64}",
            "front_mime_type": "image/jpeg",
            "back_image": "string",
            "back_mime_type": "string",
        },
    )

    assert response.status_code == 200
    assert response.json()["vision_failed"] is False
