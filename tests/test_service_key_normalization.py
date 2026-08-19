"""
data.go.kr가 주는 "일반 인증키(Encoding)"를 그대로 넣어도, httpx가 요청 시점에
다시 인코딩해서 이중 인코딩(%2B -> %252B)이 나는 문제가 없는지 검증한다.
"""
from app.pill_identification_api_client import _normalize_service_key


def test_이미_인코딩된_키는_디코딩해서_원본으로_되돌린다():
    encoded = "xrDvRipGmxED7sIegsTUJ7aZJpbBodUgp2XZYxU5%2BHdVDg27q1VpufPlqaQJvQ0k6%2Fe%2FnM4OfS1ISbajnHtPtQ%3D%3D"
    normalized = _normalize_service_key(encoded)

    assert "%" not in normalized
    assert normalized.endswith("==")  # base64 스타일 padding이 살아있어야 정상 디코딩된 것


def test_이미_디코딩된_키는_그대로_둔다():
    decoded = "xrDvRipGmxED7sIegsTUJ7aZJpbBodUgp2XZYxU5+HdVDg27q1VpufPlqaQJvQ0k6/e/nM4OfS1ISbajnHtPtQ=="
    normalized = _normalize_service_key(decoded)

    assert normalized == decoded
